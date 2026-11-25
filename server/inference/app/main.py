"""추론 서버 (위키 전체 흐름 문서의 "추론 서버 (분류 모델)").

API 서버가 내부에서 호출한다. 앱이 직접 부르지 않는다 (과제 시연용으로는 앱이 직접 부른다).
  GET  /health   상태와 모델 정보
  POST /predict  사진 1장 → 후보 Top-N + 신뢰도 + status(SUCCESS/FAILED)
  POST /detect   사진 1장 → 음식 상자마다 후보 Top-N + status (한 사진에 여러 음식. DETECT_MODEL_DIR 필요,
                 DETECT_BOWL_MODEL_DIR 가 있으면 그릇 찾기도 함께 쓴다 → app/pipeline.py)

응답 형식은 위키 API 문서 1절의 공통 규칙({"success": ..., "data"/"error": ...})을 따른다.
FAILED 여도 candidates 는 내려준다. API 서버가 RECOGNITION_ITEMS 에 저장하고, 앱에는 빈 items 로 보낸다
(위키 API 문서 4절 GET /recognitions/{resultId}).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .classifier import Classifier
from .config import load_settings
from .detector import Detector
from .pipeline import detect_items

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 서버가 켜질 때 한 번만 불러온다. 파일이 없으면 서버가 뜨지 않는다.
    app.state.settings = load_settings()
    app.state.classifier = Classifier(app.state.settings.model_dir)
    dmd = app.state.settings.detect_model_dir
    app.state.detector = Detector(dmd) if dmd else None
    bmd = app.state.settings.detect_bowl_model_dir
    app.state.bowl_detector = Detector(bmd) if bmd and dmd else None
    yield


app = FastAPI(title="food-kcal inference", lifespan=lifespan)

# 과제 시연용: Expo 웹(브라우저)에서 직접 부를 수 있게 허용한다. 휴대폰 앱(Expo Go)은 CORS 와 상관없다.
# API 서버가 생기면 앱은 이 서버를 직접 부르지 않으므로 CORS_ORIGINS 를 비워 끈다.
_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:8081,http://127.0.0.1:8081").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, e: ApiError):
    return JSONResponse(status_code=e.status, content={"success": False, "error": {"code": e.code, "message": e.message}})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, e: RequestValidationError):
    return JSONResponse(status_code=400, content={"success": False, "error": {
        "code": "INVALID_REQUEST", "message": "이미지 파일(file 필드)이 필요합니다."}})


def model_info(request: Request) -> dict:
    s = request.app.state.settings
    return {"name": s.model_name, "version": s.model_version}


@app.get("/health")
def health(request: Request):
    clf = request.app.state.classifier
    s = request.app.state.settings
    return {"success": True, "data": {
        "status": "ok",
        "model": model_info(request),
        "numClasses": len(clf.idx_to_food_id),
        "inputSize": clf.crop,
        "minConfidence": s.min_confidence,
        "topK": s.top_k,
        "detect": {"enabled": request.app.state.detector is not None, "minConfidence": s.detect_min_confidence,
                   "maxItems": s.detect_max_items, "bowl": request.app.state.bowl_detector is not None},
    }}


async def read_image(file: UploadFile, s, clf: Classifier) -> Image.Image:
    """업로드한 파일 → PIL 이미지. 크기·형식이 맞지 않으면 ApiError."""
    data = await file.read(s.max_image_bytes + 1)
    if not data:
        raise ApiError(400, "INVALID_REQUEST", "빈 파일입니다.")
    if len(data) > s.max_image_bytes:
        raise ApiError(413, "IMAGE_TOO_LARGE", f"이미지는 {s.max_image_bytes // (1024 * 1024)}MB 이하만 받습니다.")
    try:
        im, fmt = clf.open_image(data)
    except Image.DecompressionBombError:
        raise ApiError(413, "IMAGE_TOO_LARGE", "이미지 해상도가 너무 큽니다.")
    except (UnidentifiedImageError, OSError):
        raise ApiError(415, "UNSUPPORTED_IMAGE", "JPEG 또는 PNG 이미지만 받습니다.")
    if fmt not in ALLOWED_FORMATS:
        raise ApiError(415, "UNSUPPORTED_IMAGE", "JPEG 또는 PNG 이미지만 받습니다.")
    return im


def candidates_json(candidates) -> list[dict]:
    return [{"rankNo": c.rank_no, "foodId": c.food_id, "confidence": round(c.confidence, 4)} for c in candidates]


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    s = request.app.state.settings
    clf: Classifier = request.app.state.classifier
    im = await read_image(file, s, clf)

    # 추론은 CPU 작업이라 이벤트 루프를 막지 않게 스레드에서 돌린다
    candidates = await run_in_threadpool(clf.predict, im, s.top_k)
    status = "SUCCESS" if candidates[0].confidence >= s.min_confidence else "FAILED"
    return {"success": True, "data": {
        "status": status,
        "model": model_info(request),
        "minConfidence": s.min_confidence,
        "candidates": candidates_json(candidates),
    }}


@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):
    """음식 상자를 찾아 상자마다 분류한다. 상자를 못 찾으면 items 는 빈 배열 (음식이 아닌 사진 포함).

    box 는 사진(EXIF 회전 반영) 대비 0~1 비율. 앱은 사진 위에 그대로 그릴 수 있다.
    small 은 음식이 작게 찍혀(상자의 짧은 변 < DETECT_SMALL_PX) 정확도가 떨어질 수 있다는 표시다.
    source 는 상자를 찾은 모델: "food"(음식 검출) 또는 "bowl"(그릇 찾기). 그릇 상자는 기본으로 FAILED (후보만 보여 준다).
    """
    s = request.app.state.settings
    det: Detector | None = request.app.state.detector
    if det is None:
        raise ApiError(503, "DETECTOR_NOT_AVAILABLE", "검출 모델이 설정되지 않았습니다 (DETECT_MODEL_DIR).")
    clf: Classifier = request.app.state.classifier
    im = await read_image(file, s, clf)

    results = await run_in_threadpool(detect_items, im, s, clf, det, request.app.state.bowl_detector)
    w, h = im.size

    def item_json(i, it):
        b = it.box
        return {
            "itemNo": i,
            "box": {"x0": round(b.x0 / w, 4), "y0": round(b.y0 / h, 4), "x1": round(b.x1 / w, 4), "y1": round(b.y1 / h, 4)},
            "detConfidence": round(b.confidence, 4),
            "cropShortSide": round(min(b.x1 - b.x0, b.y1 - b.y0)),
            "small": min(b.x1 - b.x0, b.y1 - b.y0) < s.detect_small_px,
            "source": it.source,
            "status": it.status,
            "candidates": candidates_json(it.candidates),
        }

    items = [item_json(i, it) for i, it in enumerate(results, 1)]
    return {"success": True, "data": {
        "model": model_info(request),
        "minConfidence": s.min_confidence,
        "imageWidth": w,
        "imageHeight": h,
        "items": items,
    }}
