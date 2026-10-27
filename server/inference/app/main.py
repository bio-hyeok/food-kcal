"""추론 서버 (위키 전체 흐름 문서의 "추론 서버 (분류 모델)").

API 서버가 내부에서 호출한다. 앱이 직접 부르지 않는다.
  GET  /health   상태와 모델 정보
  POST /predict  사진 1장 → 후보 Top-N + 신뢰도 + status(SUCCESS/FAILED)

응답 형식은 위키 API 문서 1절의 공통 규칙({"success": ..., "data"/"error": ...})을 따른다.
FAILED 여도 candidates 는 내려준다. API 서버가 RECOGNITION_ITEMS 에 저장하고, 앱에는 빈 items 로 보낸다
(위키 API 문서 4절 GET /recognitions/{resultId}).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .classifier import Classifier
from .config import load_settings

ALLOWED_FORMATS = {"JPEG", "PNG"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델은 서버가 켜질 때 한 번만 불러온다. 파일이 없으면 서버가 뜨지 않는다.
    app.state.settings = load_settings()
    app.state.classifier = Classifier(app.state.settings.model_dir)
    yield


app = FastAPI(title="food-kcal inference", lifespan=lifespan)


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
    }}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    s = request.app.state.settings
    clf: Classifier = request.app.state.classifier

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

    # 추론은 CPU 작업이라 이벤트 루프를 막지 않게 스레드에서 돌린다
    candidates = await run_in_threadpool(clf.predict, im, s.top_k)
    status = "SUCCESS" if candidates[0].confidence >= s.min_confidence else "FAILED"
    return {"success": True, "data": {
        "status": status,
        "model": model_info(request),
        "minConfidence": s.min_confidence,
        "candidates": [{"rankNo": c.rank_no, "foodId": c.food_id, "confidence": round(c.confidence, 4)}
                       for c in candidates],
    }}
