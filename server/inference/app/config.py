"""환경 변수 설정. 값은 .env 나 실행 환경에서 넣는다 (저장소 루트 .env.example 참고)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_dir: str
    model_name: str
    model_version: str
    min_confidence: float
    top_k: int
    max_image_bytes: int
    # 한 사진에 여러 음식(POST /detect). DETECT_MODEL_DIR 이 없으면 /detect 만 꺼진다.
    detect_model_dir: str | None = None
    detect_min_confidence: float = 0.25
    detect_iou: float = 0.7
    # 실제 상차림은 반찬이 10개 넘게 나와 15 로 둔다
    detect_max_items: int = 15
    # 음식 상자의 짧은 변이 이보다 작으면 small=true (가까이서 다시 찍도록 안내). 128 은 val 크기별 정확도로 정했다
    detect_small_px: int = 128
    # 그릇 찾기(COCO 사전학습 YOLO 의 bowl). 없으면 음식 검출 모델만 쓴다 (app/pipeline.py)
    detect_bowl_model_dir: str | None = None
    detect_bowl_min_confidence: float = 0.15
    # 그릇 상자도 1순위를 미리 고를지. 실제 상차림 20장에서 미리 고른 것 중 틀린 것이 맞은 것보다 많아 기본은 끈다
    detect_bowl_auto_select: bool = False


def load_settings() -> Settings:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        raise RuntimeError("MODEL_DIR 환경 변수가 필요합니다 (classifier.onnx, model_meta.json 이 있는 폴더)")
    s = Settings(
        model_dir=model_dir,
        model_name=os.environ.get("MODEL_NAME", "efficientnet_b0_food"),
        model_version=os.environ.get("MODEL_VERSION", "v0.1.0"),
        # 0.30 은 v3 모델(288px)로 정한 값이다. 모델을 바꾸면 ml/scripts/confidence_threshold.py 로 다시 정한다.
        min_confidence=float(os.environ.get("RECOGNITION_MIN_CONFIDENCE", "0.30")),
        top_k=int(os.environ.get("RECOGNITION_TOP_K", "5")),
        max_image_bytes=int(os.environ.get("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
        detect_model_dir=os.environ.get("DETECT_MODEL_DIR") or None,
        # 0.25, 0.7 은 Ultralytics 기본값 (test 평가도 이 값으로 했다, ml/README.md)
        detect_min_confidence=float(os.environ.get("DETECT_MIN_CONFIDENCE", "0.25")),
        detect_iou=float(os.environ.get("DETECT_IOU", "0.7")),
        detect_max_items=int(os.environ.get("DETECT_MAX_ITEMS", "15")),
        detect_small_px=int(os.environ.get("DETECT_SMALL_PX", "128")),
        detect_bowl_model_dir=os.environ.get("DETECT_BOWL_MODEL_DIR") or None,
        # 0.15 는 실제 상차림 20장에서 0.25 보다 찾은 비율이 높았다 (아는 음식 88% → 94%)
        detect_bowl_min_confidence=float(os.environ.get("DETECT_BOWL_MIN_CONFIDENCE", "0.15")),
        detect_bowl_auto_select=os.environ.get("DETECT_BOWL_AUTO_SELECT", "false").lower() in ("1", "true", "yes"),
    )
    if not 0.0 <= s.min_confidence <= 1.0:
        raise RuntimeError("RECOGNITION_MIN_CONFIDENCE 는 0~1 사이여야 합니다")
    if s.top_k < 1:
        raise RuntimeError("RECOGNITION_TOP_K 는 1 이상이어야 합니다")
    if not (0.0 <= s.detect_min_confidence <= 1.0 and 0.0 < s.detect_iou <= 1.0):
        raise RuntimeError("DETECT_MIN_CONFIDENCE 는 0~1, DETECT_IOU 는 0 초과 1 이하여야 합니다")
    if s.detect_max_items < 1:
        raise RuntimeError("DETECT_MAX_ITEMS 는 1 이상이어야 합니다")
    if not 0.0 <= s.detect_bowl_min_confidence <= 1.0:
        raise RuntimeError("DETECT_BOWL_MIN_CONFIDENCE 는 0~1 사이여야 합니다")
    return s
