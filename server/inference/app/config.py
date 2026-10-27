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
    )
    if not 0.0 <= s.min_confidence <= 1.0:
        raise RuntimeError("RECOGNITION_MIN_CONFIDENCE 는 0~1 사이여야 합니다")
    if s.top_k < 1:
        raise RuntimeError("RECOGNITION_TOP_K 는 1 이상이어야 합니다")
    return s
