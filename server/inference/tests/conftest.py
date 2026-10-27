"""시험용 작은 ONNX 모델. 실제 모델(저장소에 없음) 없이 서버 동작을 확인한다.

모델: 이미지 채널 평균(GlobalAveragePool) × 10 → 클래스 3개 (R, G, B 순서로 foodId 5, 12, 100)
- 빨간 사진 → foodId 5 를 거의 100% 신뢰도로 고른다
- 회색 사진 → 정규화하면 0 에 가까워 세 클래스가 약 1/3 씩 → 기준값 0.5 에서 FAILED
"""
import io
import json

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper
from PIL import Image

SIZE = 32
FOOD_IDS = {0: 5, 1: 12, 2: 100}


def build_model(path):
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 3], (np.eye(3) * 10).flatten().tolist())
    b = helper.make_tensor("B", TensorProto.FLOAT, [3], [0.0, 0.0, 0.0])
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture
def model_dir(tmp_path):
    build_model(tmp_path / "classifier.onnx")
    meta = {
        "onnx": "classifier.onnx",
        "input": {"name": "image", "size": SIZE, "resize_short_side": 37, "center_crop": SIZE,
                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"name": "logits", "num_classes": 3},
        "idx_to_food_id": {str(k): v for k, v in FOOD_IDS.items()},
    }
    (tmp_path / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("RECOGNITION_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("RECOGNITION_TOP_K", "3")
    monkeypatch.setenv("MAX_IMAGE_BYTES", str(200_000))
    from app.main import app
    with TestClient(app) as c:
        yield c


def image_bytes(color, fmt="JPEG", size=(64, 48), exif_orientation=None):
    im = Image.new("RGB", size, color)
    buf = io.BytesIO()
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        im.save(buf, fmt, exif=exif)
    else:
        im.save(buf, fmt)
    return buf.getvalue()
