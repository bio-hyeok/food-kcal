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


DET_SIZE = 640
# 가짜 검출 모델이 항상 내는 상자 (입력 640 기준 cx, cy, w, h, 신뢰도)
#   128×64 사진은 640×320 으로 줄어 위아래에 160 씩 여백이 생긴다 → A 는 사진 왼쪽 절반, B 는 오른쪽 절반
DET_BOXES = [
    (160, 320, 320, 320, 0.9),   # A
    (480, 320, 320, 320, 0.8),   # B
    (165, 320, 320, 320, 0.5),   # A 와 거의 겹침 → NMS 로 지워진다
    (320, 320, 40, 40, 0.1),     # 신뢰도가 낮아 지워진다
]


# 그릇 찾기용 가짜 COCO 모델: 클래스 3개(cup, bowl, dining table) 중 bowl(1)만 쓴다. (cx, cy, w, h, 점수 3개)
BOWL_BOXES = [
    (160, 320, 300, 300, 0.0, 0.9, 0.0),   # 음식 A 와 같은 자리 → 겹쳐서 빠진다
    (320, 200, 100, 80, 0.0, 0.7, 0.0),    # 새 그릇 → 더해진다 (source=bowl)
    (480, 420, 60, 60, 0.95, 0.05, 0.0),   # 컵 → keep_classes 가 아니라 무시
    (320, 320, 640, 330, 0.0, 0.6, 0.0),   # 사진 거의 전체 → 너무 커서 버린다
]


def build_detector(path, boxes=None):
    """입력과 상관없이 boxes(기본 DET_BOXES)를 (1, 4+클래스 수, N) 으로 내는 모델. 입력 × 0 을 더해 입력과 연결해 둔다."""
    boxes = DET_BOXES if boxes is None else boxes
    const = np.array(boxes, dtype=np.float32).T[None]                   # (1, 4+클래스 수, N)
    c = helper.make_tensor("C", TensorProto.FLOAT, const.shape, const.flatten().tolist())
    zero = helper.make_tensor("Z", TensorProto.FLOAT, [], [0.0])
    shape = helper.make_tensor("S", TensorProto.INT64, [3], [1, 1, 1])
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=1),
        helper.make_node("Reshape", ["m", "S"], ["m3"]),
        helper.make_node("Mul", ["m3", "Z"], ["z"]),
        helper.make_node("Add", ["z", "C"], ["output0"]),
    ]
    graph = helper.make_graph(
        nodes, "tiny_det",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, DET_SIZE, DET_SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(const.shape))],
        initializer=[c, zero, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)


@pytest.fixture
def detect_dir(tmp_path):
    d = tmp_path / "det"
    d.mkdir()
    build_detector(d / "detector.onnx")
    meta = {"onnx": "detector.onnx", "input": {"name": "images", "size": DET_SIZE, "pad_value": 114},
            "output": {"name": "output0", "format": "cx,cy,w,h,conf"}, "classes": ["food"]}
    (d / "detector_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture
def bowl_dir(tmp_path):
    d = tmp_path / "bowl"
    d.mkdir()
    build_detector(d / "detector.onnx", BOWL_BOXES)
    meta = {"onnx": "detector.onnx", "input": {"name": "images", "size": DET_SIZE, "pad_value": 114},
            "output": {"name": "output0", "format": "cx,cy,w,h,class_scores"},
            "classes": ["cup", "bowl", "dining table"], "keep_classes": [1]}
    (d / "detector_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture
def bowl_client(model_dir, detect_dir, bowl_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DETECT_MODEL_DIR", str(detect_dir))
    monkeypatch.setenv("DETECT_BOWL_MODEL_DIR", str(bowl_dir))
    monkeypatch.setenv("RECOGNITION_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("RECOGNITION_TOP_K", "3")
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def detect_client(model_dir, detect_dir, monkeypatch):
    monkeypatch.delenv("DETECT_BOWL_MODEL_DIR", raising=False)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("DETECT_MODEL_DIR", str(detect_dir))
    monkeypatch.setenv("RECOGNITION_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("RECOGNITION_TOP_K", "3")
    from app.main import app
    with TestClient(app) as c:
        yield c


def two_color_image(size=(128, 64)):
    """왼쪽 절반 빨강(foodId 5), 오른쪽 절반 초록(foodId 12)."""
    im = Image.new("RGB", size, (0, 255, 0))
    im.paste((255, 0, 0), (0, 0, size[0] // 2, size[1]))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.delenv("DETECT_MODEL_DIR", raising=False)
    monkeypatch.delenv("DETECT_BOWL_MODEL_DIR", raising=False)
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
