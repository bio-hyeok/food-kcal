"""ONNX 음식 위치 찾기(검출) 모델. 사진 한 장 → 음식 상자 여러 개.

종류는 가리지 않고 "food" 한 클래스만 찾는다. 찾은 상자는 잘라서 분류 모델(classifier.py)에 넣는다.
여러 클래스 모델(예: COCO 사전학습 YOLO)도 쓸 수 있다. detector_meta.json 의 keep_classes(클래스 번호)만 남긴다
(실제 상차림에서 "그릇(bowl)" 찾기, server/inference/README.md).
전처리는 학습 때(Ultralytics)와 같게 한다. 값은 detector_meta.json 에서 읽는다 (ml/scripts/export_detector.py).
  RGB → 비율 유지하며 size 에 맞게 줄이고 가운데 놓기(남는 곳은 pad_value 회색) → 0~1 → NCHW
출력 (1, 4+클래스 수, N) = 가운데 x, y, 너비, 높이(입력 size 기준 픽셀), 클래스별 신뢰도
→ keep_classes 중 가장 높은 신뢰도로 거르고 겹친 상자(NMS)를 지운다.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from .classifier import ModelFilesError


@dataclass
class Box:
    x0: float  # 원본 사진 기준 픽셀
    y0: float
    x1: float
    y1: float
    confidence: float


def letterbox(im: Image.Image, size: int, pad_value: int) -> tuple[Image.Image, float, int, int]:
    """Ultralytics LetterBox(auto=False, center=True) 와 같은 크기·여백 계산. (이미지, 배율, 왼쪽 여백, 위 여백)"""
    w, h = im.size
    r = min(size / w, size / h)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, top = round(dw - 0.1), round(dh - 0.1)
    if (new_w, new_h) != (w, h):
        im = im.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad_value,) * 3)
    canvas.paste(im, (left, top))
    return canvas, r, left, top


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """상자 a (4,) 와 상자들 b (n, 4) 의 IoU. 상자는 x0, y0, x1, y1."""
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float, max_det: int) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        order = order[1:][iou(boxes[i], boxes[order[1:]]) < iou_th]
    return keep


class Detector:
    def __init__(self, model_dir: str | Path):
        md = Path(model_dir)
        meta_path = md / "detector_meta.json"
        if not meta_path.exists():
            raise ModelFilesError(f"detector_meta.json 이 없습니다: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        onnx_path = md / self.meta["onnx"]
        if not onnx_path.exists():
            raise ModelFilesError(f"ONNX 파일이 없습니다: {onnx_path}")
        self.input_name = self.meta["input"]["name"]
        self.size = int(self.meta["input"]["size"])
        self.pad_value = int(self.meta["input"].get("pad_value", 114))
        self.keep_classes = [int(c) for c in self.meta.get("keep_classes", [0])]
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def detect(self, im: Image.Image, min_confidence: float, iou_th: float, max_det: int) -> list[Box]:
        """신뢰도 높은 순서의 상자 목록. 못 찾으면 빈 목록."""
        im = im.convert("RGB")
        x, r, left, top = letterbox(im, self.size, self.pad_value)
        x = np.asarray(x, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        pred = self.session.run(None, {self.input_name: x})[0][0].T    # (N, 4 + 클래스 수)
        conf = pred[:, [4 + c for c in self.keep_classes]].max(axis=1)
        ok = conf >= min_confidence
        pred, conf = pred[ok], conf[ok]
        if not len(pred):
            return []
        cx, cy, bw, bh = pred[:, :4].T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = nms(xyxy, conf, iou_th, max_det)
        w, h = im.size
        out = []
        for i in keep:
            x0, y0, x1, y1 = xyxy[i]
            out.append(Box(
                x0=float(np.clip((x0 - left) / r, 0, w)), y0=float(np.clip((y0 - top) / r, 0, h)),
                x1=float(np.clip((x1 - left) / r, 0, w)), y1=float(np.clip((y1 - top) / r, 0, h)),
                confidence=float(conf[i]),
            ))
        return [b for b in out if b.x1 - b.x0 >= 2 and b.y1 - b.y0 >= 2]


def crop_box(im: Image.Image, box: Box, margin: float = 0.05) -> Image.Image:
    """상자 둘레에 margin 만큼 여유를 두고 자른다. 분류 학습 데이터(prepare_classifier_data.py)와 같은 5%."""
    bw, bh = box.x1 - box.x0, box.y1 - box.y0
    return im.crop((max(0, int(box.x0 - bw * margin)), max(0, int(box.y0 - bh * margin)),
                    min(im.width, int(box.x1 + bw * margin)), min(im.height, int(box.y1 + bh * margin))))
