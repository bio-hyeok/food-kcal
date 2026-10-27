"""ONNX 분류 모델 불러오기와 추론.

전처리는 학습 때(torchvision)와 똑같아야 정확도가 유지된다. 값은 model_meta.json 에서 읽는다.
  RGB → 짧은 변을 resize_short_side 로 Resize(BILINEAR) → center_crop 으로 CenterCrop → 0~1 → mean/std 정규화
torchvision 과 같은지는 ml/scripts/check_server_preprocess.py 로 확인한다.
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


class ModelFilesError(RuntimeError):
    """모델 폴더에 필요한 파일이 없거나 내용이 맞지 않음."""


@dataclass
class Candidate:
    rank_no: int
    food_id: int
    confidence: float


def resize_short_side(im: Image.Image, size: int) -> Image.Image:
    """torchvision.transforms.Resize(int) 와 같은 크기 계산 (긴 변은 버림)."""
    w, h = im.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_w, new_h = int(size * w / h), size
    if (new_w, new_h) == (w, h):
        return im
    return im.resize((new_w, new_h), Image.BILINEAR)


def center_crop(im: Image.Image, size: int) -> Image.Image:
    """torchvision.transforms.CenterCrop 과 같은 위치 계산 (반올림)."""
    w, h = im.size
    top = int(round((h - size) / 2.0))
    left = int(round((w - size) / 2.0))
    return im.crop((left, top, left + size, top + size))


class Classifier:
    def __init__(self, model_dir: str | Path):
        md = Path(model_dir)
        meta_path = md / "model_meta.json"
        if not meta_path.exists():
            raise ModelFilesError(f"model_meta.json 이 없습니다: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        onnx_path = md / self.meta["onnx"]
        if not onnx_path.exists():
            raise ModelFilesError(f"ONNX 파일이 없습니다: {onnx_path}")

        inp = self.meta["input"]
        self.input_name = inp["name"]
        self.resize = int(inp["resize_short_side"])
        self.crop = int(inp["center_crop"])
        self.mean = np.array(inp["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(inp["std"], dtype=np.float32).reshape(3, 1, 1)
        self.idx_to_food_id = {int(k): int(v) for k, v in self.meta["idx_to_food_id"].items()}
        num_classes = int(self.meta["output"]["num_classes"])
        if len(self.idx_to_food_id) != num_classes:
            raise ModelFilesError(f"클래스 수가 맞지 않습니다: 표 {len(self.idx_to_food_id)}개, 모델 {num_classes}개")

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @property
    def food_ids(self) -> list[int]:
        return sorted(self.idx_to_food_id.values())

    def preprocess(self, im: Image.Image) -> np.ndarray:
        im = center_crop(resize_short_side(im.convert("RGB"), self.resize), self.crop)
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self.mean) / self.std)[None]

    @staticmethod
    def open_image(data: bytes) -> tuple[Image.Image, str | None]:
        """바이트 → (PIL 이미지, 형식). 휴대폰 사진은 EXIF 회전 정보대로 돌린다.

        회전한 이미지는 format 이 비므로 형식은 돌리기 전에 읽어 둔다.
        """
        im = Image.open(io.BytesIO(data))
        fmt = im.format
        im.load()
        return ImageOps.exif_transpose(im), fmt

    def predict(self, im: Image.Image, top_k: int) -> list[Candidate]:
        logits = self.session.run(None, {self.input_name: self.preprocess(im)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)[:top_k]
        return [Candidate(rank_no=i + 1, food_id=self.idx_to_food_id[int(k)], confidence=float(p[k]))
                for i, k in enumerate(order)]
