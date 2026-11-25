"""검출 모델(best.pt)을 추론 서버용 폴더로 내보낸다: detector.onnx + detector_meta.json

    python scripts/export_detector.py --weights C:\\food-data\\detect_runs\\yolo26n_food\\weights\\best.pt --out C:\\food-data\\detector_v1

서버(server/inference/app/detector.py)는 detector_meta.json 값대로 전처리한다.
  RGB → 비율 유지하며 size 에 맞게 줄이고 가운데 놓기(남는 곳은 pad_value 회색) → 0~1 → NCHW
출력 (1, 5, N) = 상자 가운데 x, y, 너비, 높이(입력 size 기준 픽셀), 신뢰도. 겹친 상자 지우기(NMS)는 서버가 한다.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, simplify=True))
    shutil.copy(onnx_path, out / "detector.onnx")

    s = ort.InferenceSession(str(out / "detector.onnx"), providers=["CPUExecutionProvider"])
    inp, outp = s.get_inputs()[0], s.get_outputs()[0]
    if list(inp.shape) != [1, 3, a.imgsz, a.imgsz] or outp.shape[1] != 5:
        raise SystemExit(f"예상과 다른 ONNX 형식: 입력 {inp.shape}, 출력 {outp.shape}")
    meta = {
        "onnx": "detector.onnx",
        "input": {"name": inp.name, "size": a.imgsz, "pad_value": 114},
        "output": {"name": outp.name, "format": "cx,cy,w,h,conf"},
        "classes": ["food"],
        "weights_sha1": hashlib.sha1(Path(a.weights).read_bytes()).hexdigest(),
    }
    (out / "detector_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
