"""서버 전처리·추론이 학습 때(torchvision)와 같은지 실제 모델로 확인한다.

torch/torchvision 이 필요하므로 ml/.venv 로 실행한다 (서버 자체는 torch 없이 돈다).
    ml\\.venv\\Scripts\\python.exe server\\inference\\scripts\\check_against_torchvision.py ^
        --model-dir C:\\food-data\\models_v3 --data C:\\food-data\\processed_full

확인하는 것
1. 같은 사진의 전처리 결과(입력 텐서) 최대 차이 — 학습 쪽 torchvision 파이프라인과 비교
2. 서버 Classifier 로 test 전체를 돌린 top-1 / top-5 — export_onnx.py 결과와 같아야 한다
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classifier import Classifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--compare", type=int, default=200, help="전처리를 비교할 사진 수")
    a = ap.parse_args()

    clf = Classifier(a.model_dir)
    inp = clf.meta["input"]
    tv = transforms.Compose([
        transforms.Resize(inp["resize_short_side"]), transforms.CenterCrop(inp["center_crop"]),
        transforms.ToTensor(), transforms.Normalize(inp["mean"], inp["std"]),
    ])
    test = Path(a.data) / "test"
    files = sorted((int(d.name), p) for d in test.iterdir() if d.is_dir() for p in d.glob("*.jpg"))

    step = max(1, len(files) // a.compare)
    max_diff = 0.0
    for _, p in files[::step][: a.compare]:
        im = Image.open(p)
        max_diff = max(max_diff, float(np.abs(clf.preprocess(im) - tv(im.convert("RGB")).numpy()[None]).max()))
    print(f"전처리 비교 {min(a.compare, len(files[::step]))}장: 입력 최대 차이 {max_diff:.2e}")

    n = top1 = top5 = 0
    for true_id, p in files:
        c = clf.predict(Image.open(p), 5)
        n += 1
        top1 += c[0].food_id == true_id
        top5 += true_id in {x.food_id for x in c}
    print(f"서버 Classifier test {n}장: top-1 {top1 / n:.2%}, top-5 {top5 / n:.2%}")
    if max_diff > 1e-4:
        raise SystemExit("전처리가 학습 때와 다릅니다.")


if __name__ == "__main__":
    main()
