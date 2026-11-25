"""/detect 의 작은 설정을 정한다: 자르는 여유(margin), 좌우 뒤집어 한 번 더 보기(TTA), 너무 작은 음식 기준.

값은 val 로 고르고 test 로 확인한다 (test 로 고르면 점수가 부풀려진다).
  1) margin × TTA: 검출 상자(신뢰도 1위)를 잘라 분류한 top-1 / top-5 / 칼로리 오차 / 분류 시간
  2) 잘라 낸 음식의 크기별 정확도: 사진을 줄여 상자의 짧은 변이 N px 이 되게 하고 top-1 을 잰다
     (여러 음식 사진에서는 음식 하나가 차지하는 부분이 작아진다)

    .venv\\Scripts\\python.exe scripts\\tune_detect.py --model-dir C:\\food-data\\models_v3 --detect-model-dir C:\\food-data\\detector_v1 \\
        --detect-data C:\\food-data\\detect --split val
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classifier import Classifier  # noqa: E402
from app.detector import Detector, crop_box  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20)
SIDES = (48, 64, 96, 128, 160, 224, 320)


def probs(clf: Classifier, im: Image.Image, tta: bool) -> np.ndarray:
    x = clf.preprocess(im)
    if tta:
        x = np.concatenate([x, x[:, :, :, ::-1]], axis=0)       # 좌우 뒤집은 사진을 같은 배치로
    logits = clf.session.run(None, {clf.input_name: np.ascontiguousarray(x)})[0]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--detect-model-dir", required=True)
    ap.add_argument("--detect-data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--margins", default=",".join(map(str, MARGINS)))
    ap.add_argument("--skip-sizes", action="store_true")
    a = ap.parse_args()

    clf, det = Classifier(a.model_dir), Detector(a.detect_model_dir)
    id2idx = {v: k for k, v in clf.idx_to_food_id.items()}
    foods = {f["foodId"]: f for f in json.loads((ROOT / "app" / "src" / "foods.json").read_text(encoding="utf-8"))}
    kcal = np.array([foods[clf.idx_to_food_id[i]]["nutrition"]["kcal"] for i in range(len(clf.idx_to_food_id))])
    root = Path(a.detect_data)
    rows = [r for r in csv.DictReader(open(root / "index.csv", encoding="utf-8")) if r["split"] == a.split]
    margins = [float(m) for m in a.margins.split(",")]

    # 1) margin × TTA
    stats = {(m, t): {"top1": 0, "top5": 0, "kcal": 0.0, "sec": 0.0} for m in margins for t in (False, True)}
    images = []
    for i, r in enumerate(rows, 1):
        im = Image.open(root / "images" / a.split / (r["stem"] + ".jpg")).convert("RGB")
        truth = id2idx[int(r["food_id"])]
        boxes = det.detect(im, 0.25, 0.7, 6)
        gt = tuple(map(int, r["boxes"].split("|")[0].split()))
        images.append((im, truth, gt))
        if not boxes:
            continue
        for m in margins:
            c = crop_box(im, boxes[0], m)
            for t in (False, True):
                t0 = time.perf_counter()
                p = probs(clf, c, t)
                s = stats[(m, t)]
                s["sec"] += time.perf_counter() - t0
                order = np.argsort(-p)
                s["top1"] += order[0] == truth
                s["top5"] += truth in order[:5]
                s["kcal"] += abs(kcal[order[0]] - kcal[truth])
        if i % 200 == 0:
            print(f"{i}/{len(rows)}", flush=True)
    n = len(rows)
    grid = [{"margin": m, "tta": t, "top1": round(s["top1"] / n, 4), "top5": round(s["top5"] / n, 4),
             "kcal_err": round(s["kcal"] / n, 1), "cls_ms": round(s["sec"] / n * 1000, 1)}
            for (m, t), s in stats.items()]
    print("\n margin  TTA   top1    top5   kcal_err  분류ms")
    for g in grid:
        print(f" {g['margin']:.2f}   {'켬' if g['tta'] else '끔'}   {g['top1']:.4f}  {g['top5']:.4f}  {g['kcal_err']:7.1f}  {g['cls_ms']:6.1f}")

    # 2) 잘라 낸 음식의 크기별 정확도 (라벨 상자 기준, margin 0.05, TTA 끔)
    sizes = []
    if not a.skip_sizes:
        for side in SIDES:
            hit = 0
            for im, truth, (x0, y0, x1, y1) in images:
                s = side / min(x1 - x0, y1 - y0)
                if s >= 1:      # 이미 그보다 작은 음식은 원본 그대로
                    small, box = im, (x0, y0, x1, y1)
                else:
                    small = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
                    box = tuple(round(v * s) for v in (x0, y0, x1, y1))
                bw, bh = box[2] - box[0], box[3] - box[1]
                c = small.crop((max(0, int(box[0] - bw * .05)), max(0, int(box[1] - bh * .05)),
                                min(small.width, int(box[2] + bw * .05)), min(small.height, int(box[3] + bh * .05))))
                hit += int(np.argmax(probs(clf, c, False))) == truth
            sizes.append({"crop_short_side": side, "top1": round(hit / n, 4)})
            print(f" 음식 짧은 변 {side:4d}px → top1 {hit / n:.4f}", flush=True)

    out = Path(a.detect_model_dir) / f"tune_{a.split}.json"
    out.write_text(json.dumps({"split": a.split, "images": n, "grid": grid, "sizes": sizes}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
