"""여러 음식 사진 평가 (서버 코드 그대로). ml/scripts/make_synthetic_meals.py 의 test 합성 사진을 쓴다.

음식(정답 상자) 하나하나에 대해:
  찾음   = 신뢰도 높은 상자부터 정답 상자와 IoU 0.5 이상이면 짝짓는다 (한 상자는 한 음식에만)
  맞힘   = 짝지은 상자의 1순위(후보 5개 안) 음식이 정답
사진 하나에 대해: 헛상자 = 어느 음식과도 짝이 안 된 상자 수, 모두 맞힘 = 모든 음식을 찾고 1순위로 맞힘

    .venv\\Scripts\\python.exe scripts\\eval_multi.py --model-dir C:\\food-data\\models_v3 --data C:\\food-data\\detect_syn \\
        --detect-model-dir C:\\food-data\\detector_v1 [C:\\food-data\\detector_v2 ...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classifier import Classifier  # noqa: E402
from app.detector import Detector, crop_box  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detect-model-dir", required=True, nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-confidence", type=float, default=0.30)
    a = ap.parse_args()

    clf = Classifier(a.model_dir)
    root = Path(a.data)
    rows = [r for r in csv.DictReader(open(root / "index.csv", encoding="utf-8")) if r["split"] == a.split]
    results = {}
    for dmd in a.detect_model_dir:
        det = Detector(dmd)
        n_food = found = top1 = top5 = success = extra = all_ok = 0
        for r in rows:
            im = Image.open(root / "images" / a.split / (r["stem"] + ".jpg")).convert("RGB")
            gts = [tuple(map(int, b.split())) for b in r["boxes"].split("|")]
            ids = [int(x) for x in r["food_ids"].split("|")]
            boxes = det.detect(im, 0.25, 0.7, 6)
            used, ok_here = set(), 0
            matched = {}
            for bi, b in enumerate(boxes):                      # 신뢰도 높은 순
                cand = [(iou((b.x0, b.y0, b.x1, b.y1), g), gi) for gi, g in enumerate(gts) if gi not in used]
                if cand:
                    best = max(cand)
                    if best[0] >= 0.5:
                        used.add(best[1]); matched[best[1]] = b
            extra += len(boxes) - len(matched)
            for gi, fid in enumerate(ids):
                n_food += 1
                if gi not in matched:
                    continue
                found += 1
                cands = clf.predict(crop_box(im, matched[gi]), 5)
                top1 += cands[0].food_id == fid
                top5 += fid in [c.food_id for c in cands]
                success += cands[0].food_id == fid and cands[0].confidence >= a.min_confidence
                ok_here += cands[0].food_id == fid
            all_ok += ok_here == len(ids)
        n = len(rows)
        results[Path(dmd).name] = {
            "images": n, "foods": n_food,
            "found": round(found / n_food, 4), "top1": round(top1 / n_food, 4), "top5": round(top5 / n_food, 4),
            "top1_and_confident": round(success / n_food, 4),
            "extra_boxes_per_image": round(extra / n, 3), "all_foods_correct_images": round(all_ok / n, 4),
        }
        print(Path(dmd).name, json.dumps(results[Path(dmd).name], ensure_ascii=False), flush=True)
    out = root / f"eval_multi_{a.split}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
