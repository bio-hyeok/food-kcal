"""검출 → 잘라 내기 → 분류(v3) 파이프라인 평가. (test 사진, 분류·검출 모두 본 적 없는 사진)

1) 검출 모델 자체: test 의 mAP50 / mAP50-95 (Ultralytics val)
2) 같은 test 사진에서 분류 정확도를 세 가지로 비교한다
   - full : 사진 전체를 그대로 분류 (지금 앱·서버가 하는 방식)
   - det  : 검출 모델이 찾은 상자(신뢰도 1위)를 잘라 분류 (새 방식). 상자를 못 찾으면 사진 전체로 분류
   - gt   : 라벨 상자를 잘라 분류 (상한선. 검출이 완벽할 때)
   잘라 낼 때 상자 둘레 5% 여유는 분류 학습 데이터(prepare_classifier_data.py)와 같다.
분류는 추론 서버 코드(server/inference/app/classifier.py)를 그대로 써서 서버와 같은 결과가 나오게 한다.

    python scripts/evaluate_detect_pipeline.py --detect C:\\food-data\\detect --weights <best.pt> --cls-model-dir C:\\food-data\\models_v3
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server" / "inference"))
from app.classifier import Classifier  # noqa: E402

MARGIN = 0.05


def crop(im: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return im.crop((max(0, int(x0 - bw * MARGIN)), max(0, int(y0 - bh * MARGIN)),
                    min(im.width, int(x1 + bw * MARGIN)), min(im.height, int(y1 + bh * MARGIN))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True, help="prepare_detect_data.py 결과 폴더")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cls-model-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="결과 json (기본: weights 옆 pipeline_eval.json)")
    a = ap.parse_args()

    det_root = Path(a.detect)
    det = YOLO(a.weights)
    m = det.val(data=str(det_root / "data.yaml"), split="test", plots=False, verbose=False)
    det_metrics = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                   "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}
    print("검출 test:", det_metrics, flush=True)

    clf = Classifier(a.cls_model_dir)
    foods = {f["foodId"]: f for f in json.loads((ROOT / "app" / "src" / "foods.json").read_text(encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(det_root / "index.csv", encoding="utf-8")) if r["split"] == "test"]

    names = ("full", "det", "gt")
    top1 = dict.fromkeys(names, 0)
    top5 = dict.fromkeys(names, 0)
    kcal_err = dict.fromkeys(names, 0.0)
    no_box = multi_box = 0
    for i, r in enumerate(rows, 1):
        im = Image.open(det_root / "images" / "test" / (r["stem"] + ".jpg")).convert("RGB")
        truth = int(r["food_id"])
        gt_box = tuple(map(int, r["boxes"].split("|")[0].split()))
        res = det.predict(im, conf=a.conf, verbose=False)[0]
        if len(res.boxes) == 0:
            no_box += 1
            det_im = im
        else:
            if len(res.boxes) > 1:
                multi_box += 1
            best = int(res.boxes.conf.argmax())
            det_im = crop(im, res.boxes.xyxy[best].tolist())
        for name, x in (("full", im), ("det", det_im), ("gt", crop(im, gt_box))):
            cands = clf.predict(x, 5)
            ids = [c.food_id for c in cands]
            top1[name] += ids[0] == truth
            top5[name] += truth in ids
            kcal_err[name] += abs(foods[ids[0]]["nutrition"]["kcal"] - foods[truth]["nutrition"]["kcal"])
        if i % 200 == 0:
            print(f"{i}/{len(rows)}", flush=True)

    n = len(rows)
    result = {
        "test_images": n,
        "detector": det_metrics,
        "detector_no_box": no_box,
        "detector_multi_box": multi_box,
        "classification": {k: {"top1": round(top1[k] / n, 4), "top5": round(top5[k] / n, 4),
                               "kcal_err": round(kcal_err[k] / n, 1)} for k in names},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(a.out) if a.out else Path(a.weights).parent.parent / "pipeline_eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
