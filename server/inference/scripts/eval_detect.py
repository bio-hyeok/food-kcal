"""서버 코드(app/detector.py + app/classifier.py)로 /predict 와 /detect 방식을 비교한다.

- 음식 사진: ml prepare_detect_data.py 의 test (분류·검출 모두 본 적 없는 사진)
    /predict 방식 = 사진 전체 분류,  /detect 방식 = 첫 상자(음식 검출 신뢰도 1위)를 잘라 분류
    --bowl-model-dir 를 주면 /detect 처럼 그릇 찾기도 함께 쓴다 (app/pipeline.py)
- 음식이 아닌 사진: ml build_nonfood_set.py 결과 (배경 화면 조각, 식탁 가장자리)
    거절 = /predict 는 1순위 신뢰도 < 기준값,  /detect 는 SUCCESS 상자가 하나도 없음 (그릇 상자는 기본으로 FAILED)

    .venv\\Scripts\\python.exe scripts\\eval_detect.py --model-dir C:\\food-data\\models_v3 --detect-model-dir C:\\food-data\\detector_v1 \\
        --detect-data C:\\food-data\\detect --nonfood C:\\food-data\\nonfood
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classifier import Classifier  # noqa: E402
from app.config import Settings  # noqa: E402
from app.detector import Detector  # noqa: E402
from app.pipeline import detect_items  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--detect-model-dir", required=True)
    ap.add_argument("--bowl-model-dir", default=None)
    ap.add_argument("--detect-data", required=True)
    ap.add_argument("--nonfood", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    s = Settings(model_dir=a.model_dir, model_name="", model_version="", min_confidence=0.30, top_k=5,
                 max_image_bytes=0)   # 기준값은 서버 기본값과 같게 둔다
    clf, det = Classifier(a.model_dir), Detector(a.detect_model_dir)
    bowl = Detector(a.bowl_model_dir) if a.bowl_model_dir else None
    foods = {f["foodId"]: f for f in json.loads((ROOT / "app" / "src" / "foods.json").read_text(encoding="utf-8"))}
    kcal = lambda i: foods[i]["nutrition"]["kcal"]  # noqa: E731

    def run_detect(im):
        return detect_items(im, s, clf, det, bowl)

    rows = [r for r in csv.DictReader(open(Path(a.detect_data) / "index.csv", encoding="utf-8")) if r["split"] == "test"]
    acc = {k: {"top1": 0, "top5": 0, "kcal_err": 0.0, "rejected": 0} for k in ("predict", "detect")}
    t_pred = t_det = 0.0
    for i, r in enumerate(rows, 1):
        im = Image.open(Path(a.detect_data) / "images" / "test" / (r["stem"] + ".jpg")).convert("RGB")
        truth = int(r["food_id"])
        t0 = time.perf_counter()
        full = clf.predict(im, s.top_k)
        t1 = time.perf_counter()
        items = run_detect(im)
        t2 = time.perf_counter()
        t_pred, t_det = t_pred + t1 - t0, t_det + t2 - t1
        dc = items[0].candidates if items else full     # 상자를 못 찾으면 사진 전체 결과로 센다
        acc["predict"]["rejected"] += full[0].confidence < s.min_confidence
        acc["detect"]["rejected"] += all(it.status != "SUCCESS" for it in items)
        for k, cands in (("predict", full), ("detect", dc)):
            ids = [c.food_id for c in cands]
            acc[k]["top1"] += ids[0] == truth
            acc[k]["top5"] += truth in ids
            acc[k]["kcal_err"] += abs(kcal(ids[0]) - kcal(truth))
        if i % 300 == 0:
            print(f"음식 {i}/{len(rows)}", flush=True)
    n = len(rows)

    nonfood = sorted(p for p in Path(a.nonfood).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rej_pred = rej_det = no_box = 0
    for p in nonfood:
        im = Image.open(p).convert("RGB")
        rej_pred += clf.predict(im, 1)[0].confidence < s.min_confidence
        items = run_detect(im)
        no_box += not items
        rej_det += all(it.status != "SUCCESS" for it in items)   # 상자가 없어도 참

    result = {
        "food_test_images": n,
        "food": {k: {"top1": round(v["top1"] / n, 4), "top5": round(v["top5"] / n, 4),
                     "kcal_err": round(v["kcal_err"] / n, 1), "rejected": round(v["rejected"] / n, 4)}
                 for k, v in acc.items()},
        "cpu_ms_per_image": {"predict": round(t_pred / n * 1000, 1), "detect": round(t_det / n * 1000, 1)},
        "nonfood_images": len(nonfood),
        "nonfood_rejected": {"predict": round(rej_pred / len(nonfood), 4), "detect": round(rej_det / len(nonfood), 4),
                             "detect_no_box": round(no_box / len(nonfood), 4)},
        "settings": {"min_confidence": s.min_confidence, "detect_min_confidence": s.detect_min_confidence,
                     "detect_iou": s.detect_iou, "bowl": a.bowl_model_dir},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = Path(a.out) if a.out else Path(a.detect_model_dir) / "server_eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
