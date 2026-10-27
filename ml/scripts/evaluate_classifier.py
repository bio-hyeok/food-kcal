"""분류 모델 test 평가 (위키 전체 흐름 문서 6.8 의 지표).

입력: <data>/test (폴더 이름 = food_id), <model-dir>/classifier_best.pt, idx_to_food_id.json
출력: <out>/metrics.json, per_class.csv, confusions.csv

사용 (로컬 Windows):
    python scripts\\evaluate_classifier.py --data C:\\food-data\\processed_full --model-dir C:\\food-data\\models

지표
- top-1 / top-5 정확도
- 칼로리 오차: |예측 음식의 1인분 kcal - 정답 음식의 1인분 kcal| 의 평균 (틀린 사진은 0이 아님, 맞은 사진은 0)
- 클래스별 정확도, 자주 헷갈리는 음식 쌍과 그 칼로리 차이
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", help="결과 폴더 (기본: <model-dir>/eval_<split>)")
    ap.add_argument("--foods", default=str(REPO / "app" / "src" / "foods.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    out = Path(a.out or f"{a.model_dir}/eval_{a.split}")
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(f"{a.model_dir}/idx_to_food_id.json", encoding="utf-8") as f:
        idx_to_food_id = {int(k): v for k, v in json.load(f).items()}
    with open(a.foods, encoding="utf-8") as f:
        foods = {x["foodId"]: x for x in json.load(f)}
    # 1인분 기준 kcal (foods.json 은 basisG 기준값이라 servingSizeG 로 환산)
    kcal = {fid: x["nutrition"]["kcal"] * x["servingSizeG"] / x["nutrition"]["basisG"] for fid, x in foods.items()}

    tf_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(f"{a.data}/{a.split}", tf_eval)
    # 평가 폴더의 클래스 번호는 학습 때와 다를 수 있으니 폴더 이름(food_id)으로 맞춘다
    ds_idx_to_food_id = {idx: int(name) for name, idx in ds.class_to_idx.items()}
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=device == "cuda")

    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(idx_to_food_id))
    model.load_state_dict(torch.load(f"{a.model_dir}/classifier_best.pt", map_location=device))
    model.to(device).eval()
    print(f"device: {device}, {a.split} {len(ds)}장, 클래스 {len(ds.classes)}개")

    n = top1 = top5 = 0
    kcal_err = 0.0
    per_class = defaultdict(lambda: [0, 0])          # food_id -> [맞음, 전체]
    confusions = Counter()                           # (정답, 예측) -> 수
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device, non_blocking=True))
            pred5 = logits.topk(5, dim=1).indices.cpu().tolist()
            for p5, yi in zip(pred5, y.tolist()):
                true_id = ds_idx_to_food_id[yi]
                p5_ids = [idx_to_food_id[i] for i in p5]
                ok = p5_ids[0] == true_id
                n += 1
                top1 += ok
                top5 += true_id in p5_ids
                kcal_err += abs(kcal[p5_ids[0]] - kcal[true_id])
                per_class[true_id][0] += ok
                per_class[true_id][1] += 1
                if not ok:
                    confusions[(true_id, p5_ids[0])] += 1

    metrics = {
        "split": a.split,
        "images": n,
        "top1": round(top1 / n, 4),
        "top5": round(top5 / n, 4),
        "kcal_error_mean": round(kcal_err / n, 1),
        "kcal_error_mean_when_wrong": round(kcal_err / max(1, n - top1), 1),
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    rows = sorted(((fid, c[0] / c[1], c[1]) for fid, c in per_class.items()), key=lambda r: r[1])
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["food_id", "name_ko", "accuracy", "images"])
        for fid, acc, cnt in rows:
            w.writerow([fid, foods[fid]["nameKo"], round(acc, 4), cnt])

    with open(out / "confusions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true_id", "true_name", "pred_id", "pred_name", "count", "kcal_diff"])
        for (t, p), cnt in confusions.most_common():
            w.writerow([t, foods[t]["nameKo"], p, foods[p]["nameKo"], cnt, round(abs(kcal[p] - kcal[t]), 1)])

    print(json.dumps(metrics, ensure_ascii=False))
    print("가장 약한 음식 10개:")
    for fid, acc, cnt in rows[:10]:
        print(f"  {foods[fid]['nameKo']}({fid}) {acc:.1%} / {cnt}장")
    print("가장 자주 헷갈린 쌍 10개 (정답 → 예측, 수, kcal 차이):")
    for (t, p), cnt in confusions.most_common(10):
        print(f"  {foods[t]['nameKo']} → {foods[p]['nameKo']}, {cnt}장, {abs(kcal[p] - kcal[t]):.0f}kcal")
    print(f"결과: {out}")


if __name__ == "__main__":
    main()
