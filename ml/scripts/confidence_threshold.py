"""RECOGNITION_MIN_CONFIDENCE 정하기 (위키 전체 흐름 문서 6.8 "음식이 아닌 사진 거절 검증").

1순위 신뢰도(softmax 최대값)가 기준값 미만이면 서버는 status=FAILED 로 처리한다.
기준값을 바꿔 가며 두 가지를 함께 본다.
- 음식 사진(test)을 잘못 거절하는 비율   → 낮을수록 좋다
- 음식이 아닌 사진을 걸러 내는 비율        → 높을수록 좋다

추천 규칙: 음식 사진 거절 비율이 --max-food-reject(기본 10%) 이하인 기준값 중 가장 큰 값.
          (음식 사진을 덜 거절하는 쪽을 우선한다. 거절되면 사용자는 수동 검색을 해야 하기 때문)

입력: <model-dir>/classifier.onnx, model_meta.json (export_onnx.py 결과), <data>/test, --nonfood 폴더
출력: <model-dir>/threshold/thresholds.csv, summary.json

사용:
    python scripts\\confidence_threshold.py --model-dir C:\\food-data\\models_v3 --data C:\\food-data\\processed_full --nonfood C:\\food-data\\nonfood
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def scores(sess, ds, workers):
    """(1순위 신뢰도, 1순위 클래스 번호, 상위 5개 번호) 를 사진마다 돌려준다."""
    conf, top1, top5, labels = [], [], [], []
    for x, y in DataLoader(ds, batch_size=64, num_workers=workers):
        logits = sess.run(None, {"image": x.numpy()})[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        conf.append(p.max(1))
        top1.append(p.argmax(1))
        top5.append(np.argsort(-p, 1)[:, :5])
        labels.append(y.numpy())
    return np.concatenate(conf), np.concatenate(top1), np.concatenate(top5), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nonfood", required=True, help="하위 폴더마다 음식이 아닌 사진 (예: wallpaper/, background/)")
    ap.add_argument("--max-food-reject", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    md = Path(a.model_dir)
    meta = json.loads((md / "model_meta.json").read_text(encoding="utf-8"))
    inp = meta["input"]
    idx_to_food_id = {int(k): v for k, v in meta["idx_to_food_id"].items()}
    tf = transforms.Compose([
        transforms.Resize(inp["resize_short_side"]), transforms.CenterCrop(inp["center_crop"]),
        transforms.ToTensor(), transforms.Normalize(inp["mean"], inp["std"]),
    ])
    sess = ort.InferenceSession(str(md / meta["onnx"]), providers=["CPUExecutionProvider"])

    food = datasets.ImageFolder(f"{a.data}/test", tf)
    f_conf, f_top1, f_top5, f_y = scores(sess, food, a.workers)
    true_id = np.array([int(food.classes[i]) for i in f_y])
    pred_id = np.array([idx_to_food_id[int(i)] for i in f_top1])
    in_top5 = np.array([t in {idx_to_food_id[int(j)] for j in row} for t, row in zip(true_id, f_top5)])
    correct = pred_id == true_id

    nf = datasets.ImageFolder(a.nonfood, tf)
    n_conf, _, _, n_y = scores(sess, nf, a.workers)
    groups = {name: n_conf[n_y == i] for i, name in enumerate(nf.classes)}

    print(f"음식 test {len(f_conf)}장, 음식 아님 {len(n_conf)}장 ({', '.join(f'{k} {len(v)}' for k, v in groups.items())})")
    q = [5, 10, 25, 50, 75, 90]
    print("1순위 신뢰도 분포 (백분위 " + "/".join(map(str, q)) + ")")
    print("  음식, 맞힘 :", np.round(np.percentile(f_conf[correct], q), 3))
    print("  음식, 틀림 :", np.round(np.percentile(f_conf[~correct], q), 3))
    for k, v in groups.items():
        print(f"  음식 아님 {k}:", np.round(np.percentile(v, q), 3))

    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
        acc = f_conf >= t
        row = {
            "threshold": float(t),
            "food_reject": round(1 - acc.mean(), 4),                       # 음식 사진을 거절한 비율
            "food_reject_but_top5_ok": round((~acc & in_top5).mean(), 4),  # 거절했지만 후보 5개 안에 정답이 있던 비율
            "accepted_top1": round(correct[acc].mean(), 4) if acc.any() else None,  # 통과한 사진의 1순위 정확도
            "wrong_accepted": round((acc & ~correct).mean(), 4),           # 틀렸는데 통과한 비율
            "nonfood_reject": round((n_conf < t).mean(), 4),
        }
        for k, v in groups.items():
            row[f"nonfood_reject_{k}"] = round((v < t).mean(), 4)
        rows.append(row)

    out = md / "threshold"
    out.mkdir(exist_ok=True)
    with open(out / "thresholds.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["food_reject"] <= a.max_food_reject]
    rec = max(ok, key=lambda r: r["threshold"]) if ok else rows[0]
    summary = {"recommended": rec["threshold"], "rule": f"food_reject <= {a.max_food_reject}",
               "at_recommended": rec, "at_0.5": next(r for r in rows if r["threshold"] == 0.5),
               "food_images": int(len(f_conf)), "nonfood_images": {k: int(len(v)) for k, v in groups.items()}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n기준값 | 음식 거절 | (그중 top5 정답) | 통과 사진 정확도 | 틀렸는데 통과 | 음식 아님 거절 | " + " | ".join(groups))
    for r in rows:
        extra = " | ".join(f"{r[f'nonfood_reject_{k}']:.1%}" for k in groups)
        print(f"  {r['threshold']:.2f} | {r['food_reject']:.1%} | {r['food_reject_but_top5_ok']:.1%} | "
              f"{(r['accepted_top1'] or 0):.1%} | {r['wrong_accepted']:.1%} | {r['nonfood_reject']:.1%} | {extra}")
    print(f"\n추천 기준값: {rec['threshold']} (규칙: 음식 사진 거절 {a.max_food_reject:.0%} 이하 중 가장 큰 값)")
    print(f"결과: {out}")


if __name__ == "__main__":
    main()
