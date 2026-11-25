"""진짜 상차림 사진(인터넷 공개 사진 20장) 평가. 정답은 사람이 표시한 gt.json (상자 %, 인정 이름 목록).

    .venv\\Scripts\\python.exe scripts\\eval_real.py --model-dir C:\\food-data\\models_v3 --data C:\\food-data\\real_meals \\
        --detect-model-dir C:\\food-data\\detector_v1 C:\\food-data\\detector_v2 --bowl-model-dir C:\\food-data\\bowl_coco

상자는 서버와 같은 방법(app/pipeline.py)으로 찾는다. --bowl-model-dir 를 주면 모델마다 "그릇 없이 / 그릇과 함께" 둘 다 잰다.
음식(정답 상자)마다: 찾음 = 신뢰도 높은 상자부터 IoU 기준 이상인 정답과 짝짓기(한 상자는 한 음식에만).
아는 음식(304종 안)은 짝지은 상자의 1순위/후보 5개에 인정 이름이 있으면 맞힘.
헛상자 = 어느 정답과도 짝이 안 된 상자. 큰 상자 = 사진 넓이의 40% 이상인 상자(상 전체를 한 음식으로 본 경우).
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.classifier import Classifier  # noqa: E402
from app.config import Settings  # noqa: E402
from app.detector import Detector  # noqa: E402
from app.pipeline import detect_items  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--detect-model-dir", required=True, nargs="+")
    ap.add_argument("--bowl-model-dir", default=None)
    ap.add_argument("--min-confidence", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()
    foods = {f["foodId"]: f["nameKo"] for f in json.loads((ROOT / "app" / "src" / "foods.json").read_text(encoding="utf-8"))}
    clf = Classifier(a.model_dir)
    root = Path(a.data)
    gt = {k: v for k, v in json.loads((root / "gt.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}
    s = Settings(model_dir=a.model_dir, model_name="", model_version="", min_confidence=a.min_confidence, top_k=5,
                 max_image_bytes=0)
    bowl = Detector(a.bowl_model_dir) if a.bowl_model_dir else None
    runs = [(Path(d).name, Detector(d), None) for d in a.detect_model_dir]
    if bowl:
        runs += [(Path(d).name + "+bowl", Detector(d), bowl) for d in a.detect_model_dir]
    results, cases = {}, {}
    for name, det, bw in runs:
        c = dict(foods=0, known=0, found=0, found_known=0, top1=0, top5=0, success=0, boxes=0, extra=0, big=0,
                 wrong_auto=0)
        cases[name] = {}
        for stem, items in gt.items():
            im = ImageOps.exif_transpose(Image.open(root / "images" / f"{stem}.jpg")).convert("RGB")
            W, H = im.size
            gts = [((b[0] * W / 100, b[1] * H / 100, b[2] * W / 100, b[3] * H / 100), labels) for b, labels in items]
            its = detect_items(im, s, clf, det, bw)
            boxes = [it.box for it in its]
            c["boxes"] += len(boxes)
            c["big"] += sum((b.x1 - b.x0) * (b.y1 - b.y0) >= 0.4 * W * H for b in boxes)
            used, matched = set(), {}
            for it in its:
                b = it.box
                cand = [(iou((b.x0, b.y0, b.x1, b.y1), g[0]), gi) for gi, g in enumerate(gts) if gi not in used]
                if cand and max(cand)[0] >= a.iou:
                    gi = max(cand)[1]; used.add(gi); matched[gi] = it
            # 미리 고른(SUCCESS) 음식이 틀린 경우: 모르는 음식·음식 아닌 곳·다른 음식으로 고름 (합계 kcal 가 틀어진다)
            for it in its:
                if it.status != "SUCCESS":
                    continue
                gi = next((k for k, v in matched.items() if v is it), None)
                labels = gts[gi][1] if gi is not None else []
                c["wrong_auto"] += foods.get(it.candidates[0].food_id) not in labels
            c["extra"] += len(boxes) - len(matched)
            log = []
            for gi, (_, labels) in enumerate(gts):
                c["foods"] += 1; c["known"] += bool(labels)
                if gi not in matched:
                    if labels:
                        log.append(f"못 찾음: {labels[0]}")
                    continue
                c["found"] += 1
                if not labels:
                    continue
                c["found_known"] += 1
                cands = matched[gi].candidates
                names = [foods.get(x.food_id, str(x.food_id)) for x in cands]
                c["top1"] += names[0] in labels
                c["top5"] += any(n in labels for n in names)
                c["success"] += names[0] in labels and matched[gi].status == "SUCCESS"
                log.append(f"{'맞힘' if names[0] in labels else '틀림'}: 정답 {labels[0]} / 1순위 {names[0]} {cands[0].confidence:.2f}")
            cases[name][stem] = log
        n = len(gt)
        results[name] = {
            "images": n, "foods": c["foods"], "known_foods": c["known"],
            "found_all": round(c["found"] / c["foods"], 4), "found_known": round(c["found_known"] / c["known"], 4),
            "top1_known": round(c["top1"] / c["known"], 4), "top5_known": round(c["top5"] / c["known"], 4),
            "top1_and_confident_known": round(c["success"] / c["known"], 4),
            "top1_among_found": round(c["top1"] / max(1, c["found_known"]), 4),
            "boxes_per_image": round(c["boxes"] / n, 2), "extra_boxes_per_image": round(c["extra"] / n, 2),
            "big_boxes": c["big"], "wrong_auto_selected": c["wrong_auto"],
        }
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    out = root / f"eval_real_iou{a.iou}.json"
    out.write_text(json.dumps({"results": results, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
