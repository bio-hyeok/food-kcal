"""가짜 상차림 사진 만들기: 한 식탁 배경 위에 음식 2~5개를 붙인다. (검출 모델이 여러 음식을 배우게)

prepare_detect_data.py 결과(사진 + 음식 상자)를 재료로 쓴다. train/val/test 는 섞지 않는다
(train 합성 사진은 train 사진만, test 합성 사진은 test 사진만 쓴다).
  - 배경: 같은 split 사진에서 음식 상자 밖의 가장 넓은 띠(식탁·쟁반)를 잘라 캔버스 크기로 늘린다
  - 음식: 음식 상자 둘레 12% 를 더 잘라, 모서리를 둥글고 부드럽게 오려 붙인다
  - 크기: 캔버스 너비의 14~45% (작은 반찬 크기 포함). 이미 붙인 음식과는 거의 겹치지 않게 놓는다
  - 라벨: 붙인 자리의 음식 상자 (YOLO 형식, 클래스 0 = food)

    python scripts/make_synthetic_meals.py --detect C:\\food-data\\detect --out C:\\food-data\\detect_syn

결과: <out>/images|labels/{train,val,test}/, <out>/index.csv (split, stem, food_ids, boxes), <out>/data_combined.yaml
"""
import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CANVAS = ((1024, 768), (768, 1024))   # 가로 사진이 대부분
MARGIN = 0.12


def load_index(detect: Path):
    by = {"train": [], "val": [], "test": []}
    with open(detect / "index.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boxes = [tuple(map(int, b.split())) for b in r["boxes"].split("|")]
            by[r["split"]].append((r["stem"], int(r["food_id"]), boxes))
    return by


def background(rng, detect: Path, split: str, rows, size):
    """음식 상자 밖에서 가장 넓은 띠를 잘라 캔버스에 맞게 늘린다."""
    for _ in range(20):
        stem, _, boxes = rng.choice(rows)
        im = Image.open(detect / "images" / split / f"{stem}.jpg").convert("RGB")
        W, H = im.size
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        strips = [(0, 0, W, y0), (0, y1, W, H), (0, 0, x0, H), (x1, 0, W, H)]
        best = max(strips, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
        if (best[2] - best[0]) * (best[3] - best[1]) >= 0.12 * W * H and min(best[2] - best[0], best[3] - best[1]) >= 60:
            return im.crop(best).resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(1.5))
    return Image.new("RGB", size, (200, 190, 175))


def soft_mask(w, h):
    r = int(min(w, h) * 0.12)
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    return m.filter(ImageFilter.GaussianBlur(max(1, min(w, h) * 0.015)))


def overlap_ok(b, placed):
    for p in placed:
        ix = max(0, min(b[2], p[2]) - max(b[0], p[0])); iy = max(0, min(b[3], p[3]) - max(b[1], p[1]))
        inter = ix * iy
        small = min((b[2] - b[0]) * (b[3] - b[1]), (p[2] - p[0]) * (p[3] - p[1]))
        if inter > 0.08 * small:
            return False
    return True


def make_one(rng, detect: Path, split: str, rows):
    size = CANVAS[0] if rng.random() < 0.8 else CANVAS[1]
    W, H = size
    canvas = background(rng, detect, split, rows, size)
    k = rng.choices([2, 3, 4, 5], weights=[3, 4, 2, 1])[0]
    placed, food_ids = [], []
    used = set()
    tries = 0
    while len(placed) < k and tries < 60:
        tries += 1
        stem, food_id, boxes = rng.choice(rows)
        if food_id in used:
            continue
        im = Image.open(detect / "images" / split / f"{stem}.jpg").convert("RGB")
        x0, y0, x1, y1 = boxes[0]
        bw, bh = x1 - x0, y1 - y0
        cx0, cy0 = max(0, int(x0 - bw * MARGIN)), max(0, int(y0 - bh * MARGIN))
        cx1, cy1 = min(im.width, int(x1 + bw * MARGIN)), min(im.height, int(y1 + bh * MARGIN))
        crop = im.crop((cx0, cy0, cx1, cy1))
        target = rng.uniform(0.14, 0.45) * W * (0.85 if k >= 4 else 1.0)
        s = target / max(crop.size)
        cw, ch = max(16, round(crop.width * s)), max(16, round(crop.height * s))
        if cw >= W or ch >= H:
            continue
        crop = crop.resize((cw, ch), Image.LANCZOS)
        for _ in range(25):
            px, py = rng.randint(0, W - cw), rng.randint(0, H - ch)
            fb = (px + (x0 - cx0) * s, py + (y0 - cy0) * s, px + (x1 - cx0) * s, py + (y1 - cy0) * s)
            if overlap_ok(fb, placed):
                canvas.paste(crop, (px, py), soft_mask(cw, ch))
                placed.append(tuple(round(v) for v in fb))
                food_ids.append(food_id)
                used.add(food_id)
                break
    return canvas, placed, food_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    detect, out = Path(a.detect), Path(a.out)
    by = load_index(detect)
    rng = random.Random(a.seed)
    rows_out = []
    for split, n in (("train", a.n_train), ("val", a.n_val), ("test", a.n_test)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            canvas, boxes, food_ids = make_one(rng, detect, split, by[split])
            if len(boxes) < 2:
                continue
            stem = f"syn_{split}_{i:05d}"
            canvas.save(out / "images" / split / f"{stem}.jpg", quality=90)
            W, H = canvas.size
            (out / "labels" / split / f"{stem}.txt").write_text("".join(
                f"0 {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}\n"
                for b in boxes))
            rows_out.append((split, stem, "|".join(map(str, food_ids)), "|".join(" ".join(map(str, b)) for b in boxes)))
            if (i + 1) % 500 == 0:
                print(f"{split} {i + 1}/{n}", flush=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("split", "stem", "food_ids", "boxes"), *rows_out])
    d, s = detect.as_posix(), out.as_posix()
    (out / "data_combined.yaml").write_text(
        f"train:\n  - {d}/images/train\n  - {s}/images/train\nval:\n  - {d}/images/val\n  - {s}/images/val\n"
        f"test: {s}/images/test\nnames:\n  0: food\n", encoding="utf-8")
    (out / "data_syn_test.yaml").write_text(f"path: {s}\ntrain: images/train\nval: images/test\nnames:\n  0: food\n",
                                            encoding="utf-8")
    counts = {sp: sum(1 for r in rows_out if r[0] == sp) for sp in ("train", "val", "test")}
    foods = sum(len(r[2].split("|")) for r in rows_out)
    print("저장:", counts, "| 음식 수 합계:", foods)


if __name__ == "__main__":
    main()
