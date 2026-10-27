"""신뢰도 기준값 검증용 "음식이 아닌 사진" 모음을 만든다 (다운로드 없이 이 PC에 있는 것만 쓴다).

두 종류를 만든다.
1. wallpaper/  : Windows 기본 배경 화면(풍경·추상화)을 무작위로 잘라 낸 사진. 일반 사진 대용.
2. background/ : AI-Hub test 원본 사진에서 음식·접시 상자와 겹치지 않는 가장자리(식탁, 쟁반 등)만 잘라 낸 사진.
                 "음식 없이 식탁만 찍은 사진"에 가깝다. 사진 수가 적거나 가장자리가 좁으면 건너뛴다.

한계: 사람, 문서, 화면 캡처, 동물처럼 실제 사용자가 잘못 올릴 만한 사진은 들어 있지 않다.
      위키 6.8 의 "일반 사진 100장 이상"을 제대로 채우려면 나중에 그런 사진을 더 모아 --extra 로 넣는다.

사용:
    python scripts\\build_nonfood_set.py --out C:\\food-data\\nonfood
"""
import argparse
import csv
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageOps

MIN_SIDE = 160


def wallpaper_crops(root: Path, out: Path, per_image: int, rng: random.Random):
    out.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    for p in sorted(root.rglob("*.jpg")):
        if p.stat().st_size < 50_000 or "touchkeyboard" in str(p).lower():
            continue                      # 작은 썸네일·키보드 테마는 뺀다
        if p.stat().st_size in seen:      # 같은 그림이 여러 폴더에 있다
            continue
        seen.add(p.stat().st_size)
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        w, h = im.size
        for k in range(per_image):
            s = int(min(w, h) * rng.uniform(0.35, 0.9))
            x, y = rng.randint(0, w - s), rng.randint(0, h - s)
            im.crop((x, y, x + s, y + s)).resize((320, 320)).save(out / f"{p.stem}_{k}.jpg", quality=92)
            n += 1
    return n


def boxes_of(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    out = []
    for o in root.iter("object"):
        b = o.find("bndbox")
        if b is not None:
            out.append(tuple(int(float(b.findtext(t))) for t in ("xmin", "ymin", "xmax", "ymax")))
    return out


def background_crops(manifest: Path, src_root: Path, out: Path, limit: int, rng: random.Random):
    out.mkdir(parents=True, exist_ok=True)
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    rng.shuffle(rows)
    label_root = src_root / "[라벨]음식분류_VAL" / "xml"
    n = tried = 0
    for r in rows:
        if n >= limit:
            break
        rel = r["src"].split("/음식데이터/", 1)[1]
        img_path = src_root / Path(rel)
        xml_path = label_root / r["code"] / (img_path.stem + ".xml")
        tried += 1
        try:
            boxes = boxes_of(xml_path)
            im = Image.open(img_path).convert("RGB")   # 상자 좌표는 원본 방향 기준이라 회전하지 않는다
        except (OSError, ET.ParseError):
            continue
        if not boxes:
            continue
        w, h = im.size
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        strips = [(0, 0, x0, h), (x1, 0, w, h), (0, 0, w, y0), (0, y1, w, h)]   # 왼·오·위·아래
        strips = [s for s in strips if min(s[2] - s[0], s[3] - s[1]) >= MIN_SIDE]
        if not strips:
            continue
        sx0, sy0, sx1, sy1 = max(strips, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
        im.crop((sx0, sy0, sx1, sy1)).save(out / f"{img_path.stem}.jpg", quality=92)
        n += 1
    return n, tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wallpapers", default=r"C:\Windows\Web")
    ap.add_argument("--per-wallpaper", type=int, default=5)
    ap.add_argument("--manifest", default=r"C:\food-data\processed_full\manifest.csv")
    ap.add_argument("--src-root", default=r"G:\내 드라이브\음식데이터")
    ap.add_argument("--background", type=int, default=200, help="만들 배경 사진 수 (0이면 건너뜀)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out)
    rng = random.Random(a.seed)

    n_w = wallpaper_crops(Path(a.wallpapers), out / "wallpaper", a.per_wallpaper, rng)
    print(f"wallpaper: {n_w}장")
    if a.background:
        n_b, tried = background_crops(Path(a.manifest), Path(a.src_root), out / "background", a.background, rng)
        print(f"background: {n_b}장 (원본 {tried}장 확인)")


if __name__ == "__main__":
    main()
