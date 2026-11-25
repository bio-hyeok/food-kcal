"""음식 위치 찾기(검출) 모델용 데이터를 만든다. (한 사진에 여러 음식, 위키 1절 3단계)

분류 데이터(prepare_classifier_data.py)와 같은 원본·같은 train/val/test 배정을 쓴다.
manifest.csv 에서 split·코드마다 몇 장씩 골라, 원본 사진을 긴 변 LONG 으로 줄여 저장하고
음식 상자를 YOLO 형식(클래스 0 = food, 가운데 x/y·너비·높이를 0~1 로) 라벨로 쓴다.
종류는 분류 모델(v3)이 맡으므로 검출 모델은 "음식이 어디 있나"만 배운다. 접시(dish) 상자는 쓰지 않는다.

    python scripts/prepare_detect_data.py --out C:\\food-data\\detect

결과
    <out>/images/{train,val,test}/<stem>.jpg
    <out>/labels/{train,val,test}/<stem>.txt
    <out>/index.csv     split, food_id, code, stem, boxes (줄인 사진 기준 픽셀 x0 y0 x1 y1; 여러 개면 | 로 구분)
    <out>/skipped.csv   제외한 사진과 이유
    <out>/data.yaml     학습 설정
같은 명령을 다시 실행하면 이미 만든 사진은 건너뛴다 (Drive 를 읽다 끊겨도 이어서 한다).
"""
import argparse
import csv
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

LONG = 1024
PER_SPLIT = {"train": 20, "val": 3, "test": 5}   # 클래스(코드)마다 고르는 장수


def food_boxes(xml_path: Path, code: str):
    """(w 또는 None, h 또는 None, [상자...]) 또는 제외 이유. 규칙은 prepare_classifier_data.food_box 와 같고,
    코드와 같은 상자가 여러 개면 모두 쓴다 (검출은 음식 영역을 빠짐없이 알려 주는 것이 중요하다)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = h = None
    if size is not None and size.findtext("width") and size.findtext("height"):
        w, h = int(size.findtext("width")), int(size.findtext("height"))

    def box_of(obj):
        b = obj.find("bndbox")
        return tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))

    objs = [((o.findtext("name") or "").strip(), o) for o in root.findall("object") if o.find("bndbox") is not None]
    exact = [box_of(o) for n, o in objs if n == code]
    if exact:
        return w, h, exact
    others = [(n, o) for n, o in objs if n != "dish"]
    named = [(n, o) for n, o in others if not re.fullmatch(r"\d+", n)]
    if len(named) == 1 and len(others) == 1:
        return w, h, [box_of(named[0][1])]
    return "코드불일치" if others else "음식상자_없음"


def to_local(src: str, src_root: Path) -> Path:
    return src_root / Path(src.split("/음식데이터/", 1)[1])


def make_one(job):
    """성공하면 (boxes, None), 제외하면 (None, 이유)."""
    split, code, img_path, xml_path, out_img, out_lab = job
    if out_img.exists() and out_lab.exists():
        boxes = [tuple(map(int, l.split()[5:])) for l in out_lab.with_suffix(".box").read_text().splitlines()] \
            if out_lab.with_suffix(".box").exists() else None
        if boxes:
            return boxes, None
    try:
        fb = food_boxes(xml_path, code)
    except FileNotFoundError:
        return None, "xml_없음"
    except ET.ParseError:
        return None, "xml_읽기실패"
    if isinstance(fb, str):
        return None, fb
    w, h, raw = fb
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            sx = sy = 1.0
            if w is not None and (im.width, im.height) != (w, h):
                if abs(im.width / im.height - w / h) > 0.02:
                    return None, "크기_불일치"          # 회전 등으로 좌표를 믿을 수 없다
                sx, sy = im.width / w, im.height / h
            s = min(1.0, LONG / max(im.size))
            small = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im
            W, H = small.size
            boxes = []
            for x0, y0, x1, y1 in raw:
                b = (max(0, round(x0 * sx * s)), max(0, round(y0 * sy * s)),
                     min(W, round(x1 * sx * s)), min(H, round(y1 * sy * s)))
                if b[2] - b[0] >= 8 and b[3] - b[1] >= 8:
                    boxes.append(b)
            if not boxes:
                return None, "상자_너무작음"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lab.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_img.with_name(out_img.name + ".tmp")
            small.save(tmp, "JPEG", quality=90)
            os.replace(tmp, out_img)
    except UnidentifiedImageError:
        return None, "이미지_읽기실패"
    except OSError as e:
        return None, f"읽기오류:{type(e).__name__}"
    lines = [f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
             for x0, y0, x1, y1 in boxes]
    out_lab.write_text("\n".join(lines) + "\n")
    # 픽셀 좌표도 남겨 둔다 (평가·재실행용). YOLO 는 .box 를 읽지 않는다
    out_lab.with_suffix(".box").write_text("\n".join(f"{l} {' '.join(map(str, b))}" for l, b in zip(lines, boxes)) + "\n")
    return boxes, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=r"C:\food-data\processed_full\manifest.csv")
    ap.add_argument("--src-root", default=r"G:\내 드라이브\음식데이터")
    ap.add_argument("--limit-classes", type=int, default=0, help="시험용: 앞에서부터 이 수의 코드만 (0이면 전체)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out, src_root = Path(a.out), Path(a.src_root)
    label_root = src_root / "[라벨]음식분류_VAL" / "xml"
    by = defaultdict(list)
    with open(a.manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[(r["split"], r["code"])].append(r)
    codes = sorted({c for _, c in by})
    if a.limit_classes:
        codes = codes[: a.limit_classes]

    rng = random.Random(a.seed)
    jobs, meta = [], []
    for code in codes:
        for split, n in PER_SPLIT.items():
            rows = sorted(by.get((split, code), []), key=lambda r: r["src"])
            for r in rng.sample(rows, min(n, len(rows))):
                img = to_local(r["src"], src_root)
                stem = img.stem
                jobs.append((split, code, img, label_root / code / (stem + ".xml"),
                             out / "images" / split / (stem + ".jpg"), out / "labels" / split / (stem + ".txt")))
                meta.append((split, r["food_id"], code, stem))
    print(f"대상 {len(jobs)}장 (코드 {len(codes)}개)", flush=True)

    ok, skipped = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, ((boxes, why), m) in enumerate(zip(ex.map(make_one, jobs), meta), 1):
            if boxes:
                ok.append((*m, "|".join(" ".join(map(str, b)) for b in boxes)))
            else:
                skipped.append((*m, why))
            if i % 200 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} 저장 {len(ok)} 제외 {len(skipped)}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("split", "food_id", "code", "stem", "boxes"), *ok])
    with open(out / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("split", "food_id", "code", "stem", "reason"), *skipped])
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: food\n",
        encoding="utf-8")
    reasons = defaultdict(int)
    for *_, why in skipped:
        reasons[why] += 1
    split_n = defaultdict(int)
    for s, *_ in ok:
        split_n[s] += 1
    multi = sum(1 for *_, b in ok if "|" in b)
    print("split:", dict(split_n), "| 상자 2개 이상:", multi, "| 제외 이유:", dict(reasons))


if __name__ == "__main__":
    main()
