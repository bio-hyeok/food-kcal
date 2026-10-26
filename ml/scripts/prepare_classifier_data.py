"""분류 학습용 데이터 만들기 (Colab / Linux / WSL).

AI-Hub 74번 분류 데이터에서 음식 영역만 잘라 아래 구조로 저장한다 (위키 전체 흐름 문서 6.5~6.6).

    <dst>/train|val|test/<food_id>/<파일명>.jpg      한 변(짧은 쪽) 최대 320px
    <dst>/manifest.csv                                split, food_id, code, src, dst
    <dst>/skipped.csv                                 제외한 이미지와 이유

사용 (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !python ml/scripts/prepare_classifier_data.py \
        --src "/content/drive/MyDrive/음식데이터" --dst /content/processed --limit-per-class 20   # 먼저 작게
    # 문제 없으면 --limit-per-class 를 빼고 전체 실행 (Drive에서 읽어서 오래 걸린다)

규칙
- 클래스 = ml/label_map.csv 의 food_id. 이미지가 없는 코드(대분류 11~16)는 건너뛴다.
- 라벨 xml 의 object 중 name 이 8자리 코드인 상자가 음식 영역이다 (name=dish 는 접시라서 쓰지 않는다).
- 같은 음식을 이어서 찍은 사진이 train/test 에 갈리면 성능이 부풀려진다. 파일명 속 촬영 시각(초 단위, 10자리)을
  10분 단위로 묶어 한 묶음으로 나눈다. 이 묶음 기준은 추정이다 (어노테이션에 촬영 세션 정보가 없다).
- 이 스크립트는 로컬(Windows, Python 없음)에서 실행해 보지 못했다. 처음에는 --limit-per-class 로 확인한다.
"""
import argparse
import csv
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}
SIDE = 320          # 짧은 쪽이 이보다 크면 이 크기로 줄인다
MARGIN = 0.05       # 상자 둘레 여유 비율
SPLITS = ("train", "val", "test")
RATIO = (0.8, 0.1, 0.1)
BUCKET_SEC = 600


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_label_map(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset_label"] and r["food_id"]:
                out[r["dataset_label"]] = int(r["food_id"])
    return out


def find_images(src: Path, codes: set) -> dict:
    """code -> [이미지 경로]. [원천]음식분류_* 아래 8자리 코드 폴더를 깊이 2까지 찾는다."""
    found = defaultdict(list)
    tops = [d for d in os.scandir(src) if d.is_dir() and nfc(d.name).startswith("[원천]음식분류")]

    def walk(d, depth):
        for e in os.scandir(d):
            if not e.is_dir():
                continue
            if re.fullmatch(r"\d{8}", e.name):
                if e.name in codes:
                    for f in os.scandir(e.path):
                        if f.is_file() and Path(f.name).suffix.lower() in IMG_EXT:
                            found[e.name].append(Path(f.path))
            elif depth < 2:
                walk(e.path, depth + 1)

    for t in tops:
        walk(t.path, 0)
    return found


def find_xml_dir(src: Path) -> Path:
    lab = next(d for d in os.scandir(src) if d.is_dir() and nfc(d.name).startswith("[라벨]음식분류"))
    return Path(lab.path) / "xml"


def food_box(xml_path: Path, code: str):
    """(w, h, (x0, y0, x1, y1)) — name == code 인 상자 중 가장 큰 것. 없으면 None."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    best = None
    for obj in root.findall("object"):
        if (obj.findtext("name") or "").strip() != code:
            continue
        b = obj.find("bndbox")
        box = tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
        area = (box[2] - box[0]) * (box[3] - box[1])
        if best is None or area > best[0]:
            best = (area, box)
    return None if best is None else (w, h, best[1])


def group_key(code: str, stem: str):
    """촬영 시각(10자리 초)을 10분 단위로 묶는다. 읽지 못하면 파일 하나가 한 묶음."""
    m = re.match(r"^\d+_\d+_\d{8}_(\d{10})", stem)
    return (code, int(m.group(1)) // BUCKET_SEC) if m else (code, stem)


def assign_splits(items, rng):
    """items: [(group, path)] → {path: split}. 묶음 단위로 8:1:1에 가깝게 나눈다."""
    groups = defaultdict(list)
    for g, p in items:
        groups[g].append(p)
    keys = list(groups)
    rng.shuffle(keys)
    total = len(items)
    got = [0, 0, 0]
    out = {}
    for k in keys:
        # 목표 비율 대비 가장 모자란 split 에 넣는다
        i = max(range(3), key=lambda j: RATIO[j] * total - got[j])
        for p in groups[k]:
            out[p] = SPLITS[i]
        got[i] += len(groups[k])
    return out


def make_crop(img_path: Path, xml_path: Path, code: str, dst: Path):
    """성공하면 None, 건너뛰면 이유 문자열."""
    if not xml_path.exists():
        return "xml_없음"
    try:
        fb = food_box(xml_path, code)
    except Exception:
        return "xml_읽기실패"
    if fb is None:
        return "음식상자_없음"
    w, h, (x0, y0, x1, y1) = fb
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if (im.width, im.height) != (w, h):
                if abs(im.width / im.height - w / h) > 0.02:
                    return "크기_불일치"          # 회전 등으로 좌표를 믿을 수 없다
                sx, sy = im.width / w, im.height / h
                x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
            bw, bh = x1 - x0, y1 - y0
            if bw < 20 or bh < 20:
                return "상자_너무작음"
            box = (
                max(0, int(x0 - bw * MARGIN)), max(0, int(y0 - bh * MARGIN)),
                min(im.width, int(x1 + bw * MARGIN)), min(im.height, int(y1 + bh * MARGIN)),
            )
            crop = im.crop(box)
            s = SIDE / min(crop.size)
            if s < 1:
                crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dst, "JPEG", quality=90)
    except Exception:
        return "이미지_읽기실패"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="음식데이터 폴더")
    ap.add_argument("--dst", required=True, help="결과 폴더 (Git 밖이나 data/processed)")
    ap.add_argument("--label-map", default=str(Path(__file__).resolve().parents[1] / "label_map.csv"))
    ap.add_argument("--limit-per-class", type=int, default=0, help="클래스당 이미지 수 제한 (0=전부). 처음엔 작게")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    lmap = load_label_map(Path(a.label_map))
    found = find_images(src, set(lmap))
    xml_dir = find_xml_dir(src)
    rng = random.Random(a.seed)
    print(f"label_map {len(lmap)}개 코드, 이미지가 있는 코드 {len(found)}개")

    jobs = []   # (code, food_id, img_path, xml_path, split, dst_path)
    for code in sorted(found):
        paths = sorted(found[code])
        if a.limit_per_class:
            rng.shuffle(paths)
            paths = paths[: a.limit_per_class]
        sp = assign_splits([(group_key(code, p.stem), p) for p in paths], rng)
        for p in paths:
            out = dst / sp[p] / str(lmap[code]) / (p.stem + ".jpg")
            jobs.append((code, lmap[code], p, xml_dir / code / (p.stem + ".xml"), sp[p], out))

    skipped, done = [], []

    def run(j):
        code, fid, ip, xp, sp, out = j
        why = make_crop(ip, xp, code, out)
        return j, why

    with ThreadPoolExecutor(a.workers) as ex:
        for n, (j, why) in enumerate(ex.map(run, jobs), 1):
            (skipped if why else done).append((j, why))
            if n % 2000 == 0:
                print(f"  {n}/{len(jobs)}")

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "food_id", "code", "src", "dst"])
        for (code, fid, ip, xp, sp, out), _ in done:
            w.writerow([sp, fid, code, ip, out])
    with open(dst / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "src", "reason"])
        for (code, fid, ip, xp, sp, out), why in skipped:
            w.writerow([code, ip, why])

    print(f"\n저장 {len(done)}장 / 제외 {len(skipped)}장")
    print("split별:", dict(Counter(j[4] for j, _ in done)))
    print("제외 이유:", dict(Counter(why for _, why in skipped)))
    print("클래스 수:", len({j[1] for j, _ in done}))


if __name__ == "__main__":
    main()
