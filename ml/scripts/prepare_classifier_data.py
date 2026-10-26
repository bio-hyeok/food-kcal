"""분류 학습용 데이터 만들기 (Colab / Linux / WSL).

AI-Hub 74번 분류 데이터에서 음식 영역만 잘라 아래 구조로 저장한다 (위키 전체 흐름 문서 6.5~6.6).

    <dst>/train|val|test/<food_id>/<파일명>.jpg      한 변(짧은 쪽) 최대 320px
    <dst>/manifest.csv                                split, food_id, code, src, dst
    <dst>/skipped.csv                                 제외한 이미지와 이유 (이어 하기에도 쓴다)
    <dst>/_run_config.json                            limit/seed. 다른 설정으로 이어 하는 것을 막는다

사용 (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !python ml/scripts/prepare_classifier_data.py \
        --src "/content/drive/MyDrive/음식데이터" --dst /content/processed --limit-per-class 20   # 먼저 작게
    # 문제 없으면 --limit-per-class 를 빼고 전체 실행 (Drive에서 읽어서 오래 걸린다)

이어 하기 (끊겨도 같은 명령을 다시 실행하면 된다)
- --dst 를 Drive 안(/content/drive/MyDrive/...)으로 두면 런타임이 새로 시작돼도 만든 결과가 남는다.
- 이미 만든 파일과 제외한 이미지(skipped.csv)는 건너뛴다. 파일은 임시 이름으로 쓴 뒤 이름을 바꿔서 쓰다 끊겨도 깨진 파일이 남지 않는다.
- 이어 할 때는 --limit-per-class, --seed 를 바꾸면 안 된다 (train/val/test 배정이 달라져 평가가 오염된다). 바꾸면 중단한다.

규칙
- 클래스 = ml/label_map.csv 의 food_id. 이미지가 없는 코드(대분류 11~16)는 건너뛴다.
- 라벨 xml 의 object 중 name 이 8자리 코드인 상자가 음식 영역이다 (name=dish 는 접시라서 쓰지 않는다).
- 같은 음식을 이어서 찍은 사진이 train/test 에 갈리면 성능이 부풀려진다. 파일명 속 촬영 시각(초 단위, 10자리)을
  10분 단위로 묶어 한 묶음으로 나눈다. 이 묶음 기준은 추정이다 (어노테이션에 촬영 세션 정보가 없다).
- 이 스크립트는 로컬(Windows, Python 없음)에서 실행해 보지 못했다. 처음에는 --limit-per-class 로 확인한다.
"""
import argparse
import csv
import json
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
FALLBACK = "성공(이름대체)"


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
    """(w 또는 None, h 또는 None, (x0, y0, x1, y1), 방식) 또는 제외 이유 문자열.

    - name 이 코드와 같은 상자(가장 큰 것) → "코드"
    - 없으면, name 이 숫자가 아닌 상자가 하나뿐일 때(한글 음식명이나 깨진 글자) → "이름대체"
      (일부 xml 은 코드 대신 음식명을 쓰고, 한글이 CP949 깨진 글자로 들어 있다)
    - name 이 다른 8자리 코드이거나 00000000 이면 라벨이 의심스러워 제외한다 → "코드불일치"
    - <size> 가 없는 xml 이 있어 w, h 는 None 일 수 있다 (이미지 크기를 쓴다)
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = h = None
    if size is not None and size.findtext("width") and size.findtext("height"):
        w, h = int(size.findtext("width")), int(size.findtext("height"))

    def box_of(obj):
        b = obj.find("bndbox")
        return tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))

    objs = [(((o.findtext("name") or "").strip()), o) for o in root.findall("object") if o.find("bndbox") is not None]
    exact = [box_of(o) for n, o in objs if n == code]
    if exact:
        best = max(exact, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return w, h, best, "코드"
    others = [(n, o) for n, o in objs if n != "dish"]
    named = [(n, o) for n, o in others if not re.fullmatch(r"d+", n)]
    if len(named) == 1:
        return w, h, box_of(named[0][1]), "이름대체"
    if others:
        return "코드불일치"
    return "음식상자_없음"


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
    """성공하면 None 또는 FALLBACK, 건너뛰면 이유 문자열."""
    if not xml_path.exists():
        return "xml_없음"
    try:
        fb = food_box(xml_path, code)
    except Exception:
        return "xml_읽기실패"
    if isinstance(fb, str):
        return fb
    w, h, (x0, y0, x1, y1), how = fb
    try:
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if w is not None and (im.width, im.height) != (w, h):
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
            tmp = dst.with_name(dst.name + ".tmp")        # 쓰다 끊겨도 반쯤 쓴 .jpg 가 남지 않게
            crop.save(tmp, "JPEG", quality=90)
            os.replace(tmp, dst)
    except Exception:
        return "이미지_읽기실패"
    return FALLBACK if how == "이름대체" else None   # 성공. 이름 대체로 찾았으면 표시만 한다


def check_config(dst: Path, a) -> None:
    """같은 dst 에서 이어 할 때 설정이 바뀌면 중단한다.
    limit/seed 가 바뀌면 사진의 train/val/test 배정이 달라져서, 먼저 만든 파일과 섞이면 평가가 오염된다."""
    cfg = {"limit_per_class": a.limit_per_class, "seed": a.seed}
    p = dst / "_run_config.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        if old != cfg:
            raise SystemExit(
                f"이 dst({dst})는 {old} 설정으로 만들던 폴더입니다. 지금 설정 {cfg} 와 달라 이어 할 수 없습니다.\n"
                "설정을 되돌리거나 다른 --dst 를 쓰세요."
            )
    else:
        p.write_text(json.dumps(cfg), encoding="utf-8")


def scan_existing(dst: Path) -> set:
    """이미 만들어 둔 결과 (split, food_id, 파일명 stem). 폴더 단위로만 읽어서 Drive 에서도 빠르다."""
    have = set()
    for sp in SPLITS:
        d = dst / sp
        if not d.is_dir():
            continue
        for c in os.scandir(d):
            if c.is_dir():
                for f in os.scandir(c.path):
                    if f.name.endswith(".jpg"):          # 쓰다 만 .jpg.tmp 는 제외된다
                        have.add((sp, c.name, f.name[:-4]))
    return have


def load_skipped(path: Path) -> dict:
    """이전에 제외한 이미지 {src: 이유}. 다시 처리하지 않는다."""
    out = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["src"]] = r["reason"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="음식데이터 폴더")
    ap.add_argument("--dst", required=True, help="결과 폴더. 끊겨도 이어 하려면 Drive 안(/content/drive/...)")
    ap.add_argument("--label-map", default=str(Path(__file__).resolve().parents[1] / "label_map.csv"))
    ap.add_argument("--limit-per-class", type=int, default=0, help="클래스당 이미지 수 제한 (0=전부). 처음엔 작게")
    ap.add_argument("--workers", type=int, default=8, help="전체 실행은 12~16 권장 (Drive 대기 시간이 대부분)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    check_config(dst, a)
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

    # 이어 하기: 이미 만든 것과 이미 제외한 것은 건너뛴다
    have = scan_existing(dst)
    skip_path = dst / "skipped.csv"
    prev_skipped = load_skipped(skip_path)
    key = lambda j: (j[4], str(j[1]), j[2].stem)
    pending = [j for j in jobs if key(j) not in have and str(j[2]) not in prev_skipped]
    n_have = sum(1 for j in jobs if key(j) in have)
    n_prev = sum(1 for j in jobs if str(j[2]) in prev_skipped and key(j) not in have)
    print(f"전체 {len(jobs)}장 / 이미 완료 {n_have}장 / 이전에 제외 {n_prev}장 → 이번에 처리할 것 {len(pending)}장")

    def run(j):
        code, fid, ip, xp, sp, out = j
        return j, make_crop(ip, xp, code, out)

    fallback, new_done, new_skip = 0, 0, 0
    new_file = not skip_path.exists()
    with open(skip_path, "a", newline="", encoding="utf-8") as sf:
        sw = csv.writer(sf)
        if new_file:
            sw.writerow(["code", "src", "reason"])
        try:
            with ThreadPoolExecutor(a.workers) as ex:
                for n, (j, why) in enumerate(ex.map(run, pending), 1):
                    if why == FALLBACK:
                        fallback += 1
                        why = None
                    if why:
                        new_skip += 1
                        sw.writerow([j[0], j[2], why])
                    else:
                        new_done += 1
                        have.add((j[4], str(j[1]), j[2].stem))
                    if n % 500 == 0:
                        sf.flush()
                        if n % 2000 == 0:
                            print(f"  {n}/{len(pending)}")
        finally:
            sf.flush()

    # manifest 는 결과 폴더에 실제로 있는 파일 기준으로 다시 만든다
    rows = [(j[4], j[1], j[0], j[2], j[5]) for j in jobs if (j[4], str(j[1]), j[2].stem) in have]
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "food_id", "code", "src", "dst"])
        w.writerows(rows)
    reasons = Counter(load_skipped(skip_path).values())

    print(f"\n이번 실행: 저장 {new_done}장 (그중 이름 대체 {fallback}장) / 제외 {new_skip}장")
    print(f"누적: 저장 {len(rows)}장 / 제외 {sum(reasons.values())}장")
    print("split별:", dict(Counter(r[0] for r in rows)))
    print("제외 이유:", dict(reasons))
    print("클래스 수:", len({r[1] for r in rows}))
    if len(rows) + sum(reasons.values()) < len(jobs):
        print("※ 아직 끝나지 않았습니다. 같은 명령을 다시 실행하면 이어서 합니다.")


if __name__ == "__main__":
    main()
