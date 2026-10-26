"""분류 학습용 데이터 만들기 (Colab / Linux / WSL).

AI-Hub 74번 분류 데이터에서 음식 영역만 잘라 아래 구조로 저장한다 (위키 전체 흐름 문서 6.5~6.6).

    <dst>/train|val|test/<food_id>/<파일명>.jpg      한 변(짧은 쪽) 최대 320px
    <dst>/manifest.csv                                split, food_id, code, src, dst
    <dst>/skipped.csv                                 제외한 이미지와 이유 (이어 하기에도 쓴다)
    <dst>/_splits.csv                                 사진별 train/val/test 배정. 첫 실행에 정하고 이후 그대로 쓴다
    <dst>/_run_config.json                            limit/seed/묶음 단위/label_map. 다른 설정으로 이어 하는 것을 막는다

사용 (Colab):
    from google.colab import drive; drive.mount('/content/drive')
    !python ml/scripts/prepare_classifier_data.py \
        --src "/content/drive/MyDrive/음식데이터" --dst /content/processed --limit-per-class 20   # 먼저 작게
    # 문제 없으면 --limit-per-class 를 빼고 전체 실행 (Drive에서 읽어서 오래 걸린다)

이어 하기 (끊겨도 같은 명령을 다시 실행하면 된다)
- --dst 를 Drive 안(/content/drive/MyDrive/...)으로 두면 런타임이 새로 시작돼도 만든 결과가 남는다.
- 이미 만든 파일과 제외한 이미지(skipped.csv)는 건너뛴다. 파일은 임시 이름으로 쓴 뒤 이름을 바꿔서 쓰다 끊겨도 깨진 파일이 남지 않는다.
- train/val/test 배정은 첫 실행에 _splits.csv 로 저장하고 이후 그대로 쓴다. Drive 목록이 실행마다 조금 달라도 배정이 바뀌지 않는다.
- Drive 일시 오류(Input/output error 등)는 제외로 기록하지 않고 다음 실행에서 다시 한다.
- 이어 할 때 --limit-per-class, --seed, --bucket-sec, label_map.csv 가 바뀌면 중단한다.

규칙
- 클래스 = ml/label_map.csv 의 food_id. 이미지가 없는 코드(대분류 11~16)는 건너뛴다.
- 라벨 xml 의 object 중 name 이 8자리 코드인 상자가 음식 영역이다 (name=dish 는 접시라서 쓰지 않는다).
- 같은 음식을 이어서 찍은 사진이 train/test 에 갈리면 성능이 부풀려진다. 파일명 속 촬영 시각(초 단위, 10자리)을
  --bucket-sec(기본 3600초=1시간) 단위로 묶어 한 묶음으로 나눈다. 이 묶음 기준은 추정이다 (어노테이션에 촬영 세션 정보가 없다).
- 이 스크립트는 로컬(Windows, Python 없음)에서 실행해 보지 못했다. 처음에는 --limit-per-class 로 확인한다.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMG_EXT = {".jpg", ".jpeg", ".png"}
SIDE = 320          # 짧은 쪽이 이보다 크면 이 크기로 줄인다
MARGIN = 0.05       # 상자 둘레 여유 비율
SPLITS = ("train", "val", "test")
RATIO = (0.8, 0.1, 0.1)
CHUNK = 1000        # 한 번에 작업 대기열에 넣는 수. 셀을 중지하면 이 이상 기다리지 않는다
FALLBACK = "성공(이름대체)"
RETRY = "재시도"     # Drive 일시 오류 등. skipped.csv 에 쓰지 않고 다음 실행에서 다시 한다


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def load_label_map(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["dataset_label"] and r["food_id"]:
                out[r["dataset_label"]] = int(r["food_id"])
    return out


def find_images(src: Path, codes: set):
    """(code -> {stem: 이미지 경로}, 중복 stem 수). [원천]음식분류_* 아래 8자리 코드 폴더를 깊이 2까지 찾는다."""
    found = defaultdict(dict)
    dup = 0
    tops = [d for d in os.scandir(src) if d.is_dir() and nfc(d.name).startswith("[원천]음식분류")]

    def walk(d, depth):
        nonlocal dup
        for e in os.scandir(d):
            if not e.is_dir():
                continue
            if re.fullmatch(r"\d{8}", e.name):
                if e.name in codes:
                    for f in sorted(os.scandir(e.path), key=lambda x: x.name):
                        if f.is_file() and Path(f.name).suffix.lower() in IMG_EXT:
                            stem = Path(f.name).stem
                            if stem in found[e.name]:
                                dup += 1          # 같은 이름의 .jpg/.jpeg 가 둘 → 하나만 쓴다
                            else:
                                found[e.name][stem] = Path(f.path)
            elif depth < 2:
                walk(e.path, depth + 1)

    for t in tops:
        walk(t.path, 0)
    return found, dup


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

    objs = [((o.findtext("name") or "").strip(), o) for o in root.findall("object") if o.find("bndbox") is not None]
    exact = [box_of(o) for n, o in objs if n == code]
    if exact:
        best = max(exact, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        return w, h, best, "코드"
    others = [(n, o) for n, o in objs if n != "dish"]
    named = [(n, o) for n, o in others if not re.fullmatch(r"\d+", n)]
    if len(named) == 1 and len(others) == 1:
        return w, h, box_of(named[0][1]), "이름대체"
    if others:
        return "코드불일치"
    return "음식상자_없음"


def group_key(code: str, stem: str, bucket_sec: int):
    """촬영 시각(10자리 초)을 bucket_sec 단위로 묶는다. 읽지 못하면 파일 하나가 한 묶음."""
    m = re.match(r"^\d+_\d+_\d{8}_(\d{10})", stem)
    return (code, int(m.group(1)) // bucket_sec) if m else (code, stem)


def assign_splits(items, rng):
    """items: [(group, stem)] → {stem: split}. 묶음 단위로 8:1:1에 가깝게 나눈다."""
    groups = defaultdict(list)
    for g, s in items:
        groups[g].append(s)
    keys = list(groups)
    rng.shuffle(keys)
    total = len(items)
    got = [0, 0, 0]
    out = {}
    for k in keys:
        # 목표 비율 대비 가장 모자란 split 에 넣는다
        i = max(range(3), key=lambda j: RATIO[j] * total - got[j])
        for s in groups[k]:
            out[s] = SPLITS[i]
        got[i] += len(groups[k])
    return out


def make_crop(img_path: Path, xml_path: Path, code: str, dst: Path):
    """성공하면 None 또는 FALLBACK, 다음에 다시 할 것은 RETRY, 제외하면 이유 문자열."""
    try:
        os.stat(xml_path)
    except FileNotFoundError:
        return "xml_없음"
    except OSError:
        return RETRY
    try:
        fb = food_box(xml_path, code)
    except ET.ParseError:
        return "xml_읽기실패"
    except OSError:
        return RETRY
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
    except UnidentifiedImageError:
        return "이미지_읽기실패"                        # 이미지 파일이 아니다 (영구)
    except OSError:
        return RETRY                                    # Drive 일시 오류, 쓰기 실패, 잘린 파일
    except Exception:
        return "이미지_읽기실패"
    return FALLBACK if how == "이름대체" else None   # 성공. 이름 대체로 찾았으면 표시만 한다


def check_config(dst: Path, a, label_map_path: Path) -> None:
    """같은 dst 에서 이어 할 때 설정이 바뀌면 중단한다. 배정이 달라지거나 클래스 번호가 바뀌면 결과가 섞인다."""
    cfg = {
        "limit_per_class": a.limit_per_class,
        "seed": a.seed,
        "bucket_sec": a.bucket_sec,
        "label_map_sha1": hashlib.sha1(label_map_path.read_bytes()).hexdigest()[:12],
    }
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


def write_atomic(path: Path, rows, header) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def load_or_make_splits(dst: Path, found: dict, a) -> dict:
    """{(code, stem): split}. 처음이면 정해서 저장하고, 있으면 저장된 것을 그대로 쓴다."""
    p = dst / "_splits.csv"
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            return {(r["code"], r["stem"]): r["split"] for r in csv.DictReader(f)}
    out = {}
    for code in sorted(found):
        rng = random.Random(f"{a.seed}:{code}")      # 클래스마다 따로. 다른 클래스 목록에 영향받지 않는다
        stems = sorted(found[code])
        if a.limit_per_class:
            rng.shuffle(stems)
            stems = stems[: a.limit_per_class]
        sp = assign_splits([(group_key(code, s, a.bucket_sec), s) for s in stems], rng)
        for s in stems:
            out[(code, s)] = sp[s]
    write_atomic(p, [(c, s, sp) for (c, s), sp in sorted(out.items())], ["code", "stem", "split"])
    return out


def scan_existing(dst: Path) -> set:
    """이미 만들어 둔 결과 (split, food_id, stem). 폴더 단위로 읽는다. 크기 0인 파일은 완료로 보지 않는다."""
    have = set()
    for sp in SPLITS:
        d = dst / sp
        if not d.is_dir():
            continue
        for c in os.scandir(d):
            if c.is_dir():
                for f in os.scandir(c.path):
                    if f.name.endswith(".jpg"):          # 쓰다 만 .jpg.tmp 는 제외된다
                        try:
                            if f.stat().st_size > 0:
                                have.add((sp, c.name, f.name[:-4]))
                        except OSError:
                            pass
    return have


def load_skipped(path: Path) -> dict:
    """이전에 제외한 이미지 {(code, stem): 이유}. 다시 처리하지 않는다."""
    out = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("code") and r.get("stem") and r.get("reason"):
                    out[(r["code"], r["stem"])] = r["reason"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="음식데이터 폴더")
    ap.add_argument("--dst", required=True, help="결과 폴더. 끊겨도 이어 하려면 Drive 안(/content/drive/...)")
    ap.add_argument("--label-map", default=str(Path(__file__).resolve().parents[1] / "label_map.csv"))
    ap.add_argument("--limit-per-class", type=int, default=0, help="클래스당 이미지 수 제한 (0=전부). 처음엔 작게")
    ap.add_argument("--workers", type=int, default=8, help="전체 실행은 12~16 권장 (Drive 대기 시간이 대부분)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bucket-sec", type=int, default=3600, help="촬영 시각을 이 초 단위로 묶어 같은 묶음은 같은 split 에 둔다")
    ap.add_argument("--give-up-retries", action="store_true",
                    help="재시도 대상(계속 읽히지 않는 파일)을 이번에는 제외로 기록한다. 같은 수가 두세 번 남을 때만 쓴다")
    a = ap.parse_args()

    src, dst, lm_path = Path(a.src), Path(a.dst), Path(a.label_map)
    dst.mkdir(parents=True, exist_ok=True)
    check_config(dst, a, lm_path)
    lmap = load_label_map(lm_path)
    found, dup = find_images(src, set(lmap))
    xml_dir = find_xml_dir(src)
    print(f"label_map {len(lmap)}개 코드, 이미지가 있는 코드 {len(found)}개" + (f", 같은 이름 중복 {dup}개는 하나만 사용" if dup else ""))

    splits = load_or_make_splits(dst, found, a)
    jobs = []   # (code, food_id, img_path, xml_path, split, dst_path, stem)
    unseen = 0  # 배정에는 있는데 이번 목록에서 안 보이는 사진 (Drive 목록 문제일 수 있음)
    for (code, stem), sp in sorted(splits.items()):
        ip = found.get(code, {}).get(stem)
        if ip is None:
            unseen += 1
            continue
        out = dst / sp / str(lmap[code]) / (stem + ".jpg")
        jobs.append((code, lmap[code], ip, xml_dir / code / (stem + ".xml"), sp, out, stem))

    # 이어 하기: 이미 만든 것과 이미 제외한 것은 건너뛴다
    have = scan_existing(dst)
    # 안전 점검: 배정과 다른 split 폴더에 있는 파일이 있으면 결과가 섞인 것이다
    assigned = {(str(lmap[c]), s): sp for (c, s), sp in splits.items()}
    wrong = [(sp, fid, s) for sp, fid, s in have if assigned.get((fid, s), sp) != sp]
    if wrong:
        raise SystemExit(f"배정과 다른 split 에 있는 파일 {len(wrong)}개 (예: {wrong[:3]}). 이 dst 는 섞였습니다. 새 --dst 로 다시 하세요.")

    skip_path = dst / "skipped.csv"
    prev_skipped = load_skipped(skip_path)
    key = lambda j: (j[4], str(j[1]), j[6])
    pending = [j for j in jobs if key(j) not in have and (j[0], j[6]) not in prev_skipped]
    n_have = sum(1 for j in jobs if key(j) in have)
    n_prev = sum(1 for j in jobs if (j[0], j[6]) in prev_skipped and key(j) not in have)
    print(f"전체 {len(splits)}장 / 이미 완료 {n_have}장 / 이전에 제외 {n_prev}장 → 이번에 처리할 것 {len(pending)}장")
    if unseen:
        print(f"※ 이번 목록에서 안 보이는 사진 {unseen}장은 건너뜁니다 (Drive 목록 문제일 수 있음. 다음 실행에서 다시 봅니다).")

    def run(j):
        code, fid, ip, xp, sp, out, stem = j
        return j, make_crop(ip, xp, code, out)

    fallback, new_done, new_skip, retry, interrupted = 0, 0, 0, 0, False
    # 쓰다 끊긴 마지막 줄 뒤에 새 줄이 붙지 않게 줄바꿈으로 끝나는지 확인한다
    if skip_path.exists() and skip_path.stat().st_size > 0:
        with open(skip_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
    else:
        needs_nl = False
    new_file = not skip_path.exists() or skip_path.stat().st_size == 0
    with open(skip_path, "a", newline="", encoding="utf-8") as sf:
        if needs_nl:
            sf.write("\n")
        sw = csv.writer(sf)
        if new_file:
            sw.writerow(["code", "stem", "src", "reason"])
        with ThreadPoolExecutor(a.workers) as ex:
            try:
                for start in range(0, len(pending), CHUNK):
                    for j, why in ex.map(run, pending[start:start + CHUNK]):
                        if why == RETRY and a.give_up_retries:
                            why = "읽기오류_반복"
                        if why == RETRY:
                            retry += 1
                        elif why == FALLBACK or why is None:
                            fallback += why == FALLBACK
                            new_done += 1
                            have.add(key(j))
                        else:
                            new_skip += 1
                            sw.writerow([j[0], j[6], j[2], why])
                    sf.flush()
                    print(f"  {min(start + CHUNK, len(pending))}/{len(pending)}")
            except KeyboardInterrupt:
                interrupted = True
                ex.shutdown(wait=False, cancel_futures=True)
                print("\n중지 요청을 받았습니다. 진행 중인 몇 장만 마치고 멈춥니다. 같은 명령으로 이어 하면 됩니다.")

    # manifest 는 결과 폴더에 실제로 있는 파일 기준으로 다시 만든다
    rows = [(j[4], j[1], j[0], j[2], j[5]) for j in jobs if key(j) in have]
    write_atomic(dst / "manifest.csv", rows, ["split", "food_id", "code", "src", "dst"])
    skipped_all = load_skipped(skip_path)
    reasons = Counter(skipped_all.values())

    print(f"\n이번 실행: 저장 {new_done}장 (그중 이름 대체 {fallback}장) / 제외 {new_skip}장 / 다음에 다시 할 것 {retry}장")
    print(f"누적: 저장 {len(rows)}장 / 제외 {len(skipped_all)}장 / 전체 {len(splits)}장")
    print("split별:", dict(Counter(r[0] for r in rows)))
    print("제외 이유:", dict(reasons))
    print("클래스 수:", len({r[1] for r in rows}))
    if interrupted or len(rows) + len(skipped_all) < len(splits):
        print("※ 아직 끝나지 않았습니다. 같은 명령을 다시 실행하면 이어서 합니다.")
        if retry and not interrupted:
            print("  (다시 해도 '다음에 다시 할 것'이 같은 수로 남으면 깨진 파일입니다. 그때는 --give-up-retries 를 붙여 한 번 실행하세요.)")
    else:
        print("완료. Drive 에 다 써졌는지 확실히 하려면 마지막에 drive.flush_and_unmount() 를 실행하세요.")


if __name__ == "__main__":
    main()
