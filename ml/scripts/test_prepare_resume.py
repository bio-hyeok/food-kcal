"""prepare_classifier_data.py 의 이어 하기·제외 규칙 시험 (가짜 소규모 데이터, 1분 안쪽).

사용 (Colab 또는 Python 이 있는 곳):
    !python /content/ml/scripts/test_prepare_resume.py

마지막에 "모든 시험 통과"가 나오면 된다. 실패하면 어느 단계인지 출력된다.
실제 데이터나 Drive 는 건드리지 않는다 (임시 폴더만 쓴다).
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "prepare_classifier_data.py"

XML_OK = """<annotation><size><width>{w}</width><height>{h}</height><depth>3</depth></size>
<object><name>dish</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>390</xmax><ymax>290</ymax></bndbox></object>
<object><name>{name}</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>300</xmax><ymax>250</ymax></bndbox></object>
</annotation>"""
XML_NOSIZE = """<annotation>
<object><name>{name}</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>300</xmax><ymax>250</ymax></bndbox></object>
</annotation>"""
XML_DISH_ONLY = """<annotation><size><width>400</width><height>300</height></size>
<object><name>dish</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>390</xmax><ymax>290</ymax></bndbox></object>
</annotation>"""

# (경우, 기대 결과)  결과: "ok" 저장 / 그 밖은 skipped.csv 의 reason
CASES = [
    ("code", "ok"), ("code", "ok"), ("code", "ok"), ("code", "ok"), ("code", "ok"),
    ("code", "ok"), ("code", "ok"), ("code", "ok"), ("code", "ok"), ("code", "ok"),
    ("korean", "ok"),            # 이름대체
    ("mojibake", "ok"),          # 이름대체
    ("nosize", "ok"),
    ("other_code", "코드불일치"),
    ("zeros", "코드불일치"),
    ("dish_only", "음식상자_없음"),
    ("bad_xml", "xml_읽기실패"),
    ("bad_image", "이미지_읽기실패"),
    ("no_xml", "xml_없음"),
]


def build(root: Path):
    src = root / "src"
    codes = {"01011001": 1, "02011001": 2}
    for ci, code in enumerate(codes):
        img_dir = src / "[원천]음식분류_01" / code
        xml_dir = src / "[라벨]음식분류_VAL" / "xml" / code
        img_dir.mkdir(parents=True)
        xml_dir.mkdir(parents=True)
        for i, (kind, _) in enumerate(CASES):
            ts = 1602000000 + i * 7200          # 2시간 간격 → 사진마다 다른 묶음
            stem = f"01_011_{code}_{ts}{i:05d}"
            ip = img_dir / f"{stem}.jpg"
            if kind == "bad_image":
                ip.write_bytes(b"not an image")
            else:
                Image.new("RGB", (400, 300), (ci * 100 + i * 5, 80, 120)).save(ip, "JPEG")
            xp = xml_dir / f"{stem}.xml"
            name = {"code": code, "korean": "잣국", "mojibake": "┬½╗═╣Σ", "nosize": code,
                    "other_code": "09999999", "zeros": "00000000", "bad_image": code}.get(kind)
            if kind == "nosize":
                xp.write_text(XML_NOSIZE.format(name=name), encoding="utf-8")
            elif kind == "dish_only":
                xp.write_text(XML_DISH_ONLY, encoding="utf-8")
            elif kind == "bad_xml":
                xp.write_text("<annotation><object>", encoding="utf-8")
            elif kind == "no_xml":
                pass
            else:
                xp.write_text(XML_OK.format(w=400, h=300, name=name), encoding="utf-8")
    lm = root / "label_map.csv"
    lm.write_text("dataset_label,food_id,name_ko\n" + "".join(f"{c},{f},음식{f}\n" for c, f in codes.items()),
                  encoding="utf-8")
    return src, lm, codes


def run(src, dst, lm, *extra):
    p = subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst), "--label-map", str(lm),
                        "--workers", "4", *extra], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jpgs(dst):
    return sorted(p.relative_to(dst).as_posix() for p in dst.glob("*/*/*.jpg"))


def check(cond, msg, out=""):
    if not cond:
        print("실패:", msg)
        if out:
            print("---- 스크립트 출력 ----\n" + out)
        sys.exit(1)
    print("통과:", msg)


def main():
    root = Path(tempfile.mkdtemp(prefix="prep_test_"))
    try:
        src, lm, codes = build(root)
        dst = root / "dst"
        n_ok = sum(1 for _, e in CASES if e == "ok") * len(codes)
        n_skip = sum(1 for _, e in CASES if e != "ok") * len(codes)

        # 1) 첫 실행: 저장/제외 개수와 이유
        rc, out = run(src, dst, lm)
        check(rc == 0, "첫 실행이 정상 종료", out)
        files = jpgs(dst)
        check(len(files) == n_ok, f"저장 {n_ok}장 (실제 {len(files)})", out)
        sk = read_csv(dst / "skipped.csv")
        want = sorted(e for _, e in CASES if e != "ok")
        got = sorted(r["reason"] for r in sk if r["code"] == "01011001")
        check(got == sorted(want), f"제외 이유가 기대와 같다 {got}", out)
        check(len(sk) == n_skip, f"제외 {n_skip}장 기록", out)
        check(len(read_csv(dst / "manifest.csv")) == n_ok, "manifest 가 저장 파일 수와 같다", out)
        splits1 = (dst / "_splits.csv").read_text(encoding="utf-8")
        check(not list(dst.rglob("*.tmp")), "임시 파일이 남지 않는다")

        # 2) 바로 다시 실행 → 할 일 0
        rc, out = run(src, dst, lm)
        check(rc == 0 and "이번에 처리할 것 0장" in out, "재실행하면 처리할 것 0장", out)
        check(len(read_csv(dst / "skipped.csv")) == n_skip, "재실행해도 skipped.csv 가 늘지 않는다", out)

        # 3) 일부 삭제 + 크기 0 파일 + 남은 .tmp → 그것만 다시 만든다, 배정은 그대로
        f0, f1 = dst / files[0], dst / files[1]
        f0.unlink()
        f1.write_bytes(b"")
        (f1.parent / "junk.jpg.tmp").write_bytes(b"x")
        rc, out = run(src, dst, lm)
        check(rc == 0 and "이번에 처리할 것 2장" in out, "지운 1장 + 크기 0인 1장만 다시 처리", out)
        check(f0.exists() and f1.stat().st_size > 0, "두 파일이 다시 만들어졌다", out)
        check((dst / "_splits.csv").read_text(encoding="utf-8") == splits1, "배정(_splits.csv)이 바뀌지 않았다")

        # 4) 쓰다 끊긴 skipped.csv 마지막 줄 → 다음 실행이 망가지지 않는다
        with open(dst / "skipped.csv", "a", encoding="utf-8") as f:
            f.write("01011001,반쯤쓴줄")      # 줄바꿈 없이 끝남
        rc, out = run(src, dst, lm)
        check(rc == 0, "끊긴 줄이 있어도 정상 실행", out)

        # 5) 설정을 바꾸면 중단
        rc, out = run(src, dst, lm, "--seed", "7")
        check(rc != 0 and "이어 할 수 없습니다" in out, "seed 를 바꾸면 중단한다", out)
        rc, out = run(src, dst, lm, "--limit-per-class", "3")
        check(rc != 0 and "이어 할 수 없습니다" in out, "limit 을 바꾸면 중단한다", out)

        # 6) 다른 split 폴더에 파일이 섞이면 중단
        a = dst / files[2]
        other = "val" if a.parts[-3] != "val" else "train"
        b = dst / other / a.parts[-2] / a.name
        b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(a, b)
        rc, out = run(src, dst, lm)
        check(rc != 0 and "섞였습니다" in out, "split 이 섞이면 중단한다", out)
        b.unlink()

        # 7) 이름대체는 2장(한글, 깨진 글자) × 코드 수
        dst2 = root / "dst2"
        rc, out = run(src, dst2, lm)
        check(f"이름 대체 {2 * len(codes)}장" in out, "음식명/깨진 글자 상자만 이름 대체로 받아들인다", out)

        print("\n모든 시험 통과")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
