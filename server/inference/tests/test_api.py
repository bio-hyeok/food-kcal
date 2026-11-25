from PIL import Image

from app.classifier import center_crop, resize_short_side
from app.detector import Box, letterbox
from app.pipeline import merge_boxes
from conftest import image_bytes, two_color_image


def post(client, data, name="a.jpg", ctype="image/jpeg"):
    return client.post("/predict", files={"file": (name, data, ctype)})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["status"] == "ok"
    assert d["numClasses"] == 3
    assert d["minConfidence"] == 0.5
    assert d["model"] == {"name": "efficientnet_b0_food", "version": "v0.1.0"}


def test_predict_success(client):
    r = post(client, image_bytes((255, 0, 0)))
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["status"] == "SUCCESS"
    assert [c["rankNo"] for c in d["candidates"]] == [1, 2, 3]
    assert d["candidates"][0]["foodId"] == 5
    assert d["candidates"][0]["confidence"] > 0.99
    confs = [c["confidence"] for c in d["candidates"]]
    assert confs == sorted(confs, reverse=True)


def test_predict_png(client):
    r = post(client, image_bytes((0, 255, 0), "PNG"), "a.png", "image/png")
    assert r.status_code == 200
    assert r.json()["data"]["candidates"][0]["foodId"] == 12


def test_low_confidence_is_failed_but_keeps_candidates(client):
    # 회색 → 세 클래스가 거의 같은 확률 → 1순위가 기준값 0.5 미만
    r = post(client, image_bytes((124, 116, 104)))
    d = r.json()["data"]
    assert r.status_code == 200
    assert d["status"] == "FAILED"
    assert d["candidates"][0]["confidence"] < 0.5
    assert len(d["candidates"]) == 3      # API 서버가 분석용으로 저장할 수 있게 후보는 남긴다


def test_exif_rotated_jpeg_is_accepted(client):
    r = post(client, image_bytes((0, 0, 255), exif_orientation=6))
    assert r.status_code == 200
    assert r.json()["data"]["candidates"][0]["foodId"] == 100


def test_missing_file(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_file(client):
    r = post(client, b"")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_REQUEST"


def test_not_an_image(client):
    r = post(client, b"hello", "a.txt", "text/plain")
    assert r.status_code == 415
    assert r.json() == {"success": False, "error": {"code": "UNSUPPORTED_IMAGE", "message": "JPEG 또는 PNG 이미지만 받습니다."}}


def test_unsupported_format(client):
    r = post(client, image_bytes((255, 0, 0), "GIF"), "a.gif", "image/gif")
    assert r.status_code == 415


def test_too_large(client):
    r = post(client, b"x" * 200_001)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_cors_allows_expo_web(client):
    r = client.options("/predict", headers={"Origin": "http://localhost:8081", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8081"
    r = client.options("/predict", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


def test_resize_and_crop_match_torchvision_rules():
    im = Image.new("RGB", (400, 300))
    r = resize_short_side(im, 329)
    assert r.size == (438, 329)          # int(329 * 400 / 300) = 438
    c = center_crop(r, 288)
    assert c.size == (288, 288)
    tall = resize_short_side(Image.new("RGB", (300, 401)), 100)
    assert tall.size == (100, 133)


def test_detect_finds_each_food_and_classifies_it(detect_client):
    r = detect_client.post("/detect", files={"file": ("a.png", two_color_image(), "image/png")})
    assert r.status_code == 200
    d = r.json()["data"]
    assert (d["imageWidth"], d["imageHeight"]) == (128, 64)
    items = d["items"]
    assert [i["itemNo"] for i in items] == [1, 2]                 # 겹친 상자·낮은 신뢰도 상자는 빠진다
    assert [i["detConfidence"] for i in items] == [0.9, 0.8]
    assert items[0]["box"] == {"x0": 0.0, "y0": 0.0, "x1": 0.5, "y1": 1.0}   # 위아래 여백을 빼고 원본 비율로
    assert items[1]["box"] == {"x0": 0.5, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    assert items[0]["candidates"][0]["foodId"] == 5               # 왼쪽 빨강
    assert items[1]["candidates"][0]["foodId"] == 12              # 오른쪽 초록
    assert all(i["status"] == "SUCCESS" for i in items)
    assert (items[0]["cropShortSide"], items[0]["small"]) == (64, True)   # 64px < 128px → 작게 찍힘


def test_detect_small_flag_off_for_big_photo(detect_client):
    r = detect_client.post("/detect", files={"file": ("a.png", two_color_image((512, 256)), "image/png")})
    items = r.json()["data"]["items"]
    assert [(i["cropShortSide"], i["small"]) for i in items] == [(256, False), (256, False)]


def test_detect_no_box_gives_empty_items(detect_client, monkeypatch):
    monkeypatch.setenv("DETECT_MIN_CONFIDENCE", "0.95")
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/detect", files={"file": ("a.png", two_color_image(), "image/png")})
    assert r.status_code == 200
    assert r.json()["data"]["items"] == []


def test_detect_validates_upload_like_predict(detect_client):
    r = detect_client.post("/detect", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_detect_not_configured(client):
    r = client.post("/detect", files={"file": ("a.png", two_color_image(), "image/png")})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "DETECTOR_NOT_AVAILABLE"
    assert client.get("/health").json()["data"]["detect"]["enabled"] is False


def test_letterbox_matches_ultralytics_rules():
    canvas, r, left, top = letterbox(Image.new("RGB", (1024, 768)), 640, 114)
    assert canvas.size == (640, 640)
    assert (r, left, top) == (0.625, 0, 80)                        # 640×480 → 위아래 80
    _, r, left, top = letterbox(Image.new("RGB", (300, 401)), 640, 114)
    assert (left, top) == (80, 0)                                  # 479×640 → 왼쪽 round(80.5-0.1)=80


def test_detect_adds_bowls_without_auto_select(bowl_client):
    r = bowl_client.post("/detect", files={"file": ("a.png", two_color_image((512, 256)), "image/png")})
    items = r.json()["data"]["items"]
    assert [it["source"] for it in items] == ["food", "food", "bowl"]   # 겹친 그릇·컵·상 전체 상자는 빠진다
    assert [it["status"] for it in items[:2]] == ["SUCCESS", "SUCCESS"]
    assert items[2]["status"] == "FAILED"            # 그릇 상자는 후보만 보여 준다
    assert items[2]["box"]["y0"] < 0.5 < items[2]["box"]["x1"]
    assert bowl_client.get("/health").json()["data"]["detect"]["bowl"] is True


def test_detect_bowl_auto_select_option(bowl_client, monkeypatch):
    monkeypatch.setenv("DETECT_BOWL_AUTO_SELECT", "true")
    monkeypatch.setenv("RECOGNITION_MIN_CONFIDENCE", "0.0")
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        items = c.post("/detect", files={"file": ("a.png", two_color_image((512, 256)), "image/png")}).json()["data"]["items"]
    assert items[2]["source"] == "bowl" and items[2]["status"] == "SUCCESS"


def test_merge_drops_table_wide_food_box_and_huge_bowls():
    table = Box(0, 0, 100, 100, 0.9)                 # 상 전체를 음식 하나로 본 상자
    dish = Box(70, 70, 90, 90, 0.8)
    bowls = [Box(10, 10, 30, 30, 0.9), Box(40, 10, 60, 30, 0.8), Box(10, 40, 30, 60, 0.75), Box(0, 0, 100, 90, 0.7),
             Box(71, 71, 89, 89, 0.6)]
    kept = merge_boxes([table, dish], bowls, 100, 100)
    assert [(b.x0, src) for b, src in kept] == [(70, "food"), (10, "bowl"), (40, "bowl"), (10, "bowl")]
    # 그릇이 2개뿐이면(한 음식 사진의 곁들이 그릇 등) 큰 음식 상자를 지우지 않는다
    assert merge_boxes([table], bowls[:2], 100, 100)[0] == (table, "food")
    assert len(merge_boxes([dish], bowls, 100, 100, max_items=2)) == 2
