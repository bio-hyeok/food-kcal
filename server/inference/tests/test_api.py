from PIL import Image

from app.classifier import center_crop, resize_short_side
from conftest import image_bytes


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
