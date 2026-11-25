# server/inference — 추론 서버

음식 사진 1장을 받아 학습한 분류 모델(ONNX)로 음식 후보 Top-N과 신뢰도를 돌려주는 서버다.
위키 [전체 흐름 문서](https://github.com/bio-hyeok/food-kcal/wiki/전체-흐름-문서)의 "추론 서버 (분류 모델)"에 해당한다.

## 어디에 쓰이나

```
앱 ──POST /images, /recognitions──▶ API 서버 (아직 없음) ──POST /predict──▶ 추론 서버 (이 폴더)
                                         │                               │
                                         │◀── 후보 Top-N + 신뢰도 ────────┘
                                         ├─ RECOGNITION_RESULTS / ITEMS 에 저장
                                         └─ 후보마다 FOODS 영양정보를 붙여 GET /recognitions/{resultId} 로 응답
```

- **위키 설계에서는 앱이 직접 부르지 않는다.** API 서버가 내부에서만 부른다. 그래서 인증이 없다. 밖으로 열지 말고 내부망·localhost 에만 둔다.
- 단, **과제 시연용으로는 앱이 이 서버를 직접 부른다** (API 서버 없음, `app/README.md`). 같은 Wi-Fi 안에서만 쓰고 인터넷에 열지 않는다.
- 영양정보, DB 저장, 사용자 구분은 API 서버가 맡는다. 추론 서버는 "사진 → foodId 후보"만 한다.
- PyTorch 없이 ONNX Runtime 으로만 돈다. CPU 로 사진 1장에 수 ms (전처리 제외, 개발 PC 기준).

## 실행

```powershell
cd server\inference
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:MODEL_DIR = "C:\food-data\models_v3"
$env:DETECT_MODEL_DIR = "C:\food-data\detector_v1"   # 한 사진에 여러 음식(/detect). 없으면 /detect 만 꺼진다
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

Git Bash 에서는 `MODEL_DIR="C:\food-data\models_v3" DETECT_MODEL_DIR="C:\food-data\detector_v1" .venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8100`.

`MODEL_DIR` 에는 `ml/scripts/export_onnx.py` 가 만든 두 파일이 있어야 한다. 없으면 서버가 뜨지 않는다.
- `classifier.onnx` — 모델
- `model_meta.json` — 전처리 값, 클래스 번호 → foodId 표

모델 파일은 Git 에 올리지 않는다 (위키 6.9). 배포할 때는 스토리지나 GitHub Releases 에서 받아 `MODEL_DIR` 에 둔다.

## 환경 변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `MODEL_DIR` | (필수) | `classifier.onnx`, `model_meta.json` 이 있는 폴더 |
| `RECOGNITION_MIN_CONFIDENCE` | `0.30` | 1순위 신뢰도가 이 값 미만이면 `FAILED`. v3 모델로 정한 값 (`ml/README.md`) |
| `RECOGNITION_TOP_K` | `5` | 돌려줄 후보 수 |
| `MODEL_NAME` / `MODEL_VERSION` | `efficientnet_b0_food` / `v0.1.0` | 응답의 `model` 값. `MODEL_VERSIONS` 테이블과 맞춘다 |
| `MAX_IMAGE_BYTES` | `10485760` (10MB) | 받는 이미지 최대 크기 |
| `CORS_ORIGINS` | `http://localhost:8081,http://127.0.0.1:8081` | 브라우저에서 부를 수 있는 주소 (과제 시연용 Expo 웹). 비우면 끈다 |
| `DETECT_MODEL_DIR` | (없음) | `detector.onnx`, `detector_meta.json` 폴더 (`ml/scripts/export_detector.py`). 없으면 `/detect` 가 503 |
| `DETECT_MIN_CONFIDENCE` | `0.25` | 이 신뢰도 미만의 상자는 버린다 |
| `DETECT_IOU` | `0.7` | 이만큼 겹친 상자는 신뢰도 높은 하나만 남긴다 (NMS) |
| `DETECT_MAX_ITEMS` | `6` | 한 사진에서 돌려줄 음식 최대 수 |
| `DETECT_SMALL_PX` | `128` | 음식 상자의 짧은 변이 이보다 작으면 `small: true` (앱이 "가까이서 찍어 주세요" 안내) |

**모델을 바꾸면 `RECOGNITION_MIN_CONFIDENCE` 도 다시 정한다** (`ml/scripts/confidence_threshold.py`). 학습 설정에 따라 신뢰도 분포가 크게 달라진다.

## API

응답은 위키 API 문서 1절 공통 형식(`{"success": true, "data": ...}` / `{"success": false, "error": {...}}`)을 따른다.

### GET `/health`

```json
{ "success": true, "data": { "status": "ok", "model": { "name": "efficientnet_b0_food", "version": "v0.1.0" },
  "numClasses": 304, "inputSize": 288, "minConfidence": 0.3, "topK": 5 } }
```

### POST `/predict`

`multipart/form-data`, 필드명 `file`. JPEG, PNG 만 받는다.

```json
{
  "success": true,
  "data": {
    "status": "SUCCESS",
    "model": { "name": "efficientnet_b0_food", "version": "v0.1.0" },
    "minConfidence": 0.3,
    "candidates": [
      { "rankNo": 1, "foodId": 256, "confidence": 0.4217 },
      { "rankNo": 2, "foodId": 202, "confidence": 0.3976 }
    ]
  }
}
```

(위 값은 실제 v3 모델에 불고기 test 사진을 넣은 결과다. 1순위는 돼지고기볶음, 2순위가 정답인 불고기.)

- `status`: 1순위 `confidence` ≥ `minConfidence` 이면 `SUCCESS`, 아니면 `FAILED`.
- **`FAILED` 여도 `candidates` 는 내려준다.** API 서버는 이를 `RECOGNITION_ITEMS` 에 저장하고(기준값 조정·분석용), 앱 응답에서는 `items` 를 빈 배열로 보낸다 (위키 API 문서 4절).
- `confidence` 는 softmax 확률이다. 후보를 모두 더해도 1이 되지 않는다 (Top-N 만 보내므로).

| HTTP | code | 언제 |
| --- | --- | --- |
| 400 | `INVALID_REQUEST` | `file` 이 없거나 빈 파일 |
| 413 | `IMAGE_TOO_LARGE` | `MAX_IMAGE_BYTES` 초과, 또는 해상도가 지나치게 큼 |
| 415 | `UNSUPPORTED_IMAGE` | 이미지가 아니거나 JPEG/PNG 가 아님 |

### POST `/detect` — 한 사진에 여러 음식 (2026-11-11 추가)

`/predict` 와 같은 업로드(`file`)·같은 오류 코드. 사진에서 음식 상자를 찾고(검출), 상자마다 잘라서 분류한다.

```json
{
  "success": true,
  "data": {
    "model": { "name": "efficientnet_b0_food", "version": "v0.1.0" },
    "minConfidence": 0.3,
    "imageWidth": 1024, "imageHeight": 768,
    "items": [
      { "itemNo": 1, "box": { "x0": 0.08, "y0": 0.12, "x1": 0.52, "y1": 0.81 }, "detConfidence": 0.93,
        "status": "SUCCESS", "candidates": [ { "rankNo": 1, "foodId": 1, "confidence": 0.91 } ] }
    ]
  }
}
```

- `box` 는 사진(EXIF 회전 반영) 대비 0~1 비율. `items` 는 검출 신뢰도 높은 순서.
- `cropShortSide` 는 상자의 짧은 변(px), `small` 은 그 값이 `DETECT_SMALL_PX` 미만인지. 음식이 작게 찍히면 정확도가 떨어진다 (아래 표).
- 상자를 하나도 못 찾으면 `items` 는 빈 배열 → 앱은 "음식을 찾지 못했어요". 음식이 아닌 사진이 여기서 많이 걸러진다.
- `status` 는 상자마다 `/predict` 와 같은 규칙(1순위 신뢰도 ≥ `minConfidence`).
- `DETECT_MODEL_DIR` 이 없으면 `503 DETECTOR_NOT_AVAILABLE`.

서버 코드로 잰 결과 (`scripts/eval_detect.py`, 2026-11-11):

| | `/predict` (사진 전체) | `/detect` (상자 잘라서) |
| --- | --- | --- |
| 음식 test 1,520장 top-1 / top-5 | 74.5% / 92.8% | **82.6% / 95.8%** |
| 칼로리 오차 평균 | 47.2 kcal | **30.5 kcal** |
| 음식 사진을 거절 (잘못 거절) | 18.5% | **9.0%** |
| 음식 아닌 사진 320장 거절 | 62.2% | **86.3%** (그중 상자 없음 75.0%) |
| CPU 시간 (사진 1장, 이 PC) | 약 25ms | 약 80ms |

#### 작은 설정 조정 (`scripts/tune_detect.py`, 2026-11-25)

값은 val 912장으로 고르고 test 1,520장으로 확인했다 (test 로 고르면 점수가 부풀려진다).

| 시험한 것 | val top-1 | test top-1 | 결정 |
| --- | --- | --- | --- |
| 자르는 여유 5% (지금) | 82.2% | 82.6% | **유지** (분류 학습 데이터와 같은 값) |
| 자르는 여유 10% | 83.2% | 82.3% | val 에서만 좋았다. 차이가 우연 범위(약 ±1%p) 안 |
| 좌우 뒤집어 한 번 더 보기 (여유 5%) | 82.8% | 83.0% | **쓰지 않음**. 우연 범위 안인데 분류 시간이 약 2배 (25 → 44ms) |

음식 부분의 크기별 정확도 (val, 사진을 줄여 라벨 상자의 짧은 변을 맞춤):

| 짧은 변 | 48px | 64px | 96px | 128px | 160px | 224px | 320px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top-1 | 34.5% | 49.5% | 66.5% | 76.0% | 79.4% | 81.6% | 81.8% |

128px 아래에서 크게 떨어져 `DETECT_SMALL_PX` 를 128 로 정했다. 단 이 표는 1순위 정답률이라, 음식이 작으면 맞히더라도 신뢰도가 낮아져 `FAILED` 가 늘어난다. 여러 음식을 작게 이어 붙인 시험 사진(음식 짧은 변 149~164px)에서는 `small` 이 붙지 않았지만 3개 모두 `FAILED` 였다 (표본 1장이라 기준을 바꾸지는 않았다).

검출 전처리는 Ultralytics 와 같다: 비율을 유지해 640 에 맞게 줄이고 가운데 놓기(남는 곳 회색 114) → 0~1. 출력 (1, 5, N) 을 신뢰도로 거르고 NMS 를 서버에서 한다 (`app/detector.py`).

## 전처리 (학습 때와 같아야 한다)

`app/classifier.py` 가 `model_meta.json` 값대로 처리한다.

1. EXIF 회전 정보대로 돌린다 (휴대폰 세로 사진)
2. RGB 로 바꾼다
3. 짧은 변을 `resize_short_side`(v3: 329)로 Resize (BILINEAR, 긴 변은 torchvision 과 같이 버림)
4. 가운데 `center_crop`(v3: 288) × 288 을 자른다
5. 0~1 로 나누고 mean/std 로 정규화 → NCHW float32

학습 쪽 torchvision 과 같은지는 실제 모델로 확인한다 (torch 가 필요해서 `ml/.venv` 로 실행).

```powershell
ml\.venv\Scripts\python.exe server\inference\scripts\check_against_torchvision.py --model-dir C:\food-data\models_v3 --data C:\food-data\processed_full
```

2026-10-27 결과: 입력 최대 차이 0 (200장), 서버 코드로 test 5,793장 top-1 81.94% / top-5 96.01% — `export_onnx.py` 결과와 같다.

## 시험

```powershell
cd server\inference
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

실제 모델 대신 `tests/conftest.py` 가 만드는 작은 ONNX 모델(클래스 3개)로 돌린다. GitHub Actions `inference` 가 `server/inference/**` 가 바뀔 때 같은 시험을 돌린다.

## 알려진 한계

- 분류 학습은 **잘라 낸 음식 상자**로 했다. `/predict` 에 사진 전체를 넣으면 top-1 이 74.5% 로 떨어진다 (잘라 낸 사진 기준 81.9%). `/detect` 는 상자를 잘라 넣어 82.6% 다.
- `/predict` 는 기준값 하나로 음식 아닌 사진을 약 62%만 거른다 (`/detect` 는 86%).
- 검출 학습 사진은 거의 모두 음식이 하나다. 여러 음식이 함께 찍힌 사진의 정확도는 재지 못했다.
- 304종만 인식한다. 원천 이미지가 없는 96종은 후보에 나오지 않는다 (수동 검색).
- API 서버(`/images`, `/recognitions`, 비동기 처리, DB 저장)는 만들지 않았다. 과제 범위에서는 앱이 직접 부른다.
