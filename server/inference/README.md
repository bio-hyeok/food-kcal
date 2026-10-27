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
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

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

- 학습은 **잘라 낸 음식 상자**로 했고, 이 서버는 **사진 전체**를 받는다. 실제 정확도는 test 점수(81.9%)보다 낮을 수 있다. 원본 전체 사진 평가가 남아 있다.
- 기준값 하나로는 음식 아닌 사진을 약 62%만 거른다.
- 304종만 인식한다. 원천 이미지가 없는 96종은 후보에 나오지 않는다 (수동 검색).
- API 서버(`/images`, `/recognitions`, 비동기 처리, DB 저장)는 만들지 않았다. 과제 범위에서는 앱이 직접 부른다.
