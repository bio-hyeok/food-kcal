# food-kcal 모바일 앱

음식 사진을 고르면 음식을 자동으로 찾아 후보를 보여주고, 먹은 양을 고르면 칼로리를 보여주는 앱입니다. Expo(React Native)로 만들었습니다.

## 지금 되는 것

- 갤러리/카메라로 사진 고르기 → **추론 서버가 음식 후보 5개를 찾아 줌** (이름, 신뢰도, 1인분 칼로리)
- 후보 중 하나를 고르거나, 목록에 없으면 음식 이름 검색으로 직접 고르기
- 인분(0.5/1/1.5/2) 또는 g으로 먹은 양 입력 → 칼로리·탄수화물·단백질·지방·나트륨 표시
- 인식에 실패하면(음식이 아니거나 신뢰도가 낮음) "음식을 찾지 못했어요" 안내 후 검색으로 연결

## 과제 시연용 구조 (위키 설계와 다른 점)

```
앱 ──사진──▶ 추론 서버 (server/inference) ──▶ 후보 foodId + 신뢰도
 └─ 영양정보는 앱 안의 src/foods.json 에서 찾는다
```

위키 설계는 `앱 → API 서버 → 추론 서버 + DB` 이다. 이번 과제는 배포하지 않고 `develop` 에서 돌아가는지만 확인하므로 **API 서버를 건너뛰고 앱이 추론 서버를 직접 부른다.** 로그인, 식단 저장, 통계는 없다. API 서버를 만들면 `src/recognize.js` 만 `POST /images`, `/recognitions` 로 바꾸면 된다.

## 실행 (시연 순서)

### 1. 추론 서버 켜기

모델 파일(`classifier.onnx`, `model_meta.json`)은 Git 에 없다. `ml/scripts/export_onnx.py` 로 만든 폴더를 `MODEL_DIR` 로 준다. 자세한 내용은 [server/inference/README.md](../server/inference/README.md).

```powershell
cd server\inference
$env:MODEL_DIR = "C:\food-data\models_v3"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8100
```

- `--host 0.0.0.0` 은 휴대폰에서 접속하기 위해서다. PC 웹으로만 볼 때는 `127.0.0.1` 이어도 된다.
- 처음 켜면 **Windows 방화벽 허용 창**이 뜰 수 있다. 휴대폰으로 시연하려면 "개인 네트워크"만 허용한다.
- 켜졌는지 확인: 브라우저에서 `http://localhost:8100/health`

### 2. 앱 켜기

```bash
cd app
npm install
npx expo start
```

- **PC 웹으로 보기**: 터미널에서 `w` 를 누르거나 `npx expo start --web`. 앱은 `http://localhost:8100` 으로 서버를 부른다.
- **휴대폰(Expo Go)**: QR 코드를 스캔한다. PC 와 휴대폰이 **같은 Wi-Fi** 여야 한다. 앱은 Expo 개발 서버가 도는 PC 의 IP 로 서버를 찾는다 (예: `http://192.168.0.5:8100`).
- 서버 주소를 직접 정하려면 `app/.env` 에 적는다 (`.env` 는 Git 에 올리지 않는다).

  ```
  EXPO_PUBLIC_INFERENCE_URL=http://192.168.0.5:8100
  ```

### 3. 사진으로 확인

1. "갤러리" 또는 "카메라"로 음식 사진을 고른다.
2. "음식을 찾는 중이에요…" 뒤에 **"이 음식인가요?"** 후보 5개가 나온다.
3. 후보를 누르면 1인분 칼로리가 나온다. 인분이나 g 을 바꾸면 다시 계산한다.

연결이 안 되면 화면에 "인식 서버에 연결하지 못했어요 (주소)" 가 나온다. 서버가 켜져 있는지, 같은 Wi-Fi 인지, 방화벽을 허용했는지 확인한다.

## 알아 둘 점

- 모델은 304종만 안다. 김치, 나물, 회, 떡 등 원천 이미지가 없던 96종은 후보에 나오지 않으니 검색으로 고른다.
- test 기준 1순위 정답률 81.9%, 후보 5개 안에 정답 96.0% (잘라 낸 음식 사진 기준). 사진 전체로 찍으면 더 낮을 수 있다.
- 음식이 아닌 사진도 약 40%는 후보가 나온다 (신뢰도 기준값 0.30 의 한계, `ml/README.md`).
- 칼로리는 AI-Hub 영양DB의 한 접시 기준 값이다.

## 테스트

```bash
npm test
```

칼로리 계산(`src/calc.js`), 영양 데이터 형식, 인식 응답 처리(`src/recognize.js`)를 확인한다. 계산 규칙은 위키 [ERD 설계 문서](https://github.com/bio-hyeok/food-kcal/wiki/ERD-설계-문서) 4절을 따른다.

## 구조

| 파일 | 역할 |
| --- | --- |
| `App.js` | 화면 하나 (사진 → 후보 또는 검색 → 양 입력 → 결과) |
| `src/recognize.js` | 추론 서버 주소 정하기, 사진 보내기, 응답을 음식 데이터와 묶기 |
| `src/calc.js` | 칼로리 계산 |
| `src/foods.json` | 음식·영양 데이터 400종 (AI-Hub 74번 영양DB). 직접 고치지 말고 `ml/scripts/build_foods_json.mjs` 로 다시 만든다 |
| `src/foods.sample.json` | 계산 테스트용 샘플 11개 |
