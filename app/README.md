# food-kcal 모바일 앱

사진을 첨부하고, 음식과 먹은 양을 고르면 칼로리를 보여주는 앱입니다. Expo(React Native)로 만들었습니다.

## 지금 되는 것

- 갤러리/카메라로 사진 첨부 (화면에 표시만 함)
- 음식 이름 검색 → 선택
- 인분(0.5/1/1.5/2) 또는 g으로 먹은 양 입력 → 칼로리·탄수화물·단백질·지방·나트륨 표시

## 아직 없는 것 (의도적으로 제외)

- 사진으로 음식을 자동 인식하는 기능과 모델 학습 (사진은 참고용으로만 보여줌)
- 서버 연동, 로그인, 식단 기록 저장
- 영양 데이터는 `src/foods.sample.json`의 **샘플 값**이다. 실제 영양정보(식약처 DB, AI-Hub 등)로 교체해야 한다.

## 실행

```bash
cd app
npm install
npx expo start
```

- 휴대폰: Expo Go 앱으로 터미널의 QR 코드를 스캔한다(휴대폰과 PC가 같은 네트워크여야 함).
- 웹으로 확인: `npx expo start --web`

## 테스트

```bash
npm test
```

칼로리 계산(`src/calc.js`)과 샘플 데이터 형식을 확인한다. 계산 규칙은 위키 [ERD 설계 문서](https://github.com/bio-hyeok/food-kcal/wiki/ERD-설계-문서) 4절을 따른다.

## 구조

| 파일 | 역할 |
| --- | --- |
| `App.js` | 화면 하나(사진 → 음식 선택 → 양 입력 → 결과) |
| `src/calc.js` | 칼로리 계산 |
| `src/foods.sample.json` | 샘플 음식·영양 데이터 (ERD의 `foods` + `nutrition_facts` 모양) |
