# ml/ — 데이터 변환·학습 보조

## 영양 데이터 만들기 (`feature/food-data`)

```bash
cd ml
npm install
node scripts/build_foods_json.mjs "<음식분류 AI 데이터 영양DB.xlsx 경로>"
```

- 결과: `app/src/foods.json`(앱용 400종), `ml/label_map.csv`(`food_id`, `name_ko`. `dataset_label`은 학습 데이터와 맞출 때 채운다)
- 영양값은 **한 접시(양 Q4=100%) 기준**이다. `basisG = servingSizeG = 중량(g)`.
- `foodId`는 엑셀 행 순서(1부터)이며 모델 클래스 번호와 이어진다. **한 번 정하면 바꾸지 않는다.** 엑셀이 바뀌면 기존 행 순서는 두고 뒤에만 추가한다.
- 원본 엑셀은 AI-Hub 데이터라 저장소에 올리지 않는다.
- 양 단계(Q1~Q5) = 25 / 50 / 75 / 100 / 125 %. `먹은 양(g) = 중량 × 비율`.
