# DB 스키마

기준 문서: [ERD 설계 문서](https://github.com/bio-hyeok/food-kcal/wiki/ERD-설계-문서)

| 파일 | 내용 |
| --- | --- |
| `migrations/001_init_schema.sql` | 초기 테이블 11개와 인덱스 |
| `migrations/001_init_schema.down.sql` | 위 스키마 삭제 (되돌리기) |

## 적용

```bash
psql "$DATABASE_URL" -f server/db/migrations/001_init_schema.sql
```

## 테이블

`users` · `user_goals` · `food_images` · `model_versions` · `foods` · `food_aliases` · `nutrition_facts` · `recognition_results` · `recognition_items` · `meals` · `meal_items`

## ERD 문서 DDL과 다른 점

ERD 문서에 적힌 규칙을 DB 제약으로 옮겼습니다.

- `model_versions`: 서비스 중인 모델(`is_active = true`)은 1건만 허용 (부분 유니크 인덱스)
- `recognition_results.status`: `PENDING` / `SUCCESS` / `FAILED`
- `meals.meal_type`: `BREAKFAST` / `LUNCH` / `DINNER` / `SNACK`
- `meal_items.input_type`: `DEFAULT_SERVING` / `SERVING_RATIO` / `GRAM` / `AI_ESTIMATE`
- `meal_items`: `amount_g`와 `serving_ratio` 중 하나는 반드시 값이 있어야 함
- `recognition_items.confidence`: 0~1 범위
- `nutrition_facts.basis_g`: 값이 있다면 0보다 커야 함 (0으로 나누는 계산 방지)
