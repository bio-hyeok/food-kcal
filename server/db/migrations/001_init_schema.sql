-- food-kcal 초기 스키마 (PostgreSQL)
-- 기준 문서: Wiki "ERD 설계 문서"

-- 사용자 -------------------------------------------------------------------
CREATE TABLE users (
  user_id       BIGSERIAL PRIMARY KEY,
  email         VARCHAR(255) NOT NULL UNIQUE,
  nickname      VARCHAR(50)  NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  birth_date    DATE,
  gender        VARCHAR(10),
  created_at    TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE user_goals (
  goal_id    BIGSERIAL PRIMARY KEY,
  user_id    BIGINT NOT NULL UNIQUE REFERENCES users(user_id),
  daily_kcal INT,
  carb_g     NUMERIC(6,1),
  protein_g  NUMERIC(6,1),
  fat_g      NUMERIC(6,1),
  updated_at TIMESTAMP NOT NULL DEFAULT now()
);

-- 이미지 -------------------------------------------------------------------
CREATE TABLE food_images (
  image_id    BIGSERIAL PRIMARY KEY,
  user_id     BIGINT NOT NULL REFERENCES users(user_id),
  file_path   VARCHAR(500) NOT NULL,
  width       INT,
  height      INT,
  uploaded_at TIMESTAMP NOT NULL DEFAULT now()
);

-- 모델 버전 ----------------------------------------------------------------
CREATE TABLE model_versions (
  model_id       BIGSERIAL PRIMARY KEY,
  name           VARCHAR(100) NOT NULL,
  version        VARCHAR(30)  NOT NULL,
  dataset_source VARCHAR(200),
  metrics_json   JSONB,
  is_active      BOOLEAN NOT NULL DEFAULT FALSE,
  trained_at     TIMESTAMP,
  UNIQUE (name, version)
);

-- 서비스 중인 모델은 동시에 1건만 허용
CREATE UNIQUE INDEX uq_model_versions_active ON model_versions (is_active) WHERE is_active;

-- 음식·영양정보 --------------------------------------------------------------
CREATE TABLE foods (
  food_id        BIGSERIAL PRIMARY KEY,
  name_ko        VARCHAR(100) NOT NULL,
  name_en        VARCHAR(100),
  category       VARCHAR(50),
  serving_size_g NUMERIC(7,1),
  source         VARCHAR(50)
);
CREATE INDEX idx_foods_name_ko ON foods (name_ko);

CREATE TABLE food_aliases (
  alias_id   BIGSERIAL PRIMARY KEY,
  food_id    BIGINT NOT NULL REFERENCES foods(food_id),
  alias_name VARCHAR(100) NOT NULL
);
CREATE INDEX idx_food_aliases_name ON food_aliases (alias_name);

CREATE TABLE nutrition_facts (
  nutrition_id BIGSERIAL PRIMARY KEY,
  food_id      BIGINT NOT NULL UNIQUE REFERENCES foods(food_id),
  basis_g      NUMERIC(7,1),            -- 영양값의 기준 중량(g). NULL이면 1인분 기준
  kcal         NUMERIC(7,1) NOT NULL,
  carb_g       NUMERIC(6,1),
  protein_g    NUMERIC(6,1),
  fat_g        NUMERIC(6,1),
  sugar_g      NUMERIC(6,1),
  sodium_mg    NUMERIC(8,1),
  fiber_g      NUMERIC(6,1),
  CONSTRAINT chk_nutrition_basis_positive CHECK (basis_g IS NULL OR basis_g > 0)
);

-- 인식 결과 ----------------------------------------------------------------
CREATE TABLE recognition_results (
  result_id   BIGSERIAL PRIMARY KEY,
  image_id    BIGINT NOT NULL UNIQUE REFERENCES food_images(image_id),
  model_id    BIGINT NOT NULL REFERENCES model_versions(model_id),
  status      VARCHAR(20) NOT NULL,
  latency_ms  INT,
  analyzed_at TIMESTAMP,
  CONSTRAINT chk_recognition_status CHECK (status IN ('PENDING', 'SUCCESS', 'FAILED'))
);

CREATE TABLE recognition_items (
  item_id            BIGSERIAL PRIMARY KEY,
  result_id          BIGINT NOT NULL REFERENCES recognition_results(result_id),
  food_id            BIGINT NOT NULL REFERENCES foods(food_id),
  confidence         NUMERIC(5,4) NOT NULL,
  rank_no            INT NOT NULL,
  bbox_json          JSONB,
  estimated_amount_g NUMERIC(7,1),
  CONSTRAINT chk_recognition_confidence CHECK (confidence BETWEEN 0 AND 1)
);
CREATE INDEX idx_recognition_items_result_rank ON recognition_items (result_id, rank_no);

-- 식단 기록 ----------------------------------------------------------------
CREATE TABLE meals (
  meal_id   BIGSERIAL PRIMARY KEY,
  user_id   BIGINT NOT NULL REFERENCES users(user_id),
  image_id  BIGINT REFERENCES food_images(image_id),
  meal_type VARCHAR(20) NOT NULL,
  eaten_at  TIMESTAMP NOT NULL,
  memo      VARCHAR(300),
  CONSTRAINT chk_meals_type CHECK (meal_type IN ('BREAKFAST', 'LUNCH', 'DINNER', 'SNACK'))
);
CREATE INDEX idx_meals_user_eaten ON meals (user_id, eaten_at);

CREATE TABLE meal_items (
  meal_item_id      BIGSERIAL PRIMARY KEY,
  meal_id           BIGINT NOT NULL REFERENCES meals(meal_id) ON DELETE CASCADE,
  food_id           BIGINT NOT NULL REFERENCES foods(food_id),
  source_item_id    BIGINT REFERENCES recognition_items(item_id),
  amount_g          NUMERIC(7,1),
  serving_ratio     NUMERIC(4,2),
  input_type        VARCHAR(20) NOT NULL,
  kcal_snapshot     NUMERIC(7,1) NOT NULL,
  carb_snapshot     NUMERIC(6,1),
  protein_snapshot  NUMERIC(6,1),
  fat_snapshot      NUMERIC(6,1),
  is_user_corrected BOOLEAN NOT NULL DEFAULT FALSE,
  CONSTRAINT chk_meal_items_input_type CHECK (input_type IN ('DEFAULT_SERVING', 'SERVING_RATIO', 'GRAM', 'AI_ESTIMATE')),
  CONSTRAINT chk_meal_items_amount CHECK (amount_g IS NOT NULL OR serving_ratio IS NOT NULL)
);
CREATE INDEX idx_meal_items_meal ON meal_items (meal_id);
