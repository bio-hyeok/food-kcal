-- 001_init_schema.sql 되돌리기 (의존 관계의 역순으로 삭제)
DROP TABLE IF EXISTS meal_items;
DROP TABLE IF EXISTS meals;
DROP TABLE IF EXISTS recognition_items;
DROP TABLE IF EXISTS recognition_results;
DROP TABLE IF EXISTS nutrition_facts;
DROP TABLE IF EXISTS food_aliases;
DROP TABLE IF EXISTS foods;
DROP TABLE IF EXISTS model_versions;
DROP TABLE IF EXISTS food_images;
DROP TABLE IF EXISTS user_goals;
DROP TABLE IF EXISTS users;
