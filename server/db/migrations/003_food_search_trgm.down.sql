-- pg_trgm 확장은 다른 곳에서 쓸 수 있어 남겨 둔다.
DROP INDEX IF EXISTS idx_food_aliases_name_trgm;
DROP INDEX IF EXISTS idx_foods_name_ko_trgm;
