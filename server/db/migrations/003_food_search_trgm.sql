-- 이름에 검색어가 "포함된" 음식 검색(LIKE '%q%')용 인덱스. 검색어가 3글자 이상일 때 쓰인다.
-- shortcut: 1~2글자 검색어는 인덱스를 못 타고 순차 스캔한다. 음식이 수만 건 이하면 충분하고, 더 커지면 접두어 검색이나 전문 검색을 검토한다.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_foods_name_ko_trgm ON foods USING gin (name_ko gin_trgm_ops);
CREATE INDEX idx_food_aliases_name_trgm ON food_aliases USING gin (alias_name gin_trgm_ops);
