-- 포함 검색 결과가 맞고, 3글자 이상 검색어는 트라이그램 인덱스를 쓸 수 있는지 확인한다. 끝에서 ROLLBACK 한다.
BEGIN;

INSERT INTO foods(name_ko) VALUES ('김치찌개'), ('된장찌개'), ('비빔밥');

DO $$
DECLARE
  n int;
  plan text := '';
  r record;
BEGIN
  SELECT count(*) INTO n FROM foods WHERE name_ko LIKE '%찌개%';
  IF n <> 2 THEN RAISE EXCEPTION '포함 검색 결과가 틀림: %', n; END IF;

  -- 테이블이 작으면 플래너가 순차 스캔을 고르므로, 인덱스를 "쓸 수 있는지"만 보도록 막아 둔다.
  SET LOCAL enable_seqscan = off;
  FOR r IN EXECUTE 'EXPLAIN SELECT food_id FROM foods WHERE name_ko LIKE ''%치찌개%''' LOOP
    plan := plan || r."QUERY PLAN" || E'\n';
  END LOOP;
  IF plan NOT LIKE '%idx_foods_name_ko_trgm%' THEN
    RAISE EXCEPTION '3글자 이상 검색어가 인덱스를 쓰지 않음: %', plan;
  END IF;
END $$;

ROLLBACK;
