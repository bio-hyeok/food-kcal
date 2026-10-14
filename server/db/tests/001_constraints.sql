-- 001 스키마의 CHECK·유니크 제약이 실제로 막는지 확인한다. 끝에서 ROLLBACK 하므로 데이터는 남지 않는다.
BEGIN;

-- stmt 가 expected(SQLSTATE) 오류로 실패해야 통과. 성공하거나 다른 오류면 테스트 실패.
CREATE FUNCTION pg_temp.must_fail(stmt text, expected text) RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  BEGIN
    EXECUTE stmt;
  EXCEPTION WHEN OTHERS THEN
    IF SQLSTATE = expected THEN RETURN; END IF;
    RAISE;
  END;
  RAISE EXCEPTION '제약이 통과됨: %', stmt;
END $$;

DO $$
DECLARE u bigint; f bigint; img bigint; img2 bigint; mdl bigint; res bigint; m bigint;
BEGIN
  INSERT INTO users(email, nickname, password_hash) VALUES ('t@test.com', 't', 'x') RETURNING user_id INTO u;
  INSERT INTO foods(name_ko) VALUES ('테스트') RETURNING food_id INTO f;
  INSERT INTO food_images(user_id, file_path) VALUES (u, 'a.jpg') RETURNING image_id INTO img;
  INSERT INTO food_images(user_id, file_path) VALUES (u, 'b.jpg') RETURNING image_id INTO img2;
  INSERT INTO model_versions(name, version, is_active) VALUES ('m', 'v1', true) RETURNING model_id INTO mdl;
  INSERT INTO recognition_results(image_id, model_id, status) VALUES (img, mdl, 'SUCCESS') RETURNING result_id INTO res;
  INSERT INTO meals(user_id, meal_type, eaten_at) VALUES (u, 'LUNCH', now()) RETURNING meal_id INTO m;

  -- 23505 unique_violation, 23514 check_violation
  PERFORM pg_temp.must_fail($q$INSERT INTO model_versions(name, version, is_active) VALUES ('m', 'v2', true)$q$, '23505');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO recognition_results(image_id, model_id, status) VALUES (%s, %s, 'BAD')$q$, img2, mdl), '23514');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO recognition_items(result_id, food_id, confidence, rank_no) VALUES (%s, %s, 1.5, 1)$q$, res, f), '23514');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO nutrition_facts(food_id, basis_g, kcal) VALUES (%s, 0, 1)$q$, f), '23514');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO meals(user_id, meal_type, eaten_at) VALUES (%s, 'BRUNCH', now())$q$, u), '23514');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO meal_items(meal_id, food_id, input_type, serving_ratio, kcal_snapshot) VALUES (%s, %s, 'BAD', 1, 1)$q$, m, f), '23514');
  PERFORM pg_temp.must_fail(format($q$INSERT INTO meal_items(meal_id, food_id, input_type, kcal_snapshot) VALUES (%s, %s, 'GRAM', 1)$q$, m, f), '23514');

  -- 정상 입력은 통과해야 한다
  INSERT INTO meal_items(meal_id, food_id, input_type, serving_ratio, kcal_snapshot) VALUES (m, f, 'DEFAULT_SERVING', 1, 210);
END $$;

ROLLBACK;
