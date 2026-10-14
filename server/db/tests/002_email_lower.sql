-- 대소문자만 다른 이메일은 중복 가입이 막혀야 한다. 끝에서 ROLLBACK 한다.
BEGIN;

DO $$
BEGIN
  INSERT INTO users(email, nickname, password_hash) VALUES ('Case@Test.com', 't', 'x');
  BEGIN
    INSERT INTO users(email, nickname, password_hash) VALUES ('case@test.com', 't', 'x');
    RAISE EXCEPTION '대소문자만 다른 이메일이 중복 가입됨';
  EXCEPTION WHEN unique_violation THEN
    NULL;
  END;
  -- 다른 이메일은 통과해야 한다
  INSERT INTO users(email, nickname, password_hash) VALUES ('other@test.com', 't', 'x');
END $$;

ROLLBACK;
