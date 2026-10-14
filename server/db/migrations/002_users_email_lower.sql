-- 이메일을 대소문자 구분 없이 유일하게 한다. 원본 표기는 그대로 저장한다.
-- 조회할 때는 반드시 lower(email) = lower($1) 로 비교해야 이 인덱스를 탄다.
ALTER TABLE users DROP CONSTRAINT users_email_key;
CREATE UNIQUE INDEX uq_users_email_lower ON users (lower(email));
