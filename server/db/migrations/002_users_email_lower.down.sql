DROP INDEX IF EXISTS uq_users_email_lower;
ALTER TABLE users ADD CONSTRAINT users_email_key UNIQUE (email);
