#!/usr/bin/env bash
# 빈 DB에 마이그레이션을 순서대로 적용 → 제약 테스트 → 되돌리기 → 테이블이 0개인지 확인한다.
# 접속 정보는 PGHOST, PGUSER, PGPASSWORD, PGDATABASE 환경변수를 쓴다.
set -euo pipefail
cd "$(dirname "$0")"
run() { psql -v ON_ERROR_STOP=1 -q -f "$1"; }

for f in $(ls migrations/*.sql | grep -v '\.down\.sql$' | sort); do echo "up:   $f"; run "$f"; done
for f in $(ls tests/*.sql 2>/dev/null | sort); do echo "test: $f"; run "$f"; done
for f in $(ls migrations/*.down.sql | sort -r); do echo "down: $f"; run "$f"; done

left=$(psql -tA -c "select count(*) from information_schema.tables where table_schema = 'public'")
[ "$left" = "0" ] || { echo "되돌린 뒤 테이블이 남아 있음: $left"; exit 1; }
echo "ok"
