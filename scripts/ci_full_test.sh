#!/usr/bin/env bash
# Full CI gate: execute only against a disposable PostgreSQL/PostGIS database.
set -euo pipefail

: "${MV_TEST_DATABASE_URL:?MV_TEST_DATABASE_URL must reference the isolated CI PostgreSQL database}"
if [[ ! "$MV_TEST_DATABASE_URL" =~ ^postgres(ql)?(\+[a-z0-9_]+)?:// ]]; then
  echo "MV_TEST_DATABASE_URL must be a PostgreSQL URL; refusing SQLite or production-default test run." >&2
  exit 2
fi

export DATABASE_URL="$MV_TEST_DATABASE_URL"
export ENV="test"
export AUTH_MODE="local"
export JWT_SECRET="${JWT_SECRET:-ci-test-secret-not-for-production}"
export PYTHONDONTWRITEBYTECODE=1

# The lock must be committed and unchanged. Install all groups needed by the
# 405-candidate suite (geospatial plus heavy JEPA/WALDO ML tests).
uv lock --locked
uv sync --locked --all-extras --dev

pushd MineralVision_Final_Package >/dev/null
alembic upgrade head
popd >/dev/null

python3 scripts/verify.py
python3 scripts/verify_security_baseline.py
python3 scripts/verify_operations_baseline.py
pytest -q --disable-warnings --maxfail=1
