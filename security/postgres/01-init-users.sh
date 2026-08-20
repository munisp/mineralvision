#!/usr/bin/env bash
# Runs only during PostgreSQL first-time initialization.
# Required environment values are injected by the orchestrator, never committed.
set -Eeuo pipefail

: "${POSTGRES_APP_PASSWORD:?POSTGRES_APP_PASSWORD is required}"
: "${POSTGRES_MIGRATOR_PASSWORD:?POSTGRES_MIGRATOR_PASSWORD is required}"
: "${KEYCLOAK_DB_PASSWORD:?KEYCLOAK_DB_PASSWORD is required}"

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" --set ON_ERROR_STOP=1 \
  --set app_password="$POSTGRES_APP_PASSWORD" \
  --set migrator_password="$POSTGRES_MIGRATOR_PASSWORD" \
  --set keycloak_password="$KEYCLOAK_DB_PASSWORD" <<'SQL'
CREATE ROLE mineralvision_migrator LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD :'migrator_password';
CREATE ROLE mineralvision_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD :'app_password';
CREATE ROLE keycloak_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT PASSWORD :'keycloak_password';

CREATE DATABASE mineralvision OWNER mineralvision_migrator;
CREATE DATABASE keycloak OWNER keycloak_app;
SQL

psql --username "$POSTGRES_USER" --dbname mineralvision --set ON_ERROR_STOP=1 <<'SQL'
REVOKE ALL ON DATABASE mineralvision FROM PUBLIC;
GRANT CONNECT, TEMPORARY ON DATABASE mineralvision TO mineralvision_app;
GRANT CONNECT, TEMPORARY ON DATABASE mineralvision TO mineralvision_migrator;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO mineralvision_app;
-- Alembic runs as mineralvision_migrator. Its future tables/sequences grant only
-- the DML privileges the API needs; schema changes remain migration-only.
ALTER DEFAULT PRIVILEGES FOR USER mineralvision_migrator IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mineralvision_app;
ALTER DEFAULT PRIVILEGES FOR USER mineralvision_migrator IN SCHEMA public
  GRANT USAGE, SELECT, UPDATE ON SEQUENCES TO mineralvision_app;
SQL

psql --username "$POSTGRES_USER" --dbname keycloak --set ON_ERROR_STOP=1 <<'SQL'
REVOKE ALL ON DATABASE keycloak FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
SQL
