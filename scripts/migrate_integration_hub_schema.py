#!/usr/bin/env python3
"""Idempotently migrate the standalone integration-hub PostgreSQL schema.

The legacy integration hub used ``Base.metadata.create_all`` and therefore
cannot add columns/tables to an already deployed database. Run this script with
an approved migrator role before deploying the governed evidence/audit feature.
It intentionally refuses SQLite and never runs from the API request path.
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text


def main() -> int:
    url = os.getenv("MV_HUB_DATABASE_URL", "")
    if not url.startswith("postgresql"):
        raise RuntimeError("MV_HUB_DATABASE_URL must be a PostgreSQL URL for this migration")

    engine = create_engine(url, pool_pre_ping=True)
    statements = [
        "ALTER TABLE hub_api_keys ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(128) NOT NULL DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS ix_hub_api_keys_tenant_id ON hub_api_keys (tenant_id)",
        """
        CREATE TABLE IF NOT EXISTS hub_audit_streams (
            id SERIAL PRIMARY KEY,
            tenant_id VARCHAR(128) NOT NULL,
            stream_id VARCHAR(128) NOT NULL,
            last_sequence INTEGER NOT NULL DEFAULT 0,
            last_event_hash VARCHAR(64) NOT NULL DEFAULT '',
            active_key_id VARCHAR(128) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_hub_audit_stream_tenant_stream UNIQUE (tenant_id, stream_id)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_hub_audit_streams_tenant_id ON hub_audit_streams (tenant_id)",
        """
        CREATE TABLE IF NOT EXISTS hub_signed_audit_events (
            id SERIAL PRIMARY KEY,
            event_id VARCHAR(40) NOT NULL UNIQUE,
            tenant_id VARCHAR(128) NOT NULL,
            stream_id VARCHAR(128) NOT NULL,
            sequence INTEGER NOT NULL,
            previous_hash VARCHAR(64) NOT NULL DEFAULT '',
            event_hash VARCHAR(64) NOT NULL,
            key_id VARCHAR(128) NOT NULL,
            signature_b64 VARCHAR(256) NOT NULL,
            event_type VARCHAR(128) NOT NULL,
            actor_id VARCHAR(128) NOT NULL,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            occurred_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT uq_hub_signed_event_stream_sequence UNIQUE (tenant_id, stream_id, sequence)
        )
        """,
        "CREATE INDEX IF NOT EXISTS ix_hub_signed_events_tenant_id ON hub_signed_audit_events (tenant_id)",
        "CREATE INDEX IF NOT EXISTS ix_hub_signed_events_stream_id ON hub_signed_audit_events (stream_id)",
        "CREATE INDEX IF NOT EXISTS ix_hub_signed_events_event_hash ON hub_signed_audit_events (event_hash)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        # Runtime application role receives only append/read permissions.
        # Substitute real role names through deployment IaC, not this generic script.
    print("integration hub governed evidence and signed audit schema migration completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
