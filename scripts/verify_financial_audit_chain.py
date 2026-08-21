#!/usr/bin/env python3
"""Verify tamper-evident financial audit chains without modifying ledger data.

Usage:
  export LEDGER_DATABASE_URL='postgresql://ledger_audit_reader:...@db/mineralvision'
  export LEDGER_AUDIT_HMAC_KEY='secret-from-kms-or-secret-manager'
  python scripts/verify_financial_audit_chain.py --all
  python scripts/verify_financial_audit_chain.py --idempotency-key payment-2026-0001

The database role must have only SELECT rights on financial_transfer_audit_events
and financial_transfer_intents.  Never place the audit HMAC key in shell history,
logs, CI output, source control, or the database itself.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from collections import defaultdict
from typing import Any


def _canonical_hash(key: bytes, row: dict[str, Any]) -> str:
    event = {
        "event_type": row["event_type"],
        "intent": row["intent_payload"],
        "actor_id": row["actor_id"],
        "details": row["details"],
        "previous_hash": row["previous_hash"],
        "key_version": row["key_version"],
    }
    encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _connect(database_url: str):
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError as exc:
        raise RuntimeError("psycopg2 is required; install the locked project dependencies") from exc
    return psycopg2.connect(database_url), psycopg2.extras.RealDictCursor


def verify(database_url: str, hmac_key: str, idempotency_key: str | None) -> dict[str, Any]:
    if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        raise ValueError("LEDGER_DATABASE_URL must be a PostgreSQL URL")
    if len(hmac_key) < 32:
        raise ValueError("LEDGER_AUDIT_HMAC_KEY must contain at least 32 characters")

    query = """
        SELECT e.sequence, e.idempotency_key, e.event_type, e.actor_id, e.details,
               e.previous_hash, e.event_hash, e.key_version, e.created_at,
               i.intent_payload
        FROM financial_transfer_audit_events AS e
        JOIN financial_transfer_intents AS i USING (idempotency_key)
    """
    params: tuple[Any, ...] = ()
    if idempotency_key:
        query += " WHERE e.idempotency_key = %s"
        params = (idempotency_key,)
    query += " ORDER BY e.idempotency_key, e.sequence"

    connection, cursor_factory = _connect(database_url.replace("postgresql+psycopg2://", "postgresql://"))
    try:
        with connection, connection.cursor(cursor_factory=cursor_factory) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError("no audit events matched the requested scope")

    failures: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    previous_by_key: dict[str, str] = defaultdict(str)
    secret = hmac_key.encode("utf-8")
    for row in rows:
        row = dict(row)
        key = row["idempotency_key"]
        counts[key] += 1
        expected_previous = previous_by_key[key]
        computed_hash = _canonical_hash(secret, row)
        reasons: list[str] = []
        if not hmac.compare_digest(row["previous_hash"], expected_previous):
            reasons.append("previous_hash_mismatch")
        if not hmac.compare_digest(row["event_hash"], computed_hash):
            reasons.append("event_hmac_mismatch")
        if reasons:
            failures.append({
                "sequence": row["sequence"],
                "idempotency_key": key,
                "reasons": reasons,
                "stored_hash": row["event_hash"],
                "computed_hash": computed_hash,
            })
        previous_by_key[key] = row["event_hash"]

    return {
        "ok": not failures,
        "stream_count": len(counts),
        "event_count": len(rows),
        "events_per_stream": dict(counts),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only financial audit HMAC-chain verifier")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="verify every financial transfer event stream")
    target.add_argument("--idempotency-key", help="verify one transfer event stream")
    parser.add_argument("--database-url", default=os.getenv("LEDGER_DATABASE_URL", ""))
    parser.add_argument("--hmac-key", default=os.getenv("LEDGER_AUDIT_HMAC_KEY", ""))
    args = parser.parse_args()

    try:
        outcome = verify(args.database_url, args.hmac_key, args.idempotency_key)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
