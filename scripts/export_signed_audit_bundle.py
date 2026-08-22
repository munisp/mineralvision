#!/usr/bin/env python3
"""Export a tenant audit stream as signed JSON Lines for off-host verification.

This exporter reads only immutable signed event fields and writes no private-key
material. The output must be transferred over mTLS/allow-listed transport to an
independently administered retention-controlled collector.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package"))

from src.api.innovations.integration_hub.audit_crypto import jsonl_export  # noqa: E402
from src.api.innovations.integration_hub.db import get_session_factory  # noqa: E402
from src.api.innovations.integration_hub.models import SignedAuditEventModel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--stream", required=True)
    parser.add_argument("--after-sequence", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.getenv("ENV", "development").lower() in {"production", "preproduction", "staging"}:
        if not os.getenv("MV_HUB_DATABASE_URL", "").startswith("postgresql"):
            raise RuntimeError("production export requires a configured PostgreSQL MV_HUB_DATABASE_URL")

    session = get_session_factory()()
    try:
        events = (
            session.query(SignedAuditEventModel)
            .filter(
                SignedAuditEventModel.tenant_id == args.tenant,
                SignedAuditEventModel.stream_id == args.stream,
                SignedAuditEventModel.sequence > args.after_sequence,
            )
            .order_by(SignedAuditEventModel.sequence)
            .all()
        )
        if not events:
            raise RuntimeError("no signed audit events matched the requested export")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(jsonl_export(events), encoding="utf-8")
        print(
            f"exported {len(events)} events tenant={args.tenant!r} stream={args.stream!r} "
            f"sequence={events[0].sequence}-{events[-1].sequence} last_hash={events[-1].event_hash}"
        )
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
