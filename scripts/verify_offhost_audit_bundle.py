#!/usr/bin/env python3
"""Independently verify a MineralVision signed audit JSON Lines bundle.

The verifier intentionally takes only an audit bundle and a public-key registry.
It never connects to the source database and never loads a signing private key.

Example:
  python scripts/verify_offhost_audit_bundle.py \
    --bundle /secure/collector/tenant-a-connector-1.jsonl \
    --public-keys /secure/collector/audit-public-keys.json \
    --tenant tenant-a --stream connector:arcgis-prod-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package"))

from src.api.innovations.integration_hub.audit_crypto import (  # noqa: E402
    public_key_from_b64,
    verify_audit_chain,
)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"record on line {line_number} is not a JSON object")
        records.append(record)
    if not records:
        raise ValueError("audit bundle contains no records")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Signed JSON Lines bundle")
    parser.add_argument("--public-keys", type=Path, required=True, help="JSON object: key_id -> raw Ed25519 public key base64")
    parser.add_argument("--tenant", required=True, help="Expected tenant identifier")
    parser.add_argument("--stream", required=True, help="Expected audit stream identifier")
    parser.add_argument("--prior-hash", default="", help="Expected previous anchor hash for incremental bundles")
    parser.add_argument("--start-sequence", type=int, default=1, help="Expected first sequence number")
    args = parser.parse_args()

    key_doc = json.loads(args.public_keys.read_text(encoding="utf-8"))
    if not isinstance(key_doc, dict):
        raise ValueError("public key registry must be a JSON object")
    public_keys = {key_id: public_key_from_b64(value) for key_id, value in key_doc.items()}
    result = verify_audit_chain(
        _read_jsonl(args.bundle),
        public_keys=public_keys,
        expected_tenant_id=args.tenant,
        expected_stream_id=args.stream,
        expected_prior_hash=args.prior_hash,
        expected_start_sequence=args.start_sequence,
    )
    output = {
        "valid": result.valid,
        "event_count": result.event_count,
        "last_sequence": result.last_sequence,
        "last_event_hash": result.last_event_hash,
        "failures": list(result.failures),
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0 if result.valid else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"valid": False, "fatal_error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
