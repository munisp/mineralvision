#!/usr/bin/env python3
"""Pre-production financial integration test for a LIVE OPA server and TigerBeetle cluster.

This script calls the deployed OPA REST API using the exact input contract from
security/opa/financial_transfer.rego. It also performs a TCP liveness probe of
the TigerBeetle cluster. It deliberately DOES NOT create accounts or transfers:
actual value movement must be covered by a licensed payment-partner sandbox and
an independently reviewed payment-service adapter.

Required environment:
  OPA_URL=http://opa:8181
  TIGERBEETLE_ADDRESS=tigerbeetle:3000

Optional:
  FINANCIAL_HIGH_VALUE_MINOR=5000000  # default: USD 50,000 in minor units
"""
from __future__ import annotations

import json
import os
import socket
import sys
from urllib.parse import urlparse

import requests

OPA_URL = os.environ.get("OPA_URL", "http://opa:8181").rstrip("/")
TB_ADDRESS = os.environ.get("TIGERBEETLE_ADDRESS", "tigerbeetle:3000")
AMOUNT_MINOR = int(os.environ.get("FINANCIAL_HIGH_VALUE_MINOR", "5000000"))
OPA_DECISION_URL = f"{OPA_URL}/v1/data/mineralvision/financial/decision"


class IntegrationFailure(RuntimeError):
    """Raised when a pre-production dependency or policy contract fails."""


def opa_decision(action: str, subject: dict, transfer: dict) -> dict:
    """Call the live OPA policy REST endpoint and return its decision object."""
    response = requests.post(
        OPA_DECISION_URL,
        json={"input": {"action": action, "subject": subject, "transfer": transfer, "policy": {}}},
        timeout=5,
    )
    response.raise_for_status()
    body = response.json()
    decision = body.get("result")
    if not isinstance(decision, dict) or "allow" not in decision:
        raise IntegrationFailure(f"Malformed OPA response for {action}: {body}")
    return decision


def assert_decision(label: str, decision: dict, expected: bool) -> None:
    """Print and enforce an expected OPA allow/deny decision."""
    observed = bool(decision.get("allow"))
    print(f"{label}: {'ALLOW' if observed else 'DENY'} — {decision.get('reason', 'no_reason')}")
    if observed != expected:
        raise IntegrationFailure(f"{label}: expected allow={expected}, got {decision}")


def probe_tigerbeetle() -> None:
    """Verify a private TigerBeetle TCP listener is reachable without submitting a transfer."""
    host, sep, port_text = TB_ADDRESS.rpartition(":")
    if not sep or not host or not port_text.isdigit():
        raise IntegrationFailure("TIGERBEETLE_ADDRESS must be host:port")
    with socket.create_connection((host, int(port_text)), timeout=5):
        pass
    print(f"TigerBeetle TCP liveness: CONNECTED — {TB_ADDRESS}")


def main() -> int:
    print("=== Pre-production live financial integration test ===")
    print(f"OPA decision endpoint: {OPA_DECISION_URL}")
    print(f"High-value amount: {AMOUNT_MINOR} minor units")

    # Live component reachability checks.
    health = requests.get(f"{OPA_URL}/health?plugins=false", timeout=5)
    health.raise_for_status()
    print("OPA health: OK")
    probe_tigerbeetle()

    transfer = {
        "idempotency_key": "preprod-ci-synthetic-transfer",
        "maker_id": "maker-001",
        "amount_minor": AMOUNT_MINOR,
        "currency": "USD",
        "approver_ids": [],
        "approval_count": 0,
        "distinct_approval_count": 0,
        "approval_assurance_ok": False,
    }
    maker = {"id": "maker-001", "roles": ["financial_maker"], "mfa_verified": True}
    checker_one = {"id": "checker-001", "roles": ["financial_checker", "financial_high_value_checker"], "mfa_verified": True}
    checker_two = {"id": "checker-002", "roles": ["financial_checker", "financial_high_value_checker"], "mfa_verified": True}
    releaser = {"id": "releaser-001", "roles": ["financial_releaser"], "mfa_verified": True}

    assert_decision("maker submit", opa_decision("financial.transfer.submit", maker, transfer), True)
    assert_decision("maker self-approval", opa_decision("financial.transfer.approve", maker, transfer), False)
    assert_decision("checker one approval", opa_decision("financial.transfer.approve", checker_one, transfer), True)

    transfer["approver_ids"] = [checker_one["id"]]
    transfer["approval_count"] = 1
    transfer["distinct_approval_count"] = 1
    transfer["approval_assurance_ok"] = True
    assert_decision("checker two approval", opa_decision("financial.transfer.approve", checker_two, transfer), True)
    assert_decision("early release", opa_decision("financial.transfer.release", releaser, transfer), False)

    transfer["approver_ids"].append(checker_two["id"])
    transfer["approval_count"] = 2
    transfer["distinct_approval_count"] = 2
    assert_decision("two-checker release", opa_decision("financial.transfer.release", releaser, transfer), True)

    print("RESULT: live OPA policy and TigerBeetle reachability checks passed; no ledger transfer was submitted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, OSError, IntegrationFailure) as error:
        print(f"INTEGRATION FAILURE: {error}", file=sys.stderr)
        raise SystemExit(1)
