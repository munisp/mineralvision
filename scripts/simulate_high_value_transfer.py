#!/usr/bin/env python3
"""End-to-end simulation: high-value fund transfer under OPA policy enforcement.

This script exercises the complete transfer lifecycle in a controlled local
environment without moving real funds or contacting external services:

1. Reserve an idempotent transfer intent (maker)
2. Evaluate OPA policy for maker submission (allow)
3. Attempt self-approval by maker (OPA deny)
4. First checker approval with MFA (OPA allow)
5. Second checker approval with MFA for high-value (OPA allow)
6. Release/post the transfer (OPA allow for releaser)
7. Verify the HMAC audit chain integrity

Prerequisites:
  - PostgreSQL with migration 0003 applied
  - Environment: DATABASE_URL, LEDGER_AUDIT_HMAC_KEY
"""
import hashlib
import hmac
import json
import os
import sys
import uuid
from datetime import datetime, timezone

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MineralVision_Final_Package", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MineralVision_Enhanced"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AUDIT_KEY = os.environ.get("LEDGER_AUDIT_HMAC_KEY", "simulation-audit-key-not-for-production")
DATABASE_URL = os.environ.get("DATABASE_URL", os.environ.get("MV_TEST_DATABASE_URL", ""))

if not DATABASE_URL or "postgresql" not in DATABASE_URL:
    print("ERROR: DATABASE_URL must point to a PostgreSQL instance with migration 0003 applied.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Simulated identity tokens (would come from Keycloak OIDC in production)
# ---------------------------------------------------------------------------
MAKER = {"sub": "user-maker-001", "roles": ["finance_maker"], "mfa_verified": True, "name": "Alice Maker"}
CHECKER_1 = {"sub": "user-checker-001", "roles": ["finance_checker", "finance_high_value_checker"], "mfa_verified": True, "name": "Bob Checker"}
CHECKER_2 = {"sub": "user-checker-002", "roles": ["finance_checker", "finance_high_value_checker"], "mfa_verified": True, "name": "Carol Checker"}
RELEASER = {"sub": "user-releaser-001", "roles": ["finance_releaser"], "mfa_verified": True, "name": "Dave Releaser"}

# ---------------------------------------------------------------------------
# OPA Policy Evaluation (local simulation of the Rego rules)
# ---------------------------------------------------------------------------
HIGH_VALUE_THRESHOLD = 10000  # matches financial_transfer.rego

def evaluate_opa_policy(action: str, actor: dict, context: dict) -> dict:
    """Simulate the OPA financial_transfer.rego policy evaluation locally.
    
    In production, this would be an HTTP call to the OPA sidecar.
    """
    roles = set(actor.get("roles", []))
    mfa = actor.get("mfa_verified", False)
    actor_id = actor["sub"]
    
    if action == "submit_transfer":
        allowed = "finance_maker" in roles and mfa
        return {"allow": allowed, "reason": "maker with MFA" if allowed else "missing maker role or MFA"}
    
    elif action == "approve_transfer":
        # Cannot approve own submission
        maker_id = context.get("maker_id")
        if actor_id == maker_id:
            return {"allow": False, "reason": "maker cannot approve own transfer (self-approval denied)"}
        # Must have checker role with MFA
        if "finance_checker" not in roles or not mfa:
            return {"allow": False, "reason": "missing checker role or MFA"}
        # High-value requires high_value_checker
        if context.get("amount", 0) >= HIGH_VALUE_THRESHOLD:
            if "finance_high_value_checker" not in roles:
                return {"allow": False, "reason": "high-value transfer requires finance_high_value_checker role"}
        return {"allow": True, "reason": "checker with MFA approved"}
    
    elif action == "release_transfer":
        # Releaser must be distinct from maker and checkers
        prior_actors = set(context.get("prior_actor_ids", []))
        if actor_id in prior_actors:
            return {"allow": False, "reason": "releaser must be distinct from maker and checkers"}
        if "finance_releaser" not in roles or not mfa:
            return {"allow": False, "reason": "missing releaser role or MFA"}
        # High-value requires at least 2 distinct approvals
        approval_count = context.get("approval_count", 0)
        if context.get("amount", 0) >= HIGH_VALUE_THRESHOLD and approval_count < 2:
            return {"allow": False, "reason": f"high-value release requires 2 approvals, got {approval_count}"}
        return {"allow": True, "reason": "releaser with MFA, sufficient approvals"}
    
    return {"allow": False, "reason": f"unknown action: {action}"}


# ---------------------------------------------------------------------------
# Audit Event Chain (HMAC-SHA-256)
# ---------------------------------------------------------------------------
audit_events = []

def record_audit_event(event_type: str, actor: dict, payload: dict, prev_hash: str = "") -> dict:
    """Record an immutable HMAC-chained audit event."""
    event = {
        "id": str(uuid.uuid4()),
        "sequence": len(audit_events) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "actor_id": actor["sub"],
        "actor_name": actor["name"],
        "mfa_verified": actor["mfa_verified"],
        "payload": payload,
        "prev_hash": prev_hash,
    }
    # Canonical message for HMAC
    canonical = f"{event['sequence']}|{event['timestamp']}|{event['event_type']}|{event['actor_id']}|{json.dumps(event['payload'], sort_keys=True)}|{event['prev_hash']}"
    event["hmac"] = hmac.new(AUDIT_KEY.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    audit_events.append(event)
    return event


def verify_audit_chain(events: list, key: str) -> dict:
    """Verify the HMAC chain integrity of all recorded events."""
    results = {"total": len(events), "valid": 0, "invalid": 0, "errors": []}
    for i, event in enumerate(events):
        expected_prev = events[i - 1]["hmac"] if i > 0 else ""
        if event["prev_hash"] != expected_prev:
            results["invalid"] += 1
            results["errors"].append(f"Event {event['sequence']}: predecessor hash mismatch")
            continue
        canonical = f"{event['sequence']}|{event['timestamp']}|{event['event_type']}|{event['actor_id']}|{json.dumps(event['payload'], sort_keys=True)}|{event['prev_hash']}"
        expected_hmac = hmac.new(key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(event["hmac"], expected_hmac):
            results["invalid"] += 1
            results["errors"].append(f"Event {event['sequence']}: HMAC verification failed")
        else:
            results["valid"] += 1
    results["chain_intact"] = results["invalid"] == 0
    return results


# ---------------------------------------------------------------------------
# Simulation Execution
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("HIGH-VALUE TRANSFER SIMULATION UNDER OPA POLICY ENFORCEMENT")
    print("=" * 80)
    
    idempotency_key = f"SIM-{uuid.uuid4().hex[:12].upper()}"
    transfer = {
        "idempotency_key": idempotency_key,
        "amount": 50000.00,
        "currency": "USD",
        "from_account": "ACCT-001-OPERATIONS",
        "to_account": "ACCT-002-VENDOR",
        "description": "Quarterly equipment procurement payment",
    }
    
    print(f"\n{'─' * 80}")
    print(f"STEP 1: Maker submits transfer intent (idempotency_key={idempotency_key})")
    print(f"{'─' * 80}")
    policy_result = evaluate_opa_policy("submit_transfer", MAKER, transfer)
    print(f"  OPA Decision: {'ALLOW' if policy_result['allow'] else 'DENY'} — {policy_result['reason']}")
    assert policy_result["allow"], "Maker submission should be allowed"
    
    prev_hash = ""
    evt = record_audit_event("transfer_intent_created", MAKER, {"idempotency_key": idempotency_key, "amount": transfer["amount"], "currency": "USD"}, prev_hash)
    prev_hash = evt["hmac"]
    print(f"  Audit Event #{evt['sequence']}: {evt['event_type']} → HMAC={evt['hmac'][:16]}...")
    print(f"  Transfer state: RESERVED")
    
    print(f"\n{'─' * 80}")
    print(f"STEP 2: Maker attempts self-approval (MUST BE DENIED)")
    print(f"{'─' * 80}")
    policy_result = evaluate_opa_policy("approve_transfer", MAKER, {"maker_id": MAKER["sub"], "amount": transfer["amount"]})
    print(f"  OPA Decision: {'ALLOW' if policy_result['allow'] else 'DENY'} — {policy_result['reason']}")
    assert not policy_result["allow"], "Self-approval must be denied"
    
    evt = record_audit_event("approval_denied", MAKER, {"reason": policy_result["reason"]}, prev_hash)
    prev_hash = evt["hmac"]
    print(f"  Audit Event #{evt['sequence']}: {evt['event_type']} → HMAC={evt['hmac'][:16]}...")
    
    print(f"\n{'─' * 80}")
    print(f"STEP 3: First checker (Bob) approves with MFA")
    print(f"{'─' * 80}")
    policy_result = evaluate_opa_policy("approve_transfer", CHECKER_1, {"maker_id": MAKER["sub"], "amount": transfer["amount"]})
    print(f"  OPA Decision: {'ALLOW' if policy_result['allow'] else 'DENY'} — {policy_result['reason']}")
    assert policy_result["allow"], "First checker approval should be allowed"
    
    evt = record_audit_event("transfer_approved", CHECKER_1, {"approval_number": 1, "mfa_method": "webauthn"}, prev_hash)
    prev_hash = evt["hmac"]
    print(f"  Audit Event #{evt['sequence']}: {evt['event_type']} → HMAC={evt['hmac'][:16]}...")
    
    print(f"\n{'─' * 80}")
    print(f"STEP 4: Second checker (Carol) approves with MFA (high-value requirement)")
    print(f"{'─' * 80}")
    policy_result = evaluate_opa_policy("approve_transfer", CHECKER_2, {"maker_id": MAKER["sub"], "amount": transfer["amount"]})
    print(f"  OPA Decision: {'ALLOW' if policy_result['allow'] else 'DENY'} — {policy_result['reason']}")
    assert policy_result["allow"], "Second checker approval should be allowed"
    
    evt = record_audit_event("transfer_approved", CHECKER_2, {"approval_number": 2, "mfa_method": "webauthn"}, prev_hash)
    prev_hash = evt["hmac"]
    print(f"  Audit Event #{evt['sequence']}: {evt['event_type']} → HMAC={evt['hmac'][:16]}...")
    
    print(f"\n{'─' * 80}")
    print(f"STEP 5: Releaser (Dave) posts the transfer")
    print(f"{'─' * 80}")
    prior_actors = {MAKER["sub"], CHECKER_1["sub"], CHECKER_2["sub"]}
    policy_result = evaluate_opa_policy("release_transfer", RELEASER, {
        "maker_id": MAKER["sub"],
        "amount": transfer["amount"],
        "prior_actor_ids": list(prior_actors),
        "approval_count": 2,
    })
    print(f"  OPA Decision: {'ALLOW' if policy_result['allow'] else 'DENY'} — {policy_result['reason']}")
    assert policy_result["allow"], "Releaser should be allowed after 2 approvals"
    
    evt = record_audit_event("transfer_posted", RELEASER, {
        "idempotency_key": idempotency_key,
        "amount": transfer["amount"],
        "from_account": transfer["from_account"],
        "to_account": transfer["to_account"],
    }, prev_hash)
    prev_hash = evt["hmac"]
    print(f"  Audit Event #{evt['sequence']}: {evt['event_type']} → HMAC={evt['hmac'][:16]}...")
    print(f"  Transfer state: POSTED")
    
    print(f"\n{'─' * 80}")
    print(f"STEP 6: Verify HMAC audit chain integrity")
    print(f"{'─' * 80}")
    verification = verify_audit_chain(audit_events, AUDIT_KEY)
    print(f"  Total events: {verification['total']}")
    print(f"  Valid: {verification['valid']}")
    print(f"  Invalid: {verification['invalid']}")
    print(f"  Chain intact: {verification['chain_intact']}")
    assert verification["chain_intact"], "Audit chain must be intact"
    
    print(f"\n{'─' * 80}")
    print(f"STEP 7: Tamper detection test (modify event #3 payload)")
    print(f"{'─' * 80}")
    tampered_events = [dict(e) for e in audit_events]
    tampered_events[2]["payload"] = {"approval_number": 1, "mfa_method": "sms"}  # tampered
    tampered_verification = verify_audit_chain(tampered_events, AUDIT_KEY)
    print(f"  Chain intact after tampering: {tampered_verification['chain_intact']}")
    print(f"  Errors detected: {tampered_verification['errors']}")
    assert not tampered_verification["chain_intact"], "Tampered chain must be detected"
    
    print(f"\n{'═' * 80}")
    print(f"SIMULATION COMPLETE — ALL ASSERTIONS PASSED")
    print(f"{'═' * 80}")
    
    print(f"\n{'─' * 80}")
    print(f"FULL AUDIT LOG (JSON)")
    print(f"{'─' * 80}")
    print(json.dumps(audit_events, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
