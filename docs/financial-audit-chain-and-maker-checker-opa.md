# Financial Audit-Chain Verification and Maker-Checker OPA Controls

**Status:** Implemented engineering controls. These controls must be independently reviewed before real funds move.

## Audit-chain integrity verifier

`./scripts/verify_financial_audit_chain.py` is a read-only PostgreSQL verifier for the `financial_transfer_audit_events` table. It retrieves the transfer intent and every event in sequence order, checks that each `previous_hash` matches the predecessor event, reconstructs the canonical event, and computes the expected **HMAC-SHA-256** using the audit key and recorded key version. It returns JSON and exits nonzero if any event stream is incomplete or modified.

The verifier uses the following canonical event structure. The same ordered JSON construction is used by `PostgresTransferControlStore` when it writes an audit event.

```python
{
  "event_type": event_type,
  "intent": intent_payload,
  "actor_id": actor_id,
  "details": details,
  "previous_hash": previous_hash,
  "key_version": key_version,
}
```

| Mode | Command | Expected use |
|---|---|---|
| One transfer | `python scripts/verify_financial_audit_chain.py --idempotency-key <key>` | Case review, pre-release, dispute handling, or suspicious-event investigation. |
| Entire ledger-event set | `python scripts/verify_financial_audit_chain.py --all` | Scheduled independent integrity control and restore/DR verification. |

Set credentials only through a secret manager or controlled process environment. The verifier process needs a **read-only** `LEDGER_DATABASE_URL` and the matching `LEDGER_AUDIT_HMAC_KEY`; it must not write to financial tables or print the key.

```bash
export LEDGER_DATABASE_URL='postgresql://ledger_audit_reader:<secret>@postgres/mineralvision'
export LEDGER_AUDIT_HMAC_KEY="$(secret-manager read ledger/audit-hmac/2026-q3)"
python scripts/verify_financial_audit_chain.py --all
```

A failure is an incident signal, not an auto-repair condition. Freeze the affected transfer stream, preserve PostgreSQL WAL/backups and external SIEM evidence, compare the result with the KMS key-version record, perform an independent TigerBeetle reconciliation, and follow the financial incident procedure. Never overwrite a mismatched hash or modify a posted transfer.

## OPA maker-checker policy

The attached `security/opa/financial_transfer.rego` policy is a separate decision package for an internal payments orchestration service. The platform's current `OPAMiddleware` protects oil-spill routes only; it does **not** expose a financial REST route or automatically apply the financial policy. That separation is intentional until the service boundary has full financial-compliance approval.

The payment service must construct the OPA input from its authenticated OIDC context and its own committed PostgreSQL tables. It must **not** accept the transfer maker, amount, approver list, approval count, or MFA outcome from a browser request body.

```json
{
  "action": "financial.transfer.release",
  "subject": {
    "id": "releaser-42",
    "roles": ["financial_releaser"],
    "mfa_verified": true
  },
  "policy": {"high_value_threshold_minor": 100000},
  "transfer": {
    "maker_id": "maker-7",
    "amount_minor": 250000,
    "currency": "USD",
    "approver_ids": ["checker-8", "checker-9"],
    "approval_count": 2,
    "distinct_approval_count": 2,
    "approval_assurance_ok": true
  }
}
```

| Action | Required role and condition | Denied when |
|---|---|---|
| `financial.transfer.submit` | `financial_maker`; MFA; authenticated subject equals `maker_id` | A checker/releaser attempts to create on another maker’s behalf, amount is non-positive, or currency is absent. |
| `financial.transfer.approve` below threshold | `financial_checker`; MFA; subject differs from maker and is not already an approver | Maker self-approval, duplicate checker, or no MFA. |
| `financial.transfer.approve` at/above threshold | `financial_high_value_checker`; MFA; same distinctness checks | A normal checker tries to approve high value, maker self-approves, or approval is repeated. |
| `financial.transfer.release` at/above threshold | `financial_releaser`; MFA; distinct from maker/checkers; at least two distinct verified approvals | A maker/checker releases their own case, fewer than two approvals exist, or assurance evidence fails. |

The OPA policy therefore enforces the identity-side separation, while `RegulatedTransferService` independently validates two distinct `TransferApproval` records and stores them in `financial_transfer_approvals`. Both checks are required: policy grants an action at a point in time; durable ledger controls preserve auditable state and protect retries.

> Assigning multiple incompatible roles to one human defeats maker-checker controls. Keycloak group administration must prohibit overlapping maker, checker, high-value-checker, and releaser entitlements for the same payment scope, except for formally approved emergency access that is separately logged, time-bound, and post-reviewed.

## Test evidence

The regression suite covers stable idempotent replay, conflicting-payload rejection, maker self-approval rejection, MFA assurance rejection, durable PostgreSQL storage, HMAC-chain success, and deliberate predecessor tampering detection. Run it only against an isolated migrated PostgreSQL database:

```bash
export MV_TEST_DATABASE_URL='postgresql+psycopg2://<test-user>:<test-secret>@127.0.0.1:5432/mineralvision_test'
pytest -q \
  tests/hardening/test_regulated_transfer_controls.py \
  tests/hardening/test_postgres_transfer_control_store.py \
  tests/hardening/test_financial_audit_chain.py
```
