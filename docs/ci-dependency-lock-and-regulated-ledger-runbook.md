# CI Recovery, Dependency Lock, and Regulated Ledger Runbook

**Author:** Manus AI  
**Status:** Implementation baseline — review by platform security, finance, legal, compliance, and payment-partner teams is required before any real funds flow.

## Purpose and non-negotiable boundary

This runbook resolves the former test-collection blockers by using an isolated **PostgreSQL/PostGIS** test service and a committed universal dependency lock. It also describes the new idempotent, maker-checker, audit-controlled TigerBeetle integration baseline.

> The ledger controls are not a banking licence, money-transmitter programme, AML/KYC system, sanctions-screening system, or legal compliance certification. They deliberately expose no public transfer endpoint. Do not enable real-value transfers until the release gates at the end of this runbook have been met.

## 1. Reproduce the complete CI test environment

The repository now contains the following source-of-truth files.

| File | Function |
|---|---|
| `pyproject.toml` and `uv.lock` | Canonical, universal dependency graph for Python 3.12, including core, geospatial, ML, and development groups. |
| `requirements-ci.in` and `requirements-ci.lock` | Fully expanded pip-compatible CI dependency graph generated with `uv pip compile`. |
| `constraints-ci.txt` | Minimum reviewed toolchain versions and a block against undeclared `xhtml2pdf` production use. |
| `docker-compose.ci.yml` | Disposable PostGIS 16 test database. |
| `scripts/ci_full_test.sh` | Fail-closed full test runner. It rejects SQLite/default databases, verifies `uv.lock`, migrates the schema, runs static baseline checks, and executes `pytest`. |
| `.github/workflows/full-test.yml` | GitHub Actions workflow that runs the same PostGIS-backed gate. |

Run the complete gate locally with Docker installed:

```bash
cd mineralvision
export CI_POSTGRES_PASSWORD="use-a-local-random-test-password"
docker compose -f docker-compose.ci.yml up -d --wait
export MV_TEST_DATABASE_URL="postgresql+psycopg2://mineralvision_ci:${CI_POSTGRES_PASSWORD}@127.0.0.1:55432/mineralvision_ci"
bash scripts/ci_full_test.sh
docker compose -f docker-compose.ci.yml down --volumes --remove-orphans
```

The runner uses `MV_TEST_DATABASE_URL` exclusively. It must never receive production credentials. `tests/innovations/test_geodb.py` and `tests/innovations/test_geolibre.py` were migrated from direct SQLite fixtures to this isolated PostgreSQL contract; GeoDB requires the Compose service to have PostGIS enabled.

## 2. Regenerate and verify dependency locks

Use only a Python 3.12 resolver. A dependency update is complete only when both locks are regenerated, reviewed, audited, and committed.

```bash
cd mineralvision
uv lock
uv lock --locked
uv pip compile requirements-ci.in \
  --constraint constraints-ci.txt \
  --python-version 3.12 \
  --output-file requirements-ci.lock
uv pip sync requirements-ci.lock
pip-audit --local
```

For Pipenv consumers, use the supplied `Pipfile` and create the local lock from the same project metadata:

```bash
pipenv --python 3.12 install --dev
pipenv lock --clear
pipenv sync --dev
pipenv run test-ci
```

The GitHub workflow must be added with a token permitted to edit `.github/workflows/`. It should be required by branch protection alongside code-owner approval and a separate security scan.

## 3. Durable transfer schema provisioning

The Alembic revision `0003_financial_transfer_controls` creates three PostgreSQL tables:

| Table | Purpose |
|---|---|
| `financial_transfer_intents` | One immutable business intent per idempotency key, request hash, state, canonical payload, and durable receipt. |
| `financial_transfer_approvals` | Distinct maker-checker approval evidence, MFA assurance level, and OIDC/WebAuthn challenge reference. |
| `financial_transfer_audit_events` | Append-only event sequence with previous/event HMAC hashes and audit-key version. |

Provision only through the migration account, never the runtime transfer account:

```bash
export DATABASE_URL='postgresql+psycopg2://<migration-role>:<secret>@<postgres-host>:5432/mineralvision'
cd MineralVision_Final_Package
alembic upgrade head
alembic current
```

The runtime transfer-control process requires a non-production test or production PostgreSQL URL plus a secret-managed `LEDGER_AUDIT_HMAC_KEY` of at least 32 characters. It must run with a database role limited to `SELECT`/`INSERT`/`UPDATE` on its three financial-control tables and no schema DDL rights.

## 4. Transfer-control execution model

A transfer request starts with a client-generated business idempotency key. The stable key is hashed into a non-zero 128-bit TigerBeetle transfer ID. The same request key therefore survives browser/application retry and yields the same ledger object. TigerBeetle documents this “record once and only once” pattern and requires reuse of the original ID after uncertain network outcomes. [1]

The `RegulatedTransferService` applies the following sequence:

1. Validate positive minor-unit amount, distinct accounts, assigned ledger/code, currency, limit, purpose, actor, and external reference.
2. Require a durable PostgreSQL `PostgresTransferControlStore` when `production=True`; reject the in-memory store.
3. Reserve the idempotency key atomically. A different request hash raises `IdempotencyConflict`; an in-progress matching key returns `TransferInProgress` and requires reconciliation/retry with the same key.
4. Require the configured number of distinct approvers, prohibit maker self-approval, and require accepted step-up assurance (`aal2`/`aal3`) plus an OIDC/WebAuthn challenge reference.
5. Persist approval evidence and append a HMAC-chained `transfer_requested` audit event.
6. Submit the stable transfer ID to TigerBeetle. A retry result of `exists` is normalized to an idempotent replay, not a second debit/credit. [1] [2]
7. Persist the receipt and append `transfer_posted`; on failure append `transfer_rejected` and require reconciliation.

For card/bank/custodian operations where external settlement is not immediate, create a pending transfer, then post or void it with a **new** transfer ID after the independent settlement outcome is known. Pending transfers resolve once only and are immutable; correction requires a correcting transfer rather than alteration or deletion. [2] [3]

## 5. Minimal internal integration example

This code is for a private payment-orchestration service after OIDC/OPA authorization and external screening have passed. It is not a public API route.

```python
store = PostgresTransferControlStore(
    database_url=os.environ["LEDGER_DATABASE_URL"],
    audit_hmac_key=os.environ["LEDGER_AUDIT_HMAC_KEY"],
    key_version="2026-q3",
)
service = RegulatedTransferService(ledger.transfers, store, production=True)
receipt = await service.submit(intent, approvals, policy)
```

Required preconditions are an authenticated maker identity, independently verified approver MFA claims, sanctions/KYC/AML outcome, beneficiary/account ownership check, transaction limit decision, immutable external reference, and a working off-host audit export. Store the receipt’s transfer ID and audit-event hash in the payment case record. Never accept a server-generated idempotency key for a user-originated payment retry.

## 6. Tests and release gates

Run the control tests using a migrated disposable database:

```bash
export MV_TEST_DATABASE_URL='postgresql+psycopg2://<test-role>:<test-password>@127.0.0.1:5432/mineralvision_test'
cd MineralVision_Final_Package && alembic upgrade head && cd ..
pytest -q \
  tests/hardening/test_regulated_transfer_controls.py \
  tests/hardening/test_postgres_transfer_control_store.py
```

The tests prove one stable replay, changed-payload conflict rejection, maker-checker separation, MFA assurance enforcement, production rejection of the ephemeral store, durable intent state, and two audit events. They do not prove banking, tax, consumer-protection, sanctions, or payment-network compliance.

Before enabling any real funds flow, require written approval from legal/compliance, a payment/banking partner, finance controls, security, and operations. Complete KYC/AML/sanctions and fraud integration, daily reconciliation against external rails, transaction and velocity limits, dual-control break-glass, HSM/KMS audit-key management, independent audit log retention, load/failure testing, external penetration testing, disaster recovery exercise, and a documented incident-response plan.

## References

[1] [TigerBeetle: Reliable Transaction Submission](https://docs.tigerbeetle.com/coding/reliable-transaction-submission/)  
[2] [TigerBeetle: Transfer Reference](https://docs.tigerbeetle.com/reference/transfer/)  
[3] [TigerBeetle: Two-Phase Transfers](https://docs.tigerbeetle.com/coding/two-phase-transfers/)
