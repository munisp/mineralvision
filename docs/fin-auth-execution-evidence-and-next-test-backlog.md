# FIN/AUTH Execution Evidence and Remaining Critical-Path Test Backlog

**Date:** 21 August 2026  
**Environment:** Locked Python 3.12 dependency graph; disposable PostgreSQL/PostGIS test database; no payment-network access or real-value transfers.

## 1. Exact Focused Execution Result

The focused financial, identity, and policy test command completed successfully:

```text
collected 28 items

tests/critical/test_auth_fin_branch_coverage.py .........                [ 32%]
tests/critical/test_financial_transfer_lifecycle.py ......               [ 53%]
tests/critical/test_auth_identity_policy_boundaries.py .......           [ 78%]
tests/hardening/test_regulated_transfer_controls.py .....                [ 96%]
tests/hardening/test_postgres_transfer_control_store.py .                [100%]

======================= 28 passed, 65 warnings in 3.12s ========================
```

The warnings are existing dependency/test-environment warnings; the run produced no test failures. The raw report is retained at `audit_artifacts/fin_auth_coverage/coverage_report_final.txt`.

## 2. Exact Focused Coverage Output

The focused measurement was intentionally limited to the financial ledger, authentication middleware, shared authorization helper, OIDC validator, and OPA middleware. Its result was:

| Module | Statements | Missed | Branches | Partial Branches | Coverage |
|---|---:|---:|---:|---:|---:|
| `tigerbeetle_ledger.py` | 523 | 72 | 100 | 17 | **84%** |
| `auth_middleware.py` | 134 | 21 | 40 | 11 | **79%** |
| `authz.py` | 19 | 0 | 6 | 0 | **100%** |
| `security/oidc.py` | 73 | 4 | 20 | 6 | **89%** |
| `security/opa.py` | 67 | 4 | 22 | 3 | **92%** |
| **Focused FIN/AUTH total** | **816** | **101** | **188** | **37** | **85%** |

This is a **risk-weighted focused coverage threshold**, not a whole-repository coverage claim. The end-to-end locked full suite subsequently passed **706 tests, 2 skipped**, with **36.46%** whole-code statement coverage across 70,494 statements.

## 3. Defects and Technical Debt Fixed in This Increment

| Finding | Correction | Regression Evidence |
|---|---|---|
| Pending post/void transfer rejected synthetic zero IDs | Pending operations now resolve the original stored pending transfer before generic account validation. | FIN branch test covers pending creation, posting, voiding, balances, and missing pending records. |
| Posted pending transfer lost original economic metadata | Post completion now preserves original debit/credit accounts, amount, ledger, and code for account history. | FIN branch test checks account history includes both regular and posted movements. |
| In-memory idempotency store permitted altered request during an in-progress reservation | Store now reserves the idempotency key and request hash before execution, matching durable-store conflict semantics. | FIN-01, FIN-02, and branch conflict test. |
| Non-string password input raised an unhandled exception | Password verification catches malformed input and returns `False` fail closed. | AUTH branch test passes `None` and expects failure without exception. |
| OPA request/decision branches lacked direct coverage | Tests now cover configuration denial, service response, malformed response, explicit denial, missing identity, and policy timeout. | AUTH-04 and OPA branch test. |

## 4. Remaining Test Cases to Close the Original 1,129-Statement Critical-Path Gap

The original 85% target covered a broader 4,078-statement critical scope: identity, financial controls, oil-spill APIs, PostgreSQL persistence, WALDO/Molmo integration, and GeoDB. The FIN/AUTH subset now reaches 85%, but the broader scope must be remeasured and expanded using the following exact test cases.

### Oil-Spill and Raw Imagery Boundary Tests

| ID | Test case | Expected assertion | Priority |
|---|---|---|---|
| OIL-01 | Unregistered/mismatched ONNX model artifact | Raw inference rejects with no incident write. | P0 |
| OIL-02 | Approved model plus valid aerial image | Persisted pending-review incident, provenance, and GeoJSON footprint. | P0 |
| OIL-03 | Invalid image MIME, oversized body, malformed mask | 4xx response; no temporary file or partial incident. | P0 |
| OIL-04 | Low-confidence and out-of-area assessment | Human-review state and policy flags are set; no automated operational action. | P0 |
| OIL-05 | Concurrent review transition | Exactly one review transition and complete audit timeline. | P1 |
| OIL-06 | Model approval/revocation race | Revoked model cannot produce a new inference after the transaction boundary. | P0 |

### PostgreSQL, Alembic, and Project Isolation Tests

| ID | Test case | Expected assertion | Priority |
|---|---|---|---|
| DATA-01 | Upgrade `0001` through `0003` on an empty PostgreSQL/PostGIS database | Tables, JSONB constraints, indexes, and least-privilege grants exist. | P0 |
| DATA-02 | Migration downgrade/re-upgrade under a disposable database | Schema is reversible or explicitly fails before data loss; no silent SQLite fallback. | P0 |
| DATA-03 | Cross-project GeoDB index and lakehouse synchronization | Only requested project records export or index. | P0 |
| DATA-04 | Concurrent idempotency reservation under PostgreSQL isolation | One intent reaches posted/reconcile state; second gets receipt or controlled in-progress result. | P0 |
| DATA-05 | Audit-event key-rotation boundary | Each event validates with the declared key version; mixed key versions are rejected when unavailable. | P1 |

### WALDO/Molmo Service and Image-Ingress Tests

| ID | Test case | Expected assertion | Priority |
|---|---|---|---|
| WALDO-01 | Missing/incorrect service token | Endpoint returns 401/403; no model execution. | P0 |
| WALDO-02 | Remote URL ingestion attempt | Disabled route fails closed; no outbound request occurs. | P0 |
| WALDO-03 | Multipart and JSON image inputs at boundary sizes | Valid input works; oversize/malformed input is rejected without persistence. | P0 |
| WALDO-04 | Explicit artifact absent, hash mismatch, or unsupported model type | No implicit download/load; descriptive controlled error. | P0 |
| WALDO-05 | Pending video job disabled and enabled modes | Default fails closed; enabled mode uses random job IDs and private storage only. | P1 |
| WALDO-06 | Service timeout and malformed downstream response | Caller fails closed with sanitized error and audit event. | P0 |

### OIDC, OPA, and Edge Integration Tests

| ID | Test case | Expected assertion | Priority |
|---|---|---|---|
| AUTH-08 | Key rotation with old/new JWKS `kid` values | Valid rotation succeeds; unknown kid fails closed. | P0 |
| AUTH-09 | OPA response allows a role but project does not match | API deny unless server-derived project facts authorize it. | P0 |
| AUTH-10 | OPA circuit/timeout with fail-open flag prohibited in production | Startup/configuration rejects unsafe production setting. | P0 |
| AUTH-11 | Keycloak MFA assurance claim downgrade mid-session | Review/approval action is denied after assurance no longer satisfies policy. | P1 |
| AUTH-12 | Caddy/APISIX forwarded header spoofing | Application uses sanitized trusted proxy context only. | P1 |

## 5. Required CI Gates for the Next Increment

The next increment should not be considered complete until all of the following pass in a protected CI environment:

1. The locked full PostgreSQL/PostGIS suite.
2. A dedicated critical-path coverage job with the expanded OIL, DATA, WALDO, and AUTH suites and a minimum 85% statement/branch-informed threshold for that declared scope.
3. Database migration upgrade/downgrade tests in a new disposable cluster.
4. Contract tests against containerized OPA and Keycloak test realms with short-lived test tokens only.
5. Fuzz/property testing for image ingress, idempotency keys, authorization claims, and GeoJSON payloads.
6. Dependency, secret, static-analysis, and migration-schema checks as non-optional branch-protection checks.

> **Release interpretation:** the 85% FIN/AUTH result demonstrates coverage for the selected financial and identity boundary set. It does not replace the broader critical-path test plan or independent security/compliance approval for real-value operations.
