# MineralVision Mission-Critical Code Audit

**Author:** Manus AI  
**Audit date:** 2026-08-21  
**Repository reviewed:** `munisp/mineralvision`  
**Scope:** Application code, production configuration, authentication and authorization, model and imagery paths, PostgreSQL configuration, static/dependency findings, test health, auditability, and dormant financial/value-transfer code.

## Executive assessment

This repository is a broad exploration, oil-spill, and vision platform rather than a narrowly scoped transaction engine. The reviewed TigerBeetle ledger is **not reachable from the production FastAPI route graph** and does not itself expose a payment, withdrawal, settlement, or bank-integration endpoint. It must therefore not be described as a production flow-of-funds system. If it is ever enabled for real value transfer, it requires a separate regulated payments review, double-entry reconciliation design, idempotency keys, transaction limits, approver separation, external ledger reconciliation, and operational controls.

The audit confirmed several material implementation weaknesses. The remediation in this change set closes the highest-confidence cross-tenant CRUD, WALDO service, SSRF, unsafe temporary-file, dynamic SQL identifier, model-deserialization, and fail-open fallback issues found during the review. The application is materially stronger after these changes, but it is **not ready to be certified defect-free or mission-critical-production-safe**. The overall evidence-based score is **61/100 — conditional engineering baseline, not release approval**.

> A repository review and targeted test run cannot prove absence of defects, compromise, fraud, or operational outage. Production approval requires the exit criteria in this report, including a dependency-resolved full CI suite, external penetration test, recovery drill, and live identity-policy integration test.

| Dimension | Weight | Score | Basis |
|---|---:|---:|---|
| Authorization and tenant isolation | 20% | 75 | Owner-or-admin checks now protect projects, drillholes, samples, user profile access, and derived data; broad API-wide policy coverage remains incomplete. |
| API and service perimeter | 15% | 70 | WALDO now requires a private service token, rejects remote URL ingestion, bounds image inputs, and disables async video by default; the deployed edge stack was not exercised in this sandbox. |
| Data integrity and financial safety | 15% | 55 | PostgreSQL-only application path and validation exist. The dormant financial adapter now rejects invalid transfers, but lacks real-money controls and must remain disabled. |
| Dependency and supply-chain posture | 15% | 45 | Dependency-file resolution is not reproducibly validated, and the local environment reports nine advisories in four packages. A lockfile-driven remediation is required rather than relying on a sandbox-local package upgrade. |
| Reliability and test evidence | 20% | 52 | 33 focused regression tests passed and static validators passed. The full suite stopped during collection with ten errors caused by missing declared dependencies and stale SQLite-oriented test setup. |
| Auditability, recovery, and operations | 15% | 72 | OPA/Keycloak, audit-event, SIEM, and PITR designs are present. They require real deployment, off-host log export, backup verification, and restore-drill evidence. |
| **Weighted result** | **100%** | **61** | **Conditional baseline; no mission-critical release approval.** |

## Confirmed findings and remediation

| ID | Severity before fix | Finding | Remediation implemented | Validation |
|---|---|---|---|---|
| AUTH-01 | Critical | Project, drillhole, and sample CRUD/list/derived routes accepted authentication without consistently enforcing project ownership. | Added `api/authz.py`; applied owner-or-admin scope checks across project, drillhole, sample, assay, statistics, and derived-data paths. | New targeted authorization tests pass. |
| AUTH-02 | High | Any authenticated user could retrieve another user’s profile or permissions. | User-detail and permission endpoints now require self access or an administrator role. | Compiled and reviewed; targeted authorization helper tests pass. |
| WALDO-01 | Critical | Standalone WALDO routes were suitable for unauthenticated exposure and did not match the primary API JSON inference client. | Added mandatory production service-token guard, private client header, JSON/multipart bounded input support, and deployment wiring. | Python compilation and focused regressions pass. |
| WALDO-02 | High | `/detect/url` performed server-side URL fetching and exposed an SSRF path. | Disabled remote URL ingestion; clients must upload image bytes to `/detect`. | Source and endpoint contract reviewed. |
| WALDO-03 | High | Request-specific confidence mutations were shared across concurrent detector requests; legacy image/video handling used predictable `/tmp` names and leaked exception text. | Added detector lock and restoration, in-memory image handling, generic errors, random IDs, isolated job directories, and default-disabled video processing. | Compilation and focused regressions pass. |
| DATA-01 | High | WALDO aggregation interpolated caller-selected SQL identifiers. | Restricted grouping identifiers to `class_name` and `source_id`; production SQLite fallback is disabled unless explicitly permitted in non-production. | Source review and compilation pass. |
| ML-01 | Medium | SAM3 optional imports could leave `nn` or `Image` undefined, masking real inference errors; checkpoint loading used unrestricted `torch.load`. | Added import-safe unavailable stubs, separated Pillow/PyTorch availability, and set `weights_only=True` for adapter/checkpoint loading. | SAM3 honesty tests pass. |
| AUTH-03 | Medium | OPA URL validation did not constrain non-HTTP or credential-bearing URLs. | Added scheme, credential, and production private-policy-host validation. | New OPA configuration tests pass. |
| FIN-01 | Medium | Dormant TigerBeetle transfer methods accepted invalid amounts and self-transfers. | Added positive minor-unit, positive-account-ID, and distinct-account validation across individual and batch transfers. | New financial-validation tests pass. |

## Financial and value-transfer assessment

The TigerBeetle adapter is middleware code without an identified FastAPI transaction endpoint or external money rail. It is therefore assessed as a **dormant technical component**, not an active funds flow. The added validation prevents basic malformed transfer input, but it is not sufficient for real-money use.

A production funds feature must remain blocked until it provides immutable transaction IDs and idempotency guarantees, serializable transaction boundaries, account and currency constraints, four-eyes approval for release or reversal, daily and per-transaction limits, sanctions/AML and KYC integration where applicable, external-bank or custodian reconciliation, out-of-band fraud monitoring, and independently retained audit evidence. Any future endpoint that calls this adapter must be treated as a new high-risk change requiring threat modeling and independent review.

## Validation evidence

| Check | Result |
|---|---|
| Python compilation of changed security, authorization, vision, WALDO, and ledger modules | Passed. |
| Focused audit, SAM3 honesty, and oil-spill regression suite | **33 passed**. |
| Production readiness baseline | **6/6 passed**; Docker image check skipped because Docker is unavailable in this sandbox. |
| Static security baseline | Passed. |
| Operations baseline | Passed. |
| Local installed-environment dependency audit | Found **9 advisories in 4 packages**: `pypdf`, `setuptools`, `wheel`, and `xhtml2pdf`. These are sandbox/environment packages. A clean dependency-manifest audit must still be made reproducible in CI using a lockfile. |
| Full repository test run | **Blocked at collection**: 405 test candidates, 10 collection errors caused by missing declared geospatial/ML dependencies (`geopandas`, `xarray`, `shapely`, `scikit-learn`, `torch`) and two stale SQLite-oriented innovation tests after the PostgreSQL-only migration. |

## Residual-risk register and release gates

| Priority | Residual risk | Required closure before a mission-critical release |
|---|---|---|
| P0 | Full suite is not dependency-resolved and cannot provide complete regression evidence. | Create a locked, reproducible CI image; install every declared runtime/ML dependency; migrate stale SQLite test setup to isolated PostgreSQL; require full green suite. |
| P0 | Secure Compose, Keycloak MFA, OPA, APISIX, Caddy, and OpenAppSec controls are configuration artifacts rather than a live validated production deployment. | Deploy a staging stack, run the real MFA integration client, validate deny/allow OPA decisions, tune WAF rules, execute load and DDoS resilience tests, and record evidence. |
| P0 | Financial adapter lacks controls required for actual funds movement. | Keep it unreachable and disabled. Complete a standalone regulated payments design/review before adding any financial endpoint. |
| P1 | Dependency manifest audit is not reproducibly resolvable and local package advisories remain. | Generate lock files from a clean builder, eliminate resolver conflict, patch/remove vulnerable nonessential packages, and enforce SBOM plus vulnerability gates in CI. |
| P1 | WAF, SIEM, and audit events have not been verified through an external collector or immutable evidence store. | Validate log shipping, retention, alerting, audit export/CDC, and a forensic reconstruction exercise. |
| P1 | Video jobs are memory-tracked and default-disabled; they are not durable or multi-replica safe. | Use a durable authenticated queue with per-user/project job ownership, retention, cancellation, and quotas before enabling. |
| P2 | Bandit still flags protected URL opening, constrained SQL identifiers, and deployment-path temporary directories. | Document reviewed false positives or replace remaining deployment temp storage; re-run scanner with a reviewed baseline. |

## Release decision

**Decision: Do not authorize a mission-critical production release from the current repository evidence.** The fixed code should be committed, reviewed, and promoted to a hardened staging environment. Release approval requires closure of every P0 item, a clean reproducible dependency audit, full test-suite success, live identity/gateway/WAF validation, immutable off-host logging verification, and a PostgreSQL restore/PITR exercise.
