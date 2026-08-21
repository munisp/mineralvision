# MineralVision Financial Transfer Hardening and Security Posture

## Slide 1 — Stakeholder review: financial-control and security posture

**Subtitle:** Evidence from the locked PostgreSQL/PostGIS CI environment — 21 August 2026

- Purpose: summarize financial transfer hardening, critical security tests, current code quality, and release conditions.
- Scope: code-level controls and controlled test environments only; no production payment credentials, payment networks, or real funds were accessed.
- Decision statement: the codebase has materially improved but should not be represented as defect-free or mission-critical production approved without the remaining environment and assurance gates.

## Slide 2 — Executive evidence snapshot

| Indicator | Verified result | Interpretation |
|---|---:|---|
| Full locked CI suite | 706 passed, 2 skipped | Green reproducible code gate after fixes |
| Whole-code coverage | 36.46% | Improvement, but broad specialist modules remain under-tested |
| Focused FIN/AUTH coverage | 85% | Financial ledger and identity/policy subset reached its planned threshold |
| Prior code-only score | 78/100 | Release-candidate posture, not defect-free assurance |
| Real-value transfer status | Disabled / no HTTP endpoint | Safest state pending independent approvals |

> Coverage is a regression-control indicator, not proof of correctness or compliance.

## Slide 3 — Test and coverage evidence

**Primary visual:** `../audit_artifacts/security_coverage_evidence.png`

- Whole-code coverage rose from 36.22% to 36.46% after 22 additional focused critical tests and two defect fixes.
- FIN/AUTH scope reached 85% under a dedicated coverage configuration: 28 tests passed.
- The locked full suite ran under PostgreSQL/PostGIS with migration, security, operations, and production-readiness validators.
- The next quality goal is to extend the same 85% risk-weighted approach to oil-spill, image ingress, WALDO, and project-scoped data paths.

## Slide 4 — Financial-transfer hardening model

- **Stable idempotency:** a business idempotency key deterministically derives the TigerBeetle transfer identifier.
- **Replay safety:** changed payloads reuse no transfer; lost-response retries reconcile the existing stable transfer.
- **Maker-checker:** the maker cannot approve; two distinct step-up-authenticated approvers are required by policy.
- **Audit integrity:** durable PostgreSQL intent, approval, and event records use HMAC-SHA-256 chains; a read-only verifier detects predecessor or payload tampering.
- **Boundaries:** the control layer deliberately has no public endpoint and rejects non-durable stores in production mode.

## Slide 5 — FIN-01 through FIN-06 execution evidence

| Control | Test evidence | Result |
|---|---|---|
| FIN-01 | Concurrent same-key submission | One ledger effect; same stable receipt |
| FIN-02 | Same key, changed business payload | Idempotency conflict, no second intent |
| FIN-03 | Maker self-approval / duplicate checker | Denied |
| FIN-04 | Step-up MFA, approval quorum, policy limit | Enforced |
| FIN-05 | PostgreSQL HMAC-chain tampering | Detected by verifier |
| FIN-06 | Lost-response retry and durable-store requirement | Stable replay / fail closed |

- The ledger test double was corrected so post/void pending transfers resolve original account metadata before validation and preserve posted history.

## Slide 6 — Identity, MFA, and OPA security boundaries

| Control area | Evidence |
|---|---|
| OIDC token verification | Expired, malformed, wrong-audience, wrong-issuer, and signing-key failure cases denied |
| MFA assurance | Roles and MFA claims are explicit; password-only identity does not imply MFA |
| Tenant isolation | Owner/admin/missing project matrix tested; unauthorized cross-tenant requests fail |
| OPA enforcement | Missing identity, timeout, malformed response, and explicit deny fail closed |
| Middleware contract | Protected routes require bearer identity; only enumerated public routes bypass auth |

- Focused coverage reached: auth middleware 79%, OIDC 89%, OPA 92%, shared authorization helper 100%.

## Slide 7 — Defects and technical debt remediated in this pass

- Corrected pending-transfer post/void processing, which previously rejected synthetic zero account IDs before resolving the original pending transfer.
- Preserved underlying account, amount, ledger, and code metadata for posted pending-transfer history.
- Hardened malformed password handling so non-string input fails closed rather than raising an unhandled exception.
- Made the in-memory control store reserve idempotency keys before completion, matching durable-store conflict semantics.
- Added FIN/AUTH branch coverage for account constraints, missing accounts, local token lifecycle, OIDC configuration, OPA HTTP responses, and fail-closed policy behavior.

## Slide 8 — Remaining risk and release gates

| Gap | Current posture | Required closure evidence |
|---|---|---|
| Whole-code test coverage | 36.46% | Risk-weighted coverage expansion for deployed specialist modules |
| Oil-spill/WALDO boundaries | Not at 85% scope target | Artifact, input-fuzz, service-token, timeout, and route-contract test suites |
| Real payment operations | Not enabled | Payment-partner, KYC/AML/sanctions, reconciliation, fraud, legal/compliance, HSM/KMS, DR approval |
| Environment controls | Code configured, not production-proven | Live Keycloak/OPA/gateway/WAF/SIEM integration and external penetration testing |
| CI workflow publication | Workflow files remain permission-blocked | Workflow-authorized token, protected branches, signed artifacts |

## Slide 9 — Recommended stakeholder decisions

1. Maintain the financial control layer as internal-only until all real-value enablement gates are independently signed off.
2. Authorize the next two test increments: oil-spill/WALDO ingress and PostgreSQL/project-data integrity, each with an 85% critical-subset threshold.
3. Require a protected CI workflow with locked dependencies, SBOM/dependency scans, full PostgreSQL/PostGIS tests, and signed evidence artifacts.
4. Fund independent threat modeling, penetration testing, and restore/PITR drills before operational deployment.

## Slide 10 — Appendices and evidence locations

- `tests/critical/test_financial_transfer_lifecycle.py`
- `tests/critical/test_auth_identity_policy_boundaries.py`
- `tests/critical/test_auth_fin_branch_coverage.py`
- `scripts/verify_financial_audit_chain.py`
- `docs/critical-coverage-and-security-remediation-plan.md`
- `docs/code-only-production-readiness-2026-08-21.md`
- `audit_artifacts/full_ci_after_critical_fixes.txt`
- `audit_artifacts/fin_auth_coverage/coverage_report_final.txt`

> All figures in this presentation are from the controlled, locked CI environment and should be re-produced in the target deployment environment before a release decision.
