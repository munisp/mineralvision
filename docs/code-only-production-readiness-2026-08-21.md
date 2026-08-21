# Code-Only Production Readiness Assessment

**Assessment date:** 2026-08-21  
**Scope:** Repository source code, committed dependency graph, database migrations, and an isolated locked Python/PostgreSQL/PostGIS test environment.  
**Out of scope:** Live cloud infrastructure, production identity provider, API gateway/WAF/SIEM behavior, real model weights, external penetration testing, regulatory acceptance, disaster-recovery drill execution, and payment-processor or bank connectivity.

## Decision

> **Code-only decision: conditionally suitable for continued release-candidate hardening; not certifiable as a 100/100 or defect-free mission-critical production release.**

The clean locked suite completed with **684 passed and 2 skipped tests** using the committed dependency lock and PostgreSQL/PostGIS. The measured whole-repository line coverage is **36.22%** across 70,480 measured lines. The repository CI runner was executed end to end and passed its 35% coverage ratchet; this is deliberately not represented as a proof of correctness.

| Dimension | Score | Evidence and rationale |
|---|---:|---|
| Locked reproducibility | 92/100 | `uv.lock`, `requirements-ci.lock`, dependency inputs, and the PostgreSQL/PostGIS test service are committed. |
| Automated regression behavior | 91/100 | The complete locked suite passed: 684 passed, 2 skipped. |
| Critical-path test depth | 82/100 | Authorization, oil-spill governance, ledger controls, WALDO service boundaries, onboarding, migrations, and PostGIS paths have focused integration tests. |
| Whole-code coverage | 36/100 | 36.22% was measured by the locked full suite; broad specialist modules, optional ML pipelines, sensor integrations, and visualization/NLP packages remain lightly or untested. |
| Code security controls | 88/100 | Earlier audits fixed confirmed cross-tenant checks, WALDO service authentication and SSRF surfaces, SQL allow-listing, checkpoint deserialization, and financial audit-chain controls. |
| **Weighted code-only score** | **78/100** | A strong improvement from the earlier 61/100 audit, but insufficient evidence for a 100/100 claim. |

## Remediation Delivered in This Pass

| Finding | Remediation | Regression evidence |
|---|---|---|
| Implicit WALDO detector selection could load/download a default model in a clean environment. | Molmo/WALDO fusion now requires an explicit existing model artifact path and otherwise remains in honest Molmo-only mode. | `tests/hardening/test_waldo_dedupe.py` |
| GeoAI tests relied on an undeclared optional segmentation backend. | `scikit-image==0.26.0` is declared in the ML and project dependency groups and resolved in both locks. | `tests/innovations/test_geoai.py` |
| GeoDB indexing and lakehouse export collected unrelated records from the shared test database. | Added optional project scope to spatial indexing and lakehouse synchronization; test fixtures now seed isolated project data. | `tests/innovations/test_geodb.py` |
| GeoLibre, onboarding, prospectivity, and observability tests instantiated PostgreSQL-only JSONB metadata on SQLite. | Replaced stale SQLite fixtures with isolated PostgreSQL schemas/databases and full Alembic migration testing. | Associated innovation and observability test modules |
| Full CI had no whole-code coverage regression gate. | `scripts/ci_full_test.sh` now emits XML coverage and enforces `COVERAGE_MINIMUM` (default 35). | Locked-suite coverage result: 36% |

## Meaningful Coverage Strategy

A literal 100% line-coverage target for this repository would require testing every error branch and optional hardware, model, cloud, and external-service path. That would include unrepresentative tests and still would not prove correctness. The release policy should instead require all three conditions below.

| Gate | Current state | Required improvement |
|---|---|---|
| Whole-code regression floor | 35% enforced; 36.22% measured | Raise in 5-point increments only after green locked runs. |
| Critical-module coverage | Focused tests exist but no per-module threshold is enforced | Set explicit branch-coverage floors for auth/authz, financial transfer controls, oil-spill governance, database migrations, WALDO boundary code, and OPA/OIDC middleware. |
| Risk-based integration tests | PostgreSQL/PostGIS, migration, audit-chain, and policy tests run locally | Add controlled real OIDC/OPA/gateway tests, model artifact contract tests, and restore-drill tests in environment-specific gates. |

## Remaining Code-Level Release Conditions

The following are code/test evidence gaps, not infrastructure availability assumptions. They remain before a high-assurance mission-critical release can be endorsed.

| Priority | Release condition |
|---|---|
| P0 | Raise the critical-module branch-coverage policy and test all error/retry/rollback paths for money-moving adapters before any real-value activation. |
| P0 | Add testable payment-service endpoints only after independent reconciliation, sanction/KYC/AML integration, fraud/limit decisions, immutable external audit storage, and legal/compliance approval. |
| P1 | Add deterministic test doubles for specialist sensor and hardware adapters, then raise coverage for their failure, calibration, and data-quality paths. |
| P1 | Add model-contract tests for all configured ONNX/Torch artifacts without asserting model accuracy from synthetic fixtures. |
| P1 | Configure the unpushed workflows with a workflow-authorized token and require the locked full CI plus coverage ratchet on protected branches. |

## Reproduce the Evidence

```bash
export CI_POSTGRES_PASSWORD='local-random-test-password'
docker compose -f docker-compose.ci.yml up -d --wait
export MV_TEST_DATABASE_URL="postgresql+psycopg2://mineralvision_ci:${CI_POSTGRES_PASSWORD}@127.0.0.1:55432/mineralvision_ci"
COVERAGE_MINIMUM=35 bash scripts/ci_full_test.sh
docker compose -f docker-compose.ci.yml down --volumes --remove-orphans
```

The runner refuses a non-PostgreSQL URL, verifies the lock, applies Alembic migrations, runs release/security/operations validators, writes `coverage.xml`, and fails when the coverage floor is missed.

## Conclusion

This pass closed all failures exposed by the clean locked suite and establishes a reproducible 684-pass code baseline. The repository CI runner was also executed end to end, including migrations, release/security/operations validators, and the 35% coverage gate. It did not—and cannot honestly—turn 36.22% whole-repository coverage into a 100/100 assurance statement. The correct next objective is a risk-weighted, critical-module coverage program with independently reviewed production integration evidence.
