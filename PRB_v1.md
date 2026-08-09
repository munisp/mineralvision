# Production Readiness Baseline (PRB) v1

**Version:** 1.0
**Date:** 2025-12-19
**Status:** DRAFT

## Scope

This PRB defines the minimum requirements for MineralVision to be considered production-ready. All checks must PASS before deployment.

## Services Covered

| Service | Path | Language |
|---------|------|----------|
| API Server | MineralVision_Final_Package/src/api/ | Python |
| WALDO Detection | MineralVision_WALDO_Production_Package/src/ | Python |
| Lakehouse | MineralVision_Enhanced/lakehouse_architecture/ | Python |
| Middleware | MineralVision_Enhanced/middleware/ | Python |
| Web UI | MineralVision_Final_Package/src/ui/web/ | TypeScript |
| Mobile UI | MineralVision_Final_Package/src/ui/mobile/ | Dart |

## PRB Checks

### PRB-001: Zero Hardcoded Credentials

**Criteria:** No hardcoded passwords, secrets, tokens, or API keys in infrastructure YAMLs.

**Scan Pattern:** Files matching `*.yaml`, `*.yml` containing:
- `password:` followed by literal value (not `${VAR}` or `CHANGEME`)
- `secret:` followed by literal value
- `token:` followed by literal value
- `api_key:` or `api-key:` followed by literal value
- Kubernetes `Secret` with `data:` or `stringData:` containing base64 values

**Allowed Values:**
- Environment variable references: `${VAR}`, `$(VAR)`
- Placeholder markers: `CHANGEME`, `REPLACE_ME`, `<placeholder>`
- Empty strings or `null`
- Kubernetes `valueFrom` / `secretKeyRef` references

**Pass Condition:** Zero matches after filtering allowed values.

---

### PRB-002: Zero Mock Functions in Production Code

**Criteria:** No `generateMock*` or `generate_mock*` functions in production source files.

**Production Paths:**
- `MineralVision_Final_Package/src/` (excluding `tests/`, `test/`, `sample_data/`)
- `MineralVision_Enhanced/` (excluding `tests/`, `test/`)
- `MineralVision_WALDO_Production_Package/src/` (excluding `tests/`, `test/`, `sample_data/`)

**Scan Pattern:** Function definitions matching `def generate_mock` or `def generateMock`

**Pass Condition:** Zero matches.

---

### PRB-003: Zero TODO/FIXME/Placeholder Code

**Criteria:** No unfinished code markers in production Python files.

**Scan Patterns:**
- `# TODO` (case-insensitive)
- `# FIXME` (case-insensitive)
- `placeholder` in comments or string literals indicating incomplete implementation
- `stub` in comments indicating incomplete implementation

**Exclusions:**
- Variable names containing "placeholder" (e.g., `placeholder_image`)
- Documentation strings explaining what placeholders are
- Test files

**Pass Condition:** Zero actionable matches.

---

### PRB-004: All Python Files Compile

**Criteria:** All `.py` files pass syntax validation.

**Command:** `python3 -m py_compile <file>`

**Pass Condition:** Exit code 0 for all files.

---

### PRB-005: All Dockerfiles Build

**Criteria:** All Dockerfiles build successfully.

**Dockerfiles:**
- `MineralVision_WALDO_Production_Package/deployment/cloud/Dockerfile`

**Command:** `docker build -t test-build -f <Dockerfile> <context>`

**Pass Condition:** Exit code 0 (or SKIP if Docker not available).

---

### PRB-006: Database Persistence Verified

**Criteria:** No in-memory storage as default in production code paths.

**Scan Patterns:**
- `# In-memory storage` comments indicating temporary storage
- `sqlite.*:memory:` connection strings
- Dict-based storage used as primary data store

**Required:** Production entrypoints must use persistent database (SQLite file or PostgreSQL).

**Pass Condition:** All endpoints use database.py models, not in-memory dicts.

---

## Verification

Run: `make verify`

Output format:
```
PRB-001: PASS|FAIL (details)
PRB-002: PASS|FAIL (details)
PRB-003: PASS|FAIL (details)
PRB-004: PASS|FAIL (details)
PRB-005: PASS|FAIL (details)
PRB-006: PASS|FAIL (details)
---
OVERALL: PASS|FAIL (N/6 checks passed)
```

## Acceptance

PRB v1 is satisfied when `make verify` returns `OVERALL: PASS` with exit code 0.
