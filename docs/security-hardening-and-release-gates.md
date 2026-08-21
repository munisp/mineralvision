# MineralVision Security Hardening and Release Gates

This document provides a comprehensive checklist of implemented security controls and the remaining release gates required before the MineralVision platform can be deployed to a mission-critical production environment.

## Implemented Security Controls

The following defense-in-depth controls have been implemented, tested, and merged into the `main` branch.

### 1. Identity and Access Management
- [x] **Keycloak OIDC Integration**: Production authentication is delegated to Keycloak using OIDC/JWKS validation.
- [x] **Local Authentication Disabled**: Application-issued credentials and password management are disabled in production.
- [x] **Role-Based Access Control (RBAC)**: Roles (e.g., `user`, `reviewer`, `approver`) are extracted from OIDC claims and enforced.
- [x] **Tenant Isolation**: Cross-tenant data access is blocked by owner-or-admin authorization boundaries on all project, drillhole, and sample endpoints.
- [x] **MFA Enforcement**: High-risk actions (e.g., oil-spill review, financial transfer approval) require a verified MFA assurance claim in the OIDC token.

### 2. Policy-Based Access Control (PBAC)
- [x] **Open Policy Agent (OPA)**: All sensitive routes are deny-by-default and evaluated by an external OPA decision point.
- [x] **Maker-Checker Separation**: Financial transfers and model approvals require distinct authenticated actors.
- [x] **Policy Testing**: OPA rules are validated by a dedicated `.rego` test suite.

### 3. Application Security and WAF
- [x] **OpenAppSec Integration**: Preventive WAF policy is applied at the APISIX ingress, enforcing request size limits and blocking known attack patterns.
- [x] **Caddy Edge Proxy**: Caddy terminates TLS, sanitizes client IP headers, and drops malformed HTTP requests before they reach the API gateway.
- [x] **APISIX Allow-List**: Only explicitly declared routes are permitted through the API gateway.
- [x] **WALDO Service Hardening**: The WALDO inference service requires a service token, rejects URL-based ingestion (SSRF prevention), and limits image payload sizes.

### 4. Data Protection and Integrity
- [x] **PostgreSQL/PostGIS Migration**: SQLite fallbacks are disabled in production; all state is persisted to PostgreSQL.
- [x] **Idempotency Reservations**: Concurrent requests with the same idempotency key are safely rejected by the database.
- [x] **HMAC Audit Chain**: Financial events are chained using HMAC-SHA-256 to detect tampering.
- [x] **Model Artifact Hashing**: Raw-image inference models must match a pre-configured SHA-256 hash before loading.
- [x] **Safe Checkpoint Loading**: PyTorch models are loaded with `weights_only=True` to prevent arbitrary code execution via malicious pickles.

### 5. CI/CD and Dependency Management
- [x] **Reproducible Lockfiles**: `uv.lock` and `requirements-ci.lock` ensure deterministic builds.
- [x] **Automated Security Scans**: Bandit, pip-audit, and detect-secrets are integrated into the CI pipeline.
- [x] **Coverage Ratchet**: The CI pipeline fails if critical-path coverage drops below the measured baseline.
- [x] **Dependabot Configuration**: Automated dependency updates are configured for Python and GitHub Actions.

---

## Remaining Release Gates

Before authorizing a mission-critical production release, the following gates must be cleared by operations, compliance, and security teams.

### Gate 1: Infrastructure and SIEM Validation
- [ ] **External Log Aggregation**: Verify that Fluent Bit successfully ships Caddy, APISIX, OpenAppSec, Keycloak, and API logs to the external SIEM.
- [ ] **Alerting Rules**: Configure SIEM alerts for WAF prevention events, failed OPA decisions, and audit-chain verification failures.
- [ ] **PostgreSQL PITR Drill**: Execute a successful Point-in-Time Recovery (PITR) drill using pgBackRest from object storage.

### Gate 2: Financial and Compliance Approval
- [ ] **Financial Network Integration**: The current TigerBeetle ledger implementation is an internal control layer. Real-value transfers require integration with a licensed banking partner or payment network.
- [ ] **KYC/AML Verification**: Implement required Know Your Customer (KYC) and Anti-Money Laundering (AML) checks before enabling fund transfers.
- [ ] **Key Management**: Transition the HMAC audit key to a hardware security module (HSM) or managed KMS.

### Gate 3: Model and Data Quality
- [ ] **JEPA Benchmark Execution**: Run the reproducible `evaluate_oil_spill_promotion.py` benchmark against a real offshore dataset to verify the >97% oil-class performance claim.
- [ ] **Model Approval**: Formally approve the verified model artifact in the production database using the MFA-backed governance workflow.

### Gate 4: Independent Security Assessment
- [ ] **Penetration Testing**: Conduct a third-party penetration test against the integrated Caddy, APISIX, Keycloak, and API stack.
- [ ] **Threat Modeling**: Review the residual risk register with stakeholders and formally accept the remaining operational risks.

## References

[1] OWASP Application Security Verification Standard (ASVS): https://owasp.org/www-project-application-security-verification-standard/
[2] OWASP API Security Project: https://owasp.org/www-project-api-security/
[3] NIST Secure Software Development Framework (SSDF): https://csrc.nist.gov/pubs/sp/800/218/final
