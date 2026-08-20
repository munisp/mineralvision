# MineralVision Security Hardening and CI Benchmark Operations Runbook

**Author:** Manus AI  
**Status:** Implementation baseline; requires platform-owner review before deployment

> **Security statement:** This implementation establishes layered controls against external attack, privileged misuse, and application-layer denial of service. It cannot make any platform “bulletproof.” The remaining risk depends on identity administration, cloud/network configuration, timely patching, monitoring, recovery exercises, and disciplined operations.

## What Is Implemented

| Area | Repository implementation | Operational activation required |
|---|---|---|
| CI benchmark automation | `.github/workflows/security-and-benchmark.yml` validates configuration on security-relevant pull requests and supports a separately protected sealed-benchmark job. | Register and isolate a self-hosted runner; create the `sealed-benchmark` environment with required reviewers; set `SEALED_BENCHMARK_ROOT` as an environment variable. |
| CI supply-chain baseline | Read-only workflow permissions, pinned action commits, concurrency, dependency audit, secret-pattern checks, Dependabot, and CODEOWNERS. | Protect `main` and require code-owner review plus the security-contract status check. Workflow publication needs a GitHub token or browser action with workflow-write permission. |
| Edge protection | Caddy is the only container with public ports, terminates TLS, strips client-supplied forward headers, caps request bodies, and sets browser defenses. | Place a provider CDN/DDoS service and host firewall before Caddy; set DNS and real domains. |
| Gateway and DDoS controls | APISIX route allow-list applies separate general/API/image-inference quotas; it is private behind Caddy. | Tune quotas using authentic peak load; migrate rate counters to authenticated, clustered Redis policies for multi-gateway deployments. |
| Application protection | OpenAppSec APISIX attachment/agent topology with a local preventive-policy baseline. | Pin vendor images, verify compatibility, deploy in detection/learning first, validate blocking and false-positive handling, then enable prevention. |
| Identity and MFA | Keycloak realm export, OIDC-only production mode, short-lived asymmetric token validation, local auth routes disabled in OIDC mode, MFA claim gating for review and approval. | Replace example domains, configure an issuer-specific MFA/assurance claim mapper, enroll privileged users in WebAuthn, and test browser/redirect behavior. |
| PBAC and insider controls | OPA deny-by-default Rego policy maps oil-spill operations to roles; model approval and incident review require MFA. Keycloak/application audits are enabled/configured for export. | Review role assignment, configure log export to a separate retention-controlled system, and require two-person review for security-administrator and approver role changes. |
| Database segmentation | Application, migration, and Keycloak database roles are distinct; only Caddy exposes host ports; PostgreSQL and Redis are internal. | Inject real secrets, initialize a fresh database, migrate, verify grants, back up encrypted data, and test recovery. |

## CI/CD Benchmark Automation

The normal workflow executes on a hosted runner and contains no sealed benchmark data, deployment credentials, approval token, model artifact, or database password. Its job is to catch configuration drift, unsafe compose exposure, invalid identity JSON, policy/benchmark syntax errors, dependency issues, and accidentally committed secrets.

The sealed workflow intentionally runs only after manual dispatch, only in the GitHub environment named `sealed-benchmark`, and only on a runner carrying all labels below:

```text
self-hosted, linux, mineralvision-sealed-benchmark
```

The runner must be operated within the organizational security boundary. Mount the immutable benchmark release **outside** the repository checkout and make it read-only to the runner account. Create an environment variable named `SEALED_BENCHMARK_ROOT` with the mount path, for example `/srv/mineralvision/sealed-benchmark/release-2026-08-20`. The workflow verifies that `benchmark_manifest.csv` exists and is not writable before it evaluates anything.

The sealed job produces only the metrics report, deterministic split manifest, and API evaluation payloads. It does not upload raw aerial imagery, masks, ONNX/TorchScript artifacts, database secrets, or bearer tokens. It deliberately never calls the model approval endpoint. A qualified reviewer must independently inspect the data fingerprint, split manifest, artifact hash, and metrics before submitting evidence and requesting promotion.

### Required GitHub Repository Settings

Configure these settings through the repository administration interface. They cannot be made reliable solely by committing YAML.

| Setting | Required control |
|---|---|
| `main` branch protection | Require pull request, two approvals for security-sensitive changes, stale-approval dismissal, code-owner review, conversation resolution, and linear history. |
| Required checks | Require `CI / Ruff lint`, `CI / Byte-compile sweep`, `CI / Pytest with coverage gate`, and `Security Controls and Oil-Spill Benchmark / Validate security contracts`. |
| Workflow changes | Restrict edits under `.github/`, `security/`, authentication modules, Alembic, benchmark configuration, and secure compose through CODEOWNERS. |
| Actions policy | Allow only selected, verified actions; permit full-commit-SHA-pinned actions. Restrict self-hosted runners to repository/organization administrators. |
| Deployment environment | Create `sealed-benchmark`, restrict to named approvers, and do not attach cloud production credentials to it. Create a separate production environment requiring human approval. |
| Secrets | Store all secret values in environment/repository secret storage or an external manager; do not use workflow inputs for credentials. |

## Secure Container Deployment

The complete reference topology is `docker-compose.secure.yml`. It is intentionally separate from the development compose file because production security boundaries should not be a casual toggle on a local stack.

1. Create a dedicated Linux host or a managed container/Kubernetes platform that can run Docker, receive TLS traffic, and retain encrypted volumes. A managed website-only host is insufficient because Caddy, APISIX/OpenAppSec attachment, Keycloak, OPA, PostgreSQL, Redis, and background identity/audit services require operating-system and container control.
2. Obtain DDoS/CDN protection at the DNS/provider edge. Allow only that provider and operations network to reach the host firewall. Expose TCP 80/443 only for Caddy; never expose APISIX administration, Postgres, Redis, OPA, Keycloak administration, or internal metrics.
3. Copy `security/.env.security.example` into a protected deployment environment. Replace every `REPLACE_WITH_*` field using a secret manager. Pin all container references to vetted immutable digests under change control; the example tags are not production pins.
4. Create real domains and replace `app.example.com` and `auth.example.com` in `security/identity/mineralvision-realm.json` before importing the realm. Set the matching `PUBLIC_FQDN`, `IDP_FQDN`, `OIDC_ISSUER`, `OIDC_AUDIENCE`, and `OIDC_JWKS_URL` variables.
5. Review the initial Keycloak administrator account, then create ordinary realm administrators separately. The master realm must not be used for day-to-day application access. Enable login and administrative event export to the centralized security log sink. Keycloak supports OpenID Connect, role mapping, required actions, passkeys/WebAuthn and OTP factors; its own documentation describes these identity and event capabilities. [1]
6. Run PostgreSQL initialization only on a fresh volume, then run the dedicated `api-migrate` job. The API uses `mineralvision_app`; it is not the schema owner. Confirm that the PostgreSQL application role cannot execute schema changes.
7. Deploy OpenAppSec and APISIX first in staging. The OpenAppSec policy uses `prevent-learn`, but deployment teams should start with a reviewed detection/learning plan if their chosen vendor release or traffic profile requires it. The vendor documents an APISIX attachment plus agent topology for Docker deployments. [2]
8. Configure Caddy TLS/ACME only after the DDoS provider and DNS are in place. Caddy’s `header` directive supports the response header controls used in the supplied Caddyfile. [3]

## Keycloak MFA and OIDC/PBAC Activation

The API rejects production startup unless `AUTH_MODE=oidc`; it then validates configured Keycloak OIDC token issuer, audience, signature, expiration, and asymmetric JWKS key. It rejects HS algorithms for provider tokens. The API extracts Keycloak realm/client roles and a provider-issued MFA signal from `amr` or `acr` claims.

For privileged roles (`oil_spill_reviewer`, `oil_spill_approver`, and `security_admin`), require WebAuthn enrollment and configure the Keycloak authentication flow/claim mapper so successful WebAuthn step-up adds a recognized MFA assurance signal to access tokens. The OPA policy fails closed for incident review and promotion approval without that signal. Do **not** infer MFA merely from a role or password login.

Role assignment remains an insider-threat control point. Use dedicated groups from the realm export, limit membership changes to named administrators, require a second reviewer for approver/security-admin changes, export Keycloak administrative events, and remove access immediately when personnel or vendor assignments end.

OPA evaluates a minimal input document only: subject ID, roles, MFA status, project identifiers, HTTP method, and path. It does not receive bearer tokens, raw imagery, uploaded files, or request bodies. OPA policies are declarative Rego decisions over structured input, consistent with OPA’s documented policy model. [4]

## External Threat and DDoS Controls

APISIX uses a leaky-bucket `limit-req` plugin for sustained/short-burst traffic and a fixed-window `limit-count` control for expensive routes. The supplied `raw-image-inference` route is intentionally far stricter than ordinary API reads. APISIX supports Redis-backed counters for shared gateway deployments; the provided one-gateway baseline uses local counters and must be upgraded when scaling horizontally. [5] [6]

Rate limits protect application capacity, not the network uplink. Volumetric DDoS must be absorbed by an upstream CDN/provider service. Web attacks still require secure coding, patching, WAF monitoring, and a tested incident process; OpenAppSec is an additional control rather than a substitute for those measures.

## Required Monitoring and Response

Export and correlate the following data outside the production host: Caddy access/error logs, APISIX rate-limit denials, OpenAppSec detect/prevent events, Keycloak login and administrative events, OPA policy denials/failures, FastAPI audit events, PostgreSQL authentication/privilege events, host/container logs, and cloud firewall/CDN events. Restrict the log system so a production application administrator cannot silently modify or delete the only evidence source.

Alert at minimum on repeated MFA failures, privileged role changes, model approval attempts, OPA unavailability, policy denials by privileged identities, bursts of image-inference requests, unusual API 401/403/429/5xx ratios, WAF prevent events, database login failures, secrets rotation failures, and CI workflow/security-file changes.

## Residual Risks and Controls Not Automated Here

| Risk | Why code alone cannot close it | Required owner action |
|---|---|---|
| Volumetric DDoS | It occurs before the host/application stack. | Provision upstream CDN/DDoS protection, provider WAF, and firewall rules. |
| Compromised privileged identity | MFA lowers risk but does not eliminate session theft, social engineering, or malicious administrators. | Conditional access, managed devices, WebAuthn, short sessions, separate admin accounts, off-host audit, access reviews. |
| Vulnerable image or policy release | Container tags and WAF rules can drift. | Digest pinning, SBOM/vulnerability review, staging, patch SLAs, rollback testing. |
| Misconfigured Keycloak/MFA claims | OIDC/JWKS presence does not prove correct authentication assurance. | Test redirection, login flow, claims, logout, break-glass, recovery, and role governance before production cutover. |
| WAF false positives/negatives | Detection quality depends on traffic and tuning. | Monitor learning, conduct staged attack validation, define emergency bypass/change process with audit. |
| Insider data exfiltration | Application authorization cannot control screenshots, copied exports, or an administrator with broad infrastructure access. | Least privilege, export controls, DLP, device management, separate duties, monitoring, contractual controls. |

## References

[1] [Keycloak Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/)  
[2] [APISIX + OpenAppSec Docker integration guidance](https://apisix.apache.org/blog/2024/10/22/apisix-integrates-with-open-appsec/)  
[3] [Caddy `header` directive](https://caddyserver.com/docs/caddyfile/directives/header)  
[4] [Open Policy Agent policy language](https://www.openpolicyagent.org/docs/latest/policy-language/)  
[5] [APISIX `limit-req` plugin](https://apisix.apache.org/docs/apisix/3.10/plugins/limit-req/)  
[6] [APISIX `limit-count` plugin](https://apisix.apache.org/docs/apisix/plugins/limit-count/)
