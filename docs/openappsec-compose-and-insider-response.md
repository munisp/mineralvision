# OpenAppSec Payload Inspection, Secure Compose Deployment, and Insider-Threat Response

**Author:** Manus AI  
**Scope:** MineralVision oil-spill APIs, especially `POST /api/oil-spill/analyze/image`.

> **Security boundary:** The committed OpenAppSec policy is a validated configuration baseline, not proof of safe production blocking. It must be tested with the selected, digest-pinned OpenAppSec release in staging before a prevention decision is relied upon. The policy intentionally avoids logging request bodies, tokens, and raw aerial imagery.

## 1. Payload-Inspection Controls for Raw Aerial Imagery

The raw-image endpoint performs expensive local segmentation and accepts multipart image uploads. Its protection is layered: Caddy limits body size before proxying; APISIX exposes one explicit POST route with a strict quota; OpenAppSec inspects web-attack characteristics and malformed protocol content; FastAPI checks content type, valid image decoding, and a 25 MiB application-side image limit.

| Layer | Implemented raw-image control | Failure behavior |
|---|---|---|
| Caddy | Only the public edge; `request_body` is capped before the private proxy path. Client-supplied forwarding headers are removed. | Edge refusal; request does not reach APISIX. |
| APISIX | Exact route `/api/oil-spill/analyze/image`, method `POST`; leaky-bucket rate `2/s`, burst `4`; count quota `30/hour` per trusted client IP. | HTTP `429`; endpoint is not invoked. |
| OpenAppSec | `web-attacks` practice limits request body to `25,600 KiB`, headers to 32 KiB, URL length to 8 KiB, object depth to 32, and uses high-confidence preventive learning mode. Protocol method checks and enabled signature/schema practices are also present. | Preventive policy response is HTTP `403`. |
| FastAPI | `image/*` content type, non-empty request, 25 MiB byte limit, image parser validity, strict registered/approved model provenance, and review state. | HTTP `4xx` or `503`; no model execution for unapproved artifacts. |

The exact reusable WAF policy is `security/appsec/local_policy.yaml`. Its active policy is `prevent-learn`, assigns the `webapp-default-practice`, and outputs JSON-formatted WAF events to stdout. Request headers, body, URL path, and query are deliberately disabled under extended logging so inspection does not create a second store of sensitive aerial evidence or credentials.

OpenAppSec can use an attachment/agent pattern with APISIX; the secure Compose file places the APISIX attachment and agent in a shared IPC namespace, while keeping both behind Caddy. The OpenAppSec/APISIX integration guidance documents this attachment topology. [1]

### Activation Procedure

1. Select official, tested container images by immutable digest, not a mutable tag. Set `APPSEC_APISIX_IMAGE` and `APPSEC_AGENT_IMAGE` in the deployment secret store.
2. In a staging network, mount `security/appsec/local_policy.yaml` read-only at `/ext/appsec/local_policy.yaml`; do not add credentials to the policy file.
3. Start in an OpenAppSec detection/learning workflow appropriate for the selected release. Submit authorized tests: legitimate JPEG/PNG imagery at normal and near-limit sizes; rejected oversized images; malformed multipart data; unsupported media; known invalid HTTP methods; and a controlled attack corpus approved by the security team.
4. Review false positives and the WAF stdout event stream. Confirm no raw request body, bearer token, or image bytes appear in the logs. Only after review should the selected release’s preventive policy be activated for production.
5. Keep APISIX rate limits and the FastAPI limit enabled even when WAF prevention is active. WAF inspection does not replace workload protection or backend validation.

The current policy’s `specific-rules` list is intentionally empty. Do not invent path-specific OpenAppSec rule syntax from an example or apply it without release-specific validation. If a precise rule for `POST /api/oil-spill/analyze/image` is required beyond the existing APISIX route, author and test it against the vendor version in use, then add it under change control with a negative/positive test corpus.

## 2. Secure Compose Topology

`docker-compose.secure.yml` integrates the requested services. It is a production-oriented reference, not a substitute for managed networking, image scanning, secrets management, or an upstream DDoS provider.

```text
Internet / CDN-DDoS provider
          |
          v
      Caddy :443  (only published public port)
          |
          v
 APISIX + OpenAppSec attachment <--> OpenAppSec agent (shared IPC)
          |
          v
      MineralVision API  <------> OPA policy engine
          |       |                 ^
          |       +----JWKS-------- Keycloak (OIDC realm)
          |
          +------------------------ PostgreSQL / Redis / WALDO
```

| Service | Network placement | Security intent |
|---|---|---|
| `caddy` | `edge`; publishes only `80:80` and `443:443` | TLS termination and sole Internet entry. |
| `apisix` | `edge`, `application`; exposes port 9080 internally only | Route allow-list, per-IP quotas, WAF attachment. Admin plane is not published. |
| `appsec-agent` | `application` | Supplies inspection decisions to APISIX over shared IPC; logs are a dedicated volume. |
| `api` | `application`, `data`; port 8000 exposed internally only | OIDC and OPA-aware business API. |
| `opa` | `application`; port 8181 exposed internally only | Fail-closed authorization decisions for oil-spill requests. |
| `keycloak` | `edge`, `data`; port 8080 internal | OIDC identity provider, realm roles, MFA/session assurance. |
| `postgres`, `redis` | `data` internal network | Separate credentials and no host-published ports. |
| `api-migrate` | `data`, one-shot | Runs Alembic with a separate migrator role; application role is not the schema owner. |

The `application` and `data` networks are Docker `internal: true`. Only Caddy has host-port mappings. Containers drop Linux capabilities and use `no-new-privileges`; several are read-only with a tmpfs `/tmp`. The deployment variables belong in a secret manager or protected environment derived from `security/.env.security.example`, never in Git.

A deployment sequence is:

```bash
# On a hardened host; secrets must be injected by a secret manager or protected CI environment.
cp security/.env.security.example /secure/runtime/mineralvision.env
# Replace every REPLACE_WITH_* value outside the repository.

# Validate images, DNS, TLS, secret injection, and the Keycloak realm in staging first.
docker compose --env-file /secure/runtime/mineralvision.env \
  -f docker-compose.secure.yml config --quiet

docker compose --env-file /secure/runtime/mineralvision.env \
  -f docker-compose.secure.yml up -d
```

Use an upstream CDN/DDoS service and host firewall ahead of Caddy. Docker-level controls cannot absorb volumetric traffic before it reaches the host.

## 3. Implemented Insider-Threat Evidence and Audit Mechanisms

The application records both a general audit trail and an incident-specific timeline in PostgreSQL.

| Store | Implemented fields/events | Investigation use |
|---|---|---|
| `audit_logs` | `action`, entity type/id, optional user/IP fields, JSON details, UTC timestamp | Correlates assessments, reviews, model registration/evaluation/approval, exports, and incident-event creation. |
| `oil_spill_incident_events` | incident ID, event type, actor, JSON details, UTC timestamp | Reconstructs evidence attachment, review completion, coverage planning, explicit operational events, and export timeline. |
| `oil_spill_incidents` | reviewer, review note, review timestamp, immutable evidence summary/provenance fields | Shows current state together with reviewer attribution. |
| `oil_spill_models` and evaluation runs | model artifact SHA-256, intended domains, evaluation metrics, reviewer, approval identity/time | Detects or investigates unapproved model use and questionable promotion evidence. |
| Keycloak events | login, administrative events, brute-force events, group/role changes when exported | Detects privileged role assignment, unusual sign-in patterns, and MFA/identity anomalies. |
| Caddy/APISIX/OpenAppSec/OPA logs | edge access, rate-limit 429, WAF detect/prevent, policy allow/deny/error | Correlates external probing with identity and business actions. |

The code creates audit events for oil-spill assessment creation, incident review, coverage-plan generation, model registration, evaluation recording, model approval, incident event creation, and export. The review endpoint writes both `review_completed` to the incident timeline and `oil_spill_incident_reviewed` to `audit_logs` in the same database transaction.

### Important Audit Limitation

The current PostgreSQL tables are application audit records; they are **not cryptographically immutable**, and the API does not yet include a full user/IP/request-ID capture for every route. A database administrator could modify them. Therefore, export Keycloak admin events, Caddy/APISIX/OpenAppSec/OPA logs, database authentication/audit logs, and application audit events to a separate, retention-controlled security information and event management system. Restrict that system so routine application/database administrators cannot erase the sole copy of evidence.

## 4. Insider-Threat Incident Response Workflow

| Phase | Required action | Evidence/decision owner |
|---|---|---|
| Detect | Alert on privileged role/group changes, MFA anomalies, OPA denies for privileged identities, failed/rapid raw-image inference, unexpected exports, model approvals, and unusual DB/API activity. | Security operations. |
| Triage | Open a case; record UTC time, affected user/service, incident/model IDs, source IP/device, OIDC subject, role/MFA claims, policy decision, and related edge/WAF events. Classify safety/data/integrity impact. | Incident commander and oil-spill operations lead. |
| Preserve | Export immutable copies of relevant Keycloak, edge, WAF, OPA, PostgreSQL, `audit_logs`, and `oil_spill_incident_events` records. Hash exports and store them in access-controlled evidence storage. | Forensic custodian, not the suspected administrator. |
| Contain | Disable the Keycloak user/session, remove privileged group membership, rotate affected secrets, block an abusive source at CDN/Caddy/APISIX, and optionally set OPA policy to deny the affected role. Never delete the incident/model records to “clean up.” | Identity/security owner with documented approval. |
| Investigate | Correlate subject/role changes with reviewer/approver actions, model SHA-256, evaluation fingerprint, raw-image uploads, review notes, and API/edge timing. Determine whether the action was authorized, compromised, or malicious. | Security and domain reviewers. |
| Eradicate and recover | Correct role/policy/configuration defects, restore approved model/configuration from signed change records, revalidate OIDC/MFA/OPA controls, and resume only under explicit authorization. | Platform and security change authority. |
| Lessons learned | Preserve a sanitized post-incident report; update WAF/APISIX controls, role reviews, detection rules, CI checks, and response playbooks. | Governance owner. |

Avoid relying on the application’s `reviewer` request body field as proof of identity. The authoritative evidence for a production review is the OIDC subject and MFA claim captured by an external audit pipeline, correlated with the application record and OPA decision. A future improvement should bind review actors directly from verified `request.state.user` and store request identifiers/IP hashes in an append-only external ledger.

## References

[1] [Apache APISIX and OpenAppSec integration](https://apisix.apache.org/blog/2024/10/22/apisix-integrates-with-open-appsec/)  
[2] [OpenAppSec documentation](https://docs.openappsec.io/)  
[3] [Apache APISIX `limit-req` plugin](https://apisix.apache.org/docs/apisix/3.10/plugins/limit-req/)  
[4] [Apache APISIX `limit-count` plugin](https://apisix.apache.org/docs/apisix/plugins/limit-count/)
