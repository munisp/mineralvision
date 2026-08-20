# Security Control Research Notes

**Captured:** 2026-08-20

| Control | Relevant verified capability | Design implication |
|---|---|---|
| Apache APISIX `limit-req` | Uses a leaky-bucket request limit. It supports an address, consumer, or combined request key, and can use Redis-backed shared counters for multi-node operation. | Enforce short-burst and sustained limits at the API gateway, using a trusted client address and authenticated consumer/subject where available. Keep the admin plane private. |
| Apache APISIX `limit-count` | Applies a fixed-window quota and can reject excess requests. | Apply lower fixed-window controls to costly image-inference and authentication endpoints. |
| Caddy response headers | The Caddy `header` directive can set and remove response headers, including HSTS, content-type, clickjacking, and permissions policies. | Terminate TLS at Caddy and emit baseline browser defenses. Caddy is not the principal distributed DDoS control. |
| Keycloak | Supports OpenID Connect, role/claim mapping, passkeys, recovery codes, TOTP/HOTP, step-up authentication, and event streams. | Replace shared application-issued JWTs for production users with OIDC bearer tokens; require phishing-resistant WebAuthn for privileged accounts and keep TOTP as a recovery-compatible option. |
| OPA | Evaluates declarative Rego policies over structured inputs such as API requests and configuration data. | Use OPA as a deny-by-default policy decision point for privileged oil-spill operations and evaluate policy tests in CI. |
| OpenAppSec | Documents a deployment/agent approach and a management API. | Include an integration-ready protected-proxy service configuration but do not claim the WAF is active without enrolled management credentials and a validated policy. |

## Sources

1. [APISIX `limit-count`](https://apisix.apache.org/docs/apisix/plugins/limit-count/)
2. [APISIX `limit-req`](https://apisix.apache.org/docs/apisix/3.10/plugins/limit-req/)
3. [Caddy `header` directive](https://caddyserver.com/docs/caddyfile/directives/header)
4. [Keycloak Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/)
5. [Open Policy Agent Policy Language](https://www.openpolicyagent.org/docs/latest/policy-language/)
6. [open-appsec Documentation](https://docs.openappsec.io/)
