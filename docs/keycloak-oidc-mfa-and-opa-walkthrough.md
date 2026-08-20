# Keycloak OIDC, MFA, OPA, and APISIX Operator Walkthrough

**Scope:** Oil-spill review and model-approval access in the MineralVision secure deployment baseline.

> **Safety rule:** Do not apply the imported `browser-mfa` flow to the production realm until a named break-glass administrator and at least one test reviewer have successfully registered WebAuthn credentials in staging. The supplied import declares `webauthn-authenticator` as `REQUIRED`; applying it globally before enrollment can prevent ordinary users from signing in.

## 1. Prepare the Realm Import

The committed realm template is `security/identity/mineralvision-realm.json`. Before importing it, replace every example hostname. The Keycloak issuer, browser redirect URIs, WebAuthn relying-party ID, Caddy virtual host, and API environment must agree exactly.

| Setting | Required production value | Why it must match |
|---|---|---|
| Public application host | `app.<organization-domain>` | The web client redirect URI, web origin, Caddy public site, and WebAuthn RP ID reference it. |
| Keycloak host | `auth.<organization-domain>` | This becomes the OIDC issuer and the Keycloak Caddy site. |
| Realm | `mineralvision` | The API validates this issuer path: `https://auth.<organization-domain>/realms/mineralvision`. |
| API audience | `mineralvision-api` | The API rejects tokens whose `aud` does not match. |
| Web client | `mineralvision-web` | It is a public authorization-code client with PKCE S256, not a password-grant client. |

Update these JSON entries before import:

```text
redirectUris:                    https://app.<organization-domain>/*
webOrigins:                      https://app.<organization-domain>
post.logout.redirect.uris:       https://app.<organization-domain>/*
webAuthnPolicyRpId:              app.<organization-domain>
```

Then set the matching protected deployment variables:

```bash
export PUBLIC_FQDN="app.<organization-domain>"
export IDP_FQDN="auth.<organization-domain>"
export AUTH_MODE="oidc"
export OIDC_ISSUER="https://${IDP_FQDN}/realms/mineralvision"
export OIDC_AUDIENCE="mineralvision-api"
export OIDC_JWKS_URL="${OIDC_ISSUER}/protocol/openid-connect/certs"
export OIDC_ALLOWED_ALGORITHMS="RS256,ES256"
export OPA_ENABLED="true"
export OPA_URL="http://opa:8181"
export OPA_FAIL_CLOSED="true"
```

Never substitute an HTTP issuer in production, allow `HS*` algorithms, use wildcard origins, or configure Keycloak’s direct-access password grant for browser users.

## 2. Import and Review the Keycloak Realm

Start Keycloak behind Caddy with its dedicated PostgreSQL database, then sign in to the **master realm only with the bootstrap administrator**. Create/import the `mineralvision` realm through the administration console. Use the realm import file above, but inspect rather than blindly trust all imported settings.

The template disables self-registration, enables email verification, brute-force protection, realm/admin audit events, a short five-minute access-token lifetime, and a password policy. These should remain enabled. Keycloak supports OpenID Connect, client and realm roles, authentication flows, WebAuthn/passkeys, OTP, required actions, and event streams. [1]

Confirm the imported role-to-group mapping below. Do not assign realm roles directly to normal users; assign audited groups, and limit changes to designated identity administrators.

| Group | Realm role | Permitted business purpose |
|---|---|---|
| `oil-spill-operators` | `oil_spill_operator` | Read incidents, submit analysis, plan advisory coverage. |
| `oil-spill-reviewers` | `oil_spill_reviewer` | Read incidents and submit accountable reviews after MFA. |
| `oil-spill-evaluators` | `oil_spill_evaluator` | Register a candidate model and submit sealed-evaluation evidence. |
| `oil-spill-approvers` | `oil_spill_approver` | Approve a model only after MFA and independent review. |
| `security-administrators` | `security_admin` | Break-glass role; do not assign for routine operations. |

Create one non-production test user, add it to `oil-spill-reviewers`, and keep the master-realm bootstrap administrator separate from normal operating identities.

## 3. Configure Browser OIDC and API Token Validation

The `mineralvision-web` client must use **Authorization Code Flow with PKCE S256**. Leave implicit flow, direct access grants, and service accounts disabled. Its redirect URI and web origin must be exact HTTPS values; broad redirects introduce token exfiltration risk.

The `mineralvision-api` client is bearer-only. It does not authenticate a browser or hold a user password. The FastAPI OIDC validator obtains a signing key from `OIDC_JWKS_URL`, then verifies the token signature, allowed asymmetric algorithm, expiry, issuer, audience, issued-at time, and subject. It maps Keycloak realm/client role claims into `request.state.user.roles`.

The application refuses `ENV=production` unless `AUTH_MODE=oidc`. In OIDC mode, the legacy `/auth/login`, registration, password management, refresh, and local logout semantics are disabled; users must use Keycloak’s OIDC and end-session flows. This prevents two parallel sources of truth for production credentials.

After login, retrieve a non-sensitive decoded token view from your controlled browser or a test client and confirm all of the following before granting reviewer access:

```json
{
  "iss": "https://auth.<organization-domain>/realms/mineralvision",
  "aud": ["mineralvision-api"],
  "realm_access": {"roles": ["oil_spill_reviewer"]},
  "exp": 1735689900,
  "sub": "stable-user-identifier"
}
```

The actual `aud` format can be a string or array according to client mapper configuration, but it must satisfy the API validator. Do not use a copied access token in support tickets, shell history, source code, or long-lived documentation.

## 4. Enroll MFA and Make the Assurance Claim Verifiable

The template enables a default `webauthn-register` required action and includes a `browser-mfa` authentication flow whose WebAuthn executor is `REQUIRED`. WebAuthn is preferred for privileged identities because it is phishing-resistant when the RP ID is correct. TOTP is available only as an explicitly enabled recovery-compatible action, not as proof that every privileged session completed MFA.

Use this staged procedure.

1. In staging, authenticate the test reviewer with a temporary enrollment-safe flow, complete the WebAuthn registration action using a managed hardware or platform authenticator, and verify that the credential appears in the user’s **Credentials** list.
2. Create a dedicated copy of the browser flow for privileged testing. It must execute password authentication and the WebAuthn authenticator, then bind only after users are enrolled. Keep a separately tested break-glass administrative route that is time-limited, audited, and not used for application review work.
3. Configure the Keycloak token protocol mapper or authentication-session mapper used by your Keycloak release to emit an assurance signal **only after a completed WebAuthn/step-up execution**. MineralVision recognizes a token `amr` array containing `mfa`, `otp`, `webauthn`, or `hwk`, or an `acr` value of `mfa`, `aal2`, `aal3`, or `gold`.
4. Sign in through the actual privileged flow and verify that the **access token**, not merely the user’s group membership, contains the expected `amr` or `acr` signal. A role, a stored WebAuthn credential, or a password-only session is insufficient.
5. Test a reviewer request without the assurance signal. The API must return HTTP `403` with `policy_denied`. Repeat after the valid MFA flow; the request should reach the review endpoint if the role is present.

> If the selected Keycloak release cannot reliably issue an `amr`/`acr` value tied to the required WebAuthn execution, do not activate reviewer or approver access. Use a supported session-note/assurance mapper or a reviewed Keycloak extension, and re-run the negative and positive tests.

## 5. OPA Deny-by-Default Policy for Oil-Spill REST Calls

The API enables `OPAMiddleware` for every `/api/oil-spill/` request. In production, startup rejects a configuration with `OPA_ENABLED` disabled; with `OPA_FAIL_CLOSED=true`, an unavailable or malformed policy response returns HTTP `403`, not a permissive fallback.

The middleware sends OPA only a minimal input object. It does **not** forward access tokens, raw aerial images, masks, upload bodies, or credentials.

```json
{
  "subject": {
    "id": "keycloak-subject",
    "roles": ["oil_spill_reviewer"],
    "mfa_verified": true,
    "project_ids": []
  },
  "request": {
    "method": "POST",
    "path": "/api/oil-spill/incidents/INCIDENT_ID/review"
  },
  "resource": {"project_id": ""}
}
```

The policy maps method/path to one action, maps roles to precise permissions, and starts from `default decision := {"allow": false, ...}`. A missing route mapping becomes action `unknown` and remains denied. OPA’s Rego model evaluates declarative rules over structured input, which is the basis for this permission decision. [2]

| Endpoint pattern | OPA action | Required role | MFA required |
|---|---|---|---|
| `GET /api/oil-spill/*` | `oil_spill.read` | Any oil-spill role | No |
| `POST /api/oil-spill/analyze/*` | `oil_spill.analyze` | Operator or security administrator | No |
| `POST /api/oil-spill/coverage-plan` | `oil_spill.coverage` | Operator or security administrator | No |
| `PATCH /api/oil-spill/incidents/*/review` | `oil_spill.review` | Reviewer or security administrator | **Yes** |
| `POST /api/oil-spill/incidents/*/events` | `oil_spill.events` | Reviewer or security administrator | No |
| `POST /api/oil-spill/models` | `oil_spill.model.register` | Evaluator or security administrator | No |
| `POST /api/oil-spill/models/*/evaluations` | `oil_spill.model.evaluate` | Evaluator or security administrator | No |
| `POST /api/oil-spill/models/*/approve` | `oil_spill.model.approve` | Approver or security administrator | **Yes** |

For a direct controlled policy test inside the private `application` network, use an input file without any token material:

```bash
curl --fail --silent --show-error \
  --request POST http://opa:8181/v1/data/mineralvision/authz/decision \
  --header 'Content-Type: application/json' \
  --data @reviewer_mfa_input.json
```

A positive reviewer input should return `{"result":{"allow":true,...}}`; the same request with `"mfa_verified": false` must return an allow value of false.

## 6. APISIX Allow-List and the Complete Request Path

APISIX is private; its Admin API has no published host port. Caddy is the only public service, terminates TLS, removes client-supplied forwarding headers, supplies the trusted client IP, bounds the body size, and proxies to APISIX. APISIX exposes only four declarative route classes:

| Route | Methods | Upstream | Deny/abuse control |
|---|---|---|---|
| `/api/oil-spill/analyze/image` | `POST` | API | 2 requests/s, burst 4; max 30/hour per trusted client IP. |
| `/api/*` | API methods matched by APISIX route | API | 25 requests/s, burst 50; max 1,200/hour per trusted client IP. |
| `/health` | `GET` | API | 5 requests/s, burst 10. |
| `/*` | `GET`, `HEAD` | UI | 30 requests/s, burst 60. |

A request that matches no APISIX route is rejected before it reaches FastAPI. A rate-limited request is rejected as HTTP `429`. A request that reaches FastAPI must pass OIDC authentication, then passes through OPA for oil-spill paths. Finally, the endpoint’s own model-governance and human-review requirements apply.

```mermaid
sequenceDiagram
  participant Client
  participant Caddy
  participant APISIX
  participant API as FastAPI
  participant Keycloak
  participant OPA
  Client->>Caddy: HTTPS request + Bearer token
  Caddy->>APISIX: sanitized forwarded IP; bounded body
  APISIX->>APISIX: route allow-list and quota check
  APISIX->>API: permitted private upstream request
  API->>Keycloak: JWKS-backed issuer/audience/signature validation
  API->>OPA: roles + MFA flag + method + path only
  OPA-->>API: allow or deny
  API-->>Client: endpoint result or 401/403/429
```

APISIX controls **which paths and traffic volumes reach the API**. OIDC verifies **who holds a valid identity token**. OPA decides **whether that identity is permitted to perform the specific oil-spill action now**, including MFA gates. The endpoint then validates domain-specific requirements. These are complementary controls, not redundant replacements.

## References

[1] [Keycloak Server Administration Guide](https://www.keycloak.org/docs/latest/server_admin/)  
[2] [Open Policy Agent Policy Language](https://www.openpolicyagent.org/docs/latest/policy-language/)
