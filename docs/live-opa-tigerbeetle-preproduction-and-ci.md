# Live OPA and TigerBeetle Pre-Production Integration

This runbook describes the isolated pre-production configuration in `docker-compose.financial-preprod.yml` and the pull-request workflow in `.github/workflows/financial-preprod-integration.yml`. The environment validates live policy decisions and TigerBeetle cluster reachability with **synthetic data only**. It must never use production cluster IDs, volumes, HMAC keys, banking credentials, customer accounts, or payment-provider access.

## Architecture and Trust Boundary

The pre-production stack consists of an OPA policy service and a single-replica TigerBeetle cluster connected only to the internal `mineralvision-financial-preprod` Docker network. No service publishes a host port. OPA serves the checked-in `financial_transfer.rego` policy, while TigerBeetle listens on its private TCP endpoint after an explicit one-time data-file format operation. The harness calls OPA's REST decision endpoint and performs a TCP liveness probe to the ledger. It deliberately does not create accounts or submit transfers.

> A single-replica pre-production cluster validates integration contracts; it does not validate production availability. TigerBeetle documents deployment and replica configuration separately, including multi-node cluster operation for high availability. [1]

| Component | Pre-production value | Production boundary |
|---|---|---|
| OPA | `opa:8181`, private Docker network | Versioned bundle distribution, policy signing, HA replicas, off-host decision logs |
| TigerBeetle | `tigerbeetle:3000`, synthetic-only local volume | Independent cluster ID, multi-replica topology, durable encrypted storage, backup/recovery drill |
| Policy facts | Synthetic identities and committed-fact schema | Derived only by the private payment service from verified OIDC and PostgreSQL records |
| Transfer test | OPA decisions and TCP liveness only | Licensed payment-partner sandbox after compliance, key-management, and independent security approval |

## Local Operator Procedure

Start with an empty disposable volume. Formatting a nonempty TigerBeetle data file is destructive, so do not reuse a production or long-lived testing volume.

```bash
cd /path/to/mineralvision

docker compose -f docker-compose.financial-preprod.yml --profile bootstrap run --rm tigerbeetle-init
docker compose -f docker-compose.financial-preprod.yml up -d opa tigerbeetle

# Check live policy health.
docker compose -f docker-compose.financial-preprod.yml exec -T opa \
  opa eval --format pretty --data /policies \
  'data.mineralvision.financial.decision'

# Run the live policy and private-ledger reachability test from a network peer.
docker run --rm --network mineralvision-financial-preprod \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --security-opt no-new-privileges:true --cap-drop ALL \
  -e OPA_URL=http://opa:8181 \
  -e TIGERBEETLE_ADDRESS=tigerbeetle:3000 \
  -v "$PWD:/workspace:ro" -w /workspace python:3.12-slim sh -ceu '
    pip install --no-cache-dir requests==2.32.5
    python scripts/run_financial_preprod_integration.py
  '

docker compose -f docker-compose.financial-preprod.yml down --volumes --remove-orphans
```

The expected lifecycle is maker submission allowed, maker self-approval denied, two distinct MFA-backed high-value approvals allowed, early release denied, final distinct-releaser release allowed, and successful private TigerBeetle TCP connectivity. The harness cannot move value.

## Pull-Request CI Contract

The CI workflow triggers only when the financial policy, TigerBeetle integration overlay, financial middleware, or harness changes. It uses `contents: read` permissions, does not access GitHub environments or secrets, runs entirely on an ephemeral runner, and destroys its synthetic volume in an `always()` cleanup step. On failure it uploads only short-lived, sanitized service diagnostics.

The workflow is intentionally not a payment test. It proves that the pinned OPA policy parses and returns the expected maker-checker decisions, the disposable TigerBeetle listener is reachable, and the integration network stays private. A payment-service adapter should add a separate, manually approved test against a regulated partner's sandbox only after independent review.

## Required Pre-Production Controls

| Gate | Required evidence | Owner |
|---|---|---|
| Policy review | Four-eyes approval of Rego changes and passing `financial_transfer_test.rego` | Security engineering |
| Ledger topology | Fresh synthetic volume, isolated cluster ID, no production network route | Platform engineering |
| Identity assurance | Real Keycloak test-realm MFA claims mapped to policy input by a private payment service | Identity engineering |
| Durable facts | PostgreSQL transfer-intent and approval rows are committed before OPA release authorization | Payments engineering |
| Audit integrity | HMAC chain verifier passes, audit key comes from a test KMS key, and tamper test fails closed | Security and compliance |
| Payment safety | No payment rail reachable from ordinary PR CI | Release management |

## References

[1] [TigerBeetle Docker deployment documentation](https://docs.tigerbeetle.com/operating/deploying/docker/)

[2] [TigerBeetle deployment documentation](https://docs.tigerbeetle.com/operating/deploying/)

[3] [OPA policy bundle documentation](https://www.openpolicyagent.org/docs/management-bundles/)
