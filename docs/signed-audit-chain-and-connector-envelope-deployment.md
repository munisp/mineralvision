# Signed Audit Chain and Connector Envelope Deployment Runbook

**Prepared by Manus AI**  
**Date:** 22 August 2026

## Preconditions

A high-assurance deployment must use PostgreSQL for `MV_HUB_DATABASE_URL`, a KMS/HSM or equivalent workload-identity signing service, a tenant-bound connector identity, an independently administered off-host collector, and an approved public-key registry. The application must not load long-lived production private keys from repository files, Docker images, browser storage, or environment variables visible to general operators.

The following implementation objects are available:

| Component | Purpose |
|---|---|
| `audit_crypto.py` | Canonical hashing, Ed25519 envelope signing/verification, stream append, JSON Lines export, and independent chain verification. |
| `hub_audit_streams` | One row per tenant/stream storing the latest chain anchor and active signing key ID. |
| `hub_signed_audit_events` | Signed append-only audit events. Application roles must have `SELECT` and `INSERT` only; no `UPDATE` or `DELETE`. |
| `verify_offhost_audit_bundle.py` | Verifies exported JSON Lines evidence using only public keys. |
| `export_signed_audit_bundle.py` | Exports signed events from a PostgreSQL-backed hub without private key material. |

## Database Permissions

Create distinct database roles for application append, export, and schema migration. The connector/application role must not be able to update or delete audit events after initial deployment.

```sql
REVOKE ALL ON hub_audit_streams, hub_signed_audit_events FROM PUBLIC;
GRANT SELECT, INSERT ON hub_audit_streams, hub_signed_audit_events TO mv_hub_audit_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mv_hub_audit_app;
GRANT SELECT ON hub_audit_streams, hub_signed_audit_events TO mv_hub_audit_exporter;
GRANT ALL PRIVILEGES ON hub_audit_streams, hub_signed_audit_events TO mv_hub_migrator;
REVOKE UPDATE, DELETE ON hub_audit_streams, hub_signed_audit_events FROM mv_hub_audit_app, mv_hub_audit_exporter;
```

The service must use a separate, narrowly scoped migration role only during deployment. The database owner is not an application runtime identity.

## Key Registry Format

The off-host verifier accepts a JSON file containing raw Ed25519 public keys encoded in standard Base64. The registry must be signed, versioned, access-controlled, and distributed independently from the audit exporter.

```json
{
  "audit-2026-q3": "<base64-encoded-32-byte-Ed25519-public-key>",
  "connector-2026-q3": "<base64-encoded-32-byte-Ed25519-public-key>"
}
```

Private signing keys remain in an HSM/KMS or signing service. The application invokes a controlled signing operation or receives a short-lived delegated signing credential. For development only, `private_key_from_b64` accepts 32 raw private-key bytes encoded as Base64; this path must not be used for production key custody.

## Connector Envelope Procedure

1. The connector reads an approved source record and constructs an envelope containing tenant ID, connector ID, event type, issued time, nonce, key ID, payload hash, and payload.
2. The trusted signing component signs the canonical envelope with the active Ed25519 key.
3. MineralVision resolves `key_id` against its active public-key registry and verifies the signature, payload hash, expected tenant, envelope age, and nonce uniqueness before accepting the event.
4. The accepted event is appended to the tenant/connector audit stream.
5. Any invalid signature, unknown key, changed payload, expired envelope, or replayed nonce is rejected and itself logged through a security event path.

The receiver must persist nonces in a tenant/connector replay store with a retention period longer than the permitted envelope age. The in-process `seen_nonces` argument is a test/local primitive, not durable production replay protection.

## Append and Export Procedure

The application appends audit events through `append_signed_audit_event`. It serializes the event fields canonically, hashes the event with SHA-256, signs the resulting commitment with Ed25519, and updates the stream anchor under a database row lock. Stream-key changes are blocked until an explicit rotation procedure is added and approved.

Export after each batch or at a bounded interval. A deployment should not allow export delay to exceed the agreed recovery/audit objective.

```bash
export ENV=production
export MV_HUB_DATABASE_URL='postgresql://mv_hub_audit_exporter:<secret>@postgres/mineralvision_hub'
python scripts/export_signed_audit_bundle.py \
  --tenant northstar-mining \
  --stream connector:arcgis-prod-1 \
  --after-sequence 0 \
  --output /secure/outbox/northstar-arcgis-000001.jsonl
```

Transfer the output only through mutually authenticated TLS or an equivalently authenticated private transport to an allow-listed collector. The collector must verify before acknowledging receipt:

```bash
python scripts/verify_offhost_audit_bundle.py \
  --bundle /secure/collector/northstar-arcgis-000001.jsonl \
  --public-keys /secure/collector/audit-public-keys.json \
  --tenant northstar-mining \
  --stream connector:arcgis-prod-1 \
  --prior-hash '' \
  --start-sequence 1
```

For incremental bundles, the collector supplies its last accepted event hash and the next expected sequence. A mismatch is an incident: stop acceptance, preserve the bundle, compare source/exporter/collector logs, and escalate through the incident-response process.

## Key Rotation Procedure

| Step | Required evidence |
|---|---|
| Generate replacement key | KMS/HSM key creation record, owner, tenant/connector scope, and rotation ticket |
| Publish public key | Signed/versioned public-key registry release available to verifier before new signatures are accepted |
| Emit rotation audit event | Old key signs a `key.rotation.prepared` event; new key signs `key.rotation.activated` after activation where both paths are available |
| Switch active stream key | Controlled migration/change approval; stream anchor, old/new key IDs, timestamp, and operator identity logged |
| Retain old public key | Retain for at least the full signed-evidence verification retention period |
| Revoke compromised key | Publish effective revocation time; refuse future envelopes after that time; investigate/export all affected chain intervals |

## Required Tests and Monitoring

| Control | Acceptance evidence |
|---|---|
| Signature verification | Valid envelope accepted; changed payload, changed tenant, unknown key, invalid signature, expired envelope, and replay nonce rejected. |
| Chain verification | Valid multi-event export verifies; payload mutation, previous-hash alteration, dropped record, sequence change, and invalid key signature fail verification. |
| Permission test | Application runtime role cannot `UPDATE` or `DELETE` signed audit tables. |
| Export reconciliation | Collector batch count, first/last sequence, and last hash match exporter records. |
| Export delay alert | Alert when accepted collector anchor lags source stream by the defined interval. |
| Restore drill | Restored database export verifies against an independently held prior collector anchor. |
| Key rotation | Old and new public key registry verifies historical and rotated evidence. |

## Non-Repudiation Caveat

Ed25519 signatures authenticate possession of the signing key and protect the signed bytes from undetected alteration when key custody and verification records are trustworthy. They do not alone demonstrate a particular natural person’s legal intent. High-assurance or legally sensitive deployments must combine the implementation with identity proofing, MFA/step-up records, signer authorization policy, controlled key custody, trusted timestamps, retention/legal-hold controls, independent log custody, documented chain of custody, and legal/compliance review.

RFC 8032 defines Ed25519 as an EdDSA instance. NIST describes digital signatures as providing assurance about a claimed signatory and signed information, while audit-trail guidance notes signatures can protect against undetected modification but do not solve every deletion/modification problem. [1] [2] [3]

## References

[1] [RFC 8032: Edwards-Curve Digital Signature Algorithm](https://datatracker.ietf.org/doc/html/rfc8032)

[2] [NIST CSRC: Digital Signatures](https://csrc.nist.gov/projects/digital-signatures)

[3] [NIST SP 800-12, Chapter 18: Audit Trails](https://csrc.nist.gov/files/pubs/nistpubs/800-12-1/sp800-12-1.pdf)
