# Signed Audit Chain and Connector-Envelope Trust Model

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Scope:** Tenant-scoped, tamper-evident audit evidence and signed connector messages for high-assurance MineralVision integrations.

## Security Objective

The implementation provides three distinct assurances. First, it creates a chronological audit chain whose `previous_hash` and canonical event hash make insertion, payload alteration, reordering, and truncation detectable when an external verifier has a trusted anchor. Second, it signs every audit-event commitment and connector envelope with an Ed25519 private key and identifies the public key through a `key_id`. Third, it exports signed JSON Lines evidence that an off-host verifier can inspect without database credentials or private keys.

> **Boundary:** These controls establish cryptographic integrity and origin authentication for the configured signing key. They do not, by themselves, establish legal non-repudiation, prove a human’s intent, prevent a privileged operator from deleting every local copy, or replace identity proofing, key custody, retention governance, independent timestamping, and contractual/legal review.

## Algorithms and Canonicalization

The implementation uses SHA-256 to construct content commitments and Ed25519 digital signatures as specified in RFC 8032. [1] Canonical payloads are deterministic UTF-8 JSON generated with sorted object keys and compact separators. A signed value always contains an explicit schema version, tenant ID, key ID, creation time, sequence, previous hash where applicable, and payload/hash reference.

| Object | SHA-256 hash scope | Ed25519 signature scope |
|---|---|---|
| Audit event | Canonical event fields, including stream ID, sequence, previous hash, event type, actor, payload, and timestamp | Canonical commitment containing tenant/stream/sequence/event hash/previous hash/key ID/timestamp |
| Connector envelope | Canonical envelope fields, including tenant, connector ID, nonce, issued time, event type, payload hash, and payload | Entire canonical envelope body |
| Off-host bundle | One JSON record per signed audit event | Existing event signature is retained; verifier recomputes event hash and validates key-specific signature |

## Key Lifecycle

| Stage | Required practice |
|---|---|
| Generate | Generate Ed25519 keys in a KMS/HSM or an approved secret-management system. The application process must receive signing capability through workload identity or a short-lived brokered credential, not a permanently stored plaintext private key. |
| Identify | Assign an immutable `key_id` and publish the raw 32-byte Ed25519 public key through a protected verification-key registry. |
| Activate | Bind a connector or audit stream to an active `key_id`; record activation time, owner, tenant, and approved usage scope. |
| Rotate | Add a new active key, emit a signed rotation event under both old and new key where possible, keep prior public keys for verification, and never reuse a `key_id`. |
| Revoke | Mark compromised keys revoked with an effective timestamp; reject new envelopes after the revocation time while retaining historic verification evidence. |
| Destroy | Destroy private key material only under retention/legal-hold policy. Retain public verification keys and signed evidence for the full audit-retention period. |

## Off-Host Verification Model

The primary application database is not the verifier of record. An export job sends newline-delimited signed events, a manifest, and public-key material to an independent log store with write-once/retention controls. The receiving system validates event signatures and chain continuity before accepting an export batch, stores the last accepted stream anchor, and alerts when the next batch does not start at the expected sequence/hash.

A production deployment should use mutually authenticated TLS, an allow-listed collector endpoint, outbound retry/idempotency semantics, independent retention controls, access logging, and monitoring of export delay. Periodic external anchoring of the last event hash to an independently controlled timestamping or ledger system further strengthens deletion/truncation detection, but is not required by this code module.

## Required Operational Controls

The code must be paired with identity proofing, reviewer-role governance, OIDC/MFA, OPA policy enforcement, SIEM correlation, access review, key-custody procedures, a tested recovery path, and an independently operated export collector. NIST notes that audit logs are chronological records of activity and that digital signatures can help protect audit trails from undetected modification; it also notes that signatures alone do not prevent all deletion or modification scenarios. [2] [3]

## References

[1] [RFC 8032: Edwards-Curve Digital Signature Algorithm](https://datatracker.ietf.org/doc/html/rfc8032)

[2] [NIST CSRC: Digital Signatures](https://csrc.nist.gov/projects/digital-signatures)

[3] [NIST SP 800-12, Chapter 18: Audit Trails](https://csrc.nist.gov/files/pubs/nistpubs/800-12-1/sp800-12-1.pdf)
