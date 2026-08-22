# Compromised Ed25519 Signing-Key Rotation and Audit-Chain Recovery

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Applies to:** Integration connector envelope keys and tenant audit-stream signing keys.

## Incident Classification

Treat a suspected signing-key disclosure, unauthorized signing operation, unexplained public-key registry change, invalid signature from a trusted connector, or audit-chain verification failure as a **security incident**. The response must preserve evidence before modifying active systems. NIST key-management guidance describes the lifecycle controls needed for cryptographic keys, including planning for compromise and recovery. [1]

> **Immediate rule:** Do not delete the old public key, historic signed events, export batches, or verification failures. Retain them under legal-hold/retention policy so past evidence remains independently verifiable.

## First 60 Minutes: Containment

| Time | Action | Evidence retained | Owner |
|---|---|---|---|
| 0–10 min | Declare incident; freeze connector outbound writes and approval worker; set affected `key_id` state to **suspected compromised** in the key registry. | Incident ID, affected tenant/connector/stream, alert source, system clocks. | Security incident lead |
| 0–15 min | Disable signing permission in KMS/HSM or revoke the workload identity; do not delete the key. Block envelopes that claim the key after the declared compromise time. | KMS/HSM audit record, IAM change event, registry version. | Key custodian |
| 0–20 min | Snapshot signed-event tables, stream anchors, public-key registry, collector anchors, connector logs, OPA/OIDC decisions, and SIEM events to immutable case storage. | Hashes of snapshots, acquisition chain, collector receipts. | Forensics lead |
| 0–30 min | Force the connector into read-only evidence mode. Keep human review, but do not permit external write-back from affected tenant/connector. | Feature-flag/audit record, user/customer notification decision. | Integration lead |
| 0–60 min | Run independent off-host chain verification from the last trusted collector anchor to the latest exported bundle; identify earliest invalid/unknown signature or chain discontinuity. | Verifier JSON output, bundle hashes, expected/observed anchor. | Audit lead |

## Key Rotation Procedure

Rotation should be executed by an approved KMS/HSM or signing-service process. The code supports distinct `key_id` values and historic verification keys; it intentionally rejects new appends when the stream’s `active_key_id` differs. This prevents a silent key switch. A controlled migration/change workflow must rotate the stream anchor.

| Step | Required action | Success criterion |
|---|---|---|
| 1. Generate replacement | Generate a new Ed25519 key in HSM/KMS. Assign unique `key_id`, owner, tenant/connector scope, cryptoperiod, and state `pending`. | Private material never leaves the approved signing boundary; public key export is recorded. |
| 2. Publish verifier key | Add the new public key to the signed/versioned off-host verifier registry before accepting new signatures. Distribute registry through a separate trusted channel. | Off-host verifier recognizes both old and new `key_id`. |
| 3. Seal old interval | Export and verify the affected old-key stream interval from the collector’s last trusted anchor. Record last valid sequence and hash. | Verification result is preserved; any gap becomes an incident finding. |
| 4. Write dual-controlled rotation event | Under change approval, emit a `key.rotation.prepared` record signed by the old key if still safe, then a `key.rotation.activated` record signed by the new key. Include old/new IDs, effective time, stream anchor, approvers, and incident/change ID. | Rotation has an auditable continuity record; failure to sign with old key is explicitly recorded, not hidden. |
| 5. Update stream key | Transactionally set `hub_audit_streams.active_key_id` through a privileged migration/rotation job; the app role cannot change it. | First new-key event verifies from the sealed old anchor/rotation record. |
| 6. Revoke old key | Mark old key `compromised` with effective time, disable signing operation and connector credential, preserve public key for historic verification. | Future old-key envelopes are rejected; historic events still verify. |
| 7. Resume safely | Re-enable connector only after new-key envelope verification, audit-chain append, off-host export, OPA/OIDC checks, and destination write-back dry run pass. | Readiness checklist approved by security and integration owners. |

## Audit-Chain Recovery Decision Tree

| Verification outcome | Meaning | Required response |
|---|---|---|
| Valid chain; old key only used before compromise time | No cryptographic evidence of post-compromise misuse in verified range. | Rotate/revoke key, retain evidence, monitor heightened alerts, resume only after checklist passes. |
| Valid chain; old key signs events after compromise time | Potential unauthorized signing or clock/process-control failure. | Keep writes disabled; investigate signer/KMS logs and connector identity; treat all post-time events as untrusted until reconciled. |
| Signature mismatch | Event or signature was altered, wrong key registry used, or signer/key misuse occurred. | Preserve bundle, identify first failing event, compare database, exporter, and collector copy; do not repair history in place. |
| Previous-hash/sequence gap | Deletion, insertion, export loss, reordering, or collector-anchor mismatch is possible. | Stop acceptance; compare prior trusted anchor; export raw database interval and collector interval; create incident evidence bundle. |
| Collector unavailable but local chain valid | Off-host verification continuity is absent. | Keep source in read-only/high-risk mode; do not permit external write-back until collector recovers and backfilled export verifies. |
| Database loss with off-host valid bundle | Local operational history is lost but independently held evidence remains. | Restore database; import/reconcile as a new recovery operation; preserve bundle as source evidence and do not rewrite historic signatures. |

## Disaster Recovery Steps

1. Restore the PostgreSQL integration-hub database to an isolated recovery environment using the approved backup/PITR process.
2. Restore the exact public-key registry version(s) used for the recovery interval from the independent registry/collector.
3. Run `verify_offhost_audit_bundle.py` against each retained collector bundle in sequence, beginning at the last pre-incident trusted hash and expected sequence.
4. Compare recovered local event IDs, sequences, hashes, signatures, and stream anchors to off-host export records. Record every discrepancy; do not update existing signed rows to make them match.
5. Rebuild operational read models from verified evidence only. Keep any disputed post-compromise events marked quarantined.
6. Create a new stream or continue the existing verified stream only through an approved key-rotation event and trusted new key. Do not reuse a compromised key ID.
7. Run a full recovery drill: evidence registration, signed envelope verification, append, off-host export, independent verification, staged write-back dry run, and customer/incident communication exercise.

## Required Verification Commands

```bash
# Verify an incremental off-host bundle from the last collector anchor.
python scripts/verify_offhost_audit_bundle.py \
  --bundle /secure/collector/northstar-arcgis-000017.jsonl \
  --public-keys /secure/collector/audit-public-keys-2026-08-22.json \
  --tenant northstar-mining \
  --stream connector:arcgis-prod-1 \
  --prior-hash '<trusted_previous_hash>' \
  --start-sequence 1842

# Export a source interval only after read-only access and evidence preservation.
export ENV=production
export MV_HUB_DATABASE_URL='postgresql://mv_hub_audit_exporter:<secret>@postgres/mineralvision_hub'
python scripts/export_signed_audit_bundle.py \
  --tenant northstar-mining \
  --stream connector:arcgis-prod-1 \
  --after-sequence 1841 \
  --output /secure/outbox/recovery-interval.jsonl
```

## Closure Criteria

The incident can move from containment to closure only when the compromised signing ability is disabled, new key custody is independently confirmed, all affected intervals are verified or quarantined, collector and source anchors are reconciled, replay/expiry checks pass, connector dry-run succeeds, customer notification obligations are met, and corrective actions are tracked to closure.

## References

[1] [NIST SP 800-57 Part 1 Rev. 5: Recommendation for Key Management](https://csrc.nist.gov/pubs/sp/800/57/pt1/r5/final)

[2] [RFC 8032: Edwards-Curve Digital Signature Algorithm](https://datatracker.ietf.org/doc/html/rfc8032)
