# Partitioned Merkle Audit and Staging Key-Compromise Drill Runbook

**Prepared by Manus AI**  
**Date:** 22 August 2026

## Deployment Checklist: Partitioned Merkle Audit

| Control | Deployment requirement | Verification evidence |
|---|---|---|
| Routing epoch | Record `routing_epoch`, immutable partition count, routing-key fields, and activation time in configuration/change record. | Same tenant/connector/entity scope deterministically maps to the same partition in the epoch. |
| Tenant isolation | Include tenant in every routing digest, batch, key policy, database query, and collector anchor. | Cross-tenant event input is rejected before buffering; integration test proves no shared partition evidence. |
| Durable intake | Persist accepted event commitments and stable event IDs before acknowledging any event that cannot be replayed upstream. | Crash/restart test seals every durable item exactly once or surfaces an explicit failure. |
| Batch policy | Configure maximum events and maximum age; begin with 256 events or one second, whichever occurs first, then validate with production-like evidence-rate tests. | Batch manifests show count/age compliance and no unsealed backlog beyond agreed threshold. |
| Signature | Sign each sealed batch root with an HSM/KMS-backed Ed25519 key and an immutable `key_id`. | Off-host verifier accepts current key and historic public keys; signature failure blocks export acceptance. |
| Inclusion proof | Store/provide event leaf commitment, directed sibling path, root, batch ID, routing epoch, and partition. | Independent verifier proves inclusion for sampled and adversarially selected events. |
| Collector continuity | Collector stores last accepted root and last sequence per `(tenant, routing_epoch, partition)` and rejects discontinuity. | Source/exporter/collector anchors reconcile in a scheduled check. |
| External action gate | No external write-back while signed-batch export lag, proof failure, or collector mismatch exceeds policy. | Fault injection blocks write-back and creates a security incident event. |
| Retention | Preserve public-key registry versions, manifests, signed roots, event bundles, and rotation records under independent retention/legal-hold policy. | Restore drill verifies retained evidence after source database recovery. |

## Staging Ed25519 Compromise-Rotation Drill

The automated drill uses **ephemeral in-memory keys**, a unique synthetic tenant, and a PostgreSQL staging database. It never contacts a production KMS/HSM, collector, connector, payment rail, customer source, or destination system. A real production drill must repeat the same sequence with the approved signing service, public-key registry, mTLS collector, identity workflow, and incident process.

### Required Environment

```bash
export ENV=staging
export MV_AUDIT_DRILL_CONFIRM=ROTATE_COMPROMISED_KEY
export MV_AUDIT_DRILL_DATABASE_URL='postgresql+psycopg2://<staging-drill-role>:<secret>@<staging-db>/mineralvision_staging'
# Optional only while investigating drill rows; default is cleanup.
export MV_AUDIT_DRILL_KEEP_DATA=false
```

### Automated Command

```bash
export PYTHONPATH="$PWD/MineralVision_Final_Package:$PYTHONPATH"
python scripts/run_staging_ed25519_key_compromise_drill.py | tee staging-key-compromise-drill.json
```

### Automated Assertions

| Drill step | Expected result | Escalate if failed |
|---|---|---|
| Old-key evidence accepted before simulated compromise | One old-key signed event appends. | Connector/key configuration or audit append failure. |
| Containment | Old key is removed from active acceptance registry and a new envelope signed with old key is rejected. | Key revocation policy is not enforced. |
| Rotation | Privileged job changes active stream key only after recording old anchor. | Unauthorized key change or missing anchor. |
| New-key activation | Rotation event and new-key evidence append from old anchor. | Chain continuity or replacement signer failure. |
| Historic verification | Full chain verifies using retained historical old and new public keys. | Evidence loss, public-key registry failure, or broken rotation. |
| Cleanup | Synthetic event/stream rows are removed unless explicitly retained. | Staging cleanup/retention policy failure. |

### Observed Local Staging-Labelled Result

The controlled run completed with result `passed`. It accepted one pre-compromise old-key event, rejected further old-key envelope acceptance after containment, rotated the stream to `staging-new-rotated`, appended the rotation/new-key evidence, and independently verified a three-event chain with a nonempty final hash. This demonstrates the code path; it is not evidence that a real KMS/HSM or off-host collector drill has completed.

## Production Scale Limits and Decision Rules

The existing strict per-stream chain benchmark remains the baseline for one globally ordered stream. The Merkle design is an architecture and tested primitive, not yet a full durable batching service. Do not claim a higher production throughput until a persistent batch worker, outbox, collector, KMS/HSM, partition epoch store, and end-to-end pre-production benchmark are deployed.

| Demand profile | Design choice | Evidence required before production use |
|---|---|---|
| Less than one strict-stream SLO | Use direct signed event chain. | Complete direct-chain capacity and recovery tests. |
| Many independent connector/project streams | Partition by versioned tenant/connector/entity routing key. | Cross-partition isolation, ordering, scale, and rebalancing/epoch transition tests. |
| Moderate high-rate evidence | Durable event commitment buffer plus signed Merkle batches. | Crash recovery, duplicate-event, inclusion proof, batch anchor, export, and collector discontinuity tests. |
| Very high-rate raw telemetry | Dedicated telemetry store plus signed summary/rollup batches. | Retention, raw-to-summary traceability, sampling/audit, and incident replay evidence. |

The Merkle primitives use domain-separated SHA-256 leaf/branch hashes and signed batch commitments. Their inclusion-proof design follows general transparency-log concepts, but does not implement or claim RFC 9162 wire compatibility or consistency-proof semantics. [1]

## References

[1] [RFC 9162: Certificate Transparency Version 2.0](https://datatracker.ietf.org/doc/html/rfc9162)
