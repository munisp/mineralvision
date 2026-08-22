# Partitioned and Merkle-Batched Audit Architecture

**Prepared by Manus AI**  
**Date:** 22 August 2026

## Objective

The current signed audit stream locks one PostgreSQL anchor row per tenant/stream on every event. This retains strict sequence integrity but creates a tail-latency bottleneck under high contention. The scalable design preserves verifiability while introducing two levels of ordering:

1. **Strict local order** within a stable audit partition; and
2. **Signed Merkle batch commitments** across events in that partition.

This architecture lets unrelated connector/project/entity streams append concurrently. It does not claim a single total order across every tenant event; consumers requiring a global order must use an explicitly designated global stream and accept its throughput limits.

## Stable Partition Routing

Each event receives a deterministic partition key derived from tenant, connector, and an approved business scope such as project, site, asset, or workflow. The selected partition is:

```text
partition = SHA-256(tenant_id || connector_id || entity_scope) mod partition_count
stream_id = "partition:<partition_count>:<partition>"
```

`partition_count` is a versioned immutable configuration for the routing epoch. It must not be changed in place because doing so changes the partition selected for the same input. Scale-out creates a new routing epoch, publishes a signed transition record, and preserves verification of old partitions. Tenant IDs must always be part of the routing key; no partition record may combine events from different tenants.

| Ordering need | Recommended stream key | Scaling behavior |
|---|---|---|
| Tenant-wide legal/security action | One tenant global stream | Strict order; lower throughput |
| Connector/project evidence | Tenant + connector + project/entity partition | Parallel append across independent projects/connectors |
| High-rate sensor telemetry | External telemetry system; signed summary batches only | Very high rate; raw telemetry stays outside governance chain |
| External write-back | Tenant + connector + destination object/project | Partitioned; one target can retain strict order |

## Merkle Batch Construction

A partition appends event commitments to an in-memory or durable batch buffer. At a short interval or maximum count, the batch is sealed:

1. Canonically hash each event commitment with a leaf domain prefix `0x00`.
2. Build a binary Merkle tree using branch prefix `0x01` and SHA-256.
3. Persist each event leaf hash and its batch ID.
4. Persist one batch record containing tenant, routing epoch, partition ID, first/last local sequence, event count, Merkle root, prior batch root, key ID, and Ed25519 signature.
5. Export the signed batch root and event inclusion material to the off-host collector.

The primitives use explicit domain separation and directed inclusion proof entries. They are compatible with the core inclusion/append-only ideas standardized for transparency logs, but this implementation is **not an RFC 9162 transparency-service implementation** and does not claim its wire format or consistency-proof semantics. RFC 9162 describes Merkle inclusion and consistency proofs as tools for verifying log inclusion and append-only changes. [1]

## Verification Model

| Evidence | Verifier checks |
|---|---|
| Event inclusion | Recomputes canonical event commitment leaf, follows directed sibling proof, and obtains the signed batch root. |
| Batch signature | Resolves batch `key_id` in the public-key registry and verifies the Ed25519 batch commitment. |
| Partition continuity | Checks each batch `previous_batch_root`, sequence range, routing epoch, partition ID, and batch order. |
| Off-host continuity | Collector stores last accepted root per `(tenant, routing_epoch, partition)` and rejects discontinuity. |
| Cross-partition action | Correlates immutable action ID and signed references from every participating partition; does not presume temporal total order. |

## Batch Policy Defaults

The provided primitive uses configuration supplied by the caller. A production policy should begin conservatively with a maximum of 256 events or 1 second per batch, whichever occurs first, and validate the values against actual audit obligations. Lower intervals reduce time-to-evidence but increase signing/export load. Higher limits improve throughput but lengthen the evidence window and increase loss/recovery work if a buffer process fails before sealing.

The buffering service must write accepted event commitments to durable storage before acknowledging an event if the event cannot be reconstructed from another authoritative source. A memory-only batch buffer is appropriate only when upstream connector delivery is durable and replayable.

## Scale and Failure Rules

| Condition | Required behavior |
|---|---|
| Batch signer unavailable | Stop sealing and surface a visible back-pressure/error. Do not acknowledge unrecoverable events as audited. |
| Database leader/failover | Replay only durable unsealed commitments with stable event IDs; detect duplicate IDs before sealing. |
| Export collector unavailable | Keep bounded durable outbox; alert on export lag; stop external write-back when lag exceeds policy. |
| Partition saturation | Add routing partitions only through a new epoch; do not change a live modulus. |
| Cross-tenant input | Reject before routing, persistence, hashing, or worker queueing. |
| Batch proof mismatch | Preserve evidence, quarantine affected batch, block related write-back, and invoke audit incident response. |

## References

[1] [RFC 9162: Certificate Transparency Version 2.0](https://datatracker.ietf.org/doc/html/rfc9162)
