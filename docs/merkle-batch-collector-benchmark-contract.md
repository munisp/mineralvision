# Merkle Batch Collector and 5,000 Events/Second Benchmark Contract

**Prepared by Manus AI**  
**Date:** 22 August 2026

## Scope

This contract evaluates in-process partitioned Merkle batching and a reference off-host collector verifier. It deliberately does **not** represent PostgreSQL, KMS/HSM, mTLS, network, object storage, durable outbox, or production collector capacity. The 5,000 events/second target therefore means the batch-construction path can process a synthetic workload at or above that rate in the controlled environment; it is not a production throughput promise.

## Durable Batch Contract

A production batcher must durably store accepted event commitments before acknowledgement. A batch record contains the tenant, routing epoch, partition count and selected partition, first/last local sequence, event count, previous batch root, Merkle root, key ID, sealing time, and Ed25519 signature. An individual event has a canonical leaf commitment and a directed Merkle proof. A batch is immutable after sealing.

## Collector Contract

The collector treats `(tenant_id, routing_epoch, partition)` as an independent ordered chain. It has no global ordering across tenants or partitions. It verifies the signed batch commitment before admission, validates every supplied inclusion proof against the root, and stores the current accepted anchor per independent chain.

If the collector receives a valid batch that begins after its expected sequence or names a different previous root, it stores the batch as **pending** rather than rejecting or advancing the anchor. It drains pending batches whenever the missing predecessor arrives. A duplicate batch with the same root/range is idempotently accepted; a conflicting batch with the same chain/range but a different root is rejected as a fork. Cross-tenant events and proofs are rejected before state mutation.

## Benchmark Workload

The benchmark generates deterministic synthetic canonical event commitments across 8 partitions, with a 256-event maximum batch size. It measures creation of 5,000 events, per-event commitment hashing, partition routing, Merkle batch construction, Ed25519 batch signing, collector ingestion, and independent inclusion verification. It records duration, events/s, batches, batch-size distribution, event processing latency percentiles, collector anchor/pending behavior, and correctness results.

## Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Target rate | At least 5,000 synthetic events/s in the declared local scope. |
| Inclusion | Every generated event has a valid proof against its batch root. |
| Signature | Every batch commitment verifies using the declared public-key registry. |
| Tenant isolation | Cross-tenant batch/proof input is rejected with no anchor mutation. |
| Out-of-order delivery | Child batch remains pending until its predecessor arrives; chain then drains and advances. |
| Fork detection | Same chain/range with an alternate root is rejected. |
| Reporting | Report all failures, latency percentiles, environment limitations, and non-measured dependencies. |

RFC 9162 describes Merkle inclusion and consistency proof concepts for append-only transparency logs. This project uses the concepts as a scoped internal design and does not claim RFC 9162 protocol or wire-format compliance. [1]

## Reference

[1] [RFC 9162: Certificate Transparency Version 2.0](https://datatracker.ietf.org/doc/html/rfc9162)
