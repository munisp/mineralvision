# Signed Audit Benchmark Visual Findings

Two PostgreSQL single-stream contention benchmark charts were reviewed on 22 August 2026.

The initial 200-event run showed throughput between 310.627 and 362.881 events/second across 1, 4, 16, and 32 workers, while p95 latency increased from 3.118 ms at one worker to 330.854 ms at 32 workers.

The extended 500-event run showed throughput of 306.592, 420.896, 383.477, and 394.448 events/second at 1, 8, 32, and 64 workers respectively. It showed p95 latency of 3.306 ms, 46.662 ms, 418.995 ms, and 816.582 ms respectively. All 2,000 attempted events appended successfully across the extended run, and each 500-event chain verified with zero failures.

The charts visibly demonstrate the expected trade-off for a single append-only stream protected by a PostgreSQL row lock: throughput plateaus around 383–421 events/second, while tail latency grows materially after 8 workers. The figures exclude KMS/HSM signing latency, off-host transport/export latency, replica/network variation, and production workload interference; they are not a production capacity guarantee.
