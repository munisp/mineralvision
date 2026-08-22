# MineralVision P0/P1 Roadmap and Incumbent Integration Strategy

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Decision objective:** Establish MineralVision as a governed AI decision layer that augments existing geology, GIS, inspection, and environmental-response systems rather than requiring their replacement.

## Strategic Operating Model

MineralVision should operate as the **system of intelligence and evidence**, while incumbent applications remain the **system of record** for authoritative geological models, resource estimates, enterprise GIS layers, and regulated operational workflows. This prevents a disruptive rip-and-replace sale, reduces adoption risk, and focuses product investment on the differentiated problem: assembling field, geospatial, visual, model, and reviewer evidence into a traceable decision.

The first commercial wedge should be a narrow workflow: **exploration and environmental field intelligence**. A customer should be able to ingest drillhole/sample/imagery context, receive an explainable AI candidate or anomaly, review it in a map- and evidence-centric workspace, assign an action, and write an approved result back to the relevant incumbent system. Oil-spill response should remain an adjacent vertical until its satellite and incident-response evidence is externally validated.

> MineralVision must not position a raw model output as an authoritative geological interpretation, resource estimate, confirmed spill, source attribution, or financial action. Human approval and source-system governance remain integral to the product.

## P0 Priorities: 0–90 Days

P0 work is release-critical because it converts code breadth into one repeatable, supported workflow.

| P0 ID | Workstream | Exact scope | Exit evidence | Primary owner |
|---|---|---|---|---|
| P0-1 | Beachhead design partner program | Select one exploration/environmental workflow; recruit 3–5 design partners; document current state, data sources, decision owner, acceptance criteria, and value metric. | Signed design-partner plans and one controlled real-data pilot per partner. | Product and commercial |
| P0-2 | Canonical evidence contract | Define versioned schemas for `asset`, `site`, `sample`, `drillhole`, `observation`, `imagery_asset`, `model_run`, `candidate`, `review`, `decision`, and `source_reference`. Every record needs source system, immutable source ID, geometry/CRS, observed time, ingested time, producer, model version, and confidence/uncertainty. | OpenAPI/JSON Schema package, migration, validation suite, lineage query, and published compatibility policy. | Platform data |
| P0-3 | Review-first decision workspace | Productize the reviewer flow: evidence timeline, map, raw asset link, model explanation, uncertainty, decision/reason, assignment, export, and audit history. Implement responsive web/PWA before native expansion. | Usability test: a trained reviewer completes the selected workflow with fewer than five assisted steps and a full audit record. | Product engineering |
| P0-4 | Model evidence baseline | Create a model registry, data manifest, incident/project-disjoint split, calibration report, error taxonomy, reviewer-feedback capture, and model card template for WALDO and oil-spill workloads. | Reproducible benchmark report with frozen inputs, no fabricated accuracy claim, and an explicit reject/review threshold. | Applied ML |
| P0-5 | Operational release gate | Publish/activate CI workflows; require locked tests, security scanning, coverage ratchet, migration checks, OIDC/OPA regression tests, and an environment promotion record. | Main-branch required checks, release checklist, and a tagged staging deployment artifact. | Platform security |
| P0-6 | Incumbent connector minimum viable product | Deliver read-only Seequent/Evo metadata/reference ingestion and read/query-only ArcGIS Feature Service connector before any write-back feature. | A demo that traces one source object from the incumbent system through MineralVision evidence and back to a source hyperlink. | Integrations |
| P0-7 | Trust operations | Exercise OIDC/OPA, SIEM shipping, backup restore, incident runbook, model rollback, and customer-support escalation. | Logged tabletop exercise, PostgreSQL restore evidence, SIEM event correlation, and named on-call owners. | Operations |

### P0 Non-Negotiable Product Rules

MineralVision must not overwrite a source geological model, resource estimate, or GIS layer in the first release. It should generate a versioned **candidate layer** or **review package**, preserve the source object reference, and require human acceptance in the customer’s approved workflow. Every connector runs with a least-privilege service identity, customer-specific tenancy boundary, scoped credential, and revocable token.

## P1 Priorities: Months 4–9

P1 work deepens workflow value after the evidence layer works reliably.

| P1 ID | Workstream | Exact scope | Exit evidence |
|---|---|---|---|
| P1-1 | ArcGIS managed write-back | Add capability discovery, schema mapping, dry run, reviewer approval, idempotency key, transaction result capture, and rollback/compensating-action procedure for approved candidate outputs. | One non-production Feature Service write-back using an approved service account; output `objectId`/`globalId` stored in MineralVision lineage. |
| P1-2 | Seequent/Evo object integration | Use Seequent Evo’s object-level API for supported metadata/object-reference synchronization. Link decisions to object UUID/path; publish machine-readable review metadata rather than editing proprietary model artifacts directly. | Demo with UUID/path traceability and a customer-approved data-governance review. |
| P1-3 | Geochemistry workspace | Add assay QC, detection limits, units, standards/blanks/duplicates, compositional transformations, reproducible multivariate analysis, drillhole association, and map/section outputs. | Domain-geologist acceptance test against a representative data package. |
| P1-4 | Visual inspection operations | Add capture metadata, upload resilience, annotation queue, model-review loop, work orders, ticket export, dashboard templates, and partner import for orthomosaic/point-cloud/mesh outputs. | End-to-end inspection action completed from capture to closed ticket. |
| P1-5 | Oil-spill connector and evidence vertical | Ingest SAR/optical/AIS/weather/current data through licensed or public connectors; record acquisition latency and licensing; add source-association review and authority-facing evidence export. | Incident simulation with source confidence clearly labelled as potential, reviewed, or confirmed by the designated authority. |
| P1-6 | Interoperability SDK | Release SDK/API, webhook model, connector test harness, example ArcGIS/Seequent/QGIS/Pix4D adapters, and tenant connector administration. | External integration partner completes a sandbox connector without core-code modification. |

## P1 Priorities: Months 10–12

The final quarter makes the workflow repeatable and commercially supportable.

| P1 ID | Workstream | Exact scope | Exit evidence |
|---|---|---|---|
| P1-7 | Multi-tenant product operations | Tenant administration, retention policies, export controls, support playbooks, usage telemetry, cost attribution, and operational SLO dashboards. | Two paying/pilot tenants operate without bespoke database or code changes. |
| P1-8 | Deployment reference architecture | Production Helm/Compose/Kubernetes paths, key rotation, HA database design, backup/PITR automation, SIEM dashboards, external penetration test remediation, and disaster-recovery exercise. | Independent security assessment and a documented recovery drill meeting agreed RTO/RPO. |
| P1-9 | Commercial packaging | Define named modules, permissions, pricing metric, implementation package, support tier, training, and customer success plan. Exclude financial transfer functionality from commercial product claims until it has its own regulated delivery program. | Sales enablement pack, reference scope of work, and repeatable implementation estimate. |
| P1-10 | Evidence-led expansion decision | Compare design-partner outcomes to baseline; choose whether to scale exploration intelligence or invest further in oil-spill response. | Product council decision based on measured workflow, retention, and model-evidence metrics. |

## Exact 12-Month Sequence

| Month | Product and delivery milestone | Integration milestone | Decision gate |
|---|---|---|---|
| 1 | Select beachhead, design partners, and workflow scorecard. | Obtain sandbox access and data-governance requirements from one Seequent/Evo and one ArcGIS customer. | Reject projects without a named decision owner or usable source data. |
| 2 | Publish canonical evidence schema and source-lineage model. | Build connector capability-discovery adapters; ArcGIS read/query and Seequent object-reference proof of concept. | Architecture review: source system remains authoritative. |
| 3 | Deliver reviewer workspace MVP and immutable decision audit. | Demo one source object → MineralVision candidate → reviewer decision → source hyperlink. | Design partners approve the MVP workflow. |
| 4 | Freeze model-card and benchmark process. | Begin ArcGIS candidate-layer export in sandbox, but no automatic write-back. | No model claim without frozen external holdout evidence. |
| 5 | Add reviewer feedback loop and operational alerting. | Implement ArcGIS schema mapping, dry run, idempotency, and result capture. | Security review of connector identity and scopes. |
| 6 | Pilot workflow release candidate. | Enable supervised ArcGIS write-back for a non-production service. | Pilot success: traceability, user completion, and no unexplained data changes. |
| 7 | Build geochemistry QC and reproducibility workspace. | Implement Seequent/Evo object-reference synchronization for permitted object types. | Domain-geologist acceptance review. |
| 8 | Add visual inspection work management and model review. | Prototype Pix4D/DroneDeploy/Optelos import adapter via published or customer-approved exports/APIs. | Integration partner validates mapping and provenance. |
| 9 | Build oil-spill evidence connector and review workflow. | Add licensed/public SAR, AIS, and environmental-data adapters with lineage. | Environmental SME validates uncertainty wording and evidence bundle. |
| 10 | Harden multi-tenant administration and support operations. | Deliver connector SDK, webhooks, and sandbox documentation. | Independent developer completes a sandbox integration. |
| 11 | Run security, SIEM, PITR, and incident-response exercises. | Test connector failure modes, token revocation, schema change, and source outage handling. | Production readiness council verifies RTO/RPO and security closure. |
| 12 | Package the proven vertical; decide expansion investment. | Publish supported integration matrix and versioning/deprecation policy. | Convert pilots only if measured customer value and support readiness meet targets. |

## Seequent Integration Architecture

Seequent Evo documents a Geoscience Object API that supports object-level integrations using UUIDs or user-defined object paths. [1] MineralVision should use this as the primary identity bridge when an organisation has Evo access.

```text
Seequent / Evo                         MineralVision                         Reviewer
──────────────                         ─────────────                         ────────
Object UUID, path, metadata ───────► Connector / source registry
                                      │
                                      ├─► canonical evidence record
                                      ├─► AI candidate + model-run lineage
                                      └─► review package ───────────────► accept / reject / annotate
                                                                         │
Reference link + approved metadata ◄─ approved outcome / source reference ◄─┘
```

### Seequent Connector Rules

| Layer | Initial implementation | Expansion only after governance approval |
|---|---|---|
| Identity | Store Evo object UUID/path and customer tenant ID. | Customer-specific OAuth/client credential lifecycle and connector admin console. |
| Ingestion | Read permitted metadata and supported object references; preserve a version/checkpoint. | Incremental sync/webhook-like polling if supported by the customer’s Evo contract. |
| Interpretation | Keep MineralVision AI output as a separate candidate/evidence object. | Publish approved annotations/metadata through supported APIs. |
| Model artifacts | Link to, do not mutate, Leapfrog/Evo model artifacts. | Customer-approved workflow that imports a reviewed output using native Seequent processes. |
| Failure handling | Record source timestamp, connector version, scope, retry status, and immutable error record. | Automated reconnect only with idempotency and customer-visible status. |

The commercial message is: **“MineralVision helps teams identify, rank, explain, and govern candidates around their existing geological objects; geologists remain in their established modelling environment for authoritative interpretation.”**

## ArcGIS Enterprise Integration Architecture

ArcGIS Feature Services expose query, editing, synchronization, upload, and related capabilities depending on service configuration. [2] MineralVision should first inspect the service metadata and use only the capabilities explicitly advertised. Feature edits should be a separate approval-controlled operation because ArcGIS `addFeatures` is a POST editing operation that returns per-feature success/error results. [3]

```text
ArcGIS Enterprise Feature Service               MineralVision
─────────────────────────────────               ─────────────
Service metadata / capabilities ─────────────► connector capability registry
Query layers / attachments ──────────────────► canonical evidence + geometry normalizer
                                                │
                                                ├─► model candidate layer (read-only in MV)
                                                └─► review and approval
                                                             │
Approved candidate + schema validation ──────► dry-run report
                                                             │
             addFeatures / applyEdits POST ◄── staged write-back worker
                                                             │
objectId/globalId/result/error ─────────────► immutable lineage and audit event
```

### ArcGIS Connector Rules

1. **Discover before use.** Query service capabilities, layer schema, spatial reference, edit permissions, pagination limits, and sync support before an import or write-back. Do not assume write capability.
2. **Read first.** The first production release is read/query only. The connector imports a bounded geographic/time window and stores source URL, layer ID, object/global ID, service revision, attachment reference, and CRS.
3. **Candidate layers are separate.** MineralVision outputs appear in a separate candidate layer or external layer reference, never as an overwrite of a customer’s authoritative source layer.
4. **Human approval is mandatory.** A reviewer approves a staged edit after seeing geometry diff, attribute diff, source/model lineage, uncertainty, and destination layer.
5. **Write with idempotency and result capture.** Use a stable integration ID as a destination attribute/global-ID mapping when supported. Record request hash, response, returned `objectId`/`globalId`, service URL, actor, and timestamp. Treat partial `addFeatures` failure as an explicit exception requiring review.
6. **Use least privilege.** Use customer-managed OAuth/client credentials or service accounts restricted to the required services/layers and no administrative Portal scope.
7. **Handle failure visibly.** Detect token expiry, schema changes, ArcGIS service errors, geometry/CRS mismatch, record limit paging, and conflict responses; show connector health and do not silently retry writes without idempotency.

## Integration Security Baseline

| Control | Requirement |
|---|---|
| Tenant boundary | A connector credential, source registry entry, storage namespace, and audit stream are scoped to exactly one customer tenant. |
| Credential handling | Store secrets only in a customer-approved secret manager; display nonsecret identifiers and scope metadata, never access tokens. |
| Authorization | OIDC validates the user; OPA validates the action; source-system role/capability limits the connector. All three must allow a sensitive write. |
| Provenance | Persist source system, canonical source reference, data timestamp, ingestion timestamp, transformation version, model run, reviewer, decision, and destination result. |
| Data minimization | Import only fields, attachments, area, and time window required for the selected workflow. |
| Observability | Emit connector lifecycle, source error, retry, approval, write-back result, and revocation events to the tenant-aware audit/SIEM path. |
| Change management | Schema changes and connector upgrades require sandbox test, compatibility review, rollback path, and release note. |

## Operating Metrics

The roadmap should be governed by measurable evidence, not feature count.

| Measure | 90-day target | 6-month target | 12-month target |
|---|---:|---:|---:|
| Design partners with usable real-data workflow | 3 | 3 active pilots | 2 paid/repeatable deployments |
| Reviewed decisions with complete lineage | 95% | 99% | 99.5% |
| Connector failures with visible status and retry outcome | 100% | 100% | 100% |
| Human-reviewed model outputs with stored confidence/uncertainty | 100% | 100% | 100% |
| Source-system write-backs without untracked manual remediation | N/A | 95% in sandbox | 99% in supported production workflows |
| Model performance claim | None without benchmark | Published external holdout result | Domain-specific monitored benchmark and drift process |
| Security operational gates exercised | CI and staging | SIEM/PITR/incident exercise | Independent assessment and recovery objective evidence |

## References

[1] [Seequent Geoscience Object API](https://developer.seequent.com/docs/api/geoscience-object/geoscience-object-api)

[2] [ArcGIS Enterprise Feature Service REST API](https://developers.arcgis.com/rest/services-reference/enterprise/feature-service/)

[3] [ArcGIS Enterprise Add Features REST API](https://developers.arcgis.com/rest/services-reference/enterprise/add-features/)
