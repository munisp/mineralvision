# MineralVision P0 Graduation, Resource, and Budget Plan

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Planning status:** Operating-plan estimate, not a financial commitment or a substitute for procurement, payroll, legal, security, regulatory, or customer-contract review.

## Decision Standard

A 90-day P0 pilot should graduate to commercial deployment only when it demonstrates a repeatable, governed workflow with real partner data—not merely a functioning demo or a successful model run. Graduation requires both a minimum score and satisfaction of non-negotiable safety gates. A pilot that misses a safety gate does not graduate, even if its weighted score is high.

The recommended initial offer is a **decision and evidence layer** for a narrowly defined exploration or environmental field-intelligence workflow. The source geological/GIS platform remains the system of record. MineralVision owns candidate generation, evidence assembly, human review, decision audit, and controlled connector hand-off.

## P0 Graduation Scorecard

| Domain | Weight | Graduation success criteria | Measurement method | Non-negotiable gate |
|---|---:|---|---|---|
| Workflow and adoption | 20 | At least 3 design partners complete one defined workflow using production-like data; at least 80% of completed reviews are completed without engineering intervention; median reviewer completion time is at least 30% lower than the documented baseline or produces a clearly accepted evidence-quality improvement. | Time-stamped workflow telemetry, partner acceptance review, interview notes, and baseline comparison. | Each partner has a named operational sponsor and accepts the workflow evidence record. |
| Data lineage and integration | 20 | At least 99% of reviewed candidates contain source system, source ID, geometry/CRS, source observation time, ingest time, connector version, model version, reviewer, decision, and destination reference where applicable; 100% of connector write attempts return a recorded outcome. | Automated lineage completeness query; connector reconciliation report. | No untracked write-back, silent source overwrite, cross-tenant read, or cross-tenant write. |
| Model evidence and human governance | 20 | 100% of displayed model outputs link to a registered model and input manifest; external/holdout review set contains at least 100 independently reviewed observations in every intended pilot domain; confidence calibration is reported; every high-impact output receives human review before write-back. | Model registry, split manifest, calibration report, reviewer audit. | No unsupported accuracy, spill-confirmation, or source-attribution claim; no automatic action from model output. |
| Reliability and security | 20 | Three consecutive staging releases pass required tests; P0 critical-path coverage is at least 85%; no unresolved Critical/High severity code findings; p95 metadata/review API latency is below 1.5 seconds under agreed pilot load; authenticated evidence requests succeed at least 99.5% of the measured service window, excluding agreed upstream outages. | Locked CI results, vulnerability triage, synthetic monitoring, load test, and incident log. | OIDC/OPA authorization, tenant isolation, audit integrity, backup restore, SIEM correlation, and incident-response exercise pass. |
| Commercial repeatability | 20 | Standard implementation checklist completes within 4 weeks for a typical design partner; implementation scope is fixed; a support owner and escalation path exist; at least two partners declare willingness to continue under a paid or paid-pilot proposal. | Implementation project record, support runbook, signed intent/renewal evidence. | No bespoke production database changes or unmanaged credential sharing per customer. |

A pilot may graduate only with **at least 85/100 weighted points**, a score of at least 15/20 in each domain, and every non-negotiable gate satisfied. The product council should record the decision, known deviations, owners, expiry dates, and a rollback plan.

## Exact Technical Readiness Gates

### Data and Connector Gates

The first commercial release should support only the approved connector scope. For Seequent/Evo, that means source object reference/metadata ingestion, object UUID or path linkage, and approved review metadata rather than mutation of authoritative geological artefacts. Seequent documents object-level API integration through shared geoscience object structures and UUID/object-path references. [1]

For ArcGIS Enterprise, the connector must query advertised Feature Service capabilities before use; it must begin in read/query mode; and any approved write must use a dedicated candidate layer, human approval, idempotency, schema validation, destination result capture, and a compensating-action procedure. ArcGIS Feature Services expose capability metadata and editing/synchronization operations, so capability discovery—not a hard-coded assumption—must govern the connector. [2]

| Check | Threshold | Evidence required |
|---|---:|---|
| Source-to-candidate lineage completeness | ≥99% | Daily query and exception report |
| Connector write reconciliation | 100% of attempts | Request hash, actor, destination service/layer, returned object/global ID or error, and review reference |
| Schema/CRS validation before write | 100% | Preflight result stored with write request |
| Duplicate destination write rate | 0 unreviewed duplicates | Idempotency/reconciliation report |
| Tenant access-control tests | 100% pass | Locked CI, staging synthetic tenant test, and access-log review |
| Connector secret rotation test | At least once before graduation | Documented rotation and revocation evidence |

### ML and Decision Gates

The P0 pilot must distinguish model quality from product usefulness. A model may help reviewers only when it carries a model version, data manifest, confidence/uncertainty, and a review state. For each intended domain, the evaluation set must be isolated from training by project, incident, or site as appropriate.

| Check | Threshold | Evidence required |
|---|---:|---|
| Model-run provenance | 100% | Model ID, artifact hash, input manifest, preprocessing version, and timestamp |
| Human review on high-impact candidate | 100% | Reviewer identity, role, MFA state, outcome, and rationale |
| Per-domain external holdout | ≥100 independently reviewed observations per intended domain | Frozen manifest and dataset/version hashes |
| Calibration | Reported for each production candidate model | Reliability curve, expected calibration error, threshold policy |
| Drift monitoring | Measured at every data refresh/release | Data distribution and outcome-drift report |
| Rollback | Tested once before graduation | Prior approved model restored in staging and shown in audit log |

For oil-spill work, internal promotion controls should remain strict: oil-class F1, IoU, precision, and recall thresholds must be evaluated on sealed, incident-disjoint data. Until that evidence is independently reviewed, outputs must remain candidate assessments, not confirmed spills or source attributions.

### Reliability, Security, and Operations Gates

| Check | Target | Evidence required |
|---|---:|---|
| Locked test gate | Pass for 3 consecutive staging releases | CI build IDs and dependency lock hashes |
| Critical-path coverage | ≥85% | Coverage report covering authorization, financial controls, oil-spill review, connector, persistence, and WALDO boundary modules |
| Dependency/code vulnerability release policy | 0 unresolved Critical/High findings; Medium findings have owner/date/compensating control | Security review and exception register |
| Restore point objective (RPO) | ≤1 hour for the pilot database | WAL archive/backup evidence and restore timestamp |
| Recovery time objective (RTO) | ≤4 hours for the pilot service | Timed restore and application verification drill |
| SIEM coverage | 100% of OIDC, OPA, connector, API, WAF, and database audit event categories delivered | Correlation dashboard and missing-event test |
| Incident exercise | One tabletop plus one technical drill | Timeline, actions, and remediation record |

## Base-Case Engineering Organization

The base case assumes a North American/Western Europe blended delivery model, a narrow initial vertical, incumbent integration rather than replacement, and three to five design partners. It does not assume a regulated funds product, a proprietary satellite constellation, or custom hardware deployment.

| Function | Months 1–3 FTE | Months 4–6 FTE | Months 7–9 FTE | Months 10–12 FTE | Core accountabilities |
|---|---:|---:|---:|---:|---|
| Product leadership and customer discovery | 1.0 | 1.0 | 1.0 | 1.0 | Beachhead, design partners, scope control, packaging |
| Engineering leadership / architecture | 1.0 | 1.0 | 1.0 | 1.0 | Architecture, technical roadmap, release decisions |
| Backend and platform engineering | 2.0 | 3.0 | 3.0 | 3.0 | Evidence API, connectors, data contracts, audit, operations |
| Frontend / PWA engineering | 1.0 | 1.5 | 2.0 | 2.0 | Reviewer workspace, connector admin, field UX |
| Geospatial and data engineering | 1.0 | 1.5 | 2.0 | 2.0 | PostGIS, imagery, CRS, pipelines, data quality |
| Applied ML / MLOps | 1.0 | 1.5 | 2.0 | 2.0 | Model registry, benchmarks, calibration, drift, WALDO/oil work |
| Integrations engineering | 1.0 | 1.5 | 2.0 | 2.0 | Seequent/Evo, ArcGIS, SDK, connector reliability |
| Quality engineering | 1.0 | 1.5 | 2.0 | 2.0 | Test automation, performance, release evidence |
| Platform security / SRE | 0.75 | 1.0 | 1.5 | 1.5 | OIDC/OPA, CI, SIEM, backup, reliability, incident readiness |
| UX research and design | 0.5 | 1.0 | 1.0 | 1.0 | Workflow research, usability, information design |
| Geology/environment domain expertise | 0.5 | 1.0 | 1.0 | 1.0 | Acceptance criteria, data interpretation, trust language |
| Customer success / implementation | 0.5 | 1.0 | 1.5 | 2.0 | Pilot delivery, training, support, implementation repeatability |
| **Total** | **11.25** | **15.5** | **19.0** | **20.5** | **Base case** |

The initial team can be smaller only if the scope is narrowed further. The first roles that should not be deferred are product leadership, backend/platform, QA, geospatial/data, applied ML, integration engineering, and a real domain expert. A single generalist cannot safely substitute for these responsibilities in a mission-critical pilot.

## Budget Scenarios

The following are internal planning ranges in U.S. dollars. They are not vendor quotes, compensation offers, or forecasts. The ranges use a fully loaded employee/contractor model, including benefits, employer taxes, equipment, recruiting, and limited specialist support. As a public reference point, the U.S. Bureau of Labor Statistics lists a May 2024 median annual wage of $133,080 for software developers and $102,610 for software QA analysts/testers; senior geospatial, ML, security, product, domain, and integration specialists can exceed those medians materially. [3]

| Scenario | Average FTE over 12 months | Labour budget | Non-labour budget | Total 12-month planning range | Suitable use |
|---|---:|---:|---:|---:|---|
| Lean pilot | 8–10 | $1.7M–$2.4M | $0.45M–$0.85M | **$2.15M–$3.25M** | One narrow workflow, three partners, read-only incumbent connectors, no oil-spill data licensing build-out |
| Base commercial-readiness plan | 14–17 | $3.2M–$4.6M | $0.9M–$1.7M | **$4.1M–$6.3M** | The roadmap described here: design partners, reviewer UX, Seequent/ArcGIS integrations, model evidence, staged write-back, operational gates |
| Accelerated dual-vertical plan | 19–24 | $4.7M–$6.7M | $1.8M–$3.5M | **$6.5M–$10.2M** | Parallel exploration and oil-spill verticals, licensed data feeds, expanded ML operations, enterprise support, and independent assessments |

Non-labour estimates include cloud/storage/observability, security tooling and independent testing, customer onboarding/travel, geospatial or imagery processing, training, legal/compliance review, and optional data/partner licensing. They intentionally exclude acquisition costs, customer-specific hardware, actual payment-network costs, and potentially material commercial SAR/AIS/environmental data subscriptions.

### Base-Case Quarterly Allocation

| Quarter | Budget range | Primary use | Decision checkpoint |
|---|---:|---|---|
| Q1 | $0.85M–$1.20M | Team formation, partner discovery, canonical evidence model, PWA reviewer MVP, read-only connectors, CI/security baseline | Continue only if two partners provide usable data and agree on a decision workflow |
| Q2 | $0.95M–$1.40M | Model evidence, ArcGIS sandbox candidate workflow, staging operations, usability validation, connector hardening | Enable supervised write-back only if lineage/security/model gates pass |
| Q3 | $1.10M–$1.70M | Seequent/Evo object linking, geochemistry/inspection workflow depth, oil-spill evidence prototype, SDK and support model | Expand only if pilot usage and support burden meet target |
| Q4 | $1.20M–$2.00M | Multi-tenant operations, deployment reference architecture, security assessment, disaster-recovery drill, product packaging | Commercial release only if graduation scorecard and independent readiness gates pass |
| **Total** | **$4.10M–$6.30M** | **Base commercial-readiness plan** | **No automatic production release** |

## Budget Guardrails

The program should release budget in tranches. Q2 investment should require partner data, an accepted workflow definition, and a functioning reviewer MVP. Q3 investment should require successful model-evidence review, complete source lineage, and no unresolved P0 security issue. Q4 production-readiness spending should require a written commercial sponsor decision, recovery evidence, and external security assessment scope.

Financial-transfer code should remain isolated from the commercial roadmap. It is a security-control experiment and must not consume product budget as a payments offering until a separate licensed-partner, KYC/AML/sanctions, reconciliation, HSM/KMS, audit, and regulatory program is funded and approved.

## References

[1] [Seequent Geoscience Object API](https://developer.seequent.com/docs/api/geoscience-object/geoscience-object-api)

[2] [ArcGIS Enterprise Feature Service REST API](https://developers.arcgis.com/rest/services-reference/enterprise/feature-service/)

[3] [U.S. Bureau of Labor Statistics: Software Developers, Quality Assurance Analysts, and Testers](https://www.bls.gov/ooh/computer-and-information-technology/software-developers.htm)
