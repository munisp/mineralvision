# P0 Pilot Risk Mitigation and Fallback Plan

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Purpose:** Define the actions to take if MineralVision’s P0 pilot does not meet its 85/100 graduation score or a non-negotiable trust gate.

## Operating Principle

A P0 score miss is not a reason to extend the pilot indefinitely or conceal weaknesses. It is a structured decision point. The product council should place the pilot into one of four states: **graduate**, **remediate and repeat**, **narrow the workflow**, or **stop/exit**. Safety, tenant isolation, data lineage, and human-governance failures always override an aggregate score.

> The fallback objective is to preserve customer data, trust, source-system integrity, and learning while preventing unproven model outputs or connector actions from entering production workflows.

## Failure Scenarios and Required Actions

| Failure trigger | Detection threshold | Immediate containment | 30-day remediation path | Fallback / exit rule | Owner |
|---|---|---|---|---|---|
| Workflow adoption shortfall | Fewer than 80% of reviews complete without engineering help, or partner does not use the agreed workflow for two consecutive weeks | Freeze feature expansion; conduct workflow observation; keep source system as sole operational interface | Remove steps, reduce data fields, repair the top three usability defects, retrain the pilot team | If adoption remains below 60% after one remediation cycle, narrow to a single role/use case or stop the pilot | Product lead |
| Source lineage or connector defect | Missing source/destination lineage, untracked write, schema mismatch, duplicate external update, or tenant-boundary anomaly | Disable connector write-back; switch to read-only candidate package/export; preserve all audit evidence | Fix idempotency/schema/authorization fault; replay in sandbox; run reconciliation with customer | No write-back is re-enabled until 100% reconciliation and customer sign-off; terminate connector if the incumbent contract cannot support safe lineage | Integration lead |
| Model performance or trust shortfall | Holdout/calibration evidence fails acceptance criteria, reviewer override rate rises above the agreed threshold, or drift alert is unresolved | Disable model promotion and automatic ranking; retain human-only workflow with evidence capture | Audit labels/data split/features; recalibrate; retrain or restrict model to validated domain | Continue only as a human-reviewed evidence platform; do not claim model performance or expand model coverage | ML lead and domain SME |
| Security or privacy incident | Any confirmed cross-tenant access, broken authentication/authorization, exposed secret, audit-chain failure, or critical vulnerability | Revoke affected credentials, disable impacted endpoint/connector, preserve logs, declare incident | Perform root-cause analysis, patch, rotate secrets, execute regression and external review | No production graduation until all remediation gates, audit review, and customer notification obligations are complete | Security lead |
| Reliability or data-loss failure | Failure of RPO/RTO drill, missing SIEM events, unreconciled queue backlog, or repeated p95 latency SLO breach | Stop new onboarding and writes; switch to read-only evidence access where safe | Repair capacity, backup, observability, or queue design; run a new timed drill | If SLO cannot be met within an agreed remediation window, reduce scope/load or exit the pilot | SRE lead |
| Partner integration unavailable | API sandbox access, data-sharing approval, or service-account scope delayed beyond two weeks | Use a documented file/GeoJSON/CSV export bridge with source identifiers; do not simulate a live connector | Complete legal/security review and adapter contract in parallel | Ship only the read-only evidence workflow; defer write-back capability | Product and integration lead |
| Commercial repeatability failure | Implementation takes more than four weeks, requires bespoke schema changes, or requires unmanaged credentials | Freeze custom work and log every variance | Create configuration, mapping, and deployment templates; eliminate bespoke code paths | If two partners require incompatible bespoke work, split offerings or stop pursuing the segment | Customer implementation lead |

## Decision Thresholds

| Pilot state | Quantitative condition | Mandatory action |
|---|---|---|
| Graduate | Weighted score ≥85/100; each domain ≥15/20; every non-negotiable gate passed | Approve controlled commercial deployment, retain enhanced monitoring for 90 days |
| Remediate and repeat | Score 70–84 with no trust/safety failure | One 30-day remediation cycle with a fixed issue list and no scope expansion |
| Narrow workflow | Score 55–69, or only one role/data source demonstrates value | Retire unsupported features; repeat with one user role, one source system, and read-only connector mode |
| Stop / exit | Score below 55, any unresolved non-negotiable trust failure, or no partner value hypothesis | Preserve/export customer data, revoke credentials, remove pilot access, close findings, conduct retrospective |

## Customer-Safe Fallback Modes

MineralVision must have operationally tested fallback modes before the pilot begins.

| Capability | Default P0 mode | Safe fallback |
|---|---|---|
| Incumbent connector | Read/query and evidence linking | Customer-controlled scheduled export with immutable source IDs and checksum manifest |
| Write-back | Disabled until supervised approval phase | Downloadable review package/GeoJSON/CSV and human import in source system |
| AI candidate generation | Registered model with reviewer approval | Human-authored candidate/observation form with the same lineage/audit contract |
| Oil-spill assessment | Candidate assessment with uncertainty | Manual incident record and evidence bundle; never label unvalidated output a confirmed spill |
| Field upload | Secure managed upload | Offline encrypted capture queue or customer-approved secure transfer, followed by manual ingest/reconciliation |
| Source outage | Visible connector health status | Read-only last-known evidence marked stale with source timestamp; no cached data presented as current |

## Governance Cadence

The product council meets weekly during P0. Each meeting reviews a pilot scorecard, open exceptions, connector health, model evidence, security findings, partner feedback, and implementation variance. A gate owner must accept or reject every exception; exceptions expire automatically unless renewed with evidence.

The final P0 decision requires customer sponsor input, domain-SME sign-off, security/SRE sign-off, product ownership, and an engineering release recommendation. A commercial commitment cannot override a failed non-negotiable trust gate.
