# MineralVision — Platform Vision

**The premiere platform for mineral exploration, discovery, and compliance.**

MineralVision exists to shorten the path from a satellite anomaly to a
compliant mineral resource statement. It unifies the data, algorithms, and
reporting disciplines that exploration teams today stitch together across
disconnected desktop tools, spreadsheets, and consultants — and it does so
with evidence: working algorithm cores for geostatistics, geophysical
inversion, prospectivity machine learning, and field detection, all behind
one API and one web application.

This document states what the platform is for, who it serves, and where it
is going. Capabilities described as *roadmap* are not yet shipped; everything
else refers to code that exists in this repository.

---

## 1. The Problem

Mineral exploration generates some of the most heterogeneous data in any
industry: drillhole assays and lithology logs, ground and airborne
geophysics, hyperspectral satellite scenes, geochemical soil grids, field
photographs, GNSS surveys, and regulatory filings. Each dataset typically
lives in its own vendor tool. The consequences are familiar to every
exploration team:

- **Fragmented workflow.** Anomaly detection, target ranking, drill
  planning, resource estimation, and compliance reporting are performed in
  separate applications with manual hand-offs and copy-pasted data.
- **Unreproducible estimates.** Resource numbers are recomputed by hand for
  each reporting round; audit trails between raw assay and reported grade
  are weak or missing.
- **Compliance as an afterthought.** JORC- and NI 43-101-aligned reporting
  is bolted on at the end of the process instead of being produced from the
  same governed data that drove the decisions.
- **Field intelligence lost.** Detection of gossans, alteration, and
  mineralization indicators in field imagery is done by eye, at the pace of
  the geologist's notebook.

MineralVision is built to close these gaps with a single, API-first
platform.

## 2. The Exploration Lifecycle We Serve

MineralVision covers the full exploration lifecycle as one continuous,
data-governed pipeline:

1. **Satellite and airborne anomaly detection.** Hyperspectral and
   multispectral imagery, magnetometry, and radiometrics are ingested
   through the sensor-fusion layer and screened for alteration and
   structural anomalies.
2. **Target ranking.** Prospectivity machine-learning workflows
   (`src/api/ml/prospectivity_workflow.py`, `gold_exploration.py`,
   `lithium_exploration.py`) combine geological, geochemical, and
   geophysical evidence into ranked targets, with spatial cross-validation
   (`spatial_cv.py`) and uncertainty quantification
   (`uncertainty_quantification.py`) so that rankings come with honest
   confidence intervals.
3. **Drilling.** Drillhole collars, surveys, assays, and lithology are
   managed as first-class data (drillhole endpoints and UI), with QAQC
   tracking of standards, blanks, and duplicates from the moment assays are
   loaded.
4. **Resource estimation.** Variography, ordinary and indicator kriging,
   block modeling, and grade-shell generation
   (`src/api/geostatistics/`) turn drillhole data into estimated resources
   inside the platform — no export to a third-party estimator required.
5. **Compliant reporting.** The reporting module
   (`src/api/reporting/regulatory_reports.py`,
   `auto_report_generator.py`) generates JORC- and NI 43-101-aligned
   disclosures from the same governed datasets, with audit trails preserved
   end to end.

Each stage consumes the output of the previous one. Nothing is re-keyed;
nothing is detached from its provenance.

## 3. The Data We Unify

| Domain | What the platform handles today |
| --- | --- |
| Drillhole data | Collars, downhole surveys, assays, lithology; QAQC for standards, blanks, duplicates |
| Geostatistics | Experimental variograms, model fitting, kriging, block models, grade shells |
| Geophysics | Gravity, magnetic, and EM inversion (`src/api/geophysics/inversion.py`, `advanced_inversion.py`) |
| Hyperspectral & remote sensing | Hyperspectral adapter, magnetometry, radiometrics, LiDAR, GPR, and SEG-Y ingestion under `src/api/sensor_fusion/` |
| Field detection (WALDO) | YOLO11 + RF-DETR ensemble detection of outcrop, gossan, and alteration indicators in field imagery (`src/api/waldo/`) |
| Sensor fusion | Kalman and deep-learning fusion of multi-sensor surveys (`fusion_algorithms.py`, `kalman_fusion.py`, `deep_learning_fusion.py`) |
| Field collection | Structured field-data capture with GNSS support (`src/api/field/field_collection.py`) |

The unification is not just storage: it is a shared coordinate, project, and
provenance model, so a sample collected by a field crew is the same object a
resource geologist estimates from and a regulator audits.

## 4. Compliance as a First-Class Concern

MineralVision treats compliance frameworks as data products, not documents
assembled after the fact:

- **JORC** — reporting structures aligned to the Australasian Code for
  Reporting of Exploration Results, Mineral Resources and Ore Reserves.
- **NI 43-101** — Canadian Standards of Disclosure for Mineral Projects,
  supported by the regulatory report generator.
- **QAQC** — assay quality assurance/quality control (certified reference
  materials, blanks, duplicates) is tracked in the database and exposed
  through dedicated QAQC endpoints and UI, because no resource estimate is
  defensible without it.
- **ESG** — *roadmap*: structured capture of environmental, social, and
  governance indicators alongside exploration data, so that ESG disclosure
  inherits the same provenance guarantees as grade data.
- **Provenance and audit** — blockchain-anchored data provenance and audit
  trails (`src/api/blockchain/`, `src/api/core/audit_trails.py`) record who
  changed what, when, and why.

## 5. Who We Build For

- **Exploration geologist.** Screens satellite and geophysical anomalies,
  ranks targets, plans the next drill program. Needs prospectivity maps,
  cross-sections, and 3D visualization — all present in the web UI.
- **Resource geologist.** Owns the drillhole database, QAQC, variography,
  kriging, and block models. Needs reproducible estimates and an audit
  trail from assay to reported grade.
- **Data scientist.** Builds and validates prospectivity models. Needs the
  ML workflow layer, spatial cross-validation, uncertainty quantification,
  and a model registry — not a tangle of notebooks.
- **Regulator / investor.** Consumes JORC / NI 43-101 disclosures. Needs to
  trust that reported numbers trace back to governed data without manual
  reconstruction.
- **Field crew.** Captures samples, photographs, and GNSS-located
  observations. WALDO-assisted detection flags mineralization indicators in
  imagery at capture time instead of weeks later in the office.

## 6. What "Premiere" Means — Evidence, Not Adjectives

We claim to be the premiere platform for this domain on the strength of
working algorithm cores, not marketing language:

- **Kriging and variography** — `src/api/geostatistics/kriging.py` and
  `variography.py`: experimental variogram computation, model fitting, and
  ordinary kriging implemented in-platform.
- **Block modeling and grade shells** — `block_model.py`, `grade_shells.py`
  for resource-grade estimation and reporting volumes.
- **Geophysical inversion** — `src/api/geophysics/inversion.py` and
  `advanced_inversion.py` for gravity/magnetic/EM inverse problems.
- **Prospectivity ML** — `src/api/ml/prospectivity_workflow.py` with
  commodity-specific models (gold, lithium), spatial cross-validation, and
  uncertainty quantification.
- **WALDO detection** — `src/api/waldo/ensemble_detector.py`: YOLO11 +
  RF-DETR ensemble for field-image detection of geological indicators.
- **Sensor fusion** — Kalman and deep-learning fusion across magnetometry,
  radiometrics, hyperspectral, LiDAR, and GPR.

Premiere also means *honest*: where a capability is incomplete we say so.
Consolidation of the API into a single canonical entry point
(`src.api.main:app`), a root Docker Compose deployment, and ESG reporting
are active roadmap items, tracked in the remediation specification.

## 7. Roadmap

Near-term priorities, in order:

1. **Single canonical API application** — one `uvicorn src.api.main:app`
   entry point replacing the historical multi-entry sprawl.
2. **One-command deployment** — root `docker compose up --build` for the
   API, UI, and data services.
3. **Continuous integration** — `.github/workflows/ci.yml` running backend
   tests and UI build on every change.
4. **ESG data capture and disclosure** aligned with the same provenance
   model as resource data.
5. **Deepened WALDO coverage** — broader indicator classes and active
   learning from geologist feedback.

## 8. Non-Goals

MineralVision is a **mineral** exploration platform. It does not do
agriculture, crop monitoring, or land-suitability scoring for farming; such
code has been removed from this repository. Every module in the platform
must trace to the exploration lifecycle above. If a feature cannot be tied
to satellite anomaly detection, target ranking, drilling, resource
estimation, or compliant reporting, it does not belong here.

---

*This document is user-facing and normative: contributions should be able to
point to the section of the lifecycle they serve.*
