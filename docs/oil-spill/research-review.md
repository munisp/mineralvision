# Oil-Spill Intelligence: Research Review and MineralVision Applicability

**Author:** Manus AI  
**Scope:** Review of the seven user-supplied sources and the resulting MineralVision implementation.

## Executive Assessment

**Yes—MineralVision can be used for oil-spill analysis**, provided it is treated as a configurable environmental-intelligence platform rather than as a turnkey spill-response system. Its existing imagery, sensor-fusion, geospatial, workflow, digital-twin, autonomy-planning, reporting, audit, and high-availability components are well aligned with an oil-spill workflow. However, mineral-exploration models must not be reused as oil detectors: the detection layer needs an oil-spill-specific, versioned model trained and validated for the intended sensor, geography, sea state, illumination, and operating context.

The most directly applicable supplied research is the Scientific Data paper by De Kerf et al. It publishes a drone-RGB port dataset of **1,268 pixel-annotated images** labeled as oil, water, and other, with an EfficientNet-B4 U-Net baseline reporting oil-class F1 near **0.72**. The authors emphasize that RGB drone imaging can be practical for port monitoring but note ambiguity at gradual oil-water boundaries. [1] This supports segmentation, confidence gating, human review, and repeat-survey capabilities—not a binary “spill/no spill” alert alone.

> The implementation deliberately creates a **decision-support and evidence-management layer**. It does not autonomously notify authorities, deploy aircraft, operate cleanup robots, diagnose oil chemistry, or prescribe response actions.

## Review of Supplied Sources

| Source | What it contributes | Implementation conclusion |
|---|---|---|
| De Kerf et al., *Scientific Data* | A public drone-RGB dataset, oil/water/other semantic labels, 70/15/15 partitioning, and an EfficientNet-B4 U-Net baseline. [1] | Use a semantic-segmentation interface with model provenance and mandatory boundary/quality review. |
| `OilSpillDataset_Analysis` | Companion code intended to reproduce the dataset-paper analysis and segmentation work. [2] | Treat as a reproducibility and training reference; do not confuse research notebooks with production operations. |
| Optelos platform overview | An enterprise pattern: ingest and georeference imagery; contextualize it; create AI-ready digital twins; analyze, visualize, and report. [3] | MineralVision’s existing lakehouse, digital twin, visualization, RBAC, and reporting modules can follow this structure. |
| Multi-agent oil-spill search | A drone-search simulator with uncertainty grids, shared coverage knowledge, weather dynamics, and policy comparisons. [4] | Generate an **advisory** evidence-priority grid for an approved flight planner; do not issue flight commands automatically. |
| SDOS | A concept combining pipeline telemetry, drone imagery/fluorosensing, alerting, and robotic recovery. [5] | Reserve a sensor-evidence interface for future flow, pressure, fluorosensor, and water-quality corroboration. Keep dispatch and cleanup under human governance. |
| YOLO oil-spill segmentation | A minimal custom YOLO segmentation workflow for drone images and video. [6] | Support instance-segmentation models as an alternative adapter, but standardize all output as a probability mask and GeoJSON assessment. |
| prago-dev oil-spill detection | A satellite-image classification/web-alert prototype. [7] | Binary classification may be useful for triage, but it cannot quantify extent or spatial impact; segmentation and georeferencing are required for an incident record. |

## What the Research Means for a Production Workflow

The cited dataset establishes the strongest practical case for **low-altitude RGB segmentation in port environments**, where smaller spills, thin slicks, and lower wave action can reduce the value of conventional SAR-only monitoring. [1] However, the published performance should be treated as a baseline for that dataset and context, not as an operational guarantee. A new deployment must evaluate sensitivity, precision, calibration, and spatial errors on held-out imagery matching the actual sensor and operating area.

The supplied repositories collectively point to a more complete workflow than model inference: collect heterogeneous data, establish geospatial context, detect and delineate evidence, prioritize resurvey, allow expert confirmation, record provenance, and generate controlled reports. [2] [3] [4] [5] The implementation therefore focuses first on reliable incident evidence and human review, while retaining extension points for V-JEPA/WALDO imagery context, Sensor Fusion corroboration, Digital Twin visualization, climate/current inputs, and workflow orchestration.

## Implemented MineralVision Extension

The repository now includes a persistent oil-spill intelligence module under `MineralVision_Final_Package/src/api/oil_spill/` and API endpoints under `/api/oil-spill`.

| Capability | Implementation behavior |
|---|---|
| Mask assessment | Accepts a base64 PNG/JPEG probability or binary mask created by a named, versioned model or annotation process. |
| Component quality control | Removes connected components smaller than an explicit pixel threshold and records quality flags for empty, fragmented, low-confidence, or non-georeferenced assessments. |
| Area screening | Calculates surface area from calibrated GSD where supplied; otherwise estimates it from a rectangular WGS84 image footprint and flags the approximation. |
| Geospatial evidence | Converts retained mask components to WGS84 GeoJSON MultiPolygon features when a geographic image footprint is supplied. |
| Persistent incident record | Stores source, model provenance, metadata, geometry, metrics, review state, and audit events in SQLAlchemy-backed storage. |
| Human review | New assessments begin as `pending_review`; an operator can mark them `confirmed`, `rejected`, or `needs_resurvey`. |
| Advisory coverage plan | Generates a bounded, centre-outward priority grid around georeferenced incident evidence. It is explicitly non-executable and does not control drones. |
| Raw image inference | Supports an optional local TorchScript or ONNX adapter. It is **fail-closed** without a configured, trusted local artifact and never downloads models automatically. |

## API Workflow

| Step | Endpoint | Purpose |
|---|---|---|
| 1. Submit evidence | `POST /api/oil-spill/analyze/mask` | Convert a versioned segmentation mask into a reviewable incident record. |
| 2. Optional local inference | `POST /api/oil-spill/analyze/image` | Run an operator-configured local model; returns HTTP 503 unless model provenance and a local artifact are configured. |
| 3. Locate evidence | `GET /api/oil-spill/incidents` | List assessments by project and review status. |
| 4. Confirm assessment | `PATCH /api/oil-spill/incidents/{incident_id}/review` | Capture a human reviewer’s decision and note. |
| 5. Prepare resurvey | `POST /api/oil-spill/incidents/{incident_id}/coverage-plan` | Produce a non-executable search-priority grid for a later approved planner. |

## Deployment and Validation Requirements

Before this extension is used to support real-world operations, the following safeguards are required. First, the operator should curate and document an approved model registry, including source data, label definition, training date, version, model hash, intended sensor, confidence calibration, and validation metrics. The implementation requires `OIL_SPILL_MODEL_PATH`, `OIL_SPILL_MODEL_ENGINE`, `OIL_SPILL_MODEL_ID`, and `OIL_SPILL_MODEL_VERSION` for raw-image inference; these are documented in `.env.example`.

Second, the operator should validate against geographically and seasonally representative imagery. An RGB port model cannot be assumed to generalize to offshore satellite imagery, nighttime imagery, glare conditions, or another sea state. The research itself identifies difficult oil-water transition zones; reviews should prioritize these cases. [1] A multi-modal evidence scheme—such as camera imagery plus GNSS, fluorosensor, pipeline telemetry, or field observations—would improve incident confidence but still requires calibration and governance. [4] [5]

Finally, all confirmed notifications, airspace activity, maritime dispatch, and cleanup decisions must remain within the organization’s legal operating procedures and qualified human authority. This repository’s coverage plan is informational only and intentionally has no flight-control or messaging side effect.

## Validation Performed

The implementation was validated with six deterministic unit tests covering mask decoding, dimension validation, component filtering, area calculation, GeoJSON generation, quality flags, human-safe coverage planning, and fail-closed absence of a model. An isolated SQLite integration check also verified incident and geometry persistence, and a route verification script confirmed the six required FastAPI routes.

## References

[1] De Kerf, T. et al. “Annotated RGB images of oil spills in a port environment.” *Scientific Data* (2024). [Scientific Data article](https://www.nature.com/articles/s41597-024-03993-8); [Zenodo dataset](https://doi.org/10.5281/zenodo.10555314).

[2] Thomas De Kerf. [*OilSpillDataset_Analysis*](https://github.com/thomasdekerf/OilSpillDataset_Analysis).

[3] Optelos. [*Platform Overview*](https://optelos.com/platform-overview/).

[4] Inioluwa Ashamu. [*multi-agent-oil-spill-search*](https://github.com/Inioluwa-Ashamu/multi-agent-oil-spill-search).

[5] PRADULOP. [*SDOS*](https://github.com/PRADULOP/SDOS).

[6] tim3in. [*oil-spill-segmentation*](https://github.com/tim3in/oil-spill-segmentation).

[7] prago-dev. [*oil-spill-detection*](https://github.com/prago-dev/oil-spill-detection).
