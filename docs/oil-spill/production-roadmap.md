# Oil-Spill Intelligence Production Roadmap

**Author:** Manus AI  
**Status:** Implementation blueprint and acceptance criteria for the MineralVision oil-spill extension.

## Decision: Suitable Platform, Conditional Production Readiness

MineralVision has the core capabilities required for an oil-spill intelligence platform: computer vision, remote-sensing ingestion, geospatial visualization, digital twin concepts, workflows, audit trails, reporting, and device-oriented field collection. The first oil-spill extension added a safe assessment path, but the audit found five material gaps: an incomplete PostgreSQL migration, no model registry or evaluation gate, no temporal/JEPA consensus workflow, no oil-spill web or mobile workspace, and no deployment-grade client examples.

The implementation work in this roadmap closes those structural gaps. It does **not** certify a model for real response operations. Certification requires an independently sealed test set and data from the actual sensor, port/offshore geography, seasons, illumination, sea state, and lookalike conditions.

## Measurable Accuracy Objective

A headline claim of “above 97% accuracy” is not a valid safety or environmental-response criterion by itself. A model can obtain high pixel accuracy by predicting water in an image that contains little oil. The proposed production gate is therefore **oil-class F1 ≥ 0.97 and oil-class IoU ≥ 0.95**, with recall ≥ 0.97 and precision ≥ 0.97, on a sealed, incident-disjoint holdout set. The target must be met separately for relevant source domains (for example: daylight drone RGB, low-angle glare, port water, offshore water, and sensor-specific deployments), with 95% confidence intervals reported.

The supplied port RGB dataset is useful as a baseline but is neither sufficiently broad nor evidence of a universal 97% result. It contains 1,268 annotated drone images and reported an oil F1 near 0.72 using a U-Net/EfficientNet-B4 baseline. [1] The target is an **evaluation threshold**, not a promise.

## JEPA Improvement Strategy

V-JEPA is a self-supervised representation learner: it predicts masked spatiotemporal content in latent space and can adapt a backbone to downstream tasks with less labeled data. [2] V-JEPA 2.1 specifically describes temporally consistent dense features, which is relevant to stabilizing frame-level segmentation over aerial video. [3] JEPA should improve the segmentation system as a feature backbone and temporal quality signal, not replace a calibrated segmentation head.

| Stage | JEPA contribution | Measurable acceptance criterion |
|---|---|---|
| Domain-adaptive pretraining | Pretrain or continue pretraining on unlabeled aerial video from the intended sensor family, retaining sequence metadata and excluding the sealed test set. | Validation improvement over the non-JEPA baseline is statistically reported; no train/test location or incident leakage. |
| Dense feature decoder | Fine-tune a lightweight segmentation decoder on frozen or selectively tuned V-JEPA 2.1 features. | Compare oil F1, IoU, precision, recall, and expected calibration error against the current U-Net/YOLO baseline. |
| Temporal consensus | Aggregate aligned per-frame probabilities with a motion/registration-aware consistency gate; flag rather than suppress genuine new spill appearance. | Reduced flicker and false-positive persistence on a labeled video set without recall loss beyond the preset guardrail. |
| Shift detection | Use JEPA embedding distance against a validated reference bank to mark potential domain shift. | All shifted scenes are routed to review; no automatic model-promotion decision is made from this signal alone. |
| Continuous evaluation | Store sealed evaluation runs by model/version/domain, calculate class metrics, and allow promotion only when the formal gate is passed. | Registry reports a reproducible evaluation data fingerprint, metric values, sample count, and approval status. |

## Twenty Innovations

| # | Innovation | Initial implementation state |
|---:|---|---|
| 1 | Versioned model registry with artifact hash, intended domain, model card, and approval state. | Implemented as persistent registry/API contract. |
| 2 | Sealed evaluation-run ledger with oil F1, IoU, precision, recall, calibration, and domain stratification. | Implemented as persistent registry/API contract. |
| 3 | Formal 97% promotion gate based on oil-class metrics, not global accuracy. | Implemented as deterministic eligibility check. |
| 4 | JEPA temporal-consensus interface for multi-frame probabilities. | Implemented as an advisory quality/fusion utility; requires real JEPA features at deployment. |
| 5 | Temporal flicker and persistence diagnostics. | Implemented in sequence-quality output. |
| 6 | Domain-shift / out-of-distribution routing signal. | Implemented as a stored evidence field and review gate interface. |
| 7 | Per-incident model, calibration, and data-provenance record. | Extended incident metadata and API export. |
| 8 | Postgres-first incident persistence and Alembic migration. | Implemented with a Postgres-only configuration and migration assets. |
| 9 | GeoJSON export plus PostgreSQL JSONB indexing for spatial evidence. | Implemented through migration and export API. |
| 10 | Human review, reviewer note, and immutable audit event. | Existing feature retained and expanded in operations UX. |
| 11 | Evidence-quality flags for georeferencing, fragmentation, confidence, and calibration. | Existing feature retained and exposed in UI. |
| 12 | Incident operations dashboard with counts, filters, review queue, and response-safe states. | Implemented for web/PWA. |
| 13 | Offline-first PWA evidence queue for low-connectivity field surveys. | Implemented for small mask evidence payloads; image upload remains online-only. |
| 14 | Responsive field-capture workflow with GPS/footprint metadata. | Implemented in the web PWA and companion mobile source. |
| 15 | Mobile companion incident queue and review/capture flow. | Implemented as a Flutter companion application source package. |
| 16 | Advisory multi-drone coverage prioritization. | Existing feature retained; exposed in UI. |
| 17 | Incident GeoJSON / response-plan export for interoperable GIS workflows. | Implemented as API and client example. |
| 18 | Operational timeline with analyst review and model-evidence events. | Implemented in API/UI data model. |
| 19 | Model integration cookbook for trusted local TorchScript/ONNX artifacts. | Implemented with environment configuration and runnable client examples. |
| 20 | Safety controls that prohibit automatic notification, flight, cleanup, or model promotion. | Implemented and documented as default behavior. |

## PostgreSQL Migration Standard

The platform will use PostgreSQL as the required database. A local Postgres/PostGIS instance is used only for development validation; the sandbox service may stop when the environment hibernates and is not a production deployment.

| Layer | Requirement |
|---|---|
| Runtime configuration | `DATABASE_URL` is mandatory and must use `postgresql` / `postgresql+psycopg` / `postgresql+psycopg2`. No SQLite fallback is allowed. |
| Schema evolution | Alembic migrations are the only supported production schema path. `Base.metadata.create_all()` is not called in PostgreSQL production mode. |
| Oil-spill evidence | Incidents, models, evaluation runs, operations events, GeoJSON evidence, and indexes are created through a dedicated migration. |
| WALDO service | Its local deployment configuration receives the same PostgreSQL database service instead of a SQLite file. |
| Local validation | Install PostgreSQL with a non-default locally generated password, create a dedicated development database, apply Alembic migrations, and run persistence tests against it. |

## Acceptance Checks

1. The canonical API mounts the oil-spill router and all routes require the platform JWT policy.
2. The migration chain upgrades a fresh PostgreSQL database to the current revision.
3. No runtime `sqlite:` or SQLite fallback configuration remains in API, compose, or WALDO deployment settings.
4. Unit tests cover model promotion eligibility, temporal consensus, PostgreSQL persistence, incident review, and export behavior.
5. The web PWA supports incident triage, review, coverage-plan generation, and offline queuing of compact mask evidence.
6. The mobile companion can list incidents, review an incident, and capture/upload an analysis request using the secured API.
7. Custom PyTorch/ONNX inference remains fail-closed without a trusted local artifact and mandatory model provenance.

## References

[1] De Kerf, T. et al. “Annotated RGB images of oil spills in a port environment.” *Scientific Data* (2024). [Article](https://www.nature.com/articles/s41597-024-03993-8); [dataset](https://doi.org/10.5281/zenodo.10555314).

[2] Meta AI. [“V-JEPA: The next step toward advanced machine intelligence.”](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) (2024).

[3] Meta FAIR. [*V-JEPA 2 / V-JEPA 2.1 official PyTorch repository*](https://github.com/facebookresearch/vjepa2) (accessed 2026-08-20).
