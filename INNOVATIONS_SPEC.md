# INNOVATIONS SPEC — 20 Features (single source of truth for Wave 3)

## Global Contract (binding on ALL innovation agents)
- Location: each innovation lives at `MineralVision_Final_Package/src/api/innovations/<name>/`
  - `__init__.py` MUST export `router = APIRouter(prefix="/innovations/<name>", tags=["<name>"])`
  - Pure logic separated from HTTP layer (`logic.py` / modules), `routes.py` thin.
- NO edits to `main.py` or other agents' files. Orchestrator wires all routers at integration.
- Deps allowed: numpy, pandas, scipy, scikit-learn, fastapi, pydantic, sqlalchemy, httpx (already in requirements).
  NO torch/ultralytics/shap/geopandas-heavy ops in request paths. Optional deps via lazy import + 501-not-configured.
- REAL implementations only — working algorithms with correct math. No mocks, no `pass`, no simulated results.
- Tests: `tests/innovations/test_<name>.py` — deterministic tests with real numeric assertions. All must pass
  with: fastapi, sqlalchemy, pydantic, PyJWT, bcrypt, httpx, pytest, numpy, pandas, scipy, scikit-learn installed.
- Reuse existing real cores where relevant: `src/api/geostatistics/` (kriging.py, variography.py),
  `src/api/geophysics/inversion.py`, `src/api/ml/prospectivity_workflow.py`, `src/api/database.py` models.

---

## B1 — Resource Intelligence (agent 1)
### 1. jorc_reporter — JORC/NI 43-101 classification & report engine
- Input: block model array (x,y,z,grade[,density]), drill spacing per block, QAQC summary, search-ellipse params.
- Classification logic: distance-to-nearest-sample + sample-count + variogram range ratios → Measured/Indicated/Inferred
  per configurable rules (defaults documented vs JORC 2012 guidance). Output: classified block array +
  class tonnage/grade summary + report JSON (sections: data summary, QAQC statement, estimation params,
  classification table, grade-tonnage by class). Deterministic, unit-tested on synthetic grids.

### 2. qaqc_analyzer — QAQC anomaly detection
- Standards (CRMs): control-chart logic — expected value ±2SD warning / ±3SD failure, bias %, consecutive-failure runs.
- Blanks: contamination flag when >5× detection limit. Field duplicates: HARD (half absolute relative difference)
  pairs ranking + Thompson-Howarth precision plot data (CV vs mean). API: upload/ingest assay+QAQC rows →
  per-batch pass/fail, failures list, summary stats. Real statistics, tested on planted anomalies.

### 3. resource_monte_carlo — resource uncertainty via conditional simulation
- Sequential Gaussian simulation (simplified but correct): build covariance from variogram model (spherical/
  exponential — reuse variography.py), Cholesky solve on conditioning grid (cap grid size, document limits),
  LU/cholesky path simulation honoring data. N realizations → P10/P50/P90 tonnage & grade above cutoff.
- Tests: simulated mean ≈ data mean (tolerance), variance reduction vs unconditioned, reproducibility with seed.

### 4. block_model_grade_tonnage — block model builder + grade-tonnage API
- Build regular block model from kriged estimates (reuse kriging.py), block size/origin params, density field.
- Grade-tonnage: cutoff sweep (0..max, N steps) → tonnage, avg grade, metal content arrays; JSON + CSV export.
- Tests: monotonic tonnage decrease with rising cutoff; mass balance check (block volume × density sum).

## B2 — Targeting & ML (agent 2)
### 5. prospectivity_copilot — natural-language exploration query
- Deterministic NL parser (no LLM): tokenize → extract commodities (periodic-table/mineral lexicon), deposit
  types, regions, numeric constraints ("gold > 2 g/t", "within 5km"), intents (rank/list/count/explain).
  Compiles to structured query over projects/drillholes/assays/prospectivity scores (SQLAlchemy).
- Response: results + parsed-query echo + plain-language explanation string. Tests: 15+ query phrases → correct AST.

### 6. target_ranking — drill target ranking with explainability
- Features from prospectivity_workflow outputs + geophysical/geochemical layers; ranking via sklearn
  GradientBoosting/RF; explanation via permutation importance (implement directly — shuffle column, score drop,
  seeded) per target → top-k drivers with direction. API: rank targets for project, per-target explanation.
- Tests: importance sums ≈ total score drop (tolerance), ranking stable with seed, planted signal ranks top.

### 7. portfolio_optimizer — multi-criteria exploration portfolio
- AHP-style weighted scoring (criteria: prospectivity, cost, jurisdiction risk, ESG, logistics) with
  consistency-ratio check (Saaty CR<0.1 validation). Pareto frontier (non-dominated sort) across
  expected-value vs risk. Budget-constrained selection: greedy by ratio + local swap improvement.
- Tests: CR rejection of inconsistent matrix; frontier correctness on known sets; budget never exceeded.

### 8. geostat_drift_alerting — real-time geostatistical drift detection
- CUSUM + EWMA detectors on incoming assay/grade streams (configurable k/h thresholds); rolling
  declustered-mean & variogram-sill comparison vs baseline; alert events with magnitude/timestamp.
- API: register stream, push batches, list alerts. Tests: planted mean-shift detected within N samples;
  stationary stream yields zero alerts (seeded).

## B3 — Remote Sensing & Geophysics (agent 3)
### 9. inversion_jobs — geophysical inversion job orchestration
- Async job service: submit inversion config (mesh, survey data ref, regularization params) → job id;
  worker executes via FastAPI BackgroundTasks + SQLite job table (status, progress, artifacts, error);
  polls + artifact download. Wraps existing geophysics/inversion.py for real compute.
- Tests: full lifecycle submit→run→complete on tiny mesh; failed job records error; concurrency of 2 jobs.

### 10. hyperspectral_alteration — alteration mineral mapping
- Band-ratio indices on multi-band raster (numpy): clay (b7/b5 ASTER-style), iron oxide (b4/b2), carbonate
  (b8/b7), plus NDVI masking; configurable band mapping for ASTER/Landsat/Sentinel-2 presets; threshold +
  connected components (scipy.ndimage.label) → alteration zones with area/mean index; GeoJSON export.
- Tests: synthetic cube with planted clay signature → zone detected at correct location.

### 11. satellite_change_detection — bi-temporal change monitoring
- Inputs: two co-registered rasters (or arrays). Differencing on chosen indices (NDVI/NDMI/band),
  threshold by absolute + z-score vs scene stats, morphological open/close, connected components →
  change polygons (area, centroid, mean delta) as GeoJSON. Tests: planted disturbance detected, false-positive-free on identical scenes.

### 12. drill_telemetry — rig telemetry ingestion & auto-logging
- POST/WebSocket batch ingestion of rig telemetry (timestamp, depth, ROP, torque, RPM, vibration).
- Auto-logging: segmentation of depth series by ROP/torque regime shifts (CUSUM on ROP) → interval table;
  depth alignment to collar; MWD-vs-collar deviation check. Storage via SQLAlchemy. Tests: synthetic
  3-regime drill trace → 3 intervals detected at correct depths (tolerance).

## B4 — Compliance & Governance (agent 4)
### 13. custody_ledger — assay chain-of-custody hash ledger
- Append-only ledger: each entry = SHA-256(prev_hash + payload + timestamp + actor), HMAC-signed with
  server key (env); verification endpoint walks the chain proving integrity; tamper-detection test.
- Entities: sample batches, dispatch, lab receipt, results. Tests: chain verifies; flipping any byte breaks verification.

### 14. esg_scanner — ESG/environmental compliance gap scanner
- Rule-pack engine: JSON rule packs (water discharge, emissions, tailings, rehabilitation, community)
  each rule = field check/threshold/document-required; scanner evaluates project data → gap list with
  severity, remediation text, framework refs (GRI/IFC-style). Tests: planted gaps detected with correct severity.

### 15. submission_packager — regulatory tenement submission bundle
- Jurisdiction templates (WA-DMIRS-style annual report, generic): collects project data, drillhole exports
  (CSV), QAQC summary, expenditure table → manifest JSON + files → deterministic ZIP (sorted, fixed timestamps).
- Validation: required-section completeness check fails on missing data. Tests: zip contents match manifest; validator catches omission.

### 16. indigenous_governance — indigenous knowledge access governance
- Access tiers: public/restricted/sacred with RBAC enforcement dependency (require_role integration point),
  consent & attribution fields, per-access audit records (who/when/what tier), sacred tier = never via API bulk export.
- Tests: unauthorized role → 403; every access writes audit row; sacred bulk-export blocked.

## B5 — Platform & Collaboration (agent 5)
### 17. doc_intelligence — historical report extraction
- Text extraction: pdf → text via pypdf (add to requirements if absent); OCR path lazy (pytesseract optional → 501).
- Deterministic NLP: regex/lexicon extraction of hole IDs (pattern DH/RC/DD\d+), assay intervals
  (from-to-depth + value+unit), commodities, dates, coordinates (MGRS/UTM/decdeg) → structured tables + confidence.
- Tests: fixture text with planted entities → exact extraction.

### 18. field_sync — offline field data synchronization
- Change-log protocol: client sends ops (entity, op, base_version, payload, client_ts); server applies with
  optimistic version check; conflicts → conflict record (server wins default + both versions kept);
  per-entity monotonically increasing version; pull-since endpoint for delta download.
- Tests: concurrent edits produce conflict record; pull-since returns exactly the delta; idempotent retry.

### 19. audit_collaboration — audit trail + comments + versioning
- Append-only audit events (actor, action, entity, before/after JSON diff) + threaded comments on
  projects/drillholes/targets + project settings versioning (snapshot per change, revert endpoint).
- Tests: diff correctness, revert restores snapshot, comment threading order.

### 20. integration_hub — webhooks, API keys, SDK surface
- Webhook registry: subscribe URL+secret to event topics; delivery worker signs payload HMAC-SHA256
  (header X-MV-Signature), retries with exponential backoff (in-process worker), delivery log.
- API keys: create scoped keys (read/write), stored bcrypt-hashed, middleware dependency `require_api_key`.
- Tests: signature verification; retry on 500 then success recorded; scoped key forbidden on write.
