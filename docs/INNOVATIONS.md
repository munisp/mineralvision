# MineralVision — 22 Premiere Innovations

All 20 features are implemented as production modules under `MineralVision_Final_Package/src/api/innovations/`,
wired into the canonical API (`src.api.main:app`) under `/innovations/<name>`, and covered by deterministic
test suites in `tests/innovations/` (200+ tests, real numeric assertions, seeded reproducibility).

## Resource Intelligence

### 1. JORC / NI 43-101 Reporter (`/innovations/jorc_reporter`)
Resource classification engine: nearest-sample distance (fraction of variogram range) + sample count inside
the search ellipsoid → Measured / Indicated / Inferred, with configurable rule sets. Produces classification
tables, per-class tonnage/grade summaries, and a structured compliant-report JSON.

### 2. QAQC Analyzer (`/innovations/qaqc_analyzer`)
Laboratory quality control: Shewhart control charts on CRM standards (±2SD warning / ±3SD failure, bias %,
run rules), blank contamination flags (>5× detection limit), field-duplicate HARD ranking and
Thompson-Howarth precision data. Per-batch pass/warn/fail verdicts.

### 3. Monte Carlo Resource Uncertainty (`/innovations/resource_monte_carlo`)
Conditional simulation (SGS-equivalent): exact Gaussian conditioning via covariance from the platform's
variogram models, Cholesky sampling with seeded RNG. P10/P50/P90 tonnage, grade and metal above cutoff
across N realizations.

### 4. Block Model & Grade-Tonnage (`/innovations/block_model_grade_tonnage`)
Regular block model estimation using the platform's Ordinary Kriging core, per-block density support,
cutoff sweeps producing tonnage / average grade / metal-content curves (JSON + CSV export).

## Targeting & ML

### 5. Prospectivity Copilot (`/innovations/prospectivity_copilot`)
Natural-language exploration queries without an LLM: deterministic parser (commodity lexicon, deposit types,
regions, numeric constraints with unit normalization, spatial "within N km" intents) compiled to structured
database queries, with parsed-query echo and plain-language explanations.

### 6. Target Ranking with Explainability (`/innovations/target_ranking`)
Gradient-boosted / random-forest drill-target ranking with per-target permutation importance (implemented
directly — seeded column shuffles, score-drop measurement) returning top-k drivers with direction.

### 7. Exploration Portfolio Optimizer (`/innovations/portfolio_optimizer`)
AHP multi-criteria scoring with Saaty consistency-ratio validation, non-dominated-sort Pareto frontier
(value vs risk), and budget-constrained selection (ratio-greedy + local swap improvement).

### 8. Geostatistical Drift Alerting (`/innovations/geostat_drift_alerting`)
Two-sided CUSUM and EWMA detectors on assay/grade streams, rolling declustered-mean and variogram-sill
comparison against baseline, with transition-gated alert events.

## Remote Sensing & Geophysics

### 9. Inversion Job Orchestration (`/innovations/inversion_jobs`)
Async geophysical inversion service: submit → background execution (wrapping the platform's Gauss-Newton
solver) → poll status/progress → download results. SQLAlchemy job store with error-path recording.

### 10. Hyperspectral Alteration Mapping (`/innovations/hyperspectral_alteration`)
Band-ratio alteration indices (clay, iron-oxide, carbonate; ASTER / Landsat-8 / Sentinel-2 presets),
NDVI vegetation masking, thresholding and connected-component zoning with GeoJSON export.

### 11. Satellite Change Detection (`/innovations/satellite_change_detection`)
Bi-temporal scene differencing (band/NDVI/NDMI) with dual absolute + z-score thresholds, morphological
filtering, and connected-component disturbance polygons (area, centroid, mean delta) as GeoJSON.

### 12. Drill Telemetry Auto-Logging (`/innovations/drill_telemetry`)
Rig telemetry batch ingestion (depth, ROP, torque, RPM, vibration) with CUSUM binary segmentation for
automatic interval logging, collar-datum alignment, and MWD-vs-planned depth deviation flags.

## Compliance & Governance

### 13. Assay Chain-of-Custody Ledger (`/innovations/custody_ledger`)
Append-only hash-chain ledger (SHA-256 linkage + HMAC-SHA256 signatures) for sample batches, dispatch,
lab receipt and results, with a verification endpoint that replays the full chain to detect tampering.

### 14. ESG Compliance Gap Scanner (`/innovations/esg_scanner`)
Rule-pack engine (water, emissions, tailings, rehabilitation, community — 30 rules with GRI/IFC/GISTM
references) evaluating project data into severity-ranked gap lists with remediation guidance.

### 15. Regulatory Submission Packager (`/innovations/submission_packager`)
Jurisdiction templates (WA DMIRS annual report + generic) assembling deterministic ZIP bundles:
manifest with per-file SHA-256, sorted entries, fixed timestamps, and a completeness validator.

### 16. Indigenous Knowledge Governance (`/innovations/indigenous_governance`)
Access-tier governance (public / restricted / sacred) with role enforcement, consent and attribution
fields, per-access audit records, and sacred-tier exclusion from bulk export paths.

## Platform & Collaboration

### 17. Document Intelligence (`/innovations/doc_intelligence`)
Deterministic extraction from historical exploration reports: drill-hole IDs, assay intervals with units
and commodities, dates, and UTM/decimal-degree coordinates — with per-entity confidence scoring.

### 18. Field Offline Sync (`/innovations/field_sync`)
Delta synchronization for offline field capture: optimistic entity versioning, conflict records
(server-wins with both versions preserved), idempotent retry via client operation IDs, pull-since deltas.

### 19. Audit Trail & Collaboration (`/innovations/audit_collaboration`)
Append-only audit events with recursive before/after JSON diffs, threaded comments, and versioned
project settings with snapshot revert.

### 20. Integration Hub (`/innovations/integration_hub`)
Webhook registry with HMAC-SHA256 signed deliveries (`X-MV-Signature`), exponential-backoff retries and
delivery logs, plus scoped API keys (bcrypt-hashed, scope-enforced) for external integrations.
### 21. Commodity Discovery (`/innovations/commodity-discovery`)
API gateway to the platform's gold and lithium exploration engines: pathfinder-element scoring
(As-Sb-Te-W-Bi for gold; Li-Rb-Cs-Ta-Sn for LCT pegmatites), K/Rb fractionation and Mg/Li brine
chemistry, alteration indices, regolith models, and an end-to-end discovery workflow endpoint.

### 22. Marine Sonar (`/innovations/marine-sonar`)
Offshore mineral exploration from sonar data: multibeam bathymetry gridding with spike filtering,
terrain attributes (slope, rugosity, BPI, hillshade), dB-space backscatter texture classification
(fine/coarse sediment, rock), feature detection (channels, ridges, pinnacles), and composite target
scoring against five marine deposit models (placer gold, marine diamond, tin placer, SMS, polymetallic nodules).

---

*Each innovation reuses the platform's real scientific cores (kriging, variography, inversion,
prospectivity ML) rather than reimplementing them. No mocks, no placeholders: every endpoint computes
real results and every test asserts on real numbers.*
