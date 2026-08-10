# Frontend Data Sources (per page)

Branch `mock/frontend` (F5 silent-mockware fixes), 2026-08-10.
Legend: **API** = wired to a real backend endpoint · **EMPTY** = honest empty
state (no backend exists) · **DEMO** = visibly badged demo data (none remain
unbadged — see note at the end).

Backend route availability was verified against the live `/openapi.json`
(349 routes). Notably: **no GNSS, no mineral-monitoring, no insights-feed, and
no saved cross-section registry endpoints exist.**

## Web app (`ui/web/mineralvision-app`)

| Page | Data source after fix | What was removed |
|---|---|---|
| `gnss/GNSSPage` | **EMPTY** — probes backend, shows "No live GNSS feed available" (`api/gnss/*` modules are not mounted as HTTP routers) | Fake satellites, fake "RTK Fixed / 2.5 cm" position, Math.random() accuracy chart, pulsing green "Live" badge |
| `geology/QAQCPage` | **API** — `GET /api/qaqc` (records), `GET /api/qaqc/summary/{projectId}` (pass/fail cards); alerts derived ONLY from real failed records; **EMPTY** state when no records/failures | Fake control charts, fake duplicate scatter, fabricated ERROR alert "Blank BLK-045 0.15 g/t Au" |
| `mineral-monitoring/MineralMonitoringPage` | **API** — real `GET /api/projects` listed; monitoring panels (alerts, prospectivity, geochem, geophysics, resource estimates) are **EMPTY** honest states | Fake sites with prospectivity scores + alert badges, fake high-grade-intercept alerts, fake time series and resource estimates |
| `ai-insights/AIInsightsPage` | **API** — `GET /api/predictive-modeling/models` (registered models table); insights + analysis jobs are **EMPTY** honest states | Fake insight cards with confidence scores, fake job progress, fake stats |
| `geology/DrillholesPage` | **API** — `GET /api/drillholes` + `GET /api/projects` (names); upload modal calls real `drillholesApi.upload`; `avgGrade` shown as `-` (list endpoint doesn't provide it) | 8 hardcoded drillholes; fake 2 s upload simulation |
| `geostatistics/BlockModelPage` | **API** — `GET /api/geostatistics/block-model` (list + detail), `POST /api/geostatistics/block-model` (create modal with real project list) | 4 hardcoded models + fake detail stats |
| `geostatistics/VariographyPage` | **API** — `POST /api/geostatistics/variogram` on demand; chart renders only from real results; **EMPTY** state before first computation | Hardcoded experimental variogram + simulated "Calculating…" timer |
| `geology/CrossSectionsPage` | **API** — real drillhole register grouped by project; saved-sections panel is **EMPTY** (no section registry backend; terrain cross-section endpoint needs a DTM input) | 4 hardcoded sections with fake hole counts and fake rendering |
| `reporting/ReportsPage` | **API** — `GET /api/reports` + project names; loading/error/empty states | 4 hardcoded reports with fake statuses/authors |

## Mobile app (`ui/mobile/mineralvision-mobile`)

All four screens now use the app's existing axios clients
(`EXPO_PUBLIC_API_URL`, bearer token from SecureStore):

| Screen | Data source |
|---|---|
| `drillholes/DrillholesScreen` | **API** — `GET /api/drillholes` + projects; loading/error/empty states; pull-refresh reloads |
| `projects/ProjectsScreen` | **API** — `GET /api/projects` + real per-project drillhole counts |
| `samples/SamplesScreen` | **API** — `GET /api/samples?drillholeId=…`; "assayed" derived from presence of assays (labeled in code) |
| `projects/ProjectDetailScreen` | **API** — `GET /api/projects/{id}` + `GET /api/drillholes?projectId=…`; stats not provided by the API (samples, block models) shown as `—` |

## Remaining DEMO-badged / known items

- **Login demo fallback** (`authStore`, fixed on the QA branch): demo login only
  activates when the backend is unreachable; a real JWT is used otherwise.
- **No Demo badges needed**: every previously-mocked surface above is now
  either API-wired or an explicit empty state — nothing fabricated is
  presented as real anywhere in these pages.
- Web `CropMonitoringPage`, `JourneysPage` retain pre-existing `tsc` errors
  (MUI typing drift) unrelated to mock data — out of scope for F5.

## Verification

- `npx tsc --noEmit` clean for every touched web file (GNSSPage, QAQCPage,
  MineralMonitoringPage, AIInsightsPage, DrillholesPage, BlockModelPage,
  VariographyPage, CrossSectionsPage, ReportsPage).
- Mobile app: no `node_modules` available in this environment → changes were
  manually reviewed against the existing typed API clients; not type-checked
  (honest note).
- `vite build`: not run here — the 4 GB sandbox OOM-kills the production build
  (documented in `docs/RUNTIME_QA.md`); type-level verification used instead.
