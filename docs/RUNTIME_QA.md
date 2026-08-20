# Runtime QA — Geospatial Frontend vs Live Backend (W3 closeout)

Branch `gap/runtime-qa`, 2026-08-09. Goal: actually run the stack and verify the
map pages render with real API data — the gap left open by the frontend mission.

## Environment

- Backend: `uvicorn api.main:app --port 8000` from `MineralVision_Final_Package/src`,
  `SEED_DEMO=true ADMIN_INITIAL_PASSWORD=*** JWT_SECRET=*** DATABASE_URL=postgresql://...`,
  `PYTHONPATH=MineralVision_Final_Package` (required — `seed_demo_data` imports
  `src.api.auth_middleware`). Demo data seeded; login `admin`.
- Python deps installed: `sqlalchemy pyjwt prometheus_client xarray rasterio
  affine click-plugins cligj snuggs` (rest already present).
- Frontend: Vite **dev server** on :5173 (`npx vite`). The production build
  (`vite build`) is OOM-killed in this 4 GB sandbox (Cesium bundle) — see Limits.
- Browser QA: Playwright Chromium (`--enable-unsafe-swiftshader`).

## QA data seeded (via API, no fixtures)

- Feature registry: 2 drillhole Points, 2 sample Points, 1 tenement Polygon
  (lon/lat near 8.68 E, 9.08 N) via `POST /innovations/geotoolkit/tiles/features/register`.
- Raster: 20×20 grid over the same area via `POST /innovations/geotoolkit/tiles/raster/register`
  → `raster_id 3b46e05e0091` (targeting tiles verified: 256×256 PNG).
- Drillholes DB: `DH-QA-1`, `DH-QA-2` with WGS84 lon/lat collars via `POST /api/drillholes`.

## Verified (screenshots in `docs/screenshots/runtime-qa/`)

| Check | Evidence |
|---|---|
| `/health` healthy, DB ok | curl |
| Login with REAL JWT (demo shortcut bypassed) | 01-login, 02-after-login (`mineralvision-auth` holds backend user id) |
| `/map-explorer`: MapLibre canvas, layer toggles, nav/scale/attribution controls | 06-map-nigeria.png |
| Live feature tiles → circles at correct coords, counts Drillholes 2 / Samples 2 / Tenements 1 | 06-map-nigeria.png |
| Grade-colored circles (DH-001 2.4 orange, DH-002 5.1 red), cyan samples | 06-map-nigeria.png |
| Click popup with real properties | 07-map-popup.png |
| Tenement polygon fill/outline | 08-map-heatmap.png |
| Targeting heatmap raster layer attaches, tile requests fire (absolute URL fix), tiles are valid PNGs | 08-map-heatmap.png + curl |
| `/visualization`: Cesium viewer, `GET /api/drillholes` → `POST /innovations/geotoolkit/drillholes/scene` (200), collar points + traces, camera fly-to, stats overlay | 09-visualization3d.png |
| Honest notices: 2 seeded demo holes with projected collars are skipped with explanation | 09-visualization3d.png |
| No page errors / crashes on either page | Playwright console capture |

## Bugs found at runtime and fixed (commits on this branch)

1. **`abf4d9a`** — Frontend called `/api/innovations/...` but innovation routers are
   mounted at server root (verified via `/openapi.json`): 404s. Also
   `drillholes/scene` is **POST** with a `SceneHole[]` body, not GET; and the real
   response shape is `holes[].collar.position / trace_vertices / segments[].color_rgb`
   (0-1 floats) — the component was written against assumed shapes. Reworked
   `geotoolkit.ts` + `DrillholeScene3D.tsx` to the verified contract; scene request
   is built from real DB drillholes (projected collars skipped with notice).
2. **`d92b470`** — `authStore.login` hardcoded a demo shortcut for `admin`/`demo`
   that never touched the backend → fake token → every real API call 401s.
   Now: real backend login first; demo mode only when the backend is unreachable.
3. **`9f8c2f7`** — MapLibre geojson sources have **no `tiles` template support**
   (style-spec verified): layers silently failed to attach. Now the component
   fetches the viewport-covering z/x/y GeoJSON tiles via axios (auth token) and
   `setData`s them, debounced on `moveend`. Also: attach layers immediately
   (inline style object; `load`/`style.load` never fire when the basemap
   network hangs), dedupe edge features, tenement fill/outline layers, live
   feature counts, empty-viewport hint.
4. **`bd76e61`** — MapLibre raster sources reject relative tile URLs → heatmap
   never requested tiles; `heatmapTileUrl` now returns absolute URLs. Dedupe
   key ignores the backend's volatile `clipped` flag (tenement appeared twice).

## Honest limits

- **OSM tile network blocked** in this sandbox (`ERR_CONNECTION_REFUSED` for
  tile.openstreetmap.org): basemap renders black, but the app, data layers,
  popups and 3D scene all work — verified. In a networked environment the OSM
  basemap/imagery loads normally.
- **`vite build` OOM-killed** (4 GB sandbox, Cesium chunk). Build succeeded in
  the earlier frontend mission environment; QA used the Vite dev server instead.
  Consider `manualChunks` to split Cesium for lower-memory CI builds.
- **Assay segments = 0** in the 3D scene: `/api/drillholes/{id}/assays` is
  read-only (no POST), so QA drillholes have no assay intervals; the grade-colored
  segment rendering path was verified separately by POSTing a scene with assays
  directly (colored `color_rgb` segments returned as expected).
- Service worker registration fails in dev (`sw.js` MIME) — pre-existing,
  harmless, unrelated to maps.
- `window.__mvMap` debug hook left in `ExplorationMap` intentionally for QA.
