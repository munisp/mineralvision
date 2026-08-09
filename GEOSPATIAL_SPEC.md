# GEOSPATIAL_SPEC.md — Geospatial Stack + 10 Innovations + geoai Integration

## Global Contract (same as INNOVATIONS_SPEC)
- Backend modules at `MineralVision_Final_Package/src/api/innovations/<name>/` with `__init__.py` exporting `router = APIRouter(prefix="/innovations/<name>", tags=[...])`.
- Tests at `tests/innovations/test_<name>.py`, seeded/deterministic, assert on real numbers. NO mocks, NO skip-on-ImportError in committed tests (compute real results with available libs: numpy/scipy/shapely/pyproj/pillow/matplotlib are available; rasterio is installed).
- Dual-context imports: `try: from src.api...` / `except ImportError: from api...` when importing platform cores.
- NO edits to `main.py` — orchestrator wires routers.
- Optional heavy deps (torch/geoai/postgis/sedona): import lazily inside endpoint handlers; when unavailable return **503 with explicit detail** — never fake results.
- Commit early and often on your branch; never push; never run `git worktree prune`.

## Module A — geoai integration (`innovations/geoai`) — Agent G-A
Wraps opengeos/geoai-style workflows with honest degradation:
- `GET /capabilities` — reports which backends are importable (geoai, samgeo, torch, rasterio, shapely) with versions; never fails.
- `POST /raster/indices` — CPU-real raster analytics on an uploaded/sample array: NDVI/NDWI/iron-oxide/clay band ratios → per-pixel stats + PNG thumbnail (base64). Uses rasterio/numpy directly (real, always available).
- `POST /raster/auto-segment` — segmentation: uses geoai/samgeo if importable (lazy); otherwise a real CPU fallback = SLIC superpixels via skimage if available, else scipy ndimage label on thresholded indices. Returns region polygons as GeoJSON + area stats. Document fallback honestly in response field `backend`.
- `POST /detect/change` — bi-temporal change detection on two arrays: image differencing + Otsu threshold (skimage or numpy implementation), change mask GeoJSON; if torchgeo ChangeStar importable, offer backend="changestar".
- `POST /datasets/chips` — training chip extraction: tile a raster array into N chips with optional label mask (real numpy slicing), return manifest.
- Model registry table (SQLAlchemy, sqlite-ok) for trained model metadata.

## Module B — geodb bridge (`innovations/geodb`) — Agent G-B
Unifies API DB + lakehouse + PostGIS + Sedona:
- `POST /spatial/enable` / `GET /spatial/status` — detects dialect: if Postgres DSN configured (env `DATABASE_URL`), runs `CREATE EXTENSION IF NOT EXISTS postgis` and reports version; on sqlite reports SpatiaLite-style fallback mode (shapely-side spatial ops). Honest status.
- Geometry columns: add a `geometry_json` (GeoJSON) + centroid lat/lon + computed bbox columns approach for sqlite; geoalchemy2 `Geometry` when postgres (lazy import, 503 if postgres configured but geoalchemy2 missing).
- `POST /spatial/index/drillholes` — ingest drillhole collars from API DB into spatial index (rtree if available else sorted-grid index in pure python) → count + bbox.
- `POST /spatial/query/bbox` and `/spatial/query/near` — spatial queries over indexed entities (drillholes, samples, tenements/projects) with real results from the DB.
- `POST /lakehouse/sync` — export drillholes/samples to the lakehouse parquet storage (MineralVision_Enhanced/lakehouse_architecture parquet_storage, real write) → path + row count; `GET /lakehouse/status`.
- Sedona bridge: `GET /sedona/status` reports availability honestly; if importable expose `POST /sedona/knn` else 503.
- Update docker-compose.yml: `postgres:16` → `postgis/postgis:16-3.4` image.

## 10 Geospatial Innovations — Agents G-C1 (1–5) and G-C2 (6–10)
All under `innovations/geotoolkit/` (single module, sub-routers fine, prefix `/innovations/geotoolkit`):

1. **raster-tiles**: `GET /tiles/raster/{z}/{x}/{y}` — XYZ tile rendering from an in-request or registered raster (Web Mercator math, PIL PNG output, configurable colormap: viridis/terrain/iron-oxide). Real resampling with numpy.
2. **vector-geojson-tiles**: `GET /tiles/features/{z}/{x}/{y}` — bbox-clipped GeoJSON tile of drillholes/samples/tenements from DB (shapely clipping, Web Mercator tile bounds). (MVT optional if mapbox-vector-tile installed.)
3. **drillhole-3d**: `POST /drillholes/desurvey` — minimum-curvature desurvey (collar + survey stations: depth/azimuth/dip) → 3D trace polyline (x,y,z) + assay intervals positioned along trace; `GET /drillholes/scene` → three.js/Cesium-ready JSON scene (collars, traces, colorized by grade). Competitive gap #1.
4. **terrain-profile**: `POST /terrain/profile` — polyline → elevation profile sampled from DTM grid (bilinear), with drillhole-trace intersection markers; `POST /terrain/cross-section` → section JSON.
5. **targeting-tiles**: `POST /targeting/heatmap` + `GET /targeting/tiles/{z}/{x}/{y}` — prospectivity surface (reuse prospectivity/kriging core via dual-context import) rendered to colormap PNG tiles — AI targeting streamed to web (gaps #5+#10).
6. **spatial-overlay**: `POST /overlay/intersect|union|erase|clip` — layer overlay analysis on GeoJSON/entity layers (shapely ops), area-weighted attribute transfer (e.g., targets ∩ tenements ∩ geology) → GeoJSON + stats.
7. **change-map-service**: `POST /change/map` — wraps satellite_change_detection core; returns change polygons GeoJSON + per-class area + tile-ready raster stats. Gap #13.
8. **terrain-3d**: `POST /terrain/mesh` — DTM grid → decimated triangle mesh JSON (vertices/normals/indices + optional height-colored vertex colors) sized for three.js/Cesium; hillshade PNG option. Gap #15.
9. **tenement-guard**: `POST /tenements/check` — containment of drillholes/targets within tenement polygons (violations list), `POST /tenements/expiry-watch` — expiry date obligations & alert rules persisted to DB, `GET /tenements/alerts`. Gap #14.
10. **geo-crs-service**: `POST /crs/transform` (batch coord transform via pyproj), `GET /crs/utm-zone?lon=&lat=`, `POST /crs/detect` (infer CRS from coord ranges heuristics), `POST /geocode/grid-ref` (MGRS/grid-reference → lat/lon).

## Frontend — Agent G-D (mineralvision-app, React 18 + TS + Vite)
- Add real deps to package.json: `maplibre-gl`, `cesium` (resium optional).
- `src/components/map/ExplorationMap.tsx` — MapLibre map: OSM raster basemap style (no token), drillhole collars + tenements GeoJSON layers fetched from API (`/innovations/geotoolkit/...` endpoints), popups, layer toggle; prospectivity heatmap tile layer.
- `src/components/map/DrillholeScene3D.tsx` — Cesium viewer (Ion-free, OSM imagery) rendering drill traces + collars from `/innovations/geotoolkit/drillholes/scene`, color by grade; graceful fallback message if Cesium assets absent.
- Wire pages: replace mock map/3D sections of DrillholesPage/Visualization3DPage or add `/map-explorer` route in App.tsx.
- No fabricated data: all layers from API; show honest error state when API unreachable.
- Cannot npm-install in sandbox — code must be type-consistent; verify with `tsc --noEmit` if toolchain available, else careful manual review.

---
## STATUS (2026-08-09) — IMPLEMENTED & MERGED
- Module A geoai: `/innovations/geoai` — 7 endpoints, 17 tests
- Module B geodb: `/innovations/geodb` — 9 endpoints, 10 tests; postgis/postgis:16-3.4 in compose
- Innovations 1–5 geotoolkit: `/innovations/geotoolkit` — 13 tests
- Innovations 6–10 geotoolkit_ext: `/innovations/geotoolkit-ext` — 13 tests
- Frontend: MapLibre ExplorationMap + Cesium DrillholeScene3D, /map-explorer, docs/GEOSPATIAL_FRONTEND.md
- Total API routes: 324. Full suite: 461 passed / 3 skipped.
