# Geospatial Frontend (MineralVision)

This document describes the frontend mapping stack added on branch `gap/frontend-maps`
to `MineralVision_Final_Package/src/ui/web/mineralvision-app`.

## Stack

- **MapLibre GL JS v4** — 2D exploration map. Free OSM raster basemap, **no API tokens**.
- **Cesium ^1.144** — 3D drillhole scene viewer. Configured **Ion-free**:
  `OpenStreetMapImageryProvider` + `EllipsoidTerrainProvider`, no Cesium Ion token.
- **No Mapbox** anywhere. three.js remains declared but unused (pre-existing).

## Architecture

```
src/
  services/geotoolkit.ts        Typed client + URL builders for the geotoolkit API
  components/map/
    ExplorationMap.tsx          MapLibre map: GeoJSON tile layers + heatmap overlay
    DrillholeScene3D.tsx        Cesium viewer: collars (points) + traces (polylines)
  pages/geology/MapExplorerPage.tsx   Route /map-explorer
  pages/visualization/Visualization3DPage.tsx  Route /visualization (mock replaced)
```

### Backend endpoints consumed (all live — nothing mocked)

| Endpoint | Consumer |
|---|---|
| `GET /api/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes\|samples\|tenements` | MapLibre tiled-GeoJSON sources |
| `GET /api/innovations/geotoolkit/targeting/tiles/{raster_id}/{z}/{x}/{y}.png` | Optional raster heatmap overlay |
| `GET /api/innovations/geotoolkit/drillholes/scene` | Cesium collar points + grade-coloured trace polylines |

The app's existing axios instance (`src/services/api.ts`) supplies the base URL
(`VITE_API_URL`, falling back to same origin / the Vite dev proxy at
`localhost:8000`) and attaches the auth bearer token. MapLibre tile requests get
the same token via `transformRequest`.

### Cesium static assets (no vite-plugin-cesium)

A small inline plugin in `vite.config.ts` (`cesiumStatic()`):
- defines `CESIUM_BASE_URL = '/cesium/'`,
- serves `node_modules/cesium/Build/Cesium` under `/cesium/` in dev via middleware,
- copies the same directory to `dist/cesium/` on build.
`DrillholeScene3D` imports `cesium/Build/Cesium/Widgets/widgets.css` explicitly.

### Data contracts & defensive handling

- The drillhole scene JSON shape is normalized tolerantly (`collars[]`, `traces[]`,
  or per-`drillholes[]` entries; `points` or `vertices`; colors as 0-255 or 0-1
  RGB(A) arrays or CSS strings). Trace colours come from the backend's grade-based
  color values via `ColorMaterialProperty`.
- Coordinates are assumed WGS84 lon/lat degrees; if the scene declares a different
  `crs`, a visible warning banner is shown (reprojection is not yet implemented).
- Explicit states everywhere: loading spinner, error banner with the backend's
  message, and an "empty scene" notice when the API returns no data.

## Running

```bash
cd MineralVision_Final_Package/src/ui/web/mineralvision-app
npm install
VITE_API_URL=http://localhost:8000 npm run dev   # or rely on the /api dev proxy
```

Routes: `/map-explorer` (2D) and `/visualization` (3D) — both linked in the
sidebar under "Geology" / main nav. The 2D map has layer toggles; a targeting
heatmap can be overlaid by entering a `raster_id` in the page header.

## Verification

- `npx tsc --noEmit` — clean for all new/modified files. The only remaining
  errors are pre-existing in `CropMonitoringPage.tsx`, `JourneysPage.tsx`,
  `MineralMonitoringPage.tsx` (MUI Grid typing, untouched by this work).
- `npx tsc -p tsconfig.node.json --noEmit` — clean (added `@types/node`).
- `npx vite build` — succeeds; `dist/cesium/` contains the Cesium static assets.
- Runtime rendering against the live backend was not exercised in this
  environment (backend not running here).
