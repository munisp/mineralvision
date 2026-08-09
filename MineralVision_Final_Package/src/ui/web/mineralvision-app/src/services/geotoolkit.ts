import api, { drillholesApi, Drillhole } from './api';

/**
 * Typed client for the geospatial (geotoolkit) backend endpoints.
 *
 * Backend routes (FastAPI). NOTE: the innovations routers are mounted at the
 * server root WITHOUT the /api prefix (verified against /openapi.json):
 *   POST /innovations/geotoolkit/drillholes/scene      {holes: SceneHole[], step?}
 *   GET  /innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes|samples|tenements
 *   POST /innovations/geotoolkit/tiles/features/register
 *   POST /innovations/geotoolkit/tiles/raster/register
 *   GET  /innovations/geotoolkit/targeting/tiles/{raster_id}/{z}/{x}/{y}
 *   POST /innovations/geotoolkit/targeting/heatmap
 */

export type GeoToolkitLayer = 'drillholes' | 'samples' | 'tenements';

export interface ApiConfig {
  /**
   * Base URL of the FastAPI backend WITHOUT any path prefix and without a
   * trailing slash. Defaults to the same origin when served behind the Vite
   * dev proxy / a reverse proxy.
   */
  baseUrl?: string;
}

const RAW_BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? '';

/** Resolve the effective backend base URL (empty string = same origin/proxy). */
export function resolveBaseUrl(baseUrl?: string): string {
  const base = (baseUrl ?? RAW_BASE_URL).replace(/\/+$/, '');
  return base;
}

/** URL for one concrete GeoJSON feature tile. */
export function featureTileZxyUrl(
  layer: GeoToolkitLayer,
  z: number,
  x: number,
  y: number,
  baseUrl?: string,
): string {
  return `${resolveBaseUrl(baseUrl)}/innovations/geotoolkit/tiles/features/${z}/${x}/${y}?layer=${layer}`;
}

/** Same URL with concrete z/x/y — used for availability probes. */
export function featureTileProbeUrl(layer: GeoToolkitLayer, baseUrl?: string): string {
  return `${resolveBaseUrl(baseUrl)}/innovations/geotoolkit/tiles/features/0/0/0?layer=${layer}`;
}

/** URL template for targeting heatmap raster PNG tiles. */
export function heatmapTileUrl(rasterId: string, baseUrl?: string): string {
  return `${resolveBaseUrl(baseUrl)}/innovations/geotoolkit/targeting/tiles/${encodeURIComponent(
    rasterId,
  )}/{z}/{x}/{y}`;
}

// ---------------------------------------------------------------------------
// Drillhole 3D scene types (match the backend's actual response shape)
// ---------------------------------------------------------------------------

export interface SceneCollarInput {
  easting: number;
  northing: number;
  elevation?: number;
}

export interface SceneSurveyStation {
  depth: number;
  azimuth: number;
  dip: number;
}

export interface SceneAssayInterval {
  from_depth: number;
  to_depth: number;
  value: number;
  element?: string;
}

export interface SceneHoleInput {
  hole_id: string;
  collar: SceneCollarInput;
  survey: SceneSurveyStation[];
  assays?: SceneAssayInterval[];
}

export interface SceneSegment {
  from_depth: number;
  to_depth: number;
  value: number;
  element?: string;
  start: [number, number, number];
  end: [number, number, number];
  /** Grade-based colour, 0-1 RGB floats (verified against live backend). */
  color_rgb: [number, number, number];
}

/** One hole of the DrillholeScene response. */
export interface SceneHoleOut {
  hole_id: string;
  collar: { position: [number, number, number] };
  trace_vertices: [number, number, number][];
  segments?: SceneSegment[];
  grade_range?: [number, number];
}

/** Actual top-level response of POST /drillholes/scene. */
export interface DrillholeScene {
  type?: string;
  coordinate_order?: string[];
  n_holes?: number;
  holes?: SceneHoleOut[];
  crs?: string;
  [key: string]: unknown;
}

export const geotoolkitApi = {
  /** POST the scene request and return the desurveyed drillhole scene. */
  postDrillholeScene: (holes: SceneHoleInput[], step = 10) =>
    api.post<DrillholeScene>('/innovations/geotoolkit/drillholes/scene', { holes, step }),
};

/** Returns true when a collar looks like WGS84 lon/lat degrees. */
function isLonLat(x: number, y: number): boolean {
  return Math.abs(x) <= 180 && Math.abs(y) <= 90;
}

/**
 * Build a scene request from the drillhole database. Collars, azimuth, dip and
 * total depth all come from real DB records — the survey is the collar
 * orientation held constant over the hole depth (straight-hole assumption is
 * the standard desurvey fallback and is documented in the UI).
 *
 * Holes whose collars are clearly projected (not lon/lat degrees) are skipped
 * and reported, since the geotoolkit scene consumer renders in WGS84.
 */
export async function fetchDrillholeSceneFromDatabase(): Promise<{
  scene: DrillholeScene;
  skipped: string[];
}> {
  const response = await drillholesApi.list();
  const drillholes: Drillhole[] = response.data;
  const skipped: string[] = [];
  const holes: SceneHoleInput[] = [];

  for (const d of drillholes) {
    const { x, y, z } = d.collar;
    if (!isLonLat(x, y)) {
      skipped.push(`${d.holeId} (projected collar ${x}, ${y})`);
      continue;
    }
    const azimuth = typeof d.azimuth === 'number' ? d.azimuth : 0;
    const dip = typeof d.dip === 'number' && d.dip !== 0 ? d.dip : -90;
    holes.push({
      hole_id: d.holeId,
      collar: { easting: x, northing: y, elevation: z },
      survey: [
        { depth: 0, azimuth, dip },
        { depth: d.totalDepth, azimuth, dip },
      ],
    });
  }

  if (holes.length === 0) {
    return { scene: { type: 'DrillholeScene', n_holes: 0, holes: [] }, skipped };
  }

  const sceneResponse = await geotoolkitApi.postDrillholeScene(holes);
  return { scene: sceneResponse.data, skipped };
}
