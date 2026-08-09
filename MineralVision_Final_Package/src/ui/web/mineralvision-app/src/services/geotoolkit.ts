import api from './api';

/**
 * Typed client for the geospatial (geotoolkit) backend endpoints.
 *
 * Backend routes (FastAPI, mounted under /api):
 *   GET /api/innovations/geotoolkit/drillholes/scene
 *       -> three.js-ready drillhole scene JSON (collars + downhole traces)
 *   GET /api/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes|samples|tenements
 *       -> GeoJSON FeatureCollection tiles for MapLibre `geojson` tile sources
 *   GET /api/innovations/geotoolkit/targeting/tiles/{raster_id}/{z}/{x}/{y}.png
 *       -> targeting heatmap raster PNG tiles
 *   GET /api/innovations/geotoolkit-ext/tenements/*
 *       -> tenement registry resources
 */

export type GeoToolkitLayer = 'drillholes' | 'samples' | 'tenements';

export interface ApiConfig {
  /**
   * Base URL of the FastAPI backend WITHOUT the /api prefix and without a
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

/** URL for the GeoJSON feature tiles used as MapLibre tile sources. */
export function featureTileUrl(layer: GeoToolkitLayer, baseUrl?: string): string {
  return `${resolveBaseUrl(baseUrl)}/api/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=${layer}`;
}

/** URL template for targeting heatmap raster PNG tiles. */
export function heatmapTileUrl(rasterId: string, baseUrl?: string): string {
  return `${resolveBaseUrl(baseUrl)}/api/innovations/geotoolkit/targeting/tiles/${encodeURIComponent(
    rasterId,
  )}/{z}/{x}/{y}.png`;
}

// ---------------------------------------------------------------------------
// Drillhole 3D scene types
// ---------------------------------------------------------------------------

/** RGB(A) color as delivered by the backend: 0-255 or 0-1 components, or hex. */
export type SceneColor = [number, number, number] | [number, number, number, number] | string;

export interface SceneCollar {
  id?: string;
  hole_id?: string;
  x: number;
  y: number;
  z: number;
  [key: string]: unknown;
}

export interface SceneTrace {
  hole_id?: string;
  /** Downhole polyline vertices: [[x, y, z], ...] */
  points?: [number, number, number][];
  vertices?: [number, number, number][];
  /** Grade-based color from the backend. */
  color?: SceneColor;
  /** Mean/representative grade used for the color ramp (display only). */
  grade?: number;
  [key: string]: unknown;
}

/** Top-level drillhole scene payload. Keys are tolerant to snake/camel case. */
export interface DrillholeScene {
  collars?: SceneCollar[];
  traces?: SceneTrace[];
  drillholes?: Array<{
    hole_id?: string;
    collar?: SceneCollar;
    trace?: SceneTrace[] | [number, number, number][];
    color?: SceneColor;
    [key: string]: unknown;
  }>;
  bounds?: {
    min_x?: number; min_y?: number; min_z?: number;
    max_x?: number; max_y?: number; max_z?: number;
  } | [number, number, number, number, number, number];
  crs?: string;
  [key: string]: unknown;
}

export const geotoolkitApi = {
  /** Fetch the three.js-ready drillhole scene JSON. */
  getDrillholeScene: () => api.get<DrillholeScene>('/api/innovations/geotoolkit/drillholes/scene'),
};
