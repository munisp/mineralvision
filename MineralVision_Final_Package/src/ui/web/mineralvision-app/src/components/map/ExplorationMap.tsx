import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import maplibregl, { Map as MapLibreMap, MapMouseEvent, MapGeoJSONFeature } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { AlertTriangle, Loader2 } from 'lucide-react';
import api from '../../services/api';
import {
  GeoToolkitLayer,
  featureTileProbeUrl,
  featureTileZxyUrl,
  heatmapTileUrl,
} from '../../services/geotoolkit';

export interface ExplorationMapProps {
  /** Backend base URL without any path prefix (default: same origin / Vite proxy). */
  apiBaseUrl?: string;
  /** Initial visibility per layer. */
  initialLayers?: Partial<Record<GeoToolkitLayer, boolean>>;
  /** Targeting raster id; when set, a heatmap PNG tile overlay is added. */
  heatmapRasterId?: string;
  /** Attribute used for data-driven circle colouring (e.g. "grade"). */
  colorAttribute?: string;
  className?: string;
}

const ALL_LAYERS: GeoToolkitLayer[] = ['drillholes', 'samples', 'tenements'];

const LAYER_STYLE: Record<
  GeoToolkitLayer,
  { color: string; radius: number; label: string }
> = {
  drillholes: { color: '#f59e0b', radius: 6, label: 'Drillholes' },
  samples: { color: '#22d3ee', radius: 4, label: 'Samples' },
  tenements: { color: '#a78bfa', radius: 5, label: 'Tenements' },
};

/** Free OSM raster basemap style — no API tokens required. */
const OSM_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxzoom: 19,
    },
  },
  layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
};

type LoadState = 'loading' | 'ready' | 'error';

interface LayerVisibility {
  drillholes: boolean;
  samples: boolean;
  tenements: boolean;
  heatmap: boolean;
}

interface FeatureCollection {
  type: 'FeatureCollection';
  features: MapGeoJSONFeature['geometry'] extends never
    ? never[]
    : Array<{ type: 'Feature'; geometry: unknown; properties: Record<string, unknown> }>;
  [key: string]: unknown;
}

const EMPTY_FC: FeatureCollection = { type: 'FeatureCollection', features: [] };

/** Web-mercator tile indices for a lon/lat point at zoom z. */
function lonToTileX(lon: number, z: number): number {
  return Math.floor(((lon + 180) / 360) * 2 ** z);
}
function latToTileY(lat: number, z: number): number {
  const rad = (lat * Math.PI) / 180;
  return Math.floor(
    ((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * 2 ** z,
  );
}

export default function ExplorationMap({
  apiBaseUrl,
  initialLayers,
  heatmapRasterId,
  colorAttribute = 'grade',
  className,
}: ExplorationMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const loadingRef = useRef(0);

  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [dataCounts, setDataCounts] = useState<Record<GeoToolkitLayer, number>>({
    drillholes: 0,
    samples: 0,
    tenements: 0,
  });
  const [visibility, setVisibility] = useState<LayerVisibility>({
    drillholes: initialLayers?.drillholes ?? true,
    samples: initialLayers?.samples ?? true,
    tenements: initialLayers?.tenements ?? true,
    heatmap: true,
  });

  const hasHeatmap = Boolean(heatmapRasterId);

  const circleColorExpr = useCallback(
    (fallback: string): maplibregl.ExpressionSpecification =>
      [
        'case',
        ['has', colorAttribute],
        [
          'interpolate',
          ['linear'],
          ['coalesce', ['to-number', ['get', colorAttribute]], 0],
          0,
          '#2c7bb6',
          0.5,
          '#abd9e9',
          1,
          '#ffffbf',
          2.5,
          '#fdae61',
          5,
          '#d7191c',
        ],
        fallback,
      ] as unknown as maplibregl.ExpressionSpecification,
    [colorAttribute],
  );

  const applyVisibility = useCallback((map: MapLibreMap, vis: LayerVisibility) => {
    for (const layer of ALL_LAYERS) {
      const layerId = `geotoolkit-${layer}`;
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, 'visibility', vis[layer] ? 'visible' : 'none');
      }
    }
    if (map.getLayer('geotoolkit-heatmap')) {
      map.setLayoutProperty('geotoolkit-heatmap', 'visibility', vis.heatmap ? 'visible' : 'none');
    }
  }, []);

  const toggleLayer = useCallback(
    (key: keyof LayerVisibility) => {
      setVisibility((prev) => {
        const next = { ...prev, [key]: !prev[key] };
        if (mapRef.current) applyVisibility(mapRef.current, next);
        return next;
      });
    },
    [applyVisibility],
  );

  const heatmapTiles = useMemo(
    () => (heatmapRasterId ? heatmapTileUrl(heatmapRasterId, apiBaseUrl) : null),
    [heatmapRasterId, apiBaseUrl],
  );

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    let cancelled = false;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: OSM_STYLE,
      center: [134.0, -25.0],
      zoom: 3,
    });
    mapRef.current = map;
    // Debug/QA hook: inspect the live map from the console.
    (window as unknown as { __mvMap?: MapLibreMap }).__mvMap = map;
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');

    /**
     * Fetch the geotoolkit feature tiles covering the current viewport and
     * merge them into each layer's GeoJSON source. (The backend serves
     * GeoJSON z/x/y tiles; MapLibre geojson sources do not support tile
     * templates, so we load them explicitly — via the axios client, which
     * carries the auth token.)
     */
    const loadViewportData = async () => {
      const loadId = ++loadingRef.current;
      const bounds = map.getBounds();
      const z = Math.min(10, Math.max(4, Math.floor(map.getZoom())));
      const x0 = Math.max(0, lonToTileX(bounds.getWest(), z));
      const x1 = Math.min(2 ** z - 1, lonToTileX(bounds.getEast(), z));
      const y0 = Math.max(0, latToTileY(bounds.getNorth(), z));
      const y1 = Math.min(2 ** z - 1, latToTileY(bounds.getSouth(), z));
      // Cap the number of tiles per request burst at low zooms.
      if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return;

      const counts: Record<GeoToolkitLayer, number> = { drillholes: 0, samples: 0, tenements: 0 };
      for (const layer of ALL_LAYERS) {
        const merged: FeatureCollection = { type: 'FeatureCollection', features: [] };
        for (let x = x0; x <= x1; x++) {
          for (let y = y0; y <= y1; y++) {
            try {
              const resp = await api.get<FeatureCollection>(
                featureTileZxyUrl(layer, z, x, y, apiBaseUrl),
              );
              if (cancelled || loadId !== loadingRef.current) return;
              const features = Array.isArray(resp.data?.features) ? resp.data.features : [];
              merged.features.push(...features);
            } catch {
              // Individual tile failures are non-fatal; other tiles still render.
            }
          }
        }
        // Dedupe: a feature near a tile edge appears in multiple tiles (the
        // backend marks clipped copies with a volatile `clipped` flag, which
        // is excluded from the identity key).
        const seen = new Set<string>();
        merged.features = merged.features.filter((f) => {
          const { clipped: _clipped, ...stableProps } = f.properties ?? {};
          const key = JSON.stringify([
            (f.geometry as { type?: string })?.type,
            stableProps,
          ]);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        counts[layer] = merged.features.length;
        const source = map.getSource(`geotoolkit-src-${layer}`) as maplibregl.GeoJSONSource | undefined;
        source?.setData(merged as GeoJSON.FeatureCollection);
      }
      setDataCounts(counts);
    };

    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    const scheduleLoad = () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => void loadViewportData(), 300);
    };

    const setupLayers = async () => {
      try {
        // Probe the tile API once so backend/auth failures surface explicitly.
        const probes = ALL_LAYERS.map((layer) =>
          api.get(featureTileProbeUrl(layer, apiBaseUrl), {
            validateStatus: (s) => s < 500,
          }),
        );
        const results = await Promise.allSettled(probes);
        if (cancelled) return;
        const failed = results.find(
          (r) => r.status === 'rejected' || (r.status === 'fulfilled' && r.value.status >= 400),
        );
        if (failed) {
          const msg =
            failed.status === 'rejected'
              ? `Could not reach geotoolkit tile API: ${
                  failed.reason instanceof Error ? failed.reason.message : String(failed.reason)
                }`
              : `Geotoolkit tile API returned HTTP ${failed.value.status}`;
          setLoadState('error');
          setErrorMessage(msg);
          return;
        }

        for (const layer of ALL_LAYERS) {
          const sourceId = `geotoolkit-src-${layer}`;
          map.addSource(sourceId, {
            type: 'geojson',
            data: EMPTY_FC as GeoJSON.FeatureCollection,
          });
          map.addLayer({
            id: `geotoolkit-${layer}`,
            type: 'circle',
            source: sourceId,
            filter: ['==', ['geometry-type'], 'Point'],
            paint: {
              'circle-radius': LAYER_STYLE[layer].radius,
              'circle-color': circleColorExpr(LAYER_STYLE[layer].color),
              'circle-stroke-color': '#0f172a',
              'circle-stroke-width': 1,
              'circle-opacity': 0.85,
            },
          });

          if (layer === 'tenements') {
            // Tenements are polygons: render a fill + outline beneath the points.
            map.addLayer({
              id: 'geotoolkit-tenements-fill',
              type: 'fill',
              source: sourceId,
              filter: ['==', ['geometry-type'], 'Polygon'],
              paint: { 'fill-color': '#a78bfa', 'fill-opacity': 0.15 },
            });
            map.addLayer({
              id: 'geotoolkit-tenements-outline',
              type: 'line',
              source: sourceId,
              filter: ['==', ['geometry-type'], 'Polygon'],
              paint: { 'line-color': '#a78bfa', 'line-width': 2 },
            });
          }

          map.on(
            'click',
            `geotoolkit-${layer}`,
            (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
              const feature = e.features?.[0];
              if (!feature) return;
              popupRef.current?.remove();
              const rows = Object.entries(feature.properties ?? {})
                .map(
                  ([k, v]) =>
                    `<tr><td style="padding:1px 6px 1px 0;color:#64748b">${k}</td><td>${String(
                      v,
                    )}</td></tr>`,
                )
                .join('');
              popupRef.current = new maplibregl.Popup({ maxWidth: '320px' })
                .setLngLat(e.lngLat)
                .setHTML(
                  `<div style="font:12px/1.4 sans-serif"><strong>${LAYER_STYLE[layer].label}</strong><table>${rows}</table></div>`,
                )
                .addTo(map);
            },
          );
          map.on('mouseenter', `geotoolkit-${layer}`, () => {
            map.getCanvas().style.cursor = 'pointer';
          });
          map.on('mouseleave', `geotoolkit-${layer}`, () => {
            map.getCanvas().style.cursor = '';
          });
        }

        if (heatmapTiles) {
          map.addSource('geotoolkit-heatmap-src', {
            type: 'raster',
            tiles: [heatmapTiles],
            tileSize: 256,
          });
          map.addLayer(
            {
              id: 'geotoolkit-heatmap',
              type: 'raster',
              source: 'geotoolkit-heatmap-src',
              paint: { 'raster-opacity': 0.7 },
            },
            `geotoolkit-${ALL_LAYERS[0]}`,
          );
        }

        applyVisibility(map, visibility);
        setLoadState('ready');
        setErrorMessage(null);
        // Initial data load + reload on viewport changes (debounced).
        void loadViewportData();
        map.on('moveend', scheduleLoad);
      } catch (err) {
        if (!cancelled) {
          setLoadState('error');
          setErrorMessage(err instanceof Error ? err.message : 'Failed to initialise map layers');
        }
      }
    };

    // The style is an inline object, so sources can be added immediately.
    void setupLayers();

    return () => {
      cancelled = true;
      if (debounceTimer) clearTimeout(debounceTimer);
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBaseUrl, heatmapTiles]);

  const totalFeatures = dataCounts.drillholes + dataCounts.samples + dataCounts.tenements;

  return (
    <div className={`relative w-full h-full min-h-[400px] ${className ?? ''}`}>
      <div ref={containerRef} className="absolute inset-0 rounded-xl overflow-hidden" />

      {/* Layer toggle control */}
      <div className="absolute top-3 left-3 z-10 bg-card/95 border border-border rounded-lg shadow p-3 space-y-2">
        <p className="text-xs font-semibold text-foreground uppercase tracking-wide">Layers</p>
        {ALL_LAYERS.map((layer) => (
          <label
            key={layer}
            className="flex items-center gap-2 text-sm text-foreground cursor-pointer"
          >
            <input
              type="checkbox"
              checked={visibility[layer]}
              onChange={() => toggleLayer(layer)}
              className="accent-primary"
            />
            <span
              className="inline-block w-3 h-3 rounded-full"
              style={{ backgroundColor: LAYER_STYLE[layer].color }}
            />
            {LAYER_STYLE[layer].label}
            {loadState === 'ready' && (
              <span className="text-xs text-muted-foreground">({dataCounts[layer]})</span>
            )}
          </label>
        ))}
        {hasHeatmap && (
          <label className="flex items-center gap-2 text-sm text-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={visibility.heatmap}
              onChange={() => toggleLayer('heatmap')}
              className="accent-primary"
            />
            <span className="inline-block w-3 h-3 rounded bg-gradient-to-r from-blue-500 via-yellow-400 to-red-500" />
            Targeting heatmap
          </label>
        )}
        {loadState === 'ready' && totalFeatures === 0 && (
          <p className="text-xs text-muted-foreground max-w-[180px]">
            No features in this viewport — pan/zoom to your project area or register features via
            the geotoolkit API.
          </p>
        )}
      </div>

      {loadState === 'loading' && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/60">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <span className="ml-2 text-sm text-foreground">Loading geospatial layers…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="absolute inset-x-4 bottom-4 z-20 flex items-start gap-2 bg-destructive/10 border border-destructive/40 text-destructive rounded-lg p-3 text-sm">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load map layers</p>
            <p>{errorMessage ?? 'Unknown error contacting the geotoolkit API.'}</p>
          </div>
        </div>
      )}
    </div>
  );
}
