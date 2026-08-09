import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import maplibregl, { Map as MapLibreMap, MapMouseEvent, MapGeoJSONFeature } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { AlertTriangle, Loader2 } from 'lucide-react';
import api from '../../services/api';
import { useAuthStore } from '../../store/authStore';
import {
  GeoToolkitLayer,
  featureTileProbeUrl,
  featureTileUrl,
  heatmapTileUrl,
  resolveBaseUrl,
} from '../../services/geotoolkit';

export interface ExplorationMapProps {
  /** Backend base URL without /api prefix (default: same origin / Vite proxy). */
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
  layers: [
    { id: 'osm', type: 'raster', source: 'osm' },
  ],
};

type LoadState = 'loading' | 'ready' | 'error';

interface LayerVisibility {
  drillholes: boolean;
  samples: boolean;
  tenements: boolean;
  heatmap: boolean;
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

  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
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

  const applyVisibility = useCallback(
    (map: MapLibreMap, vis: LayerVisibility) => {
      for (const layer of ALL_LAYERS) {
        const layerId = `geotoolkit-${layer}`;
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', vis[layer] ? 'visible' : 'none');
        }
      }
      if (map.getLayer('geotoolkit-heatmap')) {
        map.setLayoutProperty(
          'geotoolkit-heatmap',
          'visibility',
          vis.heatmap ? 'visible' : 'none',
        );
      }
    },
    [],
  );

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
    const token = useAuthStore.getState().token;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: OSM_STYLE,
      center: [134.0, -25.0],
      zoom: 3,
      // Forward the app's auth token on tile requests so protected backends work.
      transformRequest: (url) => {
        const base = resolveBaseUrl(apiBaseUrl) || window.location.origin;
        if (token && url.startsWith(base)) {
          return { url, headers: { Authorization: `Bearer ${token}` } };
        }
        return { url };
      },
    });
    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    map.addControl(new maplibregl.ScaleControl(), 'bottom-left');
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right');

    map.on('error', (e) => {
      // Surface tile/source errors honestly instead of a silently blank map.
      const message = e.error?.message ?? 'Map tile or source error';
      // Ignore aborted requests during teardown.
      if (!/abort/i.test(message)) {
        setLoadState((s) => (s === 'ready' ? s : 'error'));
        setErrorMessage(message);
      }
    });

    map.on('load', async () => {
      if (cancelled) return;
      try {
        // Probe the first tile of each enabled layer via the axios client so
        // backend/auth failures produce an explicit error state up front.
        const probes = ALL_LAYERS.map((layer) =>
          api.get(featureTileProbeUrl(layer, apiBaseUrl), {
            validateStatus: (s) => s < 500,
          }),
        );
        const results = await Promise.allSettled(probes);
        const failed = results.find(
          (r) => r.status === 'rejected' || (r.status === 'fulfilled' && r.value.status >= 400),
        );
        if (cancelled) return;
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
          // MapLibre supports tiled GeoJSON sources; the published types do
          // not yet declare `tiles` on geojson sources, hence the cast.
          map.addSource(sourceId, {
            type: 'geojson',
            tiles: [featureTileUrl(layer, apiBaseUrl)],
          } as unknown as maplibregl.GeoJSONSourceSpecification);
          map.addLayer({
            id: `geotoolkit-${layer}`,
            type: 'circle',
            source: sourceId,
            paint: {
              'circle-radius': LAYER_STYLE[layer].radius,
              'circle-color': circleColorExpr(LAYER_STYLE[layer].color),
              'circle-stroke-color': '#0f172a',
              'circle-stroke-width': 1,
              'circle-opacity': 0.85,
            },
          });

          map.on('click', `geotoolkit-${layer}`, (e: MapMouseEvent & { features?: MapGeoJSONFeature[] }) => {
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
          });
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
          // Insert the heatmap beneath the vector circle layers.
          const firstVector = `geotoolkit-${ALL_LAYERS[0]}`;
          map.addLayer(
            {
              id: 'geotoolkit-heatmap',
              type: 'raster',
              source: 'geotoolkit-heatmap-src',
              paint: { 'raster-opacity': 0.7 },
            },
            firstVector,
          );
        }

        applyVisibility(map, visibility);
        setLoadState('ready');
        setErrorMessage(null);
      } catch (err) {
        if (!cancelled) {
          setLoadState('error');
          setErrorMessage(err instanceof Error ? err.message : 'Failed to initialise map layers');
        }
      }
    });

    return () => {
      cancelled = true;
      popupRef.current?.remove();
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBaseUrl, heatmapTiles]);

  return (
    <div className={`relative w-full h-full min-h-[400px] ${className ?? ''}`}>
      <div ref={containerRef} className="absolute inset-0 rounded-xl overflow-hidden" />

      {/* Layer toggle control */}
      <div className="absolute top-3 left-3 z-10 bg-card/95 border border-border rounded-lg shadow p-3 space-y-2">
        <p className="text-xs font-semibold text-foreground uppercase tracking-wide">Layers</p>
        {ALL_LAYERS.map((layer) => (
          <label key={layer} className="flex items-center gap-2 text-sm text-foreground cursor-pointer">
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
