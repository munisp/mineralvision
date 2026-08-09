import { useEffect, useRef, useState } from 'react';
import {
  Cartesian3,
  Color,
  ColorMaterialProperty,
  ConstantProperty,
  EllipsoidTerrainProvider,
  ImageryLayer,
  OpenStreetMapImageryProvider,
  Rectangle,
  Viewer,
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { fetchDrillholeSceneFromDatabase } from '../../services/geotoolkit';

export interface DrillholeScene3DProps {
  className?: string;
}

type LoadState = 'loading' | 'ready' | 'error' | 'empty';

/**
 * 3D drillhole scene rendered in Cesium.
 *
 * Data flow (verified against the live backend during runtime QA):
 *   1. Drillholes are read from the database (`GET /api/drillholes`).
 *   2. Their real collar + azimuth/dip/total-depth are POSTed to
 *      `/innovations/geotoolkit/drillholes/scene` (desurvey is server-side).
 *   3. The response `holes[]` carry `collar.position`, `trace_vertices` and
 *      assay `segments[]` with backend-computed `color_rgb` (0-1 floats).
 *
 * Traces are rendered as a neutral base polyline plus one coloured polyline
 * per assay segment (grade colour comes from the backend, not the frontend).
 */
export default function DrillholeScene3D({ className }: DrillholeScene3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [notices, setNotices] = useState<string[]>([]);
  const [stats, setStats] = useState<{ collars: number; segments: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return;
    let cancelled = false;

    const viewer = new Viewer(containerRef.current, {
      // Token-free stack: OSM imagery + smooth ellipsoid terrain. No Cesium Ion.
      baseLayer: new ImageryLayer(
        new OpenStreetMapImageryProvider({
          url: 'https://tile.openstreetmap.org/',
        }),
      ),
      terrainProvider: new EllipsoidTerrainProvider(),
      baseLayerPicker: false,
      geocoder: false,
      animation: false,
      timeline: false,
      sceneModePicker: true,
      navigationHelpButton: false,
    });
    viewerRef.current = viewer;

    (async () => {
      try {
        const { scene, skipped } = await fetchDrillholeSceneFromDatabase();
        if (cancelled) return;

        const uiNotices: string[] = [];
        if (skipped.length > 0) {
          uiNotices.push(
            `Skipped ${skipped.length} hole(s) with projected (non-lon/lat) collars: ${skipped.join(', ')}`,
          );
        }
        const crs = (scene.crs ?? '').toString().toLowerCase();
        if (crs && !crs.includes('4326') && !crs.includes('wgs')) {
          uiNotices.push(
            `Scene declares CRS "${scene.crs}" — coordinates are rendered as WGS84 lon/lat.`,
          );
        }

        const holes = scene.holes ?? [];
        if (holes.length === 0) {
          setNotices(uiNotices);
          setStats({ collars: 0, segments: 0 });
          setLoadState('empty');
          return;
        }

        let segmentCount = 0;
        const allX: number[] = [];
        const allY: number[] = [];

        for (const hole of holes) {
          const [cx, cy, cz] = hole.collar.position;
          allX.push(cx);
          allY.push(cy);

          viewer.entities.add({
            id: `collar-${hole.hole_id}`,
            name: `Collar ${hole.hole_id}`,
            position: Cartesian3.fromDegrees(cx, cy, cz),
            point: {
              pixelSize: 8,
              color: Color.YELLOW,
              outlineColor: Color.BLACK,
              outlineWidth: 1,
            },
            description: `Drillhole collar: ${hole.hole_id}`,
          });

          // Neutral base trace (full desurveyed path).
          if (hole.trace_vertices && hole.trace_vertices.length >= 2) {
            const positions = hole.trace_vertices.map(([x, y, z]) => {
              allX.push(x);
              allY.push(y);
              return Cartesian3.fromDegrees(x, y, z);
            });
            viewer.entities.add({
              id: `trace-${hole.hole_id}`,
              name: `Trace ${hole.hole_id}`,
              polyline: {
                positions,
                width: new ConstantProperty(2),
                material: new ColorMaterialProperty(Color.GRAY.withAlpha(0.7)),
                clampToGround: false,
              },
              description: `Drillhole trace: ${hole.hole_id}`,
            });
          }

          // Grade-coloured assay segments (colour from backend color_rgb).
          for (const [i, seg] of (hole.segments ?? []).entries()) {
            const [r, g, b] = seg.color_rgb;
            const color = Color.fromBytes(
              Math.round(r * 255),
              Math.round(g * 255),
              Math.round(b * 255),
            );
            viewer.entities.add({
              id: `seg-${hole.hole_id}-${i}`,
              name: `${hole.hole_id} ${seg.from_depth}-${seg.to_depth}m`,
              polyline: {
                positions: [
                  Cartesian3.fromDegrees(...seg.start),
                  Cartesian3.fromDegrees(...seg.end),
                ],
                width: new ConstantProperty(6),
                material: new ColorMaterialProperty(color),
                clampToGround: false,
              },
              description: `${hole.hole_id}: ${seg.from_depth}–${seg.to_depth} m — ${seg.element ?? 'grade'} ${seg.value}`,
            });
            segmentCount += 1;
          }
        }

        // Fly the camera to the data bounds (clamped to valid lon/lat).
        const west = Math.max(-180, Math.min(...allX));
        const east = Math.min(180, Math.max(...allX));
        const south = Math.max(-90, Math.min(...allY));
        const north = Math.min(90, Math.max(...allY));
        if (Number.isFinite(west) && Number.isFinite(south)) {
          viewer.camera.flyTo({
            destination: Rectangle.fromDegrees(
              west - 0.005,
              south - 0.005,
              east + 0.005,
              north + 0.005,
            ),
            duration: 1.5,
          });
        }

        setNotices(uiNotices);
        setStats({ collars: holes.length, segments: segmentCount });
        setLoadState('ready');
      } catch (err) {
        if (!cancelled) {
          setLoadState('error');
          setErrorMessage(
            err instanceof Error
              ? err.message
              : 'Failed to load drillhole scene from the geotoolkit API.',
          );
        }
      }
    })();

    return () => {
      cancelled = true;
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  return (
    <div className={`relative w-full h-full min-h-[400px] ${className ?? ''}`}>
      <div ref={containerRef} className="absolute inset-0 rounded-xl overflow-hidden" />

      {loadState === 'loading' && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/60">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
          <span className="ml-2 text-sm text-foreground">Loading drillhole scene…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="absolute inset-x-4 bottom-4 z-20 flex items-start gap-2 bg-destructive/10 border border-destructive/40 text-destructive rounded-lg p-3 text-sm">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load 3D drillhole scene</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'empty' && (
        <div className="absolute inset-x-4 bottom-4 z-20 bg-card border border-border text-foreground rounded-lg p-3 text-sm space-y-1">
          <p>
            No drillholes with WGS84 lon/lat collars found in the database. Add drillholes (or
            import collars in lon/lat) and reload.
          </p>
          {notices.map((n, i) => (
            <p key={i} className="text-amber-500">{n}</p>
          ))}
        </div>
      )}

      {loadState === 'ready' && stats && (
        <div className="absolute top-3 left-3 z-20 bg-card/95 border border-border rounded-lg shadow px-3 py-2 text-xs text-foreground space-y-1">
          <p>
            Holes: <strong>{stats.collars}</strong> · Assay segments: <strong>{stats.segments}</strong>
          </p>
          <p className="text-muted-foreground">
            Traces desurveyed server-side from collar azimuth/dip (straight-hole fallback).
          </p>
          {notices.map((n, i) => (
            <p key={i} className="text-amber-500 max-w-xs">{n}</p>
          ))}
        </div>
      )}
    </div>
  );
}
