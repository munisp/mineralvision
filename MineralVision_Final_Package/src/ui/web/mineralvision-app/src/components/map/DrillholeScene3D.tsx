import { useEffect, useRef, useState } from 'react';
import {
  Cartesian3,
  Color,
  ImageryLayer,
  ColorMaterialProperty,
  ConstantProperty,
  EllipsoidTerrainProvider,
  OpenStreetMapImageryProvider,
  Rectangle,
  Viewer,
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { geotoolkitApi, DrillholeScene, SceneColor } from '../../services/geotoolkit';

export interface DrillholeScene3DProps {
  className?: string;
}

type LoadState = 'loading' | 'ready' | 'error' | 'empty';

interface NormalizedCollar {
  id: string;
  x: number;
  y: number;
  z: number;
}

interface NormalizedTrace {
  holeId: string;
  points: [number, number, number][];
  color: Color;
  grade: number | null;
}

/** Convert a backend scene color (0-255 or 0-1 RGB(A) array, or hex string) to Cesium.Color. */
function toCesiumColor(c: SceneColor | undefined, fallback: Color): Color {
  if (c === undefined) return fallback;
  if (typeof c === 'string') {
    try {
      return Color.fromCssColorString(c);
    } catch {
      return fallback;
    }
  }
  const [r, g, b] = c;
  const a = c.length === 4 ? c[3] : 1;
  const scale = Math.max(r, g, b, a) > 1 ? 255 : 1;
  return Color.fromBytes(
    Math.round(r * (scale === 255 ? 1 : 255)),
    Math.round(g * (scale === 255 ? 1 : 255)),
    Math.round(b * (scale === 255 ? 1 : 255)),
    Math.round(a * (scale === 255 ? 1 : 255)),
  );
}

/** Tolerantly normalize the flexible scene JSON into flat collar + trace lists. */
function normalizeScene(scene: DrillholeScene): {
  collars: NormalizedCollar[];
  traces: NormalizedTrace[];
} {
  const collars: NormalizedCollar[] = [];
  const traces: NormalizedTrace[] = [];

  for (const c of scene.collars ?? []) {
    if (typeof c.x === 'number' && typeof c.y === 'number') {
      collars.push({ id: String(c.hole_id ?? c.id ?? `collar-${collars.length}`), x: c.x, y: c.y, z: c.z ?? 0 });
    }
  }

  const pushTrace = (
    holeId: string,
    pts: [number, number, number][] | undefined,
    color: SceneColor | undefined,
    grade: number | undefined,
  ) => {
    if (!pts || pts.length < 2) return;
    traces.push({
      holeId,
      points: pts,
      color: toCesiumColor(color, Color.ORANGE),
      grade: typeof grade === 'number' ? grade : null,
    });
  };

  for (const t of scene.traces ?? []) {
    pushTrace(
      String(t.hole_id ?? `trace-${traces.length}`),
      t.points ?? t.vertices,
      t.color,
      t.grade,
    );
  }

  for (const d of scene.drillholes ?? []) {
    const holeId = String(d.hole_id ?? `dh-${traces.length}`);
    if (d.collar && typeof d.collar.x === 'number' && typeof d.collar.y === 'number') {
      collars.push({ id: holeId, x: d.collar.x, y: d.collar.y, z: d.collar.z ?? 0 });
    }
    if (Array.isArray(d.trace) && d.trace.length > 0) {
      if (Array.isArray(d.trace[0])) {
        // trace given directly as vertices array
        pushTrace(holeId, d.trace as [number, number, number][], d.color, undefined);
      } else {
        for (const seg of d.trace as Array<{ points?: [number, number, number][]; vertices?: [number, number, number][]; color?: SceneColor; grade?: number }>) {
          pushTrace(holeId, seg.points ?? seg.vertices, seg.color ?? d.color, seg.grade);
        }
      }
    }
  }

  return { collars, traces };
}

export default function DrillholeScene3D({ className }: DrillholeScene3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [crsWarning, setCrsWarning] = useState<string | null>(null);
  const [stats, setStats] = useState<{ collars: number; traces: number } | null>(null);

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
        const response = await geotoolkitApi.getDrillholeScene();
        if (cancelled) return;
        const scene = response.data;
        const { collars, traces } = normalizeScene(scene);

        if (collars.length === 0 && traces.length === 0) {
          setLoadState('empty');
          setStats({ collars: 0, traces: 0 });
          return;
        }

        // The scene JSON is expected to carry WGS84 lon/lat degrees. Warn (but
        // still render) when the declared CRS suggests otherwise.
        const crs = (scene.crs ?? '').toString().toLowerCase();
        if (crs && !crs.includes('4326') && !crs.includes('wgs')) {
          setCrsWarning(
            `Scene declares CRS "${scene.crs}" — coordinates are rendered as WGS84 lon/lat and may be misplaced until reprojection support is added.`,
          );
        }

        for (const collar of collars) {
          viewer.entities.add({
            id: `collar-${collar.id}`,
            name: `Collar ${collar.id}`,
            position: Cartesian3.fromDegrees(collar.x, collar.y, collar.z),
            point: {
              pixelSize: 8,
              color: Color.YELLOW,
              outlineColor: Color.BLACK,
              outlineWidth: 1,
            },
            description: `Drillhole collar: ${collar.id}`,
          });
        }

        for (const trace of traces) {
          const positions = trace.points.map(([x, y, z]) => Cartesian3.fromDegrees(x, y, z));
          viewer.entities.add({
            id: `trace-${trace.holeId}-${positions.length}`,
            name: `Trace ${trace.holeId}`,
            polyline: {
              positions,
              width: new ConstantProperty(3),
              // Grade-based colour delivered by the backend scene JSON.
              material: new ColorMaterialProperty(trace.color),
              clampToGround: false,
            },
            description:
              trace.grade !== null
                ? `Drillhole ${trace.holeId} — grade: ${trace.grade}`
                : `Drillhole ${trace.holeId}`,
          });
        }

        // Fly the camera to the data bounds.
        const xs = [...collars.map((c) => c.x), ...traces.flatMap((t) => t.points.map((p) => p[0]))];
        const ys = [...collars.map((c) => c.y), ...traces.flatMap((t) => t.points.map((p) => p[1]))];
        const west = Math.min(...xs);
        const east = Math.max(...xs);
        const south = Math.min(...ys);
        const north = Math.max(...ys);
        viewer.camera.flyTo({
          destination: Rectangle.fromDegrees(
            west - 0.01,
            south - 0.01,
            east + 0.01,
            north + 0.01,
          ),
          duration: 1.5,
        });

        setStats({ collars: collars.length, traces: traces.length });
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
        <div className="absolute inset-x-4 bottom-4 z-20 bg-card border border-border text-foreground rounded-lg p-3 text-sm">
          The drillhole scene endpoint returned no collars or traces yet. Add drillhole data via the
          backend and reload.
        </div>
      )}

      {loadState === 'ready' && stats && (
        <div className="absolute top-3 left-3 z-20 bg-card/95 border border-border rounded-lg shadow px-3 py-2 text-xs text-foreground space-y-1">
          <p>
            Collars: <strong>{stats.collars}</strong> · Traces: <strong>{stats.traces}</strong>
          </p>
          {crsWarning && <p className="text-amber-500 max-w-xs">{crsWarning}</p>}
        </div>
      )}
    </div>
  );
}
