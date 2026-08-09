import DrillholeScene3D from '../../components/map/DrillholeScene3D';

/**
 * 3D Visualization page.
 *
 * The previous implementation was a pure mock (static layer list and a
 * placeholder canvas). It now renders the real drillhole scene delivered by
 * the geotoolkit backend (`/api/innovations/geotoolkit/drillholes/scene`)
 * in a token-free Cesium viewer (OSM imagery + ellipsoid terrain).
 */
export default function Visualization3DPage() {
  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">3D Visualization</h1>
          <p className="text-muted-foreground">
            Interactive 3D drillhole scene — collars and grade-coloured traces from the geotoolkit
            API
          </p>
        </div>
      </div>

      <div className="flex-1 bg-card border border-border rounded-xl overflow-hidden min-h-[500px]">
        <DrillholeScene3D />
      </div>

      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <span>Left-drag: rotate</span>
        <span>Right-drag: pan</span>
        <span>Scroll: zoom</span>
      </div>
    </div>
  );
}
