import { useState } from 'react';
import { Map as MapIcon } from 'lucide-react';
import ExplorationMap from '../../components/map/ExplorationMap';

/**
 * 2D exploration map: drillholes / samples / tenements GeoJSON tile layers on a
 * free OSM basemap, with an optional targeting heatmap raster overlay.
 * All data is loaded live from the geotoolkit backend — nothing is mocked.
 */
export default function MapExplorerPage() {
  const [rasterId, setRasterId] = useState('');
  const [activeRasterId, setActiveRasterId] = useState<string | undefined>(undefined);

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <MapIcon className="h-6 w-6" />
            Map Explorer
          </h1>
          <p className="text-muted-foreground">
            Drillholes, samples and tenements served live from the geotoolkit tile API
          </p>
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setActiveRasterId(rasterId.trim() || undefined);
          }}
        >
          <input
            value={rasterId}
            onChange={(e) => setRasterId(e.target.value)}
            placeholder="Targeting raster ID (optional)"
            className="px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm w-64"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm"
          >
            Load heatmap
          </button>
        </form>
      </div>

      <div className="flex-1 bg-card border border-border rounded-xl overflow-hidden min-h-[500px]">
        {/* Key remounts the map when the heatmap raster changes */}
        <ExplorationMap key={activeRasterId ?? 'none'} heatmapRasterId={activeRasterId} />
      </div>
    </div>
  );
}
