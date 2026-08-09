import { useState, useRef } from 'react';
import {
  Eye,
  EyeOff,
  Layers,
  Box,
  Database,
  Mountain,
  Download,
  Settings,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  Maximize2,
} from 'lucide-react';

interface LayerItem {
  id: string;
  name: string;
  type: 'drillholes' | 'surface' | 'blockmodel' | 'section';
  visible: boolean;
  opacity: number;
}

const mockLayers: LayerItem[] = [
  { id: '1', name: 'Drillholes', type: 'drillholes', visible: true, opacity: 100 },
  { id: '2', name: 'Topography', type: 'surface', visible: true, opacity: 80 },
  { id: '3', name: 'Block Model - Zone A', type: 'blockmodel', visible: true, opacity: 70 },
  { id: '4', name: 'Ore Shell 0.5 g/t', type: 'surface', visible: false, opacity: 60 },
  { id: '5', name: 'Section A-A\'', type: 'section', visible: false, opacity: 100 },
];

const layerIcons = {
  drillholes: Database,
  surface: Mountain,
  blockmodel: Box,
  section: Layers,
};

export default function Visualization3DPage() {
  const [layers, setLayers] = useState(mockLayers);
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'perspective' | 'top' | 'front' | 'side'>('perspective');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  const toggleLayerVisibility = (id: string) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === id ? { ...layer, visible: !layer.visible } : layer
      )
    );
  };

  const updateLayerOpacity = (id: string, opacity: number) => {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === id ? { ...layer, opacity } : layer
      )
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">3D Visualization</h1>
          <p className="text-muted-foreground">Interactive 3D view of geological data</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export Image
          </button>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Layers</h2>
            <div className="space-y-2">
              {layers.map((layer) => {
                const Icon = layerIcons[layer.type];
                return (
                  <div
                    key={layer.id}
                    className={`p-3 rounded-lg transition-colors ${
                      selectedLayer === layer.id
                        ? 'bg-primary/10 border border-primary/50'
                        : 'bg-background hover:bg-secondary border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => toggleLayerVisibility(layer.id)}
                        className="text-muted-foreground hover:text-foreground"
                      >
                        {layer.visible ? (
                          <Eye className="h-4 w-4" />
                        ) : (
                          <EyeOff className="h-4 w-4" />
                        )}
                      </button>
                      <Icon className="h-4 w-4 text-muted-foreground" />
                      <button
                        onClick={() => setSelectedLayer(layer.id)}
                        className="flex-1 text-left text-sm text-foreground"
                      >
                        {layer.name}
                      </button>
                    </div>
                    {selectedLayer === layer.id && (
                      <div className="mt-3 pt-3 border-t border-border">
                        <label className="block text-xs text-muted-foreground mb-1">
                          Opacity: {layer.opacity}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          value={layer.opacity}
                          onChange={(e) => updateLayerOpacity(layer.id, Number(e.target.value))}
                          className="w-full"
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">View Options</h2>
            <div className="grid grid-cols-2 gap-2">
              {(['perspective', 'top', 'front', 'side'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-2 text-sm rounded-lg capitalize ${
                    viewMode === mode
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-background text-foreground hover:bg-secondary'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Color Scale</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Variable</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Au (g/t)</option>
                  <option>Cu (%)</option>
                  <option>Lithology</option>
                </select>
              </div>
              <div className="h-4 rounded bg-gradient-to-r from-blue-500 via-yellow-500 to-red-500" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>0.0</span>
                <span>2.5</span>
                <span>5.0</span>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-card border border-border rounded-xl overflow-hidden">
            <div className="flex items-center justify-between p-3 border-b border-border">
              <div className="flex items-center gap-2">
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <Move className="h-4 w-4" />
                </button>
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <ZoomIn className="h-4 w-4" />
                </button>
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <ZoomOut className="h-4 w-4" />
                </button>
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <RotateCcw className="h-4 w-4" />
                </button>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground capitalize">{viewMode} View</span>
                <button
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded"
                >
                  <Maximize2 className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div
              ref={canvasRef}
              className={`bg-slate-900 flex items-center justify-center ${
                isFullscreen ? 'fixed inset-0 z-50' : 'h-96 lg:h-[600px]'
              }`}
            >
              <div className="text-center">
                <Box className="h-20 w-20 text-muted-foreground mx-auto mb-4" />
                <p className="text-foreground font-medium">3D Visualization Engine</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Interactive 3D rendering would appear here
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Powered by Three.js / React Three Fiber
                </p>
                <div className="mt-6 flex items-center justify-center gap-4 text-xs text-muted-foreground">
                  <span>Left-click: Rotate</span>
                  <span>Right-click: Pan</span>
                  <span>Scroll: Zoom</span>
                </div>
              </div>

              {isFullscreen && (
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="absolute top-4 right-4 p-2 bg-card/80 rounded-lg text-foreground hover:bg-card"
                >
                  Exit Fullscreen
                </button>
              )}
            </div>

            <div className="p-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-4">
                <span>Objects: 156</span>
                <span>Triangles: 1.2M</span>
                <span>FPS: 60</span>
              </div>
              <div>
                Camera: X: 456,500 Y: 7,654,200 Z: 1,500
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
