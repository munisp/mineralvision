import { useState } from 'react';
import {
  Plus,
  Layers,
  Download,
  Settings,
  ZoomIn,
  ZoomOut,
  Move,
  RotateCcw,
} from 'lucide-react';

const mockSections = [
  { id: '1', name: 'Section A-A\'', azimuth: 45, origin: { x: 456500, y: 7654000 }, length: 1200, drillholes: 12 },
  { id: '2', name: 'Section B-B\'', azimuth: 45, origin: { x: 456600, y: 7654100 }, length: 1000, drillholes: 8 },
  { id: '3', name: 'Section C-C\'', azimuth: 135, origin: { x: 456700, y: 7654200 }, length: 800, drillholes: 6 },
  { id: '4', name: 'Section D-D\'', azimuth: 135, origin: { x: 456800, y: 7654300 }, length: 900, drillholes: 7 },
];

export default function CrossSectionsPage() {
  const [selectedSection, setSelectedSection] = useState<string | null>('1');
  const [showNewSectionModal, setShowNewSectionModal] = useState(false);
  const [zoom, setZoom] = useState(100);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Cross Sections</h1>
          <p className="text-muted-foreground">Create and view geological cross sections</p>
        </div>
        <button
          onClick={() => setShowNewSectionModal(true)}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Section
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Sections</h2>
            <div className="space-y-2">
              {mockSections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setSelectedSection(section.id)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedSection === section.id
                      ? 'bg-primary/10 border border-primary/50'
                      : 'bg-background hover:bg-secondary border border-transparent'
                  }`}
                >
                  <p className="font-medium text-foreground">{section.name}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Az: {section.azimuth} | {section.drillholes} holes
                  </p>
                </button>
              ))}
            </div>
          </div>

          {selectedSection && (
            <div className="bg-card border border-border rounded-xl p-4">
              <h2 className="text-sm font-semibold text-foreground mb-3">Section Properties</h2>
              {(() => {
                const section = mockSections.find((s) => s.id === selectedSection);
                if (!section) return null;
                return (
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Azimuth</span>
                      <span className="text-foreground">{section.azimuth}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Origin X</span>
                      <span className="text-foreground">{section.origin.x}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Origin Y</span>
                      <span className="text-foreground">{section.origin.y}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Length</span>
                      <span className="text-foreground">{section.length}m</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Drillholes</span>
                      <span className="text-foreground">{section.drillholes}</span>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Display Options</h2>
            <div className="space-y-3">
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded border-input" />
                <span className="text-sm text-foreground">Drillholes</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded border-input" />
                <span className="text-sm text-foreground">Lithology</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" defaultChecked className="rounded border-input" />
                <span className="text-sm text-foreground">Assay Grades</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded border-input" />
                <span className="text-sm text-foreground">Surfaces</span>
              </label>
              <label className="flex items-center gap-2">
                <input type="checkbox" className="rounded border-input" />
                <span className="text-sm text-foreground">Block Model</span>
              </label>
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
                <button
                  onClick={() => setZoom((z) => Math.min(z + 25, 200))}
                  className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded"
                >
                  <ZoomIn className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setZoom((z) => Math.max(z - 25, 50))}
                  className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded"
                >
                  <ZoomOut className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setZoom(100)}
                  className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded"
                >
                  <RotateCcw className="h-4 w-4" />
                </button>
                <span className="text-sm text-muted-foreground ml-2">{zoom}%</span>
              </div>
              <div className="flex items-center gap-2">
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <Settings className="h-4 w-4" />
                </button>
                <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <Download className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="h-96 lg:h-[600px] bg-slate-900 flex items-center justify-center relative overflow-hidden">
              {selectedSection ? (
                <div className="text-center">
                  <Layers className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-foreground font-medium">
                    {mockSections.find((s) => s.id === selectedSection)?.name}
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Cross section visualization would render here
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Showing drillholes, lithology intervals, and grade values
                  </p>
                </div>
              ) : (
                <div className="text-center">
                  <Layers className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">Select a section to view</p>
                </div>
              )}

              <div className="absolute bottom-4 left-4 bg-card/80 backdrop-blur-sm rounded-lg p-3 text-xs">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-2 bg-yellow-500"></div>
                    <span className="text-muted-foreground">Oxide</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-2 bg-blue-500"></div>
                    <span className="text-muted-foreground">Transition</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-2 bg-gray-500"></div>
                    <span className="text-muted-foreground">Fresh</span>
                  </div>
                </div>
              </div>

              <div className="absolute bottom-4 right-4 bg-card/80 backdrop-blur-sm rounded-lg p-2 text-xs text-muted-foreground">
                Scale: 1:{(1000 / (zoom / 100)).toFixed(0)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {showNewSectionModal && (
        <NewSectionModal onClose={() => setShowNewSectionModal(false)} />
      )}
    </div>
  );
}

function NewSectionModal({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = useState({
    name: '',
    azimuth: 0,
    originX: 0,
    originY: 0,
    length: 1000,
    width: 50,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Creating section:', formData);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Create New Section</h2>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Section Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Section E-E'"
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              required
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Azimuth</label>
              <input
                type="number"
                value={formData.azimuth}
                onChange={(e) => setFormData({ ...formData, azimuth: Number(e.target.value) })}
                min={0}
                max={360}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Length (m)</label>
              <input
                type="number"
                value={formData.length}
                onChange={(e) => setFormData({ ...formData, length: Number(e.target.value) })}
                min={100}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Origin X</label>
              <input
                type="number"
                value={formData.originX}
                onChange={(e) => setFormData({ ...formData, originX: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Origin Y</label>
              <input
                type="number"
                value={formData.originY}
                onChange={(e) => setFormData({ ...formData, originY: Number(e.target.value) })}
                className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Swath Width (m)</label>
            <input
              type="number"
              value={formData.width}
              onChange={(e) => setFormData({ ...formData, width: Number(e.target.value) })}
              min={10}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              Create Section
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
