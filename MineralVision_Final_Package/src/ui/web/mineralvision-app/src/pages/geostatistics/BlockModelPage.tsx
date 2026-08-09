import { useState } from 'react';
import { Plus, Download, Eye, Settings } from 'lucide-react';

const mockBlockModels = [
  { id: '1', name: 'Copper Ridge - Zone A', cells: 125000, tonnage: 45.2, grade: 1.45, status: 'estimated' },
  { id: '2', name: 'Copper Ridge - Zone B', cells: 89000, tonnage: 32.1, grade: 1.12, status: 'estimated' },
  { id: '3', name: 'Golden Valley - Main', cells: 156000, tonnage: 28.5, grade: 2.85, status: 'classified' },
  { id: '4', name: 'Lithium Flats - Brine', cells: 45000, tonnage: 12.8, grade: 850, status: 'draft' },
];

export default function BlockModelPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>('1');
  const [showNewModelModal, setShowNewModelModal] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Block Models</h1>
          <p className="text-muted-foreground">Create and manage resource block models</p>
        </div>
        <button
          onClick={() => setShowNewModelModal(true)}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Block Model
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Block Models</h2>
            <div className="space-y-2">
              {mockBlockModels.map((model) => (
                <button
                  key={model.id}
                  onClick={() => setSelectedModel(model.id)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    selectedModel === model.id
                      ? 'bg-primary/10 border border-primary/50'
                      : 'bg-background hover:bg-secondary border border-transparent'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-foreground">{model.name}</p>
                    <span className={`px-2 py-0.5 text-xs rounded-full ${
                      model.status === 'classified' ? 'bg-green-500/10 text-green-500' :
                      model.status === 'estimated' ? 'bg-blue-500/10 text-blue-500' :
                      'bg-yellow-500/10 text-yellow-500'
                    }`}>
                      {model.status}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {model.cells.toLocaleString()} cells | {model.tonnage} Mt
                  </p>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          {selectedModel && (() => {
            const model = mockBlockModels.find((m) => m.id === selectedModel);
            if (!model) return null;
            return (
              <>
                <div className="bg-card border border-border rounded-xl p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-foreground">{model.name}</h2>
                    <div className="flex items-center gap-2">
                      <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                        <Eye className="h-4 w-4" />
                      </button>
                      <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                        <Download className="h-4 w-4" />
                      </button>
                      <button className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                        <Settings className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-background rounded-lg">
                      <p className="text-xs text-muted-foreground">Total Cells</p>
                      <p className="text-xl font-bold text-foreground">{model.cells.toLocaleString()}</p>
                    </div>
                    <div className="p-3 bg-background rounded-lg">
                      <p className="text-xs text-muted-foreground">Tonnage</p>
                      <p className="text-xl font-bold text-foreground">{model.tonnage} Mt</p>
                    </div>
                    <div className="p-3 bg-background rounded-lg">
                      <p className="text-xs text-muted-foreground">Avg Grade</p>
                      <p className="text-xl font-bold text-foreground">{model.grade} {model.id === '4' ? 'ppm' : 'g/t'}</p>
                    </div>
                    <div className="p-3 bg-background rounded-lg">
                      <p className="text-xs text-muted-foreground">Status</p>
                      <p className="text-xl font-bold text-foreground capitalize">{model.status}</p>
                    </div>
                  </div>
                </div>

                <div className="bg-card border border-border rounded-xl p-5">
                  <h2 className="text-lg font-semibold text-foreground mb-4">Model Definition</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Origin</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">X</span>
                          <span className="text-foreground font-mono">456,000</span>
                        </div>
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">Y</span>
                          <span className="text-foreground font-mono">7,654,000</span>
                        </div>
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">Z</span>
                          <span className="text-foreground font-mono">800</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-muted-foreground mb-2">Cell Size</h3>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">X</span>
                          <span className="text-foreground font-mono">10m</span>
                        </div>
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">Y</span>
                          <span className="text-foreground font-mono">10m</span>
                        </div>
                        <div className="flex justify-between p-2 bg-background rounded">
                          <span className="text-muted-foreground">Z</span>
                          <span className="text-foreground font-mono">5m</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-card border border-border rounded-xl p-5">
                  <h2 className="text-lg font-semibold text-foreground mb-4">Resource Classification</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 px-3 text-muted-foreground">Category</th>
                          <th className="text-right py-2 px-3 text-muted-foreground">Tonnage (Mt)</th>
                          <th className="text-right py-2 px-3 text-muted-foreground">Grade (g/t)</th>
                          <th className="text-right py-2 px-3 text-muted-foreground">Metal (koz)</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-border/50">
                          <td className="py-2 px-3 text-foreground">Measured</td>
                          <td className="py-2 px-3 text-right text-foreground">12.5</td>
                          <td className="py-2 px-3 text-right text-foreground">1.82</td>
                          <td className="py-2 px-3 text-right text-foreground">732</td>
                        </tr>
                        <tr className="border-b border-border/50">
                          <td className="py-2 px-3 text-foreground">Indicated</td>
                          <td className="py-2 px-3 text-right text-foreground">22.8</td>
                          <td className="py-2 px-3 text-right text-foreground">1.45</td>
                          <td className="py-2 px-3 text-right text-foreground">1,063</td>
                        </tr>
                        <tr className="border-b border-border/50">
                          <td className="py-2 px-3 text-foreground">Inferred</td>
                          <td className="py-2 px-3 text-right text-foreground">9.9</td>
                          <td className="py-2 px-3 text-right text-foreground">1.12</td>
                          <td className="py-2 px-3 text-right text-foreground">357</td>
                        </tr>
                        <tr className="bg-primary/5">
                          <td className="py-2 px-3 font-bold text-foreground">Total</td>
                          <td className="py-2 px-3 text-right font-bold text-foreground">45.2</td>
                          <td className="py-2 px-3 text-right font-bold text-foreground">1.48</td>
                          <td className="py-2 px-3 text-right font-bold text-foreground">2,152</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            );
          })()}
        </div>
      </div>

      {showNewModelModal && (
        <NewBlockModelModal onClose={() => setShowNewModelModal(false)} />
      )}
    </div>
  );
}

function NewBlockModelModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-lg">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Create Block Model</h2>
        </div>
        <form className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Model Name</label>
            <input type="text" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Origin X</label>
              <input type="number" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Origin Y</label>
              <input type="number" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Origin Z</label>
              <input type="number" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Cell X (m)</label>
              <input type="number" defaultValue={10} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Cell Y (m)</label>
              <input type="number" defaultValue={10} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Cell Z (m)</label>
              <input type="number" defaultValue={5} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Blocks X</label>
              <input type="number" defaultValue={50} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Blocks Y</label>
              <input type="number" defaultValue={50} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Blocks Z</label>
              <input type="number" defaultValue={50} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
          </div>
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary">
              Cancel
            </button>
            <button type="submit" className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90">
              Create Model
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
