import { useState } from 'react';
import { Play } from 'lucide-react';

export default function KrigingPage() {
  const [krigingType, setKrigingType] = useState('ordinary');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleRun = async () => {
    setIsRunning(true);
    setProgress(0);
    for (let i = 0; i <= 100; i += 10) {
      await new Promise((r) => setTimeout(r, 300));
      setProgress(i);
    }
    setIsRunning(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Kriging Estimation</h1>
          <p className="text-muted-foreground">Perform geostatistical estimation using kriging</p>
        </div>
        <button
          onClick={handleRun}
          disabled={isRunning}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 disabled:opacity-50"
        >
          <Play className="h-4 w-4" />
          {isRunning ? `Running... ${progress}%` : 'Run Kriging'}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Kriging Type</h2>
            <div className="space-y-2">
              {['ordinary', 'simple', 'universal', 'indicator'].map((type) => (
                <label key={type} className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="krigingType"
                    value={type}
                    checked={krigingType === type}
                    onChange={(e) => setKrigingType(e.target.value)}
                    className="text-primary"
                  />
                  <span className="text-sm text-foreground capitalize">{type} Kriging</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Search Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Search Radius (m)</label>
                <input type="number" defaultValue={200} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Min Samples</label>
                <input type="number" defaultValue={4} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Max Samples</label>
                <input type="number" defaultValue={16} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Max per Octant</label>
                <input type="number" defaultValue={4} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Variogram Model</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Type</span>
                <span className="text-foreground">Spherical</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Nugget</span>
                <span className="text-foreground">0.05</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Sill</span>
                <span className="text-foreground">0.63</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Range</span>
                <span className="text-foreground">180m</span>
              </div>
            </div>
            <button className="w-full mt-3 px-3 py-2 text-sm border border-input rounded-lg text-foreground hover:bg-secondary">
              Edit Variogram
            </button>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Block Model Selection</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Block Model</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>Copper Ridge - Zone A</option>
                  <option>Copper Ridge - Zone B</option>
                  <option>Golden Valley - Main</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Variable</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>Au (g/t)</option>
                  <option>Cu (%)</option>
                  <option>Ag (g/t)</option>
                </select>
              </div>
            </div>
            <div className="mt-4 p-4 bg-background rounded-lg">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Total Blocks</p>
                  <p className="text-xl font-bold text-foreground">125,000</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Blocks to Estimate</p>
                  <p className="text-xl font-bold text-foreground">98,450</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Composites</p>
                  <p className="text-xl font-bold text-foreground">4,567</p>
                </div>
              </div>
            </div>
          </div>

          {isRunning && (
            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Estimation Progress</h2>
              <div className="space-y-4">
                <div className="w-full bg-secondary rounded-full h-3">
                  <div
                    className="bg-primary h-3 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="grid grid-cols-4 gap-4 text-sm text-center">
                  <div>
                    <p className="text-muted-foreground">Blocks Estimated</p>
                    <p className="font-medium text-foreground">{Math.floor(98450 * progress / 100).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Avg Samples Used</p>
                    <p className="font-medium text-foreground">12.4</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Avg Kriging Var</p>
                    <p className="font-medium text-foreground">0.082</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Est. Time</p>
                    <p className="font-medium text-foreground">{Math.ceil((100 - progress) * 0.3)}s</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Results Summary</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 px-3 text-muted-foreground">Statistic</th>
                    <th className="text-right py-2 px-3 text-muted-foreground">Estimate</th>
                    <th className="text-right py-2 px-3 text-muted-foreground">Variance</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 text-foreground">Mean</td>
                    <td className="py-2 px-3 text-right text-foreground">1.45 g/t</td>
                    <td className="py-2 px-3 text-right text-muted-foreground">0.082</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 text-foreground">Std Dev</td>
                    <td className="py-2 px-3 text-right text-foreground">0.78 g/t</td>
                    <td className="py-2 px-3 text-right text-muted-foreground">-</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 px-3 text-foreground">CV</td>
                    <td className="py-2 px-3 text-right text-foreground">0.54</td>
                    <td className="py-2 px-3 text-right text-muted-foreground">-</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-foreground">Slope of Regression</td>
                    <td className="py-2 px-3 text-right text-foreground">0.92</td>
                    <td className="py-2 px-3 text-right text-muted-foreground">-</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
