import { useState } from 'react';
import {
  Play,
  Download,
} from 'lucide-react';
import { Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Line, ComposedChart } from 'recharts';

const mockExperimentalVariogram = [
  { lag: 0, gamma: 0, pairs: 0 },
  { lag: 25, gamma: 0.15, pairs: 245 },
  { lag: 50, gamma: 0.28, pairs: 512 },
  { lag: 75, gamma: 0.38, pairs: 789 },
  { lag: 100, gamma: 0.45, pairs: 1024 },
  { lag: 125, gamma: 0.52, pairs: 1156 },
  { lag: 150, gamma: 0.58, pairs: 1089 },
  { lag: 175, gamma: 0.62, pairs: 945 },
  { lag: 200, gamma: 0.65, pairs: 812 },
  { lag: 225, gamma: 0.67, pairs: 678 },
  { lag: 250, gamma: 0.68, pairs: 534 },
];

const sphericalModel = (h: number, nugget: number, sill: number, range: number) => {
  if (h === 0) return 0;
  if (h >= range) return nugget + sill;
  return nugget + sill * (1.5 * (h / range) - 0.5 * Math.pow(h / range, 3));
};

export default function VariographyPage() {
  const [selectedVariable, setSelectedVariable] = useState('Au');
  const [lagDistance, setLagDistance] = useState(25);
  const [numLags, setNumLags] = useState(10);
  const [direction, setDirection] = useState({ azimuth: 0, dip: 0, tolerance: 22.5 });
  const [modelParams, setModelParams] = useState({ nugget: 0.05, sill: 0.63, range: 180 });
  const [isCalculating, setIsCalculating] = useState(false);

  const modeledVariogram = mockExperimentalVariogram.map((point) => ({
    ...point,
    model: sphericalModel(point.lag, modelParams.nugget, modelParams.sill, modelParams.range),
  }));

  const handleCalculate = async () => {
    setIsCalculating(true);
    await new Promise((resolve) => setTimeout(resolve, 1500));
    setIsCalculating(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Variography</h1>
          <p className="text-muted-foreground">Calculate and model experimental variograms</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export
          </button>
          <button
            onClick={handleCalculate}
            disabled={isCalculating}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {isCalculating ? 'Calculating...' : 'Calculate'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Data Selection</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Variable</label>
                <select
                  value={selectedVariable}
                  onChange={(e) => setSelectedVariable(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                >
                  <option value="Au">Au (g/t)</option>
                  <option value="Cu">Cu (%)</option>
                  <option value="Ag">Ag (g/t)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Project</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Copper Ridge</option>
                  <option>Golden Valley</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Domain</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>All</option>
                  <option>Oxide</option>
                  <option>Transition</option>
                  <option>Fresh</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Variogram Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Lag Distance (m)</label>
                <input
                  type="number"
                  value={lagDistance}
                  onChange={(e) => setLagDistance(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Number of Lags</label>
                <input
                  type="number"
                  value={numLags}
                  onChange={(e) => setNumLags(Number(e.target.value))}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Azimuth</label>
                <input
                  type="number"
                  value={direction.azimuth}
                  onChange={(e) => setDirection({ ...direction, azimuth: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Dip</label>
                <input
                  type="number"
                  value={direction.dip}
                  onChange={(e) => setDirection({ ...direction, dip: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Angular Tolerance</label>
                <input
                  type="number"
                  value={direction.tolerance}
                  onChange={(e) => setDirection({ ...direction, tolerance: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Model Fitting</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Model Type</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Spherical</option>
                  <option>Exponential</option>
                  <option>Gaussian</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Nugget</label>
                <input
                  type="number"
                  step="0.01"
                  value={modelParams.nugget}
                  onChange={(e) => setModelParams({ ...modelParams, nugget: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Sill</label>
                <input
                  type="number"
                  step="0.01"
                  value={modelParams.sill}
                  onChange={(e) => setModelParams({ ...modelParams, sill: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Range (m)</label>
                <input
                  type="number"
                  value={modelParams.range}
                  onChange={(e) => setModelParams({ ...modelParams, range: Number(e.target.value) })}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="bg-card border border-border rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-foreground">Experimental Variogram - {selectedVariable}</h2>
              <div className="flex items-center gap-4 text-sm">
                <span className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-blue-500"></span>
                  Experimental
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-3 h-0.5 bg-green-500"></span>
                  Model
                </span>
              </div>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={modeledVariogram}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    dataKey="lag"
                    stroke="#9ca3af"
                    fontSize={12}
                    label={{ value: 'Lag Distance (m)', position: 'bottom', fill: '#9ca3af' }}
                  />
                  <YAxis
                    stroke="#9ca3af"
                    fontSize={12}
                    label={{ value: 'Gamma', angle: -90, position: 'left', fill: '#9ca3af' }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                    formatter={(value: number, name: string) => [
                      value.toFixed(3),
                      name === 'gamma' ? 'Experimental' : 'Model',
                    ]}
                  />
                  <Scatter dataKey="gamma" fill="#3b82f6" />
                  <Line type="monotone" dataKey="model" stroke="#10b981" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-muted-foreground">Model Parameters</h3>
              <div className="mt-3 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Nugget (C0)</span>
                  <span className="text-sm font-medium text-foreground">{modelParams.nugget.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Sill (C)</span>
                  <span className="text-sm font-medium text-foreground">{modelParams.sill.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Range (a)</span>
                  <span className="text-sm font-medium text-foreground">{modelParams.range}m</span>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-muted-foreground">Statistics</h3>
              <div className="mt-3 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Total Pairs</span>
                  <span className="text-sm font-medium text-foreground">7,784</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Data Variance</span>
                  <span className="text-sm font-medium text-foreground">0.72</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Nugget Effect</span>
                  <span className="text-sm font-medium text-foreground">7.4%</span>
                </div>
              </div>
            </div>

            <div className="bg-card border border-border rounded-xl p-4">
              <h3 className="text-sm font-medium text-muted-foreground">Model Fit</h3>
              <div className="mt-3 space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">R-squared</span>
                  <span className="text-sm font-medium text-foreground">0.987</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">RMSE</span>
                  <span className="text-sm font-medium text-foreground">0.023</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <span className="text-sm font-medium text-green-500">Good Fit</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
