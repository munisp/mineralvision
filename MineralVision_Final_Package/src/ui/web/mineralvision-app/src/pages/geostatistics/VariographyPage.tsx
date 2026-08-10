import { useEffect, useState } from 'react';
import { Play, AlertTriangle, Loader2 } from 'lucide-react';
import { Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart } from 'recharts';
import { geostatisticsApi, projectsApi, Project } from '../../services/api';

interface VariogramPoint {
  lag?: number;
  lagDistance?: number;
  gamma?: number;
  semivariance?: number;
  pairs?: number;
  nPairs?: number;
  [key: string]: unknown;
}

interface VariogramResult {
  experimental?: VariogramPoint[];
  lags?: VariogramPoint[];
  points?: VariogramPoint[];
  [key: string]: unknown;
}

/** Normalize whichever key the variogram endpoint returns. */
function extractPoints(result: VariogramResult): Array<{ lag: number; gamma: number; pairs: number }> {
  const raw = result.experimental ?? result.lags ?? result.points ?? [];
  return raw.map((p) => ({
    lag: p.lag ?? p.lagDistance ?? 0,
    gamma: p.gamma ?? p.semivariance ?? 0,
    pairs: p.pairs ?? p.nPairs ?? 0,
  }));
}

/**
 * Variography — computes the experimental variogram on demand via the real
 * backend (`POST /api/geostatistics/variogram`) for a real project. No chart
 * is shown until a real computation returns; the previous version displayed a
 * hardcoded variogram and simulated calculation with a timer.
 */
export default function VariographyPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [selectedVariable, setSelectedVariable] = useState('Au');
  const [lagDistance, setLagDistance] = useState(25);
  const [numLags, setNumLags] = useState(10);
  const [isCalculating, setIsCalculating] = useState(false);
  const [points, setPoints] = useState<Array<{ lag: number; gamma: number; pairs: number }> | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    projectsApi
      .list()
      .then((r) => {
        setProjects(r.data);
        setProjectId(r.data[0]?.id ?? '');
      })
      .catch((err) =>
        setErrorMessage(err instanceof Error ? err.message : 'Failed to load projects'),
      );
  }, []);

  const handleCalculate = async () => {
    if (!projectId) {
      setErrorMessage('Select a project first.');
      return;
    }
    setIsCalculating(true);
    setErrorMessage(null);
    try {
      const resp = await geostatisticsApi.calculateVariogram({
        projectId,
        variable: selectedVariable,
        lagDistance,
        numLags,
        directions: [0],
      });
      setPoints(extractPoints(resp.data as VariogramResult));
    } catch (err) {
      setPoints(null);
      setErrorMessage(err instanceof Error ? err.message : 'Variogram calculation failed');
    } finally {
      setIsCalculating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Variography</h1>
          <p className="text-muted-foreground">
            Calculate experimental variograms from real project data
          </p>
        </div>
        <button
          onClick={() => void handleCalculate()}
          disabled={isCalculating || !projectId}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 disabled:opacity-50"
        >
          {isCalculating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {isCalculating ? 'Calculating…' : 'Calculate'}
        </button>
      </div>

      {errorMessage && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <p>{errorMessage}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Data Selection</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Project</label>
                <select
                  value={projectId}
                  onChange={(e) => setProjectId(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm"
                >
                  {projects.length === 0 && <option value="">No projects available</option>}
                  {projects.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>
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
            </div>
          </div>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-card border border-border rounded-xl p-5 h-full min-h-[400px] flex flex-col">
            <h2 className="text-lg font-semibold text-foreground mb-4">Experimental Variogram</h2>
            {points === null ? (
              <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground text-center px-8">
                No variogram computed yet. Choose a project and variable, then press Calculate —
                the chart is rendered only from real backend results.
              </div>
            ) : points.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground text-center px-8">
                The backend returned no variogram points — the project likely has insufficient
                assay data for variable {selectedVariable}.
              </div>
            ) : (
              <div className="flex-1">
                <ResponsiveContainer width="100%" height={380}>
                  <ComposedChart data={points}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="lag" stroke="#94a3b8" label={{ value: 'Lag (m)', position: 'insideBottom', offset: -5 }} />
                    <YAxis stroke="#94a3b8" label={{ value: 'γ(h)', angle: -90, position: 'insideLeft' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      formatter={(value: number, name: string) => [value, name === 'gamma' ? 'γ(h)' : name]}
                    />
                    <Scatter dataKey="gamma" fill="#3b82f6" />
                  </ComposedChart>
                </ResponsiveContainer>
                <p className="text-xs text-muted-foreground mt-2">
                  {points.length} lag points · computed by /api/geostatistics/variogram
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
