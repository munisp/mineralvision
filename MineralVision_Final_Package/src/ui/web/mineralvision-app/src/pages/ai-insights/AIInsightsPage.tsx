import { useEffect, useState } from 'react';
import { AlertTriangle, Brain, CheckCircle, Loader2, RefreshCw } from 'lucide-react';
import api from '../../services/api';

interface PredictiveModel {
  id?: string;
  name?: string;
  type?: string;
  status?: string;
  accuracy?: number;
  created_at?: string;
  createdAt?: string;
  [key: string]: unknown;
}

type LoadState = 'loading' | 'ready' | 'error';

/**
 * AI Insights page.
 *
 * The backend has NO insights feed and NO analysis-job registry (verified
 * against /openapi.json). The closest real services are the predictive-
 * modeling endpoints, so this page shows the real registered models
 * (GET /api/predictive-modeling/models) and honest empty states for the
 * insight cards and analysis jobs that were previously fabricated.
 */
export default function AIInsightsPage() {
  const [models, setModels] = useState<PredictiveModel[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const load = async () => {
    setLoadState('loading');
    setErrorMessage(null);
    try {
      const resp = await api.get<PredictiveModel[]>('/api/predictive-modeling/models');
      setModels(Array.isArray(resp.data) ? resp.data : []);
      setLoadState('ready');
    } catch (err) {
      setLoadState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to load AI models');
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Brain className="h-6 w-6" />
            AI Insights
          </h1>
          <p className="text-muted-foreground">
            AI-generated exploration insights and analysis jobs
          </p>
        </div>
        <button
          onClick={() => void load()}
          className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {loadState === 'loading' && (
        <div className="bg-card border border-border rounded-xl p-10 flex items-center justify-center gap-2">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="text-sm text-foreground">Loading AI services…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to reach AI services</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'ready' && (
        <>
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-2">Insights</h2>
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              No insights available. The backend does not yet expose an insights feed — previously
              shown example insights were removed rather than presented as real.
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-2">Analysis Jobs</h2>
            <p className="text-sm text-muted-foreground py-1">
              No analysis jobs. There is no job registry endpoint yet; training jobs can be started
              and polled via the predictive-modeling API
              (<code>/api/predictive-modeling/train</code>).
            </p>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">
              Registered Predictive Models ({models.length})
            </h2>
            {models.length === 0 ? (
              <p className="text-sm text-muted-foreground py-1">
                No models registered. Train a model via the predictive-modeling API to populate
                this list.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-muted-foreground border-b border-border">
                      <th className="py-2 pr-4">Name</th>
                      <th className="py-2 pr-4">Type</th>
                      <th className="py-2 pr-4">Status</th>
                      <th className="py-2">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.map((m, i) => (
                      <tr key={m.id ?? i} className="border-b border-border/50">
                        <td className="py-2 pr-4 text-foreground">{m.name ?? m.id ?? '—'}</td>
                        <td className="py-2 pr-4 text-foreground">{m.type ?? '—'}</td>
                        <td className="py-2 pr-4 text-foreground">{m.status ?? '—'}</td>
                        <td className="py-2 text-muted-foreground">
                          {m.created_at ?? m.createdAt ?? '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
