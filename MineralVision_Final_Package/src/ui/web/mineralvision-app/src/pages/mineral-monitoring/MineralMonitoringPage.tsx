import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, Loader2, MapPin, RefreshCw } from 'lucide-react';
import { projectsApi, Project } from '../../services/api';

type LoadState = 'loading' | 'ready' | 'error';

/**
 * Mineral Monitoring dashboard.
 *
 * The backend exposes NO mineral-monitoring endpoints (no site prospectivity
 * series, no geochemical/geophysical monitoring, no alert rules — verified
 * against /openapi.json). The previous version rendered a fully fabricated
 * dashboard: fake sites with prospectivity scores and alert badges, fake
 * high-grade-intercept alerts, fake resource estimates.
 *
 * What is real: the project register (GET /api/projects). Real projects are
 * listed below; every monitoring-specific panel shows an honest empty state
 * until a data source exists.
 */
export default function MineralMonitoringPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const load = async () => {
    setLoadState('loading');
    setErrorMessage(null);
    try {
      const resp = await projectsApi.list();
      setProjects(Array.isArray(resp.data) ? resp.data : []);
      setLoadState('ready');
    } catch (err) {
      setLoadState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to load projects');
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
          <h1 className="text-2xl font-bold text-foreground">Mineral Monitoring</h1>
          <p className="text-muted-foreground">
            Site monitoring, prospectivity tracking and alerts
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
          <span className="text-sm text-foreground">Loading projects…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load projects</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'ready' && (
        <>
          {/* Alerts — no monitoring/alert-rule backend exists */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-2">Monitoring Alerts</h2>
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-1">
              <CheckCircle className="h-4 w-4 text-green-500" />
              No alerts. The backend has no monitoring alert-rule service yet — nothing is
              fabricated here. Alerts will appear once a real alert source is connected.
            </div>
          </div>

          {/* Real projects (the only real "sites" that exist) */}
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">
              Registered Projects ({projects.length})
            </h2>
            {projects.length === 0 ? (
              <p className="text-sm text-muted-foreground py-1">
                No projects registered. Create a project to start monitoring exploration sites.
              </p>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {projects.map((p) => (
                  <div key={p.id} className="border border-border rounded-lg p-4 bg-background">
                    <div className="flex items-center gap-2 mb-1">
                      <MapPin className="h-4 w-4 text-primary" />
                      <h3 className="text-sm font-semibold text-foreground">{p.name}</h3>
                    </div>
                    <p className="text-xs text-muted-foreground">{p.location}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Commodities: {p.commodities?.join(', ') || '—'} · Status: {p.status}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Honest empty states for monitoring analytics */}
          {[
            {
              title: 'Prospectivity Tracking',
              note: 'No prospectivity time-series service exists in the backend. Connect a prospectivity/targeting data source to populate this panel.',
            },
            {
              title: 'Geochemical Monitoring',
              note: 'No geochemical monitoring stream is configured. Real assay data can be browsed under Geology → Drillholes.',
            },
            {
              title: 'Geophysical Surveys',
              note: 'No geophysical survey monitoring endpoint exists yet.',
            },
            {
              title: 'Resource Estimates',
              note: 'No resource estimation service is wired to this dashboard. Block models live under Geostatistics → Block Models.',
            },
          ].map((panel) => (
            <div key={panel.title} className="bg-card border border-border rounded-xl p-4">
              <h2 className="text-sm font-semibold text-foreground mb-2">{panel.title}</h2>
              <p className="text-sm text-muted-foreground">{panel.note}</p>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
