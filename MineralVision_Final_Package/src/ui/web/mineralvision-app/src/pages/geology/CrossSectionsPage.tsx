import { useEffect, useState } from 'react';
import { AlertTriangle, Layers, Loader2, RefreshCw } from 'lucide-react';
import { drillholesApi, projectsApi, Project } from '../../services/api';

interface ProjectHoleCount {
  project: Project;
  holes: number;
}

type LoadState = 'loading' | 'ready' | 'error';

/**
 * Cross Sections page.
 *
 * The backend has no saved-section registry: the only cross-section service
 * (`POST /innovations/geotoolkit/terrain/cross-section`) is a compute endpoint
 * that requires a DTM grid as input. The previous version showed four
 * hardcoded sections with fake drillhole counts — removed.
 *
 * Real data shown here: the drillhole register grouped by project (the input
 * a geologist would section). The section browser itself is an honest empty
 * state until a section-definition/DTM data source exists.
 */
export default function CrossSectionsPage() {
  const [counts, setCounts] = useState<ProjectHoleCount[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const load = async () => {
    setLoadState('loading');
    setErrorMessage(null);
    try {
      const [projectsResp, holesResp] = await Promise.all([
        projectsApi.list(),
        drillholesApi.list(),
      ]);
      const byProject = new Map<string, number>();
      for (const h of holesResp.data) {
        byProject.set(h.projectId, (byProject.get(h.projectId) ?? 0) + 1);
      }
      setCounts(projectsResp.data.map((p) => ({ project: p, holes: byProject.get(p.id) ?? 0 })));
      setLoadState('ready');
    } catch (err) {
      setLoadState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to load drillhole data');
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
          <h1 className="text-2xl font-bold text-foreground">Cross Sections</h1>
          <p className="text-muted-foreground">Create and view geological cross sections</p>
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
          <span className="text-sm text-foreground">Loading drillhole data…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load drillhole data</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'ready' && (
        <>
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Saved Sections
            </h2>
            <p className="text-sm text-muted-foreground">
              No saved sections. The backend does not yet persist section definitions, and the
              terrain cross-section service requires a DTM grid input — connect a DTM/section data
              source to enable section rendering.
            </p>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">
              Drillholes available for sectioning (real register)
            </h2>
            {counts.length === 0 ? (
              <p className="text-sm text-muted-foreground">No projects registered.</p>
            ) : (
              <div className="space-y-2">
                {counts.map(({ project, holes }) => (
                  <div
                    key={project.id}
                    className="flex items-center justify-between p-3 rounded-lg bg-background border border-transparent"
                  >
                    <div>
                      <p className="font-medium text-foreground">{project.name}</p>
                      <p className="text-xs text-muted-foreground">{project.location}</p>
                    </div>
                    <span className="text-sm text-foreground">{holes} hole(s)</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
