import { useEffect, useState } from 'react';
import { Plus, AlertTriangle, Loader2, RefreshCw } from 'lucide-react';
import api, { BlockModel, projectsApi, Project } from '../../services/api';

type LoadState = 'loading' | 'ready' | 'error';

/**
 * Block Models — wired to the real geostatistics API:
 *   GET  /api/geostatistics/block-model   (list)
 *   POST /api/geostatistics/block-model   (create)
 * Replaces a fully hardcoded model list and detail panels.
 */
export default function BlockModelPage() {
  const [models, setModels] = useState<BlockModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showNewModelModal, setShowNewModelModal] = useState(false);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const load = async () => {
    setLoadState('loading');
    setErrorMessage(null);
    try {
      const resp = await api.get<BlockModel[]>('/api/geostatistics/block-model');
      const list = Array.isArray(resp.data) ? resp.data : [];
      setModels(list);
      setSelectedModel((prev) => prev ?? list[0]?.id ?? null);
      setLoadState('ready');
    } catch (err) {
      setLoadState('error');
      setErrorMessage(err instanceof Error ? err.message : 'Failed to load block models');
    }
  };

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selected = models.find((m) => m.id === selectedModel) ?? null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Block Models</h1>
          <p className="text-muted-foreground">Create and manage resource block models</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void load()}
            className="p-2 border border-input rounded-lg text-foreground hover:bg-secondary"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
          <button
            onClick={() => setShowNewModelModal(true)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            New Block Model
          </button>
        </div>
      </div>

      {loadState === 'loading' && (
        <div className="bg-card border border-border rounded-xl p-10 flex items-center justify-center gap-2">
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
          <span className="text-sm text-foreground">Loading block models…</span>
        </div>
      )}

      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <p className="font-semibold">Failed to load block models</p>
            <p>{errorMessage}</p>
          </div>
        </div>
      )}

      {loadState === 'ready' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-card border border-border rounded-xl p-4">
              <h2 className="text-sm font-semibold text-foreground mb-3">
                Block Models ({models.length})
              </h2>
              {models.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No block models yet. Create one with the button above.
                </p>
              ) : (
                <div className="space-y-2">
                  {models.map((model) => (
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
                        <span className="px-2 py-0.5 text-xs rounded-full bg-blue-500/10 text-blue-500">
                          {model.classification || 'unclassified'}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {model.cellCount.toLocaleString()} cells · {model.tonnage} Mt
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2">
            {selected ? (
              <div className="bg-card border border-border rounded-xl p-5">
                <h2 className="text-lg font-semibold text-foreground mb-4">{selected.name}</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground">Cells</p>
                    <p className="text-lg font-semibold text-foreground">
                      {selected.cellCount.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Tonnage (Mt)</p>
                    <p className="text-lg font-semibold text-foreground">{selected.tonnage}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Grade</p>
                    <p className="text-lg font-semibold text-foreground">{selected.grade}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Classification</p>
                    <p className="text-lg font-semibold text-foreground">
                      {selected.classification || '—'}
                    </p>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-4">
                  Project: {selected.projectId}
                </p>
              </div>
            ) : (
              <div className="bg-card border border-border rounded-xl p-10 text-center text-sm text-muted-foreground">
                Select a block model to view its details.
              </div>
            )}
          </div>
        </div>
      )}

      {showNewModelModal && (
        <NewBlockModelModal
          onClose={() => setShowNewModelModal(false)}
          onCreated={() => {
            setShowNewModelModal(false);
            void load();
          }}
        />
      )}
    </div>
  );
}

function NewBlockModelModal({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: () => void;
}) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    projectsApi
      .list()
      .then((r) => {
        setProjects(r.data);
        setProjectId(r.data[0]?.id ?? '');
      })
      .catch(() => setError('Could not load projects'));
  }, []);

  const submit = async () => {
    if (!projectId) {
      setError('Select a project first.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await api.post('/api/geostatistics/block-model', {
        projectId,
        origin: { x: 0, y: 0, z: 0 },
        cellSize: { x: 25, y: 25, z: 10 },
        dimensions: { nx: 20, ny: 20, nz: 10 },
      });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create block model');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md p-6 space-y-4">
        <h2 className="text-xl font-semibold text-foreground">New Block Model</h2>
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
        <p className="text-xs text-muted-foreground">
          Creates an empty 20×20×10 model (25×25×10 m cells) via the geostatistics API. Estimation
          runs separately (Kriging page).
        </p>
        {error && <p className="text-sm text-destructive">{error}</p>}
        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary text-sm"
          >
            Cancel
          </button>
          <button
            onClick={() => void submit()}
            disabled={submitting || !projectId}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm disabled:opacity-50"
          >
            {submitting ? 'Creating…' : 'Create'}
          </button>
        </div>
      </div>
    </div>
  );
}
