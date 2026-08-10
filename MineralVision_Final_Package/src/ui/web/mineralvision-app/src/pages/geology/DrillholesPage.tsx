import { useState, useCallback, useEffect } from 'react';
import { drillholesApi, projectsApi, Drillhole as ApiDrillhole } from '../../services/api';
import { useDropzone } from 'react-dropzone';
import {
  Search,
  Filter,
  Upload,
  Download,
  Eye,
  Edit,
  Trash2,
  FileSpreadsheet,
} from 'lucide-react';

interface Drillhole {
  id: string;
  holeId: string;
  project: string;
  collar: { x: number; y: number; z: number };
  totalDepth: number;
  azimuth: number;
  dip: number;
  status: 'completed' | 'in-progress' | 'planned';
  assayCount: number;
  avgGrade: number | null;
}


const statusColors = {
  completed: 'bg-green-500/10 text-green-500',
  'in-progress': 'bg-yellow-500/10 text-yellow-500',
  planned: 'bg-blue-500/10 text-blue-500',
};

export default function DrillholesPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [projectFilter, setProjectFilter] = useState('all');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedHoles, setSelectedHoles] = useState<string[]>([]);
  const [drillholes, setDrillholes] = useState<Drillhole[]>([]);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Live data: drillhole register + project names from the backend.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadState('loading');
      try {
        const [holesResp, projectsResp] = await Promise.all([
          drillholesApi.list(),
          projectsApi.list(),
        ]);
        if (cancelled) return;
        const projectNames = new Map(projectsResp.data.map((p) => [p.id, p.name]));
        const mapped: Drillhole[] = holesResp.data.map((h: ApiDrillhole) => ({
          id: h.id,
          holeId: h.holeId,
          project: projectNames.get(h.projectId) ?? h.projectId,
          collar: h.collar,
          totalDepth: h.totalDepth,
          azimuth: h.azimuth ?? 0,
          dip: h.dip ?? -90,
          status: (h.status as Drillhole['status']) ?? 'planned',
          assayCount: h.assayCount,
          // The list endpoint does not compute average grades; shown as '-'.
          avgGrade: null,
        }));
        setDrillholes(mapped);
        setLoadState('ready');
      } catch (err) {
        if (!cancelled) {
          setLoadState('error');
          setErrorMessage(err instanceof Error ? err.message : 'Failed to load drillholes');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredDrillholes = drillholes.filter((hole) => {
    const matchesSearch = hole.holeId.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesProject = projectFilter === 'all' || hole.project === projectFilter;
    return matchesSearch && matchesProject;
  });

  const projects = [...new Set(drillholes.map((h) => h.project))];

  const toggleSelectAll = () => {
    if (selectedHoles.length === filteredDrillholes.length) {
      setSelectedHoles([]);
    } else {
      setSelectedHoles(filteredDrillholes.map((h) => h.id));
    }
  };

  const toggleSelect = (id: string) => {
    setSelectedHoles((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Drillholes</h1>
          <p className="text-muted-foreground">Manage drillhole data across all projects</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Download className="h-4 w-4" />
            Export
          </button>
          <button
            onClick={() => setShowUploadModal(true)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
          >
            <Upload className="h-4 w-4" />
            Import Data
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search drillholes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <select
            value={projectFilter}
            onChange={(e) => setProjectFilter(e.target.value)}
            className="px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="all">All Projects</option>
            {projects.map((project) => (
              <option key={project} value={project}>{project}</option>
            ))}
          </select>
        </div>
      </div>

      {selectedHoles.length > 0 && (
        <div className="flex items-center gap-4 p-3 bg-primary/10 rounded-lg">
          <span className="text-sm text-foreground">{selectedHoles.length} selected</span>
          <button className="text-sm text-primary hover:underline">Composite</button>
          <button className="text-sm text-primary hover:underline">Desurvey</button>
          <button className="text-sm text-primary hover:underline">Export</button>
          <button className="text-sm text-destructive hover:underline">Delete</button>
        </div>
      )}

      {loadState === 'loading' && (
        <div className="bg-card border border-border rounded-xl p-10 text-center text-sm text-muted-foreground">
          Loading drillholes…
        </div>
      )}
      {loadState === 'error' && (
        <div className="bg-destructive/10 border border-destructive/40 text-destructive rounded-xl p-4 text-sm">
          Failed to load drillholes: {errorMessage}
        </div>
      )}
      {loadState === 'ready' && drillholes.length === 0 && (
        <div className="bg-card border border-border rounded-xl p-10 text-center text-sm text-muted-foreground">
          No drillholes in the register. Upload a collar file or create drillholes via the API.
        </div>
      )}
      {loadState === 'ready' && drillholes.length > 0 && (
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="text-left py-3 px-4">
                  <input
                    type="checkbox"
                    checked={selectedHoles.length === filteredDrillholes.length && filteredDrillholes.length > 0}
                    onChange={toggleSelectAll}
                    className="rounded border-input"
                  />
                </th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Hole ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Project</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Collar (X, Y, Z)</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Depth (m)</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Az/Dip</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Assays</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Avg Grade</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredDrillholes.map((hole) => (
                <tr key={hole.id} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-3 px-4">
                    <input
                      type="checkbox"
                      checked={selectedHoles.includes(hole.id)}
                      onChange={() => toggleSelect(hole.id)}
                      className="rounded border-input"
                    />
                  </td>
                  <td className="py-3 px-4 font-medium text-foreground">{hole.holeId}</td>
                  <td className="py-3 px-4 text-muted-foreground">{hole.project}</td>
                  <td className="py-3 px-4 text-sm text-muted-foreground font-mono">
                    {hole.collar.x.toFixed(0)}, {hole.collar.y.toFixed(0)}, {hole.collar.z.toFixed(0)}
                  </td>
                  <td className="py-3 px-4 text-right text-foreground">{hole.totalDepth}</td>
                  <td className="py-3 px-4 text-right text-muted-foreground">{hole.azimuth}/{hole.dip}</td>
                  <td className="py-3 px-4 text-right text-foreground">{hole.assayCount}</td>
                  <td className="py-3 px-4 text-right text-foreground">
                    {hole.avgGrade !== null ? hole.avgGrade.toFixed(2) : '-'}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full capitalize ${statusColors[hole.status]}`}>
                      {hole.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                        <Eye className="h-4 w-4" />
                      </button>
                      <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                        <Edit className="h-4 w-4" />
                      </button>
                      <button className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      )}

      {showUploadModal && (
        <UploadModal onClose={() => setShowUploadModal(false)} />
      )}
    </div>
  );
}

function UploadModal({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles((prev) => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
  });

  const handleUpload = async () => {
    setUploading(true);
    // Upload requires a project id; use the first project from the register.
    try {
      const { projectsApi, drillholesApi } = await import('../../services/api');
      const projects = (await projectsApi.list()).data;
      if (!projects[0]) throw new Error('No project exists — create a project first.');
      for (const file of files) {
        await drillholesApi.upload(file, projects[0].id);
      }
      onClose();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-lg">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Import Drillhole Data</h2>
          <p className="text-sm text-muted-foreground mt-1">
            Upload CSV or Excel files containing collar, survey, assay, or lithology data
          </p>
        </div>
        <div className="p-6 space-y-4">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive ? 'border-primary bg-primary/5' : 'border-input hover:border-primary/50'
            }`}
          >
            <input {...getInputProps()} />
            <FileSpreadsheet className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-foreground font-medium">
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse
            </p>
            <p className="text-xs text-muted-foreground mt-2">
              Supports CSV, XLS, XLSX
            </p>
          </div>

          {files.length > 0 && (
            <div className="space-y-2">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-3 bg-background rounded-lg">
                  <div className="flex items-center gap-3">
                    <FileSpreadsheet className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium text-foreground">{file.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() => setFiles((prev) => prev.filter((_, i) => i !== index))}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <div className="flex gap-3 pt-4">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary"
            >
              Cancel
            </button>
            <button
              onClick={handleUpload}
              disabled={files.length === 0 || uploading}
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
            >
              {uploading ? 'Uploading...' : 'Upload'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
