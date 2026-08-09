import { useState, useCallback } from 'react';
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

const mockDrillholes: Drillhole[] = [
  { id: '1', holeId: 'DDH-2024-156', project: 'Copper Ridge', collar: { x: 456789, y: 7654321, z: 1250 }, totalDepth: 450, azimuth: 45, dip: -60, status: 'completed', assayCount: 45, avgGrade: 0.85 },
  { id: '2', holeId: 'DDH-2024-155', project: 'Copper Ridge', collar: { x: 456812, y: 7654298, z: 1248 }, totalDepth: 380, azimuth: 45, dip: -60, status: 'completed', assayCount: 38, avgGrade: 1.12 },
  { id: '3', holeId: 'DDH-2024-154', project: 'Copper Ridge', collar: { x: 456756, y: 7654345, z: 1252 }, totalDepth: 520, azimuth: 90, dip: -55, status: 'completed', assayCount: 52, avgGrade: 0.72 },
  { id: '4', holeId: 'DDH-2024-153', project: 'Copper Ridge', collar: { x: 456801, y: 7654312, z: 1249 }, totalDepth: 290, azimuth: 45, dip: -60, status: 'in-progress', assayCount: 15, avgGrade: null },
  { id: '5', holeId: 'DDH-2024-152', project: 'Copper Ridge', collar: { x: 456778, y: 7654334, z: 1251 }, totalDepth: 410, azimuth: 135, dip: -65, status: 'completed', assayCount: 41, avgGrade: 0.95 },
  { id: '6', holeId: 'GV-2024-089', project: 'Golden Valley', collar: { x: 523456, y: 6789012, z: 380 }, totalDepth: 320, azimuth: 0, dip: -90, status: 'completed', assayCount: 64, avgGrade: 2.45 },
  { id: '7', holeId: 'GV-2024-088', project: 'Golden Valley', collar: { x: 523478, y: 6789034, z: 382 }, totalDepth: 280, azimuth: 0, dip: -90, status: 'completed', assayCount: 56, avgGrade: 3.12 },
  { id: '8', holeId: 'LF-2024-012', project: 'Lithium Flats', collar: { x: 612345, y: 7456789, z: 2340 }, totalDepth: 150, azimuth: 0, dip: -90, status: 'completed', assayCount: 30, avgGrade: 850 },
];

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

  const filteredDrillholes = mockDrillholes.filter((hole) => {
    const matchesSearch = hole.holeId.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesProject = projectFilter === 'all' || hole.project === projectFilter;
    return matchesSearch && matchesProject;
  });

  const projects = [...new Set(mockDrillholes.map((h) => h.project))];

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
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setUploading(false);
    onClose();
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
