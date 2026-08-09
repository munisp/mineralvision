import { useState } from 'react';
import {
  Plus,
  FileText,
  Download,
  Eye,
  Clock,
  Trash2,
} from 'lucide-react';

interface Report {
  id: string;
  name: string;
  type: 'ni43-101' | 'jorc' | 'custom';
  project: string;
  status: 'draft' | 'review' | 'approved' | 'published';
  createdAt: string;
  updatedAt: string;
  author: string;
}

const mockReports: Report[] = [
  { id: '1', name: 'Copper Ridge Technical Report', type: 'ni43-101', project: 'Copper Ridge', status: 'review', createdAt: '2024-01-10', updatedAt: '2024-01-15', author: 'John Smith' },
  { id: '2', name: 'Golden Valley Resource Estimate', type: 'jorc', project: 'Golden Valley', status: 'approved', createdAt: '2024-01-05', updatedAt: '2024-01-12', author: 'Jane Doe' },
  { id: '3', name: 'Q4 2023 Exploration Summary', type: 'custom', project: 'All Projects', status: 'published', createdAt: '2023-12-20', updatedAt: '2024-01-02', author: 'Mike Johnson' },
  { id: '4', name: 'Lithium Flats Preliminary Assessment', type: 'ni43-101', project: 'Lithium Flats', status: 'draft', createdAt: '2024-01-14', updatedAt: '2024-01-14', author: 'Sarah Wilson' },
];

const statusColors = {
  draft: 'bg-gray-500/10 text-gray-500',
  review: 'bg-yellow-500/10 text-yellow-500',
  approved: 'bg-blue-500/10 text-blue-500',
  published: 'bg-green-500/10 text-green-500',
};

const typeLabels = {
  'ni43-101': 'NI 43-101',
  'jorc': 'JORC',
  'custom': 'Custom',
};

export default function ReportsPage() {
  const [selectedType, setSelectedType] = useState('all');
  const [showNewReportModal, setShowNewReportModal] = useState(false);

  const filteredReports = mockReports.filter(
    (report) => selectedType === 'all' || report.type === selectedType
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Reports</h1>
          <p className="text-muted-foreground">Generate and manage regulatory reports</p>
        </div>
        <button
          onClick={() => setShowNewReportModal(true)}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Report
        </button>
      </div>

      <div className="flex items-center gap-2">
        {['all', 'ni43-101', 'jorc', 'custom'].map((type) => (
          <button
            key={type}
            onClick={() => setSelectedType(type)}
            className={`px-4 py-2 text-sm rounded-lg ${
              selectedType === type
                ? 'bg-primary text-primary-foreground'
                : 'bg-card text-foreground hover:bg-secondary border border-border'
            }`}
          >
            {type === 'all' ? 'All Reports' : typeLabels[type as keyof typeof typeLabels]}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredReports.map((report) => (
          <div
            key={report.id}
            className="bg-card border border-border rounded-xl p-5 hover:border-primary/50 transition-colors"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full capitalize ${statusColors[report.status]}`}>
                {report.status}
              </span>
            </div>

            <h3 className="font-semibold text-foreground mb-1">{report.name}</h3>
            <p className="text-sm text-muted-foreground mb-3">{report.project}</p>

            <div className="flex items-center gap-2 text-xs text-muted-foreground mb-4">
              <span className="px-2 py-0.5 bg-secondary rounded">
                {typeLabels[report.type]}
              </span>
              <span>by {report.author}</span>
            </div>

            <div className="flex items-center justify-between text-xs text-muted-foreground pt-3 border-t border-border">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Updated {new Date(report.updatedAt).toLocaleDateString()}
              </span>
              <div className="flex items-center gap-1">
                <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <Eye className="h-4 w-4" />
                </button>
                <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded">
                  <Download className="h-4 w-4" />
                </button>
                <button className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded">
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-lg font-semibold text-foreground mb-4">Report Templates</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 bg-background rounded-lg border border-border hover:border-primary/50 cursor-pointer">
            <h3 className="font-medium text-foreground">NI 43-101 Technical Report</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Canadian securities regulatory standard for mineral projects
            </p>
          </div>
          <div className="p-4 bg-background rounded-lg border border-border hover:border-primary/50 cursor-pointer">
            <h3 className="font-medium text-foreground">JORC Code Report</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Australasian code for reporting exploration results and mineral resources
            </p>
          </div>
          <div className="p-4 bg-background rounded-lg border border-border hover:border-primary/50 cursor-pointer">
            <h3 className="font-medium text-foreground">Custom Report</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Create a custom report with your own sections and content
            </p>
          </div>
        </div>
      </div>

      {showNewReportModal && (
        <NewReportModal onClose={() => setShowNewReportModal(false)} />
      )}
    </div>
  );
}

function NewReportModal({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = useState({
    name: '',
    type: 'ni43-101',
    project: '',
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Create New Report</h2>
        </div>
        <form className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Report Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
              placeholder="e.g., Q1 2024 Technical Report"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Report Type</label>
            <select
              value={formData.type}
              onChange={(e) => setFormData({ ...formData, type: e.target.value })}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
            >
              <option value="ni43-101">NI 43-101 Technical Report</option>
              <option value="jorc">JORC Code Report</option>
              <option value="custom">Custom Report</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Project</label>
            <select
              value={formData.project}
              onChange={(e) => setFormData({ ...formData, project: e.target.value })}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
            >
              <option value="">Select a project</option>
              <option value="copper-ridge">Copper Ridge</option>
              <option value="golden-valley">Golden Valley</option>
              <option value="lithium-flats">Lithium Flats</option>
            </select>
          </div>
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90"
            >
              Create Report
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
