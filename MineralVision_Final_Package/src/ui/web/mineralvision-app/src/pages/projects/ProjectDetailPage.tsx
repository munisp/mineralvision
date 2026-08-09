import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowLeft,
  MapPin,
  Calendar,
  Database,
  Box,
  FileText,
  Upload,
  Play,
  Download,
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const mockProject = {
  id: '1',
  name: 'Copper Ridge Project',
  description: 'Porphyry copper-gold exploration in the Western Cordillera. This project targets a large-scale porphyry copper-gold system with potential for significant resource expansion.',
  location: 'Nevada, USA',
  commodities: ['Copper', 'Gold'],
  status: 'active',
  createdAt: '2023-06-15',
  updatedAt: '2024-01-15',
  stats: {
    drillholes: 156,
    totalMeters: 45680,
    samples: 12450,
    blockModels: 3,
    reports: 5,
  },
};

const mockDrillholeData = [
  { month: 'Jul', meters: 2500, holes: 8 },
  { month: 'Aug', meters: 3200, holes: 12 },
  { month: 'Sep', meters: 4100, holes: 15 },
  { month: 'Oct', meters: 3800, holes: 14 },
  { month: 'Nov', meters: 5200, holes: 18 },
  { month: 'Dec', meters: 4500, holes: 16 },
  { month: 'Jan', meters: 6100, holes: 22 },
];

const mockRecentDrillholes = [
  { id: 'DDH-2024-156', depth: 450, status: 'completed', grade: 0.85 },
  { id: 'DDH-2024-155', depth: 380, status: 'completed', grade: 1.12 },
  { id: 'DDH-2024-154', depth: 520, status: 'completed', grade: 0.72 },
  { id: 'DDH-2024-153', depth: 290, status: 'in-progress', grade: null },
  { id: 'DDH-2024-152', depth: 410, status: 'completed', grade: 0.95 },
];

export default function ProjectDetailPage() {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'drillholes', label: 'Drillholes' },
    { id: 'models', label: 'Block Models' },
    { id: 'reports', label: 'Reports' },
    { id: 'settings', label: 'Settings' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/projects"
          className="p-2 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-lg"
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-foreground">{mockProject.name}</h1>
          <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
            <span className="flex items-center gap-1">
              <MapPin className="h-4 w-4" />
              {mockProject.location}
            </span>
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              Updated {new Date(mockProject.updatedAt).toLocaleDateString()}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Import Data
          </button>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2">
            <Play className="h-4 w-4" />
            Run Analysis
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <StatCard icon={Database} label="Drillholes" value={mockProject.stats.drillholes} />
        <StatCard icon={Database} label="Total Meters" value={mockProject.stats.totalMeters.toLocaleString()} />
        <StatCard icon={Database} label="Samples" value={mockProject.stats.samples.toLocaleString()} />
        <StatCard icon={Box} label="Block Models" value={mockProject.stats.blockModels} />
        <StatCard icon={FileText} label="Reports" value={mockProject.stats.reports} />
      </div>

      <div className="border-b border-border">
        <nav className="flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Drilling Progress</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={mockDrillholeData}>
                    <defs>
                      <linearGradient id="colorMeters" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="month" stroke="#9ca3af" fontSize={12} />
                    <YAxis stroke="#9ca3af" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                      }}
                    />
                    <Area type="monotone" dataKey="meters" stroke="#3b82f6" fillOpacity={1} fill="url(#colorMeters)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Project Description</h2>
              <p className="text-muted-foreground">{mockProject.description}</p>
              <div className="flex flex-wrap gap-2 mt-4">
                {mockProject.commodities.map((commodity) => (
                  <span
                    key={commodity}
                    className="px-3 py-1 text-sm font-medium rounded-full bg-primary/10 text-primary"
                  >
                    {commodity}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Recent Drillholes</h2>
              <div className="space-y-3">
                {mockRecentDrillholes.map((hole) => (
                  <div key={hole.id} className="flex items-center justify-between p-3 bg-background rounded-lg">
                    <div>
                      <p className="font-medium text-foreground">{hole.id}</p>
                      <p className="text-sm text-muted-foreground">{hole.depth}m depth</p>
                    </div>
                    <div className="text-right">
                      {hole.grade !== null ? (
                        <p className="font-medium text-foreground">{hole.grade} g/t</p>
                      ) : (
                        <p className="text-sm text-muted-foreground">In progress</p>
                      )}
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        hole.status === 'completed' ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'
                      }`}>
                        {hole.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <Link
                to="/drillholes"
                className="block text-center text-sm text-primary hover:underline mt-4"
              >
                View all drillholes
              </Link>
            </div>

            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Quick Actions</h2>
              <div className="space-y-2">
                <button className="w-full px-4 py-2 text-left text-sm text-foreground hover:bg-secondary rounded-lg flex items-center gap-3">
                  <Upload className="h-4 w-4 text-muted-foreground" />
                  Import Drillhole Data
                </button>
                <button className="w-full px-4 py-2 text-left text-sm text-foreground hover:bg-secondary rounded-lg flex items-center gap-3">
                  <Play className="h-4 w-4 text-muted-foreground" />
                  Run Variography
                </button>
                <button className="w-full px-4 py-2 text-left text-sm text-foreground hover:bg-secondary rounded-lg flex items-center gap-3">
                  <Box className="h-4 w-4 text-muted-foreground" />
                  Create Block Model
                </button>
                <button className="w-full px-4 py-2 text-left text-sm text-foreground hover:bg-secondary rounded-lg flex items-center gap-3">
                  <Download className="h-4 w-4 text-muted-foreground" />
                  Export Report
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'drillholes' && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-muted-foreground">Drillhole management interface - View, edit, and analyze drillhole data</p>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-muted-foreground">Block model management - Create and visualize resource models</p>
        </div>
      )}

      {activeTab === 'reports' && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-muted-foreground">Report generation - NI 43-101, JORC, and custom reports</p>
        </div>
      )}

      {activeTab === 'settings' && (
        <div className="bg-card border border-border rounded-xl p-5">
          <p className="text-muted-foreground">Project settings and configuration</p>
        </div>
      )}
    </div>
  );
}

function StatCard({ icon: Icon, label, value }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string | number }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
          <Icon className="h-5 w-5 text-primary" />
        </div>
        <div>
          <p className="text-sm text-muted-foreground">{label}</p>
          <p className="text-xl font-bold text-foreground">{value}</p>
        </div>
      </div>
    </div>
  );
}
