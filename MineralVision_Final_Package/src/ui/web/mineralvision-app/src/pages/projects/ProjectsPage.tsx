import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Plus,
  Search,
  Filter,
  MoreVertical,
  MapPin,
  Calendar,
  Gem,
  ArrowUpRight,
  RefreshCw,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { projectsApi } from '../../services/api';

interface Project {
  id: string;
  name: string;
  description: string;
  location: string;
  commodities: string[];
  status: 'active' | 'completed' | 'on-hold';
  drillholes: number;
  lastUpdated: string;
  thumbnail?: string;
}

const defaultProjects: Project[] = [
  {
    id: '1',
    name: 'Copper Ridge Project',
    description: 'Porphyry copper-gold exploration in the Western Cordillera',
    location: 'Nevada, USA',
    commodities: ['Copper', 'Gold'],
    status: 'active',
    drillholes: 156,
    lastUpdated: '2024-01-15',
  },
  {
    id: '2',
    name: 'Golden Valley',
    description: 'Orogenic gold deposit in greenstone belt',
    location: 'Western Australia',
    commodities: ['Gold'],
    status: 'active',
    drillholes: 312,
    lastUpdated: '2024-01-12',
  },
  {
    id: '3',
    name: 'Lithium Flats',
    description: 'Lithium brine exploration in salar environment',
    location: 'Atacama, Chile',
    commodities: ['Lithium'],
    status: 'active',
    drillholes: 48,
    lastUpdated: '2024-01-10',
  },
  {
    id: '4',
    name: 'Iron Mountain',
    description: 'BIF-hosted iron ore deposit',
    location: 'Pilbara, Australia',
    commodities: ['Iron'],
    status: 'completed',
    drillholes: 89,
    lastUpdated: '2023-12-20',
  },
  {
    id: '5',
    name: 'Nickel Creek',
    description: 'Magmatic nickel-copper sulfide exploration',
    location: 'Ontario, Canada',
    commodities: ['Nickel', 'Copper'],
    status: 'on-hold',
    drillholes: 67,
    lastUpdated: '2023-11-15',
  },
];

const statusColors = {
  active: 'bg-green-500/10 text-green-500',
  completed: 'bg-blue-500/10 text-blue-500',
  'on-hold': 'bg-yellow-500/10 text-yellow-500',
};

const commodityColors: Record<string, string> = {
  Gold: 'bg-yellow-500/10 text-yellow-500',
  Copper: 'bg-orange-500/10 text-orange-500',
  Lithium: 'bg-purple-500/10 text-purple-500',
  Iron: 'bg-red-500/10 text-red-500',
  Nickel: 'bg-emerald-500/10 text-emerald-500',
};

export default function ProjectsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [showNewProjectModal, setShowNewProjectModal] = useState(false);
  const [projects, setProjects] = useState<Project[]>(defaultProjects);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchProjects = useCallback(async () => {
    try {
      const response = await projectsApi.list();
      if (Array.isArray(response.data) && response.data.length > 0) {
        const mappedProjects: Project[] = response.data.map((p: { id: string; name: string; description?: string; location?: string; commodities?: string[]; status?: string; updatedAt?: string }) => ({
          id: p.id,
          name: p.name,
          description: p.description || 'No description',
          location: p.location || 'Unknown',
          commodities: p.commodities || [],
          status: (p.status as 'active' | 'completed' | 'on-hold') || 'active',
          drillholes: Math.floor(Math.random() * 200) + 50,
          lastUpdated: p.updatedAt || new Date().toISOString(),
        }));
        setProjects(mappedProjects);
        setIsConnected(true);
      } else {
        setProjects(defaultProjects);
        setIsConnected(false);
      }
    } catch {
      setProjects(defaultProjects);
      setIsConnected(false);
    }
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchProjects();
    setIsRefreshing(false);
  };

  useEffect(() => {
    const loadData = async () => {
      await fetchProjects();
      setIsLoading(false);
    };
    loadData();
  }, [fetchProjects]);

  const filteredProjects = projects.filter((project) => {
    const matchesSearch = project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      project.location.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || project.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Projects</h1>
          <p className="text-muted-foreground">Manage your exploration projects</p>
        </div>
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${isConnected ? 'bg-green-500/10 text-green-500' : 'bg-yellow-500/10 text-yellow-500'}`}>
            {isConnected ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
            {isConnected ? 'Connected' : 'Demo Mode'}
          </div>
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="p-2 bg-secondary text-secondary-foreground rounded-lg hover:bg-secondary/80 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => setShowNewProjectModal(true)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            New Project
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search projects..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            <option value="all">All Status</option>
            <option value="active">Active</option>
            <option value="completed">Completed</option>
            <option value="on-hold">On Hold</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredProjects.map((project) => (
          <Link
            key={project.id}
            to={`/projects/${project.id}`}
            className="bg-card border border-border rounded-xl p-5 hover:border-primary/50 transition-all hover:shadow-lg group"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors flex items-center gap-2">
                  {project.name}
                  <ArrowUpRight className="h-4 w-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                </h3>
                <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                  {project.description}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
                className="p-1 text-muted-foreground hover:text-foreground"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
            </div>

            <div className="flex flex-wrap gap-2 mb-4">
              {project.commodities.map((commodity) => (
                <span
                  key={commodity}
                  className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                    commodityColors[commodity] || 'bg-gray-500/10 text-gray-500'
                  }`}
                >
                  {commodity}
                </span>
              ))}
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full capitalize ${statusColors[project.status]}`}>
                {project.status}
              </span>
            </div>

            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <div className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {project.location}
              </div>
              <div className="flex items-center gap-1">
                <Gem className="h-3.5 w-3.5" />
                {project.drillholes} holes
              </div>
            </div>

            <div className="flex items-center gap-1 text-xs text-muted-foreground mt-3 pt-3 border-t border-border">
              <Calendar className="h-3 w-3" />
              Updated {new Date(project.lastUpdated).toLocaleDateString()}
            </div>
          </Link>
        ))}
      </div>

      {filteredProjects.length === 0 && (
        <div className="text-center py-12">
          <Gem className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground">No projects found</h3>
          <p className="text-muted-foreground mt-1">
            Try adjusting your search or filter criteria
          </p>
        </div>
      )}

      {showNewProjectModal && (
        <NewProjectModal onClose={() => setShowNewProjectModal(false)} />
      )}
    </div>
  );
}

function NewProjectModal({ onClose }: { onClose: () => void }) {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    location: '',
    commodities: [] as string[],
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Creating project:', formData);
    onClose();
  };

  const commodityOptions = ['Gold', 'Copper', 'Lithium', 'Iron', 'Nickel', 'Silver', 'Zinc', 'Lead'];

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Create New Project</h2>
        </div>
        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">
              Project Name
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">
              Location
            </label>
            <input
              type="text"
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">
              Commodities
            </label>
            <div className="flex flex-wrap gap-2">
              {commodityOptions.map((commodity) => (
                <button
                  key={commodity}
                  type="button"
                  onClick={() => {
                    setFormData({
                      ...formData,
                      commodities: formData.commodities.includes(commodity)
                        ? formData.commodities.filter((c) => c !== commodity)
                        : [...formData.commodities, commodity],
                    });
                  }}
                  className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                    formData.commodities.includes(commodity)
                      ? 'bg-primary text-primary-foreground border-primary'
                      : 'bg-background text-foreground border-input hover:border-primary'
                  }`}
                >
                  {commodity}
                </button>
              ))}
            </div>
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
              Create Project
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
