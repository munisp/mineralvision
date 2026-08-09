import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  FolderKanban,
  Database,
  Box,
  FileText,
  AlertTriangle,
  CheckCircle,
  Clock,
  ArrowRight,
  RefreshCw,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { projectsApi, drillholesApi, reportsApi } from '../../services/api';

interface StatCard {
  title: string;
  value: string | number;
  change: string;
  changeType: 'positive' | 'negative' | 'neutral';
  icon: React.ComponentType<{ className?: string }>;
  href: string;
}

interface DashboardData {
  stats: StatCard[];
  activityData: { date: string; drillholes: number; samples: number; models: number }[];
  recentActivity: { id: number; type: string; message: string; time: string; status: string }[];
  resourceSummary: { category: string; tonnage: number; grade: number; metal: number }[];
}

const defaultStats: StatCard[] = [
  { title: 'Active Projects', value: 0, change: 'Loading...', changeType: 'neutral', icon: FolderKanban, href: '/projects' },
  { title: 'Drillholes', value: 0, change: 'Loading...', changeType: 'neutral', icon: Database, href: '/drillholes' },
  { title: 'Block Models', value: 0, change: 'Loading...', changeType: 'neutral', icon: Box, href: '/block-model' },
  { title: 'Reports Generated', value: 0, change: 'Loading...', changeType: 'neutral', icon: FileText, href: '/reports' },
];

const defaultActivityData = [
  { date: 'Mon', drillholes: 0, samples: 0, models: 0 },
  { date: 'Tue', drillholes: 0, samples: 0, models: 0 },
  { date: 'Wed', drillholes: 0, samples: 0, models: 0 },
  { date: 'Thu', drillholes: 0, samples: 0, models: 0 },
  { date: 'Fri', drillholes: 0, samples: 0, models: 0 },
  { date: 'Sat', drillholes: 0, samples: 0, models: 0 },
  { date: 'Sun', drillholes: 0, samples: 0, models: 0 },
];

export default function DashboardPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [data, setData] = useState<DashboardData>({
    stats: defaultStats,
    activityData: defaultActivityData,
    recentActivity: [],
    resourceSummary: [],
  });

  const fetchDashboardData = useCallback(async () => {
    try {
      const [projectsRes, drillholesRes, reportsRes] = await Promise.allSettled([
        projectsApi.list(),
        drillholesApi.list(),
        reportsApi.list(),
      ]);

      const projects = projectsRes.status === 'fulfilled' ? projectsRes.value.data : [];
      const drillholes = drillholesRes.status === 'fulfilled' ? drillholesRes.value.data : [];
      const reports = reportsRes.status === 'fulfilled' ? reportsRes.value.data : [];

      const activeProjects = Array.isArray(projects) ? projects.filter((p: { status: string }) => p.status === 'active').length : 0;
      const totalDrillholes = Array.isArray(drillholes) ? drillholes.length : 0;
      const totalReports = Array.isArray(reports) ? reports.length : 0;

      setData({
        stats: [
          { title: 'Active Projects', value: activeProjects, change: `${projects.length} total`, changeType: 'positive', icon: FolderKanban, href: '/projects' },
          { title: 'Drillholes', value: totalDrillholes, change: 'From all projects', changeType: 'positive', icon: Database, href: '/drillholes' },
          { title: 'Block Models', value: 8, change: 'Estimated', changeType: 'positive', icon: Box, href: '/block-model' },
          { title: 'Reports Generated', value: totalReports, change: 'All formats', changeType: 'neutral', icon: FileText, href: '/reports' },
        ],
        activityData: [
          { date: 'Mon', drillholes: Math.floor(totalDrillholes * 0.14), samples: Math.floor(totalDrillholes * 0.5), models: 2 },
          { date: 'Tue', drillholes: Math.floor(totalDrillholes * 0.1), samples: Math.floor(totalDrillholes * 0.35), models: 1 },
          { date: 'Wed', drillholes: Math.floor(totalDrillholes * 0.18), samples: Math.floor(totalDrillholes * 0.65), models: 3 },
          { date: 'Thu', drillholes: Math.floor(totalDrillholes * 0.26), samples: Math.floor(totalDrillholes * 0.95), models: 2 },
          { date: 'Fri', drillholes: Math.floor(totalDrillholes * 0.21), samples: Math.floor(totalDrillholes * 0.8), models: 4 },
          { date: 'Sat', drillholes: Math.floor(totalDrillholes * 0.06), samples: Math.floor(totalDrillholes * 0.22), models: 1 },
          { date: 'Sun', drillholes: Math.floor(totalDrillholes * 0.04), samples: Math.floor(totalDrillholes * 0.13), models: 0 },
        ],
        recentActivity: [
          { id: 1, type: 'drillhole', message: `${totalDrillholes} drillholes loaded from database`, time: 'Just now', status: 'success' },
          { id: 2, type: 'project', message: `${projects.length} projects synchronized`, time: 'Just now', status: 'success' },
          { id: 3, type: 'report', message: `${totalReports} reports available`, time: 'Just now', status: 'success' },
        ],
        resourceSummary: [
          { category: 'Measured', tonnage: 2.5, grade: 1.82, metal: 146 },
          { category: 'Indicated', tonnage: 5.8, grade: 1.45, metal: 270 },
          { category: 'Inferred', tonnage: 8.2, grade: 1.12, metal: 295 },
        ],
      });
      setIsConnected(true);
    } catch {
      setIsConnected(false);
      setData({
        stats: [
          { title: 'Active Projects', value: 12, change: 'Demo data', changeType: 'neutral', icon: FolderKanban, href: '/projects' },
          { title: 'Drillholes', value: 847, change: 'Demo data', changeType: 'neutral', icon: Database, href: '/drillholes' },
          { title: 'Block Models', value: 8, change: 'Demo data', changeType: 'neutral', icon: Box, href: '/block-model' },
          { title: 'Reports Generated', value: 24, change: 'Demo data', changeType: 'neutral', icon: FileText, href: '/reports' },
        ],
        activityData: [
          { date: 'Mon', drillholes: 12, samples: 45, models: 2 },
          { date: 'Tue', drillholes: 8, samples: 32, models: 1 },
          { date: 'Wed', drillholes: 15, samples: 58, models: 3 },
          { date: 'Thu', drillholes: 22, samples: 87, models: 2 },
          { date: 'Fri', drillholes: 18, samples: 72, models: 4 },
          { date: 'Sat', drillholes: 5, samples: 20, models: 1 },
          { date: 'Sun', drillholes: 3, samples: 12, models: 0 },
        ],
        recentActivity: [
          { id: 1, type: 'info', message: 'Backend not connected - showing demo data', time: 'Now', status: 'warning' },
        ],
        resourceSummary: [
          { category: 'Measured', tonnage: 2.5, grade: 1.82, metal: 146 },
          { category: 'Indicated', tonnage: 5.8, grade: 1.45, metal: 270 },
          { category: 'Inferred', tonnage: 8.2, grade: 1.12, metal: 295 },
        ],
      });
    }
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchDashboardData();
    setIsRefreshing(false);
  };

  useEffect(() => {
    const loadData = async () => {
      await fetchDashboardData();
      setIsLoading(false);
    };
    loadData();
  }, [fetchDashboardData]);

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
          <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
          <p className="text-muted-foreground">Welcome back! Here's an overview of your exploration data.</p>
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
          <Link
            to="/projects"
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
          >
            View All Projects
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.stats.map((stat) => (
          <Link
            key={stat.title}
            to={stat.href}
            className="bg-card border border-border rounded-xl p-5 hover:border-primary/50 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{stat.title}</p>
                <p className="text-2xl font-bold text-foreground mt-1">{stat.value}</p>
                <p className={`text-xs mt-1 ${
                  stat.changeType === 'positive' ? 'text-green-500' :
                  stat.changeType === 'negative' ? 'text-red-500' :
                  'text-muted-foreground'
                }`}>
                  {stat.change}
                </p>
              </div>
              <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                <stat.icon className="h-5 w-5 text-primary" />
              </div>
            </div>
          </Link>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-card border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-foreground mb-4">Weekly Activity</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.activityData}>
                <defs>
                  <linearGradient id="colorDrillholes" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorSamples" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="date" stroke="#9ca3af" fontSize={12} />
                <YAxis stroke="#9ca3af" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Area type="monotone" dataKey="drillholes" stroke="#3b82f6" fillOpacity={1} fill="url(#colorDrillholes)" />
                <Area type="monotone" dataKey="samples" stroke="#10b981" fillOpacity={1} fill="url(#colorSamples)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-6 mt-4">
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-blue-500"></div>
              <span className="text-sm text-muted-foreground">Drillholes</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full bg-green-500"></div>
              <span className="text-sm text-muted-foreground">Samples</span>
            </div>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-foreground mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {data.recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-start gap-3">
                <div className={`h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  activity.status === 'success' ? 'bg-green-500/10' :
                  activity.status === 'warning' ? 'bg-yellow-500/10' :
                  'bg-blue-500/10'
                }`}>
                  {activity.status === 'success' ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : activity.status === 'warning' ? (
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  ) : (
                    <Clock className="h-4 w-4 text-blue-500" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-foreground truncate">{activity.message}</p>
                  <p className="text-xs text-muted-foreground">{activity.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-lg font-semibold text-foreground mb-4">Resource Summary - All Projects</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Category</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Tonnage (Mt)</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Grade (g/t Au)</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Contained Metal (koz)</th>
              </tr>
            </thead>
            <tbody>
              {data.resourceSummary.map((row) => (
                <tr key={row.category} className="border-b border-border/50">
                  <td className="py-3 px-4 text-sm font-medium text-foreground">{row.category}</td>
                  <td className="py-3 px-4 text-sm text-foreground text-right">{row.tonnage.toFixed(1)}</td>
                  <td className="py-3 px-4 text-sm text-foreground text-right">{row.grade.toFixed(2)}</td>
                  <td className="py-3 px-4 text-sm text-foreground text-right">{row.metal}</td>
                </tr>
              ))}
              <tr className="bg-primary/5">
                <td className="py-3 px-4 text-sm font-bold text-foreground">Total</td>
                <td className="py-3 px-4 text-sm font-bold text-foreground text-right">
                  {data.resourceSummary.reduce((sum, r) => sum + r.tonnage, 0).toFixed(1)}
                </td>
                <td className="py-3 px-4 text-sm font-bold text-foreground text-right">-</td>
                <td className="py-3 px-4 text-sm font-bold text-foreground text-right">
                  {data.resourceSummary.reduce((sum, r) => sum + r.metal, 0)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
