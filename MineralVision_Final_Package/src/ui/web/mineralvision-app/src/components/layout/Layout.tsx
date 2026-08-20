import { useState } from 'react';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import {
  LayoutDashboard,
  FolderKanban,
  Database,
  CheckCircle,
  Layers,
  BarChart3,
  Grid3X3,
  Box,
  Eye,
  Magnet,
  FileText,
  Radio,
  Settings,
  Users,
  LogOut,
  Menu,
  X,
  ChevronDown,
  Mountain,
  Navigation,
  Brain,
  Video,
  Workflow,
  Leaf,
  Gem,
  Map as MapIcon,
  Waves,
} from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  children?: NavItem[];
  permission?: string;
}

const navigation: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Projects', href: '/projects', icon: FolderKanban },
  {
    name: 'Geology',
    href: '#',
    icon: Database,
    children: [
      { name: 'Drillholes', href: '/drillholes', icon: Database },
      { name: 'QA/QC', href: '/qaqc', icon: CheckCircle },
      { name: 'Cross Sections', href: '/cross-sections', icon: Layers },
      { name: 'Map Explorer', href: '/map-explorer', icon: MapIcon },
    ],
  },
  {
    name: 'Geostatistics',
    href: '#',
    icon: BarChart3,
    children: [
      { name: 'Variography', href: '/variography', icon: BarChart3 },
      { name: 'Kriging', href: '/kriging', icon: Grid3X3 },
      { name: 'Block Model', href: '/block-model', icon: Box },
    ],
  },
  { name: '3D Visualization', href: '/visualization', icon: Eye },
  { name: 'Geophysics', href: '/inversion', icon: Magnet },
  { name: 'Reports', href: '/reports', icon: FileText },
  { name: 'Oil Spill Operations', href: '/oil-spill', icon: Waves },
  { name: 'Sensor Fusion', href: '/sensor-fusion', icon: Radio },
  { name: 'Enhanced GNSS', href: '/gnss', icon: Navigation },
  { name: 'AI Insights', href: '/ai-insights', icon: Brain },
  { name: 'Molmo2 Video', href: '/molmo2', icon: Video },
  { name: 'User Journeys', href: '/journeys', icon: Workflow },
  { name: 'Crop Monitoring', href: '/crop-monitoring', icon: Leaf },
  { name: 'Mineral Monitoring', href: '/mineral-monitoring', icon: Gem },
];

const adminNavigation: NavItem[] = [
  { name: 'Users', href: '/admin/users', icon: Users, permission: 'admin' },
  { name: 'Settings', href: '/settings', icon: Settings },
];

function NavLink({ item, isActive, onClick }: { item: NavItem; isActive: boolean; onClick?: () => void }) {
  const [isOpen, setIsOpen] = useState(false);
  const Icon = item.icon;

  if (item.children) {
    return (
      <div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`w-full flex items-center justify-between px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
            isActive
              ? 'bg-primary text-primary-foreground'
              : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
          }`}
        >
          <span className="flex items-center gap-3">
            <Icon className="h-5 w-5" />
            {item.name}
          </span>
          <ChevronDown className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>
        {isOpen && (
          <div className="ml-4 mt-1 space-y-1">
            {item.children.map((child) => (
              <Link
                key={child.href}
                to={child.href}
                onClick={onClick}
                className="flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
              >
                <child.icon className="h-4 w-4" />
                {child.name}
              </Link>
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <Link
      to={item.href}
      onClick={onClick}
      className={`flex items-center gap-3 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
        isActive
          ? 'bg-primary text-primary-foreground'
          : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
      }`}
    >
      <Icon className="h-5 w-5" />
      {item.name}
    </Link>
  );
}

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout, hasPermission } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const isActive = (href: string) => {
    if (href === '#') return false;
    return location.pathname === href || location.pathname.startsWith(href + '/');
  };

  return (
    <div className="min-h-screen flex">
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-card border-r border-border transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between h-16 px-4 border-b border-border">
            <Link to="/dashboard" className="flex items-center gap-2">
              <Mountain className="h-8 w-8 text-primary" />
              <span className="text-xl font-bold text-foreground">MineralVision</span>
            </Link>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden text-muted-foreground hover:text-foreground"
            >
              <X className="h-6 w-6" />
            </button>
          </div>

          <nav className="flex-1 overflow-y-auto p-4 space-y-1">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                item={item}
                isActive={isActive(item.href)}
                onClick={() => setSidebarOpen(false)}
              />
            ))}

            <div className="pt-4 mt-4 border-t border-border">
              <p className="px-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                Admin
              </p>
              {adminNavigation.map((item) => {
                if (item.permission && !hasPermission(item.permission)) {
                  return null;
                }
                return (
                  <NavLink
                    key={item.name}
                    item={item}
                    isActive={isActive(item.href)}
                    onClick={() => setSidebarOpen(false)}
                  />
                );
              })}
            </div>
          </nav>

          <div className="p-4 border-t border-border">
            <div className="flex items-center gap-3 mb-3">
              <div className="h-10 w-10 rounded-full bg-primary/20 flex items-center justify-center">
                <span className="text-sm font-medium text-primary">
                  {user?.firstName?.[0]}{user?.lastName?.[0]}
                </span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {user?.firstName} {user?.lastName}
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  {user?.roles?.[0] || 'User'}
                </p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="w-full flex items-center gap-3 px-3 py-2 text-sm font-medium text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded-lg transition-colors"
            >
              <LogOut className="h-5 w-5" />
              Sign out
            </button>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center px-4 lg:px-6">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden text-muted-foreground hover:text-foreground mr-4"
          >
            <Menu className="h-6 w-6" />
          </button>
          
          <div className="flex-1" />
          
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground hidden sm:block">
              {user?.email}
            </span>
          </div>
        </header>

        <main className="flex-1 overflow-auto p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
