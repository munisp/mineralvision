import { useState, useEffect, useCallback } from 'react';
import {
  Plus,
  Search,
  User,
  Edit,
  Trash2,
  Key,
  RefreshCw,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { AxiosError } from 'axios';
import { onboardingApi, usersApi } from '../../services/api';

interface UserData {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  roles: string[];
  status: 'active' | 'inactive' | 'pending';
  lastLogin: string;
  createdAt: string;
}

const defaultUsers: UserData[] = [
  { id: '1', username: 'jsmith', email: 'john.smith@example.com', firstName: 'John', lastName: 'Smith', roles: ['admin'], status: 'active', lastLogin: '2024-01-15T10:30:00', createdAt: '2023-06-15' },
  { id: '2', username: 'jdoe', email: 'jane.doe@example.com', firstName: 'Jane', lastName: 'Doe', roles: ['resource_geologist'], status: 'active', lastLogin: '2024-01-15T09:15:00', createdAt: '2023-07-20' },
  { id: '3', username: 'mjohnson', email: 'mike.johnson@example.com', firstName: 'Mike', lastName: 'Johnson', roles: ['geologist'], status: 'active', lastLogin: '2024-01-14T16:45:00', createdAt: '2023-08-10' },
  { id: '4', username: 'swilson', email: 'sarah.wilson@example.com', firstName: 'Sarah', lastName: 'Wilson', roles: ['geologist'], status: 'inactive', lastLogin: '2024-01-10T11:20:00', createdAt: '2023-09-05' },
  { id: '5', username: 'rbrown', email: 'robert.brown@example.com', firstName: 'Robert', lastName: 'Brown', roles: ['viewer'], status: 'pending', lastLogin: '', createdAt: '2024-01-14' },
];

const roleColors: Record<string, string> = {
  admin: 'bg-red-500/10 text-red-500',
  resource_geologist: 'bg-purple-500/10 text-purple-500',
  geologist: 'bg-blue-500/10 text-blue-500',
  viewer: 'bg-gray-500/10 text-gray-500',
};

const statusColors = {
  active: 'bg-green-500/10 text-green-500',
  inactive: 'bg-gray-500/10 text-gray-500',
  pending: 'bg-yellow-500/10 text-yellow-500',
};

export default function UsersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState('all');
  const [showNewUserModal, setShowNewUserModal] = useState(false);
  const [users, setUsers] = useState<UserData[]>(defaultUsers);
  const [isLoading, setIsLoading] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const fetchUsers = useCallback(async () => {
    try {
      const response = await usersApi.list();
      if (Array.isArray(response.data) && response.data.length > 0) {
        setUsers(response.data);
        setIsConnected(true);
      } else {
        setUsers(defaultUsers);
        setIsConnected(false);
      }
    } catch {
      setUsers(defaultUsers);
      setIsConnected(false);
    }
  }, []);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchUsers();
    setIsRefreshing(false);
  };

  useEffect(() => {
    const loadData = async () => {
      await fetchUsers();
      setIsLoading(false);
    };
    loadData();
  }, [fetchUsers]);

  const filteredUsers = users.filter((user) => {
    const matchesSearch =
      user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      `${user.firstName} ${user.lastName}`.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRole = roleFilter === 'all' || user.roles.includes(roleFilter);
    return matchesSearch && matchesRole;
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
          <h1 className="text-2xl font-bold text-foreground">User Management</h1>
          <p className="text-muted-foreground">Manage users and their permissions</p>
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
            onClick={() => setShowNewUserModal(true)}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            Add User
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Total Users</p>
          <p className="text-2xl font-bold text-foreground">{users.length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Active</p>
          <p className="text-2xl font-bold text-green-500">{users.filter((u) => u.status === 'active').length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Inactive</p>
          <p className="text-2xl font-bold text-gray-500">{users.filter((u) => u.status === 'inactive').length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Pending</p>
          <p className="text-2xl font-bold text-yellow-500">{users.filter((u) => u.status === 'pending').length}</p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search users..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-background border border-input rounded-lg text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>
        <select
          value={roleFilter}
          onChange={(e) => setRoleFilter(e.target.value)}
          className="px-3 py-2 bg-background border border-input rounded-lg text-foreground"
        >
          <option value="all">All Roles</option>
          <option value="admin">Admin</option>
          <option value="resource_geologist">Resource Geologist</option>
          <option value="geologist">Geologist</option>
          <option value="viewer">Viewer</option>
        </select>
      </div>

      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">User</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Email</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Roles</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Last Login</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((user) => (
                <tr key={user.id} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium text-foreground">{user.firstName} {user.lastName}</p>
                        <p className="text-sm text-muted-foreground">@{user.username}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-muted-foreground">{user.email}</td>
                  <td className="py-3 px-4">
                    <div className="flex flex-wrap gap-1">
                      {user.roles.map((role) => (
                        <span
                          key={role}
                          className={`px-2 py-0.5 text-xs font-medium rounded-full ${roleColors[role] || 'bg-gray-500/10 text-gray-500'}`}
                        >
                          {role.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full capitalize ${statusColors[user.status]}`}>
                      {user.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-muted-foreground">
                    {user.lastLogin ? new Date(user.lastLogin).toLocaleString() : 'Never'}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-1">
                      <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded" title="Edit">
                        <Edit className="h-4 w-4" />
                      </button>
                      <button className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-secondary rounded" title="Reset Password">
                        <Key className="h-4 w-4" />
                      </button>
                      <button className="p-1.5 text-muted-foreground hover:text-destructive hover:bg-destructive/10 rounded" title="Delete">
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

      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-lg font-semibold text-foreground mb-4">Role Permissions</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3 text-muted-foreground">Permission</th>
                <th className="text-center py-2 px-3 text-muted-foreground">Admin</th>
                <th className="text-center py-2 px-3 text-muted-foreground">Resource Geologist</th>
                <th className="text-center py-2 px-3 text-muted-foreground">Geologist</th>
                <th className="text-center py-2 px-3 text-muted-foreground">Viewer</th>
              </tr>
            </thead>
            <tbody>
              {[
                { name: 'View Projects', admin: true, rg: true, geo: true, viewer: true },
                { name: 'Edit Projects', admin: true, rg: true, geo: true, viewer: false },
                { name: 'Delete Projects', admin: true, rg: false, geo: false, viewer: false },
                { name: 'Run Estimations', admin: true, rg: true, geo: true, viewer: false },
                { name: 'Approve Reports', admin: true, rg: true, geo: false, viewer: false },
                { name: 'Manage Users', admin: true, rg: false, geo: false, viewer: false },
              ].map((perm) => (
                <tr key={perm.name} className="border-b border-border/50">
                  <td className="py-2 px-3 text-foreground">{perm.name}</td>
                  <td className="py-2 px-3 text-center">{perm.admin ? '✓' : '—'}</td>
                  <td className="py-2 px-3 text-center">{perm.rg ? '✓' : '—'}</td>
                  <td className="py-2 px-3 text-center">{perm.geo ? '✓' : '—'}</td>
                  <td className="py-2 px-3 text-center">{perm.viewer ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {showNewUserModal && (
        <NewUserModal onClose={() => setShowNewUserModal(false)} />
      )}
    </div>
  );
}

const INVITABLE_ROLES = [
  'viewer',
  'geologist',
  'resource_geologist',
  'field_technician',
  'investor',
  'regulator',
  'custodian',
  'org_admin',
];

function NewUserModal({ onClose }: { onClose: () => void }) {
  const [email, setEmail] = useState('');
  const [role, setRole] = useState('viewer');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<{ kind: 'success' | 'error'; message: string } | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setFeedback(null);
    try {
      // Invite into the first organization the current user belongs to.
      const orgsResponse = await onboardingApi.myOrgs();
      const org = orgsResponse.data.organizations[0];
      if (!org) {
        setFeedback({
          kind: 'error',
          message: 'You do not belong to any organization yet. Create one before inviting.',
        });
        return;
      }
      const result = await onboardingApi.invite(org.id, { email, role });
      setFeedback({
        kind: 'success',
        message: `Invitation sent to ${result.data.email} (${result.data.role}) in ${org.name} — delivery: ${result.data.email_delivery}.`,
      });
    } catch (err) {
      const detail = (err as AxiosError)?.response?.data as { detail?: string } | undefined;
      setFeedback({
        kind: 'error',
        message: detail?.detail || 'Failed to send invitation. Please try again.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-xl w-full max-w-md">
        <div className="p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Invite Stakeholder</h2>
        </div>
        <form className="p-6 space-y-4" onSubmit={handleSubmit}>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Role</label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value)}
              className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground"
            >
              {INVITABLE_ROLES.map((r) => (
                <option key={r} value={r}>
                  {r.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </div>
          {feedback && (
            <div
              className={`text-sm px-3 py-2 rounded-lg ${
                feedback.kind === 'success'
                  ? 'bg-green-500/10 text-green-500'
                  : 'bg-red-500/10 text-red-500'
              }`}
            >
              {feedback.message}
            </div>
          )}
          <div className="flex gap-3 pt-4">
            <button type="button" onClick={onClose} className="flex-1 px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
            >
              {isSubmitting ? 'Sending…' : 'Send Invite'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
