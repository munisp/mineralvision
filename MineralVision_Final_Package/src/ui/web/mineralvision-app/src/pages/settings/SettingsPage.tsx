import { useState } from 'react';
import {
  User,
  Bell,
  Shield,
  Palette,
  Globe,
  Database,
  Key,
  Save,
} from 'lucide-react';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('profile');

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'appearance', label: 'Appearance', icon: Palette },
    { id: 'regional', label: 'Regional', icon: Globe },
    { id: 'data', label: 'Data & Storage', icon: Database },
    { id: 'api', label: 'API Keys', icon: Key },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-muted-foreground">Manage your account and application preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-card border border-border rounded-xl p-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                  activeTab === tab.id
                    ? 'bg-primary/10 text-primary'
                    : 'text-foreground hover:bg-secondary'
                }`}
              >
                <tab.icon className="h-4 w-4" />
                <span className="text-sm font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="lg:col-span-3">
          {activeTab === 'profile' && <ProfileSettings />}
          {activeTab === 'notifications' && <NotificationSettings />}
          {activeTab === 'security' && <SecuritySettings />}
          {activeTab === 'appearance' && <AppearanceSettings />}
          {activeTab === 'regional' && <RegionalSettings />}
          {activeTab === 'data' && <DataSettings />}
          {activeTab === 'api' && <ApiSettings />}
        </div>
      </div>
    </div>
  );
}

function ProfileSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Profile Settings</h2>
      <div className="space-y-6">
        <div className="flex items-center gap-6">
          <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
            <User className="h-10 w-10 text-primary" />
          </div>
          <div>
            <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 text-sm">
              Upload Photo
            </button>
            <p className="text-xs text-muted-foreground mt-2">JPG, PNG or GIF. Max 2MB.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">First Name</label>
            <input type="text" defaultValue="John" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Last Name</label>
            <input type="text" defaultValue="Smith" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Email</label>
            <input type="email" defaultValue="john.smith@example.com" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
          </div>
          <div>
            <label className="block text-sm font-medium text-foreground mb-1.5">Job Title</label>
            <input type="text" defaultValue="Senior Geologist" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Bio</label>
          <textarea rows={3} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground resize-none" placeholder="Tell us about yourself..." />
        </div>

        <div className="flex justify-end">
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2">
            <Save className="h-4 w-4" />
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}

function NotificationSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Notification Preferences</h2>
      <div className="space-y-4">
        {[
          { label: 'Email notifications', description: 'Receive email updates about your projects' },
          { label: 'QA/QC alerts', description: 'Get notified when QA/QC checks fail' },
          { label: 'Report completion', description: 'Notify when reports are generated' },
          { label: 'Data import status', description: 'Updates on data import progress' },
          { label: 'Team mentions', description: 'When someone mentions you in comments' },
        ].map((item) => (
          <div key={item.label} className="flex items-center justify-between p-4 bg-background rounded-lg">
            <div>
              <p className="font-medium text-foreground">{item.label}</p>
              <p className="text-sm text-muted-foreground">{item.description}</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" defaultChecked className="sr-only peer" />
              <div className="w-11 h-6 bg-secondary rounded-full peer peer-checked:bg-primary peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all"></div>
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}

function SecuritySettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Security Settings</h2>
      <div className="space-y-6">
        <div>
          <h3 className="font-medium text-foreground mb-4">Change Password</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Current Password</label>
              <input type="password" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">New Password</label>
              <input type="password" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">Confirm New Password</label>
              <input type="password" className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground" />
            </div>
            <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90">
              Update Password
            </button>
          </div>
        </div>

        <div className="pt-6 border-t border-border">
          <h3 className="font-medium text-foreground mb-4">Two-Factor Authentication</h3>
          <div className="flex items-center justify-between p-4 bg-background rounded-lg">
            <div>
              <p className="font-medium text-foreground">Enable 2FA</p>
              <p className="text-sm text-muted-foreground">Add an extra layer of security to your account</p>
            </div>
            <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary">
              Enable
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function AppearanceSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Appearance</h2>
      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-foreground mb-3">Theme</label>
          <div className="grid grid-cols-3 gap-4">
            {['light', 'dark', 'system'].map((theme) => (
              <button
                key={theme}
                className={`p-4 rounded-lg border text-center capitalize ${
                  theme === 'dark' ? 'border-primary bg-primary/10' : 'border-input hover:border-primary/50'
                }`}
              >
                {theme}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-foreground mb-3">Accent Color</label>
          <div className="flex gap-3">
            {['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'].map((color) => (
              <button
                key={color}
                className="h-10 w-10 rounded-full border-2 border-transparent hover:border-white"
                style={{ backgroundColor: color }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function RegionalSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Regional Settings</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Language</label>
          <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
            <option>English (US)</option>
            <option>English (UK)</option>
            <option>Spanish</option>
            <option>French</option>
            <option>Portuguese</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Timezone</label>
          <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
            <option>UTC-08:00 Pacific Time</option>
            <option>UTC-05:00 Eastern Time</option>
            <option>UTC+00:00 GMT</option>
            <option>UTC+10:00 Australian Eastern</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Units</label>
          <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
            <option>Metric (m, kg)</option>
            <option>Imperial (ft, lb)</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-foreground mb-1.5">Coordinate System</label>
          <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
            <option>WGS84</option>
            <option>UTM</option>
            <option>Local Grid</option>
          </select>
        </div>
      </div>
    </div>
  );
}

function DataSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">Data & Storage</h2>
      <div className="space-y-6">
        <div className="p-4 bg-background rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-foreground">Storage Used</span>
            <span className="text-foreground font-medium">12.5 GB / 50 GB</span>
          </div>
          <div className="w-full bg-secondary rounded-full h-2">
            <div className="bg-primary h-2 rounded-full" style={{ width: '25%' }} />
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between p-4 bg-background rounded-lg">
            <div>
              <p className="font-medium text-foreground">Export All Data</p>
              <p className="text-sm text-muted-foreground">Download all your project data</p>
            </div>
            <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary">
              Export
            </button>
          </div>
          <div className="flex items-center justify-between p-4 bg-background rounded-lg">
            <div>
              <p className="font-medium text-foreground">Clear Cache</p>
              <p className="text-sm text-muted-foreground">Free up space by clearing cached data</p>
            </div>
            <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary">
              Clear
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ApiSettings() {
  return (
    <div className="bg-card border border-border rounded-xl p-6">
      <h2 className="text-lg font-semibold text-foreground mb-6">API Keys</h2>
      <div className="space-y-4">
        <div className="p-4 bg-background rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div>
              <p className="font-medium text-foreground">Production Key</p>
              <p className="text-sm text-muted-foreground">Created Jan 10, 2024</p>
            </div>
            <button className="px-3 py-1 text-sm border border-input rounded text-foreground hover:bg-secondary">
              Revoke
            </button>
          </div>
          <code className="block p-2 bg-card rounded text-sm text-muted-foreground font-mono">
            mv_prod_••••••••••••••••
          </code>
        </div>

        <button className="w-full px-4 py-2 border border-dashed border-input rounded-lg text-foreground hover:bg-secondary">
          + Generate New API Key
        </button>
      </div>
    </div>
  );
}
