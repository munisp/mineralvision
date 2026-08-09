import { useState } from 'react';
import {
  CheckCircle,
  AlertTriangle,
  XCircle,
  RefreshCw,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ScatterChart, Scatter } from 'recharts';

const mockControlChartData = [
  { batch: 1, value: 2.45, expected: 2.50 },
  { batch: 2, value: 2.52, expected: 2.50 },
  { batch: 3, value: 2.48, expected: 2.50 },
  { batch: 4, value: 2.55, expected: 2.50 },
  { batch: 5, value: 2.43, expected: 2.50 },
  { batch: 6, value: 2.51, expected: 2.50 },
  { batch: 7, value: 2.78, expected: 2.50 },
  { batch: 8, value: 2.49, expected: 2.50 },
  { batch: 9, value: 2.46, expected: 2.50 },
  { batch: 10, value: 2.53, expected: 2.50 },
];

const mockDuplicateData = [
  { original: 1.2, duplicate: 1.18 },
  { original: 2.5, duplicate: 2.48 },
  { original: 0.8, duplicate: 0.82 },
  { original: 3.1, duplicate: 3.05 },
  { original: 1.9, duplicate: 1.95 },
  { original: 0.5, duplicate: 0.52 },
  { original: 2.8, duplicate: 2.75 },
  { original: 1.5, duplicate: 1.48 },
];

const mockQAQCSummary = [
  { type: 'Standards', total: 245, passed: 238, failed: 7, rate: 97.1 },
  { type: 'Blanks', total: 122, passed: 120, failed: 2, rate: 98.4 },
  { type: 'Field Duplicates', total: 156, passed: 148, failed: 8, rate: 94.9 },
  { type: 'Lab Duplicates', total: 89, passed: 86, failed: 3, rate: 96.6 },
  { type: 'Umpire Assays', total: 45, passed: 43, failed: 2, rate: 95.6 },
];

const mockAlerts = [
  { id: 1, type: 'warning', message: 'CRM-Au-01 exceeded 2 standard deviations in batch 7', time: '2 hours ago' },
  { id: 2, type: 'error', message: 'Blank sample BLK-045 returned 0.15 g/t Au (threshold: 0.05)', time: '4 hours ago' },
  { id: 3, type: 'warning', message: 'Field duplicate HARD of DDH-2024-152 shows 12% variance', time: '1 day ago' },
];

export default function QAQCPage() {
  const [selectedStandard, setSelectedStandard] = useState('CRM-Au-01');
  const [dateRange, setDateRange] = useState('30d');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">QA/QC Analysis</h1>
          <p className="text-muted-foreground">Monitor quality control metrics and identify issues</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="px-3 py-2 bg-background border border-input rounded-lg text-foreground"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="all">All time</option>
          </select>
          <button className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {mockQAQCSummary.map((item) => (
          <div key={item.type} className="bg-card border border-border rounded-xl p-4">
            <p className="text-sm text-muted-foreground">{item.type}</p>
            <div className="flex items-end justify-between mt-2">
              <div>
                <p className="text-2xl font-bold text-foreground">{item.rate}%</p>
                <p className="text-xs text-muted-foreground">{item.passed}/{item.total} passed</p>
              </div>
              {item.rate >= 95 ? (
                <CheckCircle className="h-8 w-8 text-green-500" />
              ) : item.rate >= 90 ? (
                <AlertTriangle className="h-8 w-8 text-yellow-500" />
              ) : (
                <XCircle className="h-8 w-8 text-red-500" />
              )}
            </div>
          </div>
        ))}
      </div>

      {mockAlerts.length > 0 && (
        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-foreground mb-4">Recent Alerts</h2>
          <div className="space-y-3">
            {mockAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`flex items-start gap-3 p-3 rounded-lg ${
                  alert.type === 'error' ? 'bg-red-500/10' : 'bg-yellow-500/10'
                }`}
              >
                {alert.type === 'error' ? (
                  <XCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
                ) : (
                  <AlertTriangle className="h-5 w-5 text-yellow-500 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <p className="text-sm text-foreground">{alert.message}</p>
                  <p className="text-xs text-muted-foreground mt-1">{alert.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Control Chart - Standards</h2>
            <select
              value={selectedStandard}
              onChange={(e) => setSelectedStandard(e.target.value)}
              className="px-3 py-1.5 text-sm bg-background border border-input rounded-lg text-foreground"
            >
              <option value="CRM-Au-01">CRM-Au-01 (2.50 g/t)</option>
              <option value="CRM-Au-02">CRM-Au-02 (0.85 g/t)</option>
              <option value="CRM-Cu-01">CRM-Cu-01 (1.25%)</option>
            </select>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockControlChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="batch" stroke="#9ca3af" fontSize={12} />
                <YAxis stroke="#9ca3af" fontSize={12} domain={[2.2, 2.8]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <ReferenceLine y={2.50} stroke="#10b981" strokeDasharray="5 5" label="Expected" />
                <ReferenceLine y={2.60} stroke="#f59e0b" strokeDasharray="3 3" label="+2SD" />
                <ReferenceLine y={2.40} stroke="#f59e0b" strokeDasharray="3 3" label="-2SD" />
                <ReferenceLine y={2.65} stroke="#ef4444" strokeDasharray="3 3" label="+3SD" />
                <ReferenceLine y={2.35} stroke="#ef4444" strokeDasharray="3 3" label="-3SD" />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center justify-center gap-6 mt-4 text-xs">
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-green-500"></span> Expected</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-500"></span> Warning (2SD)</span>
            <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500"></span> Failure (3SD)</span>
          </div>
        </div>

        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-lg font-semibold text-foreground mb-4">Duplicate Analysis</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="original" name="Original" stroke="#9ca3af" fontSize={12} label={{ value: 'Original (g/t)', position: 'bottom', fill: '#9ca3af' }} />
                <YAxis dataKey="duplicate" name="Duplicate" stroke="#9ca3af" fontSize={12} label={{ value: 'Duplicate (g/t)', angle: -90, position: 'left', fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                />
                <Scatter data={mockDuplicateData} fill="#3b82f6" />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 4, y: 4 }]} stroke="#10b981" strokeDasharray="5 5" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4 text-center">
            <div>
              <p className="text-sm text-muted-foreground">HARD</p>
              <p className="text-lg font-semibold text-foreground">4.2%</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Correlation</p>
              <p className="text-lg font-semibold text-foreground">0.987</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Bias</p>
              <p className="text-lg font-semibold text-foreground">-0.8%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-card border border-border rounded-xl p-5">
        <h2 className="text-lg font-semibold text-foreground mb-4">Blank Analysis</h2>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Sample ID</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Batch</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Au (g/t)</th>
                <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Threshold</th>
                <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              {[
                { id: 'BLK-048', batch: 'B-2024-048', value: 0.02, threshold: 0.05, status: 'pass' },
                { id: 'BLK-047', batch: 'B-2024-047', value: 0.01, threshold: 0.05, status: 'pass' },
                { id: 'BLK-046', batch: 'B-2024-046', value: 0.03, threshold: 0.05, status: 'pass' },
                { id: 'BLK-045', batch: 'B-2024-045', value: 0.15, threshold: 0.05, status: 'fail' },
                { id: 'BLK-044', batch: 'B-2024-044', value: 0.02, threshold: 0.05, status: 'pass' },
              ].map((blank) => (
                <tr key={blank.id} className="border-b border-border/50">
                  <td className="py-3 px-4 font-medium text-foreground">{blank.id}</td>
                  <td className="py-3 px-4 text-muted-foreground">{blank.batch}</td>
                  <td className="py-3 px-4 text-right text-foreground">{blank.value.toFixed(2)}</td>
                  <td className="py-3 px-4 text-right text-muted-foreground">{blank.threshold.toFixed(2)}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      blank.status === 'pass' ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'
                    }`}>
                      {blank.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
