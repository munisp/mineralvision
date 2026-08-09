import { useState, useEffect } from 'react';
import {
  Satellite,
  Signal,
  SignalHigh,
  SignalLow,
  SignalZero,
  BatteryLow,
  BatteryMedium,
  BatteryFull,
  MapPin,
  Target,
  Compass,
  Activity,
  Wifi,
  WifiOff,
  Play,
  Pause,
  Settings,
  Smartphone,
  Plane,
  Globe,
  Crosshair,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Constellation {
  name: string;
  code: string;
  satellites: number;
  inUse: number;
  avgSnr: number;
  color: string;
  weight: number;
}

interface GNSSPosition {
  latitude: number;
  longitude: number;
  altitude: number;
  accuracy: number;
  hdop: number;
  vdop: number;
  pdop: number;
  fixType: string;
  timestamp: string;
}

interface SatelliteInfo {
  prn: number;
  constellation: string;
  elevation: number;
  azimuth: number;
  snr: number;
  inUse: boolean;
}

const constellations: Constellation[] = [
  { name: 'GPS', code: 'G', satellites: 12, inUse: 8, avgSnr: 42, color: '#3b82f6', weight: 1.0 },
  { name: 'GLONASS', code: 'R', satellites: 8, inUse: 6, avgSnr: 38, color: '#ef4444', weight: 0.95 },
  { name: 'Galileo', code: 'E', satellites: 10, inUse: 7, avgSnr: 44, color: '#22c55e', weight: 1.05 },
  { name: 'BeiDou', code: 'C', satellites: 14, inUse: 9, avgSnr: 40, color: '#f59e0b', weight: 0.98 },
];

const mockSatellites: SatelliteInfo[] = [
  { prn: 1, constellation: 'GPS', elevation: 45, azimuth: 120, snr: 42, inUse: true },
  { prn: 3, constellation: 'GPS', elevation: 65, azimuth: 45, snr: 48, inUse: true },
  { prn: 7, constellation: 'GPS', elevation: 30, azimuth: 200, snr: 35, inUse: true },
  { prn: 14, constellation: 'GPS', elevation: 55, azimuth: 310, snr: 44, inUse: true },
  { prn: 2, constellation: 'GLONASS', elevation: 40, azimuth: 90, snr: 38, inUse: true },
  { prn: 5, constellation: 'GLONASS', elevation: 70, azimuth: 180, snr: 42, inUse: true },
  { prn: 1, constellation: 'Galileo', elevation: 50, azimuth: 270, snr: 46, inUse: true },
  { prn: 4, constellation: 'Galileo', elevation: 35, azimuth: 30, snr: 40, inUse: true },
  { prn: 3, constellation: 'BeiDou', elevation: 60, azimuth: 150, snr: 44, inUse: true },
  { prn: 8, constellation: 'BeiDou', elevation: 25, azimuth: 240, snr: 32, inUse: false },
];

const mockPosition: GNSSPosition = {
  latitude: 6.5244,
  longitude: 3.3792,
  altitude: 42.5,
  accuracy: 0.025,
  hdop: 0.8,
  vdop: 1.2,
  pdop: 1.4,
  fixType: 'RTK Fixed',
  timestamp: new Date().toISOString(),
};

const mockAccuracyHistory = Array.from({ length: 30 }, (_, i) => ({
  time: `${i}s`,
  horizontal: 0.02 + Math.random() * 0.02,
  vertical: 0.03 + Math.random() * 0.03,
}));

// mockConstellationData used for radar chart visualization
const _mockConstellationData = constellations.map(c => ({
  constellation: c.name,
  satellites: c.inUse,
  snr: c.avgSnr,
  weight: c.weight * 100,
}));
void _mockConstellationData; // Suppress unused variable warning

const correctionTypes= [
  { name: 'Ionospheric', model: 'NeQuick', status: 'active', correction: '-2.3m' },
  { name: 'Tropospheric', model: 'GPT2w', status: 'active', correction: '-1.8m' },
  { name: 'Clock Bias', model: 'PPP', status: 'active', correction: '+15.2ns' },
  { name: 'Orbit', model: 'Precise', status: 'active', correction: '+0.8m' },
];

export default function GNSSPage() {
  const [isTracking, setIsTracking] = useState(true);
  const [position, setPosition] = useState(mockPosition);
  const [powerMode, setPowerMode] = useState<'high' | 'balanced' | 'low'>('high');
  const [ntripConnected, setNtripConnected] = useState(true);
  const [selectedMode, setSelectedMode] = useState<'field' | 'mobile' | 'drone'>('field');
  const [accuracyHistory, setAccuracyHistory] = useState(mockAccuracyHistory);

  useEffect(() => {
    if (isTracking) {
      const interval = setInterval(() => {
        setPosition(prev => ({
          ...prev,
          latitude: prev.latitude + (Math.random() - 0.5) * 0.00001,
          longitude: prev.longitude + (Math.random() - 0.5) * 0.00001,
          accuracy: 0.02 + Math.random() * 0.01,
          timestamp: new Date().toISOString(),
        }));
        setAccuracyHistory(prev => [
          ...prev.slice(1),
          {
            time: `${parseInt(prev[prev.length - 1].time) + 1}s`,
            horizontal: 0.02 + Math.random() * 0.02,
            vertical: 0.03 + Math.random() * 0.03,
          },
        ]);
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isTracking]);

  const totalSatellites = constellations.reduce((sum, c) => sum + c.satellites, 0);
  const totalInUse = constellations.reduce((sum, c) => sum + c.inUse, 0);

  const getSignalIcon = (snr: number) => {
    if (snr >= 40) return <SignalHigh className="h-4 w-4 text-green-500" />;
    if (snr >= 30) return <Signal className="h-4 w-4 text-yellow-500" />;
    if (snr >= 20) return <SignalLow className="h-4 w-4 text-orange-500" />;
    return <SignalZero className="h-4 w-4 text-red-500" />;
  };

  const getBatteryIcon = () => {
    if (powerMode === 'high') return <BatteryFull className="h-4 w-4 text-green-500" />;
    if (powerMode === 'balanced') return <BatteryMedium className="h-4 w-4 text-yellow-500" />;
    return <BatteryLow className="h-4 w-4 text-red-500" />;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Enhanced GNSS</h1>
          <p className="text-muted-foreground">Multi-constellation positioning with PPP/RTK support</p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 px-3 py-1.5 bg-card border border-border rounded-lg">
            <span className="text-sm text-muted-foreground">Mode:</span>
            <select
              value={selectedMode}
              onChange={(e) => setSelectedMode(e.target.value as 'field' | 'mobile' | 'drone')}
              className="bg-transparent text-foreground text-sm font-medium border-none focus:outline-none"
            >
              <option value="field">Field Collection</option>
              <option value="mobile">Mobile</option>
              <option value="drone">Drone Survey</option>
            </select>
          </div>
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Configure
          </button>
          <button
            onClick={() => setIsTracking(!isTracking)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isTracking
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}
          >
            {isTracking ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isTracking ? 'Stop' : 'Start'}
          </button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Fix Type</p>
            <Target className="h-4 w-4 text-green-500" />
          </div>
          <p className="text-2xl font-bold text-green-500">{position.fixType}</p>
          <p className="text-xs text-muted-foreground mt-1">cm-level accuracy</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Satellites</p>
            <Satellite className="h-4 w-4 text-primary" />
          </div>
          <p className="text-2xl font-bold text-foreground">{totalInUse} / {totalSatellites}</p>
          <p className="text-xs text-muted-foreground mt-1">4 constellations active</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">Accuracy</p>
            <Crosshair className="h-4 w-4 text-primary" />
          </div>
          <p className="text-2xl font-bold text-foreground">{(position.accuracy * 100).toFixed(1)} cm</p>
          <p className="text-xs text-muted-foreground mt-1">HDOP: {position.hdop.toFixed(2)}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm text-muted-foreground">NTRIP Status</p>
            {ntripConnected ? <Wifi className="h-4 w-4 text-green-500" /> : <WifiOff className="h-4 w-4 text-red-500" />}
          </div>
          <p className={`text-2xl font-bold ${ntripConnected ? 'text-green-500' : 'text-red-500'}`}>
            {ntripConnected ? 'Connected' : 'Disconnected'}
          </p>
          <p className="text-xs text-muted-foreground mt-1">RTK corrections active</p>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Position Info */}
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Current Position</h2>
            <MapPin className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Latitude</p>
                <p className="text-lg font-mono text-foreground">{position.latitude.toFixed(8)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Longitude</p>
                <p className="text-lg font-mono text-foreground">{position.longitude.toFixed(8)}</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-xs text-muted-foreground">Altitude (m)</p>
                <p className="text-lg font-mono text-foreground">{position.altitude.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Accuracy (m)</p>
                <p className="text-lg font-mono text-green-500">{position.accuracy.toFixed(3)}</p>
              </div>
            </div>
            <div className="pt-3 border-t border-border">
              <p className="text-xs text-muted-foreground mb-2">Dilution of Precision</p>
              <div className="grid grid-cols-3 gap-2">
                <div className="bg-background rounded-lg p-2 text-center">
                  <p className="text-xs text-muted-foreground">HDOP</p>
                  <p className="text-sm font-bold text-foreground">{position.hdop.toFixed(2)}</p>
                </div>
                <div className="bg-background rounded-lg p-2 text-center">
                  <p className="text-xs text-muted-foreground">VDOP</p>
                  <p className="text-sm font-bold text-foreground">{position.vdop.toFixed(2)}</p>
                </div>
                <div className="bg-background rounded-lg p-2 text-center">
                  <p className="text-xs text-muted-foreground">PDOP</p>
                  <p className="text-sm font-bold text-foreground">{position.pdop.toFixed(2)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Constellation Status */}
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Constellations</h2>
            <Globe className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-3">
            {constellations.map((constellation) => (
              <div key={constellation.code} className="bg-background rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: constellation.color }}
                    />
                    <span className="font-medium text-foreground">{constellation.name}</span>
                    <span className="text-xs text-muted-foreground">({constellation.code})</span>
                  </div>
                  {getSignalIcon(constellation.avgSnr)}
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">
                    {constellation.inUse}/{constellation.satellites} satellites
                  </span>
                  <span className="text-muted-foreground">
                    SNR: {constellation.avgSnr} dB
                  </span>
                  <span className="text-muted-foreground">
                    Weight: {(constellation.weight * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${(constellation.inUse / constellation.satellites) * 100}%`,
                      backgroundColor: constellation.color,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Atmospheric Corrections */}
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Corrections</h2>
            <Activity className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-3">
            {correctionTypes.map((correction) => (
              <div key={correction.name} className="bg-background rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-foreground">{correction.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    correction.status === 'active'
                      ? 'bg-green-500/20 text-green-500'
                      : 'bg-yellow-500/20 text-yellow-500'
                  }`}>
                    {correction.status}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Model: {correction.model}</span>
                  <span className="font-mono text-primary">{correction.correction}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-muted-foreground">NTRIP Caster</span>
              <button
                onClick={() => setNtripConnected(!ntripConnected)}
                className={`text-xs px-2 py-1 rounded ${
                  ntripConnected
                    ? 'bg-green-500/20 text-green-500'
                    : 'bg-red-500/20 text-red-500'
                }`}
              >
                {ntripConnected ? 'Connected' : 'Connect'}
              </button>
            </div>
            <p className="text-xs text-muted-foreground">rtk2go.com:2101/NEAREST</p>
          </div>
        </div>
      </div>

      {/* Accuracy Chart */}
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">Position Accuracy History</h2>
          <div className="flex items-center gap-2">
            {isTracking && (
              <span className="flex items-center gap-1 text-sm text-green-500">
                <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
                Live
              </span>
            )}
          </div>
        </div>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={accuracyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
              <YAxis stroke="#9ca3af" fontSize={12} domain={[0, 0.1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}cm`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                formatter={(value: number) => [`${(value * 100).toFixed(2)} cm`, '']}
              />
              <Line type="monotone" dataKey="horizontal" stroke="#3b82f6" strokeWidth={2} dot={false} name="Horizontal" />
              <Line type="monotone" dataKey="vertical" stroke="#22c55e" strokeWidth={2} dot={false} name="Vertical" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center justify-center gap-6 mt-4 text-sm">
          <span className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-blue-500"></span>
            Horizontal Accuracy
          </span>
          <span className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-green-500"></span>
            Vertical Accuracy
          </span>
        </div>
      </div>

      {/* Mobile & Drone Features */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mobile Features */}
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Mobile GNSS</h2>
            <Smartphone className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Power Mode</span>
              <div className="flex items-center gap-2">
                {getBatteryIcon()}
                <select
                  value={powerMode}
                  onChange={(e) => setPowerMode(e.target.value as 'high' | 'balanced' | 'low')}
                  className="bg-background border border-input rounded px-2 py-1 text-sm text-foreground"
                >
                  <option value="high">High Accuracy</option>
                  <option value="balanced">Balanced</option>
                  <option value="low">Battery Saver</option>
                </select>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Sensor Fusion</span>
              <span className="text-sm text-green-500">IMU + GNSS Active</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">A-GPS</span>
              <span className="text-sm text-green-500">Enabled</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Geofencing</span>
              <span className="text-sm text-foreground">3 zones active</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Motion State</span>
              <span className="text-sm text-foreground">Stationary</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Offline Cache</span>
              <span className="text-sm text-foreground">2.4 MB corrections</span>
            </div>
          </div>
        </div>

        {/* Drone Features */}
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-foreground">Drone GNSS</h2>
            <Plane className="h-5 w-5 text-primary" />
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Survey Mode</span>
              <select className="bg-background border border-input rounded px-2 py-1 text-sm text-foreground">
                <option>RTK Real-time</option>
                <option>PPK Post-process</option>
                <option>Autonomous</option>
              </select>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Survey Grade</span>
              <span className="text-sm text-green-500">Engineering (2cm)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Base Station</span>
              <span className="text-sm text-foreground">BASE-001 (1.2km)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Lever Arm</span>
              <span className="text-sm text-foreground">Calibrated</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Kalman Filter</span>
              <span className="text-sm text-green-500">Converged</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Quality Score</span>
              <span className="text-sm font-bold text-green-500">98.5%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Satellite Sky View Placeholder */}
      <div className="bg-card border border-border rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-foreground">Satellite Sky View</h2>
          <Compass className="h-5 w-5 text-primary" />
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-64 flex items-center justify-center bg-background rounded-lg relative">
            {/* Sky plot visualization */}
            <div className="absolute inset-4 rounded-full border-2 border-border"></div>
            <div className="absolute inset-12 rounded-full border border-border"></div>
            <div className="absolute inset-20 rounded-full border border-border"></div>
            <div className="absolute w-full h-0.5 bg-border top-1/2"></div>
            <div className="absolute h-full w-0.5 bg-border left-1/2"></div>
            {mockSatellites.slice(0, 8).map((sat) => {
              const angle = (sat.azimuth * Math.PI) / 180;
              const radius = ((90 - sat.elevation) / 90) * 100;
              const x = 50 + radius * Math.sin(angle) * 0.4;
              const y = 50 - radius * Math.cos(angle) * 0.4;
              const color = constellations.find(c => c.name === sat.constellation)?.color || '#888';
              return (
                <div
                  key={`${sat.constellation}-${sat.prn}`}
                  className="absolute w-4 h-4 rounded-full flex items-center justify-center text-[8px] font-bold text-white"
                  style={{
                    left: `${x}%`,
                    top: `${y}%`,
                    backgroundColor: color,
                    transform: 'translate(-50%, -50%)',
                  }}
                >
                  {sat.prn}
                </div>
              );
            })}
            <div className="absolute top-2 left-1/2 -translate-x-1/2 text-xs text-muted-foreground">N</div>
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-xs text-muted-foreground">S</div>
            <div className="absolute left-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">W</div>
            <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">E</div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-foreground mb-3">Satellite List</h3>
            <div className="space-y-2 max-h-56 overflow-y-auto">
              {mockSatellites.map((sat) => {
                const color = constellations.find(c => c.name === sat.constellation)?.color || '#888';
                return (
                  <div
                    key={`${sat.constellation}-${sat.prn}`}
                    className="flex items-center justify-between bg-background rounded-lg px-3 py-2"
                  >
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                      <span className="text-sm font-medium text-foreground">
                        {sat.constellation[0]}{sat.prn}
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>El: {sat.elevation}</span>
                      <span>Az: {sat.azimuth}</span>
                      <span>SNR: {sat.snr}</span>
                      {sat.inUse ? (
                        <span className="text-green-500">In Use</span>
                      ) : (
                        <span className="text-muted-foreground">Tracked</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
