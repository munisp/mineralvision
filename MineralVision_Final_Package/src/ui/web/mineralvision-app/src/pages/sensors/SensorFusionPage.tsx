import { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Wifi,
  WifiOff,
  Play,
  Pause,
  RefreshCw,
  Thermometer,
  Gauge,
  Radio,
  Zap,
  CheckCircle,
  Layers,
  ChevronRight,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface Sensor {
  id: string;
  name: string;
  type: 'magnetometer' | 'gpr' | 'radiometric' | 'em' | 'lidar';
  status: 'online' | 'offline' | 'error';
  lastReading: string;
  value: number;
  unit: string;
}

const mockSensors: Sensor[] = [
  { id: '1', name: 'MAG-001', type: 'magnetometer', status: 'online', lastReading: '2024-01-15T10:30:00', value: 52450, unit: 'nT' },
  { id: '2', name: 'GPR-001', type: 'gpr', status: 'online', lastReading: '2024-01-15T10:30:00', value: 12.5, unit: 'm depth' },
  { id: '3', name: 'RAD-001', type: 'radiometric', status: 'online', lastReading: '2024-01-15T10:30:00', value: 145, unit: 'cps' },
  { id: '4', name: 'EM-001', type: 'em', status: 'offline', lastReading: '2024-01-15T09:15:00', value: 0, unit: 'mS/m' },
  { id: '5', name: 'LIDAR-001', type: 'lidar', status: 'online', lastReading: '2024-01-15T10:30:00', value: 1250, unit: 'm' },
];

const mockTimeSeriesData = Array.from({ length: 20 }, (_, i) => ({
  time: `${10 + Math.floor(i / 4)}:${(i % 4) * 15}`,
  mag: 52400 + Math.random() * 100,
  rad: 140 + Math.random() * 20,
}));

const sensorIcons = {
  magnetometer: Gauge,
  gpr: Radio,
  radiometric: Thermometer,
  em: Activity,
  lidar: Activity,
};

// Interactive Fusion Demo Component
type FusionScenario = 'magnetic_survey' | 'gpr_subsurface' | 'multi_sensor';

interface FusionPipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed';
  duration?: number;
}

// Magnetic anomaly points
const magneticAnomalies = [
  { id: 'M1', x: 120, y: 100, intensity: 52800, type: 'high' },
  { id: 'M2', x: 280, y: 150, intensity: 52650, type: 'medium' },
  { id: 'M3', x: 400, y: 120, intensity: 52900, type: 'high' },
  { id: 'M4', x: 200, y: 220, intensity: 52400, type: 'low' },
  { id: 'M5', x: 350, y: 260, intensity: 52750, type: 'medium' },
];

// GPR subsurface features
const gprFeatures = [
  { id: 'G1', x1: 80, y1: 150, x2: 200, y2: 180, depth: 2.5, type: 'pipe' },
  { id: 'G2', x1: 250, y1: 120, x2: 380, y2: 140, depth: 4.2, type: 'void' },
  { id: 'G3', x1: 150, y1: 220, x2: 320, y2: 250, depth: 1.8, type: 'layer' },
];

// Multi-sensor fusion targets
const fusionTargets = [
  { id: 'F1', x: 180, y: 140, confidence: 94, sensors: ['MAG', 'RAD', 'GPR'] },
  { id: 'F2', x: 320, y: 180, confidence: 87, sensors: ['MAG', 'EM'] },
  { id: 'F3', x: 420, y: 220, confidence: 82, sensors: ['RAD', 'LIDAR'] },
  { id: 'F4', x: 250, y: 280, confidence: 78, sensors: ['MAG', 'RAD'] },
];

function FusionDemoOverlay({ scenario, show }: { scenario: FusionScenario; show: boolean }) {
  if (!show) return null;

  const getIntensityColor = (type: string) => {
    if (type === 'high') return '#ef4444';
    if (type === 'medium') return '#f59e0b';
    return '#22c55e';
  };

  if (scenario === 'magnetic_survey') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Survey grid */}
        <g stroke="#4b5563" strokeWidth="0.5" opacity="0.3">
          {[0, 1, 2, 3, 4].map(i => <line key={`h${i}`} x1="20" y1={20 + i * 70} x2="540" y2={20 + i * 70} />)}
          {[0, 1, 2, 3, 4, 5, 6].map(i => <line key={`v${i}`} x1={20 + i * 85} y1="20" x2={20 + i * 85} y2="300" />)}
        </g>

        {/* Magnetic contours */}
        <ellipse cx="200" cy="150" rx="100" ry="70" fill="none" stroke="#3b82f6" strokeWidth="1" opacity="0.3" />
        <ellipse cx="380" cy="180" rx="80" ry="60" fill="none" stroke="#3b82f6" strokeWidth="1" opacity="0.3" />

        {/* Anomaly points */}
        {magneticAnomalies.map((point, idx) => (
          <g key={point.id} className="animate-fade-in" style={{ animationDelay: `${idx * 100}ms` }}>
            <circle cx={point.x} cy={point.y} r="20" fill={getIntensityColor(point.type)} opacity="0.2" />
            <circle cx={point.x} cy={point.y} r="10" fill={getIntensityColor(point.type)} opacity="0.5" />
            <circle cx={point.x} cy={point.y} r="5" fill={getIntensityColor(point.type)} stroke="white" strokeWidth="1" />
            <text x={point.x} y={point.y - 25} fill="white" fontSize="9" textAnchor="middle">{point.intensity} nT</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(440, 20)">
          <rect x="0" y="0" width="100" height="60" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">Mag Intensity</text>
          <circle cx="18" cy="28" r="4" fill="#ef4444" />
          <text x="28" y="32" fill="#9ca3af" fontSize="8">High (&gt;52700)</text>
          <circle cx="18" cy="42" r="4" fill="#f59e0b" />
          <text x="28" y="46" fill="#9ca3af" fontSize="8">Medium</text>
          <circle cx="18" cy="56" r="4" fill="#22c55e" />
          <text x="28" y="60" fill="#9ca3af" fontSize="8">Low (&lt;52500)</text>
        </g>

        <rect x="10" y="285" width="150" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#3b82f6" fontSize="10" fontWeight="bold">Magnetic Survey (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'gpr_subsurface') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Ground surface */}
        <line x1="20" y1="80" x2="540" y2="80" stroke="#22c55e" strokeWidth="2" />
        <text x="30" y="70" fill="#22c55e" fontSize="10">Ground Surface</text>

        {/* Depth scale */}
        <g transform="translate(20, 80)">
          {[0, 1, 2, 3, 4].map(i => (
            <g key={i}>
              <line x1="0" y1={i * 50} x2="10" y2={i * 50} stroke="#9ca3af" strokeWidth="1" />
              <text x="-5" y={i * 50 + 4} fill="#9ca3af" fontSize="8" textAnchor="end">{i}m</text>
            </g>
          ))}
        </g>

        {/* GPR features */}
        {gprFeatures.map((feature, idx) => (
          <g key={feature.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            {feature.type === 'pipe' && (
              <>
                <ellipse cx={(feature.x1 + feature.x2) / 2} cy={80 + feature.depth * 40} rx="30" ry="15" fill="#f59e0b" opacity="0.3" stroke="#f59e0b" strokeWidth="2" />
                <text x={(feature.x1 + feature.x2) / 2} y={80 + feature.depth * 40 + 5} fill="white" fontSize="9" textAnchor="middle">Pipe</text>
              </>
            )}
            {feature.type === 'void' && (
              <>
                <rect x={feature.x1} y={80 + feature.depth * 40 - 15} width={feature.x2 - feature.x1} height="30" fill="#ef4444" opacity="0.3" stroke="#ef4444" strokeWidth="2" rx="4" />
                <text x={(feature.x1 + feature.x2) / 2} y={80 + feature.depth * 40 + 5} fill="white" fontSize="9" textAnchor="middle">Void</text>
              </>
            )}
            {feature.type === 'layer' && (
              <>
                <line x1={feature.x1} y1={80 + feature.depth * 40} x2={feature.x2} y2={80 + feature.depth * 40} stroke="#8b5cf6" strokeWidth="3" strokeDasharray="5,3" />
                <text x={(feature.x1 + feature.x2) / 2} y={80 + feature.depth * 40 - 8} fill="#c4b5fd" fontSize="9" textAnchor="middle">Layer Interface</text>
              </>
            )}
            <text x={(feature.x1 + feature.x2) / 2} y={80 + feature.depth * 40 + 20} fill="#9ca3af" fontSize="8" textAnchor="middle">{feature.depth}m depth</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(440, 100)">
          <rect x="0" y="0" width="100" height="70" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">GPR Features</text>
          <ellipse cx="18" cy="30" rx="8" ry="4" fill="#f59e0b" opacity="0.5" />
          <text x="32" y="34" fill="#9ca3af" fontSize="8">Pipe/Utility</text>
          <rect x="10" y="42" width="16" height="8" fill="#ef4444" opacity="0.5" rx="2" />
          <text x="32" y="50" fill="#9ca3af" fontSize="8">Void/Cavity</text>
          <line x1="10" y1="62" x2="26" y2="62" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="3,2" />
          <text x="32" y="66" fill="#9ca3af" fontSize="8">Layer</text>
        </g>

        <rect x="10" y="285" width="160" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#f59e0b" fontSize="10" fontWeight="bold">GPR Subsurface (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'multi_sensor') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Survey boundary */}
        <polygon points="30,30 530,25 540,290 40,300" fill="none" stroke="#22c55e" strokeWidth="2" strokeDasharray="8,4" className="animate-pulse" />

        {/* Fusion targets */}
        {fusionTargets.map((target, idx) => (
          <g key={target.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <circle cx={target.x} cy={target.y} r="35" fill="#8b5cf6" opacity="0.15" />
            <circle cx={target.x} cy={target.y} r="25" fill="#8b5cf6" opacity="0.25" />
            <circle cx={target.x} cy={target.y} r="15" fill="#8b5cf6" opacity="0.4" />
            {/* Crosshair */}
            <line x1={target.x - 20} y1={target.y} x2={target.x + 20} y2={target.y} stroke="#c4b5fd" strokeWidth="1" />
            <line x1={target.x} y1={target.y - 20} x2={target.x} y2={target.y + 20} stroke="#c4b5fd" strokeWidth="1" />
            <circle cx={target.x} cy={target.y} r="6" fill="#8b5cf6" stroke="white" strokeWidth="2" />
            {/* Confidence badge */}
            <rect x={target.x - 15} y={target.y - 45} width="30" height="16" fill="#1f2937" rx="3" />
            <text x={target.x} y={target.y - 33} fill="#22c55e" fontSize="10" textAnchor="middle" fontWeight="bold">{target.confidence}%</text>
            {/* Sensor badges */}
            <text x={target.x} y={target.y + 35} fill="#9ca3af" fontSize="7" textAnchor="middle">{target.sensors.join(' + ')}</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(430, 20)">
          <rect x="0" y="0" width="110" height="55" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">Fusion Targets</text>
          <circle cx="18" cy="30" r="6" fill="#8b5cf6" />
          <text x="30" y="34" fill="#9ca3af" fontSize="8">High confidence</text>
          <text x="10" y="48" fill="#9ca3af" fontSize="7">Multi-sensor correlation</text>
        </g>

        <rect x="10" y="285" width="170" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#8b5cf6" fontSize="10" fontWeight="bold">Multi-Sensor Fusion (Demo)</text>
      </svg>
    );
  }

  return null;
}

function InteractiveFusionDemo() {
  const [scenario, setScenario] = useState<FusionScenario>('magnetic_survey');
  const [isProcessing, setIsProcessing] = useState(false);
  const [fusionComplete, setFusionComplete] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [pipelineStages, setPipelineStages] = useState<FusionPipelineStage[]>([]);
  const processingRef = useRef<boolean>(false);

  const getInitialStages = (): FusionPipelineStage[] => [
    { name: 'Ingest', status: 'pending' },
    { name: 'Calibrate', status: 'pending' },
    { name: 'Filter', status: 'pending' },
    { name: 'Fuse', status: 'pending' },
    { name: 'Interpolate', status: 'pending' },
  ];

  const stageDurations = [80, 100, 120, 200, 150];

  const runFusion = async () => {
    if (processingRef.current) return;
    processingRef.current = true;
    setIsProcessing(true);
    setFusionComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());

    for (let i = 0; i < 5; i++) {
      setCurrentStageIdx(i);
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'running' } : s));
      await new Promise(r => setTimeout(r, stageDurations[i]));
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'completed', duration: stageDurations[i] } : s));
      if (i === 2) setShowOverlay(true);
    }

    setIsProcessing(false);
    setFusionComplete(true);
    processingRef.current = false;
  };

  useEffect(() => {
    setFusionComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());
  }, [scenario]);

  const scenarioLabels = {
    magnetic_survey: 'Magnetic Survey',
    gpr_subsurface: 'GPR Subsurface',
    multi_sensor: 'Multi-Sensor Fusion',
  };

  const scenarioDescriptions = {
    magnetic_survey: 'Fuse magnetometer data to identify magnetic anomalies',
    gpr_subsurface: 'Process GPR data to map subsurface features',
    multi_sensor: 'Combine multiple sensors for high-confidence targets',
  };

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Layers className="h-5 w-5 text-primary" />
            Interactive Fusion Demo
          </h2>
          <p className="text-sm text-muted-foreground">Select a scenario and run the fusion pipeline</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value as FusionScenario)}
            disabled={isProcessing}
            className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground"
          >
            <option value="magnetic_survey">Magnetic Survey</option>
            <option value="gpr_subsurface">GPR Subsurface</option>
            <option value="multi_sensor">Multi-Sensor Fusion</option>
          </select>
          <button
            onClick={runFusion}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {isProcessing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
            {isProcessing ? 'Processing...' : 'Run Fusion'}
          </button>
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 mb-4">
        {pipelineStages.map((stage, idx) => (
          <div key={stage.name} className="flex items-center">
            <div className={`flex flex-col items-center p-2 rounded-lg border min-w-[70px] transition-all duration-300 ${
              stage.status === 'completed' ? 'bg-green-500/10 border-green-500/30' :
              stage.status === 'running' ? 'bg-blue-500/10 border-blue-500/30' :
              'bg-secondary/50 border-border'
            }`}>
              <div className={`p-1 rounded-full mb-1 ${
                stage.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                stage.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                'bg-secondary text-muted-foreground'
              }`}>
                {stage.status === 'running' ? <RefreshCw className="h-3 w-3 animate-spin" /> :
                 stage.status === 'completed' ? <CheckCircle className="h-3 w-3" /> : <Activity className="h-3 w-3" />}
              </div>
              <span className="text-xs font-medium text-foreground">{stage.name}</span>
              {stage.status === 'completed' && stage.duration && (
                <span className="text-xs text-green-400">{stage.duration}ms</span>
              )}
            </div>
            {idx < pipelineStages.length - 1 && <ChevronRight className="h-3 w-3 text-muted-foreground mx-1" />}
          </div>
        ))}
      </div>

      {/* Visualization panel */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              {scenario === 'magnetic_survey' && <Gauge className="h-10 w-10 text-blue-600/50 mx-auto mb-2" />}
              {scenario === 'gpr_subsurface' && <Radio className="h-10 w-10 text-orange-600/50 mx-auto mb-2" />}
              {scenario === 'multi_sensor' && <Layers className="h-10 w-10 text-purple-600/50 mx-auto mb-2" />}
              <span className="text-gray-500 text-sm">{scenarioLabels[scenario]}</span>
              <div className="mt-1 text-xs text-gray-600">{scenarioDescriptions[scenario]}</div>
            </div>
          </div>
          <FusionDemoOverlay scenario={scenario} show={showOverlay} />
          {isProcessing && (
            <div className="absolute top-2 left-2 bg-blue-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <RefreshCw className="h-3 w-3 animate-spin" /> Processing...
            </div>
          )}
          {fusionComplete && (
            <div className="absolute top-2 left-2 bg-green-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <CheckCircle className="h-3 w-3" /> Fusion Complete
            </div>
          )}
        </div>

        {/* Results summary */}
        <div className="lg:col-span-2 bg-secondary/30 rounded-lg p-4">
          {!fusionComplete && !isProcessing && (
            <div className="text-center py-6">
              <Layers className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <h3 className="text-sm font-medium text-foreground mb-1">Ready to Fuse</h3>
              <p className="text-xs text-muted-foreground">Click "Run Fusion" to start</p>
            </div>
          )}
          {isProcessing && (
            <div className="text-center py-6">
              <RefreshCw className="h-8 w-8 text-primary mx-auto mb-2 animate-spin" />
              <h3 className="text-sm font-medium text-foreground mb-1">Processing...</h3>
              <p className="text-xs text-muted-foreground">Running {pipelineStages[currentStageIdx]?.name || 'pipeline'}...</p>
            </div>
          )}
          {fusionComplete && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-400" />
                Fusion Results
              </h3>
              {scenario === 'magnetic_survey' && (
                <>
                  <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded">
                    <span className="text-xs text-blue-400">5 magnetic anomalies detected</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Peak:</strong> 52,900 nT</p>
                    <p><strong className="text-foreground">Action:</strong> Ground-truth M1, M3</p>
                  </div>
                </>
              )}
              {scenario === 'gpr_subsurface' && (
                <>
                  <div className="p-2 bg-orange-500/10 border border-orange-500/30 rounded">
                    <span className="text-xs text-orange-400">3 subsurface features mapped</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Max depth:</strong> 4.2m</p>
                    <p><strong className="text-foreground">Action:</strong> Verify void at G2</p>
                  </div>
                </>
              )}
              {scenario === 'multi_sensor' && (
                <>
                  <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                    <span className="text-xs text-purple-400">4 fusion targets identified</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Best match:</strong> 94% (F1)</p>
                    <p><strong className="text-foreground">Action:</strong> Prioritize F1, F2</p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 text-xs text-muted-foreground bg-secondary/30 rounded p-2">
        <strong className="text-foreground">Synthetic Demo:</strong> This visualization uses simulated data to demonstrate sensor fusion pipelines.
      </div>
    </div>
  );
}

export default function SensorFusionPage() {
  const [sensors] = useState(mockSensors);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedSensor, setSelectedSensor] = useState<string | null>('1');

  const onlineSensors = sensors.filter((s) => s.status === 'online').length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Sensor Fusion</h1>
          <p className="text-muted-foreground">Real-time sensor data integration and monitoring</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
          <button
            onClick={() => setIsStreaming(!isStreaming)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              isStreaming
                ? 'bg-red-500 text-white hover:bg-red-600'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}
          >
            {isStreaming ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {isStreaming ? 'Stop Stream' : 'Start Stream'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Total Sensors</p>
          <p className="text-2xl font-bold text-foreground">{sensors.length}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Online</p>
          <p className="text-2xl font-bold text-green-500">{onlineSensors}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Offline</p>
          <p className="text-2xl font-bold text-red-500">{sensors.length - onlineSensors}</p>
        </div>
        <div className="bg-card border border-border rounded-xl p-4">
          <p className="text-sm text-muted-foreground">Data Rate</p>
          <p className="text-2xl font-bold text-foreground">{isStreaming ? '10 Hz' : '0 Hz'}</p>
        </div>
      </div>

      {/* Interactive Fusion Demo */}
      <InteractiveFusionDemo />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Connected Sensors</h2>
            <div className="space-y-2">
              {sensors.map((sensor) => {
                const Icon = sensorIcons[sensor.type];
                return (
                  <button
                    key={sensor.id}
                    onClick={() => setSelectedSensor(sensor.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedSensor === sensor.id
                        ? 'bg-primary/10 border border-primary/50'
                        : 'bg-background hover:bg-secondary border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`h-8 w-8 rounded-lg flex items-center justify-center ${
                        sensor.status === 'online' ? 'bg-green-500/10' : 'bg-red-500/10'
                      }`}>
                        <Icon className={`h-4 w-4 ${
                          sensor.status === 'online' ? 'text-green-500' : 'text-red-500'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <p className="font-medium text-foreground">{sensor.name}</p>
                          {sensor.status === 'online' ? (
                            <Wifi className="h-4 w-4 text-green-500" />
                          ) : (
                            <WifiOff className="h-4 w-4 text-red-500" />
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground capitalize">{sensor.type}</p>
                      </div>
                    </div>
                    {sensor.status === 'online' && (
                      <div className="mt-2 pt-2 border-t border-border">
                        <p className="text-lg font-bold text-foreground">
                          {sensor.value.toLocaleString()} <span className="text-sm font-normal text-muted-foreground">{sensor.unit}</span>
                        </p>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-card border border-border rounded-xl p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-foreground">Real-Time Data</h2>
              <div className="flex items-center gap-2">
                {isStreaming && (
                  <span className="flex items-center gap-1 text-sm text-green-500">
                    <span className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></span>
                    Live
                  </span>
                )}
              </div>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockTimeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
                  <YAxis yAxisId="left" stroke="#3b82f6" fontSize={12} />
                  <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Line yAxisId="left" type="monotone" dataKey="mag" stroke="#3b82f6" strokeWidth={2} dot={false} name="Magnetic (nT)" />
                  <Line yAxisId="right" type="monotone" dataKey="rad" stroke="#10b981" strokeWidth={2} dot={false} name="Radiometric (cps)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex items-center justify-center gap-6 mt-4 text-sm">
              <span className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-blue-500"></span>
                Magnetic (nT)
              </span>
              <span className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-green-500"></span>
                Radiometric (cps)
              </span>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Fusion Configuration</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Fusion Algorithm</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>Kalman Filter</option>
                  <option>Extended Kalman Filter</option>
                  <option>Particle Filter</option>
                  <option>Deep Learning Fusion</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Sampling Rate</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>1 Hz</option>
                  <option>5 Hz</option>
                  <option>10 Hz</option>
                  <option>20 Hz</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Coordinate System</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>WGS84</option>
                  <option>UTM Zone 11N</option>
                  <option>Local Grid</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-1">Output Format</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground">
                  <option>GeoTIFF</option>
                  <option>CSV</option>
                  <option>GeoJSON</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
