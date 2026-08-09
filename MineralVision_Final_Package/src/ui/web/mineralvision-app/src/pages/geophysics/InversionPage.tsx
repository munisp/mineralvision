import { useState, useEffect, useRef } from 'react';
import { Play, Upload, Magnet, Activity, Zap, CheckCircle, RefreshCw, Layers, ChevronRight, Target } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const mockConvergenceData = [
  { iteration: 1, misfit: 15.2, regularization: 0.85 },
  { iteration: 2, misfit: 8.4, regularization: 0.72 },
  { iteration: 3, misfit: 4.2, regularization: 0.58 },
  { iteration: 4, misfit: 2.8, regularization: 0.45 },
  { iteration: 5, misfit: 1.9, regularization: 0.38 },
  { iteration: 6, misfit: 1.4, regularization: 0.32 },
  { iteration: 7, misfit: 1.1, regularization: 0.28 },
  { iteration: 8, misfit: 0.95, regularization: 0.25 },
];

// Interactive Inversion Demo Component
type InversionScenario = 'magnetic_susceptibility' | 'gravity_density' | 'resistivity';

interface InversionPipelineStage {
  name: string;
  status: 'pending' | 'running' | 'completed';
  duration?: number;
}

// Magnetic susceptibility anomalies
const magneticAnomalies = [
  { id: 'S1', x: 150, y: 80, depth: 50, susceptibility: 0.045, shape: 'ellipse' },
  { id: 'S2', x: 350, y: 120, depth: 100, susceptibility: 0.032, shape: 'rect' },
  { id: 'S3', x: 250, y: 200, depth: 75, susceptibility: 0.028, shape: 'ellipse' },
];

// Gravity density bodies
const gravityBodies = [
  { id: 'D1', x: 120, y: 100, width: 80, height: 60, density: 2.8, type: 'intrusive' },
  { id: 'D2', x: 300, y: 150, width: 120, height: 80, density: 3.2, type: 'ore' },
  { id: 'D3', x: 400, y: 220, width: 60, height: 40, density: 2.5, type: 'sediment' },
];

// Resistivity layers
const resistivityLayers = [
  { id: 'R1', y1: 60, y2: 100, resistivity: 500, label: 'Overburden' },
  { id: 'R2', y1: 100, y2: 180, resistivity: 50, label: 'Conductive Zone' },
  { id: 'R3', y1: 180, y2: 280, resistivity: 1000, label: 'Basement' },
];

function InversionDemoOverlay({ scenario, show }: { scenario: InversionScenario; show: boolean }) {
  if (!show) return null;

  if (scenario === 'magnetic_susceptibility') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Surface line */}
        <line x1="20" y1="50" x2="540" y2="50" stroke="#22c55e" strokeWidth="2" />
        <text x="30" y="40" fill="#22c55e" fontSize="10">Surface</text>

        {/* Depth scale */}
        <g transform="translate(20, 50)">
          {[0, 50, 100, 150, 200].map((d, i) => (
            <g key={i}>
              <line x1="0" y1={i * 50} x2="10" y2={i * 50} stroke="#9ca3af" strokeWidth="1" />
              <text x="-5" y={i * 50 + 4} fill="#9ca3af" fontSize="8" textAnchor="end">{d}m</text>
            </g>
          ))}
        </g>

        {/* Susceptibility bodies */}
        {magneticAnomalies.map((body, idx) => (
          <g key={body.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            {body.shape === 'ellipse' ? (
              <ellipse cx={body.x} cy={50 + body.depth} rx="50" ry="30" fill="#3b82f6" opacity="0.4" stroke="#3b82f6" strokeWidth="2" />
            ) : (
              <rect x={body.x - 40} y={50 + body.depth - 20} width="80" height="40" fill="#3b82f6" opacity="0.4" stroke="#3b82f6" strokeWidth="2" rx="4" />
            )}
            <text x={body.x} y={50 + body.depth + 5} fill="white" fontSize="10" textAnchor="middle">{body.susceptibility} SI</text>
            <text x={body.x} y={50 + body.depth - 35} fill="#93c5fd" fontSize="8" textAnchor="middle">{body.id}</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(440, 60)">
          <rect x="0" y="0" width="100" height="50" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">Susceptibility</text>
          <rect x="10" y="22" width="20" height="10" fill="#3b82f6" opacity="0.5" rx="2" />
          <text x="35" y="30" fill="#9ca3af" fontSize="8">High (SI)</text>
          <text x="10" y="44" fill="#9ca3af" fontSize="7">3D model bodies</text>
        </g>

        <rect x="10" y="285" width="180" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#3b82f6" fontSize="10" fontWeight="bold">Magnetic Inversion (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'gravity_density') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Surface line */}
        <line x1="20" y1="50" x2="540" y2="50" stroke="#22c55e" strokeWidth="2" />
        <text x="30" y="40" fill="#22c55e" fontSize="10">Surface</text>

        {/* Density bodies */}
        {gravityBodies.map((body, idx) => (
          <g key={body.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <rect 
              x={body.x} 
              y={body.y} 
              width={body.width} 
              height={body.height} 
              fill={body.type === 'ore' ? '#f59e0b' : body.type === 'intrusive' ? '#8b5cf6' : '#6b7280'} 
              opacity="0.5" 
              stroke={body.type === 'ore' ? '#f59e0b' : body.type === 'intrusive' ? '#8b5cf6' : '#6b7280'} 
              strokeWidth="2" 
              rx="4" 
            />
            <text x={body.x + body.width / 2} y={body.y + body.height / 2 + 4} fill="white" fontSize="10" textAnchor="middle">{body.density} g/cc</text>
            <text x={body.x + body.width / 2} y={body.y - 8} fill="#d1d5db" fontSize="8" textAnchor="middle">{body.type}</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(430, 60)">
          <rect x="0" y="0" width="110" height="70" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">Density Bodies</text>
          <rect x="10" y="22" width="12" height="8" fill="#f59e0b" opacity="0.6" rx="1" />
          <text x="28" y="30" fill="#9ca3af" fontSize="8">Ore body</text>
          <rect x="10" y="36" width="12" height="8" fill="#8b5cf6" opacity="0.6" rx="1" />
          <text x="28" y="44" fill="#9ca3af" fontSize="8">Intrusive</text>
          <rect x="10" y="50" width="12" height="8" fill="#6b7280" opacity="0.6" rx="1" />
          <text x="28" y="58" fill="#9ca3af" fontSize="8">Sediment</text>
        </g>

        <rect x="10" y="285" width="160" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#f59e0b" fontSize="10" fontWeight="bold">Gravity Inversion (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'resistivity') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 560 320" preserveAspectRatio="none">
        {/* Surface line */}
        <line x1="20" y1="50" x2="540" y2="50" stroke="#22c55e" strokeWidth="2" />
        <text x="30" y="40" fill="#22c55e" fontSize="10">Surface</text>

        {/* Resistivity layers */}
        {resistivityLayers.map((layer, idx) => (
          <g key={layer.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <rect 
              x="40" 
              y={layer.y1} 
              width="480" 
              height={layer.y2 - layer.y1} 
              fill={layer.resistivity < 100 ? '#ef4444' : layer.resistivity < 800 ? '#f59e0b' : '#22c55e'} 
              opacity="0.3" 
              stroke={layer.resistivity < 100 ? '#ef4444' : layer.resistivity < 800 ? '#f59e0b' : '#22c55e'} 
              strokeWidth="1" 
            />
            <text x="280" y={(layer.y1 + layer.y2) / 2 + 4} fill="white" fontSize="11" textAnchor="middle">{layer.label}</text>
            <text x="530" y={(layer.y1 + layer.y2) / 2 + 4} fill="#d1d5db" fontSize="9" textAnchor="end">{layer.resistivity} ohm-m</text>
          </g>
        ))}

        {/* Conductive anomaly */}
        <ellipse cx="200" cy="140" rx="40" ry="25" fill="#ef4444" opacity="0.6" stroke="#ef4444" strokeWidth="2" className="animate-pulse" />
        <text x="200" y="145" fill="white" fontSize="9" textAnchor="middle">10 ohm-m</text>

        {/* Legend */}
        <g transform="translate(430, 200)">
          <rect x="0" y="0" width="110" height="70" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="14" fill="white" fontSize="9" fontWeight="bold">Resistivity</text>
          <rect x="10" y="22" width="12" height="8" fill="#ef4444" opacity="0.5" rx="1" />
          <text x="28" y="30" fill="#9ca3af" fontSize="8">Low (&lt;100)</text>
          <rect x="10" y="36" width="12" height="8" fill="#f59e0b" opacity="0.5" rx="1" />
          <text x="28" y="44" fill="#9ca3af" fontSize="8">Medium</text>
          <rect x="10" y="50" width="12" height="8" fill="#22c55e" opacity="0.5" rx="1" />
          <text x="28" y="58" fill="#9ca3af" fontSize="8">High (&gt;800)</text>
        </g>

        <rect x="10" y="285" width="170" height="25" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="302" fill="#ef4444" fontSize="10" fontWeight="bold">Resistivity Inversion (Demo)</text>
      </svg>
    );
  }

  return null;
}

function InteractiveInversionDemo() {
  const [scenario, setScenario] = useState<InversionScenario>('magnetic_susceptibility');
  const [isProcessing, setIsProcessing] = useState(false);
  const [inversionComplete, setInversionComplete] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [pipelineStages, setPipelineStages] = useState<InversionPipelineStage[]>([]);
  const processingRef = useRef<boolean>(false);

  const getInitialStages = (): InversionPipelineStage[] => [
    { name: 'Load Mesh', status: 'pending' },
    { name: 'Forward Model', status: 'pending' },
    { name: 'Sensitivity', status: 'pending' },
    { name: 'Invert', status: 'pending' },
    { name: 'Regularize', status: 'pending' },
    { name: 'Export', status: 'pending' },
  ];

  const stageDurations = [80, 150, 200, 300, 150, 100];

  const runInversion = async () => {
    if (processingRef.current) return;
    processingRef.current = true;
    setIsProcessing(true);
    setInversionComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());

    for (let i = 0; i < 6; i++) {
      setCurrentStageIdx(i);
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'running' } : s));
      await new Promise(r => setTimeout(r, stageDurations[i]));
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'completed', duration: stageDurations[i] } : s));
      if (i === 3) setShowOverlay(true);
    }

    setIsProcessing(false);
    setInversionComplete(true);
    processingRef.current = false;
  };

  useEffect(() => {
    setInversionComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());
  }, [scenario]);

  const scenarioLabels = {
    magnetic_susceptibility: 'Magnetic Susceptibility',
    gravity_density: 'Gravity Density',
    resistivity: 'DC Resistivity',
  };

  const scenarioDescriptions = {
    magnetic_susceptibility: 'Invert magnetic data to recover 3D susceptibility distribution',
    gravity_density: 'Invert gravity data to recover 3D density contrast model',
    resistivity: 'Invert DC resistivity data to recover subsurface conductivity',
  };

  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Interactive Inversion Demo
          </h2>
          <p className="text-sm text-muted-foreground">Select a scenario and run the inversion pipeline</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value as InversionScenario)}
            disabled={isProcessing}
            className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground"
          >
            <option value="magnetic_susceptibility">Magnetic Susceptibility</option>
            <option value="gravity_density">Gravity Density</option>
            <option value="resistivity">DC Resistivity</option>
          </select>
          <button
            onClick={runInversion}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {isProcessing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
            {isProcessing ? 'Inverting...' : 'Run Inversion'}
          </button>
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="flex items-center gap-1 overflow-x-auto pb-2 mb-4">
        {pipelineStages.map((stage, idx) => (
          <div key={stage.name} className="flex items-center">
            <div className={`flex flex-col items-center p-2 rounded-lg border min-w-[65px] transition-all duration-300 ${
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
                 stage.status === 'completed' ? <CheckCircle className="h-3 w-3" /> : <Layers className="h-3 w-3" />}
              </div>
              <span className="text-xs font-medium text-foreground">{stage.name}</span>
              {stage.status === 'completed' && stage.duration && (
                <span className="text-xs text-green-400">{stage.duration}ms</span>
              )}
            </div>
            {idx < pipelineStages.length - 1 && <ChevronRight className="h-3 w-3 text-muted-foreground mx-0.5" />}
          </div>
        ))}
      </div>

      {/* Visualization panel */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3 relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              {scenario === 'magnetic_susceptibility' && <Magnet className="h-10 w-10 text-blue-600/50 mx-auto mb-2" />}
              {scenario === 'gravity_density' && <Activity className="h-10 w-10 text-orange-600/50 mx-auto mb-2" />}
              {scenario === 'resistivity' && <Activity className="h-10 w-10 text-red-600/50 mx-auto mb-2" />}
              <span className="text-gray-500 text-sm">{scenarioLabels[scenario]}</span>
              <div className="mt-1 text-xs text-gray-600">{scenarioDescriptions[scenario]}</div>
            </div>
          </div>
          <InversionDemoOverlay scenario={scenario} show={showOverlay} />
          {isProcessing && (
            <div className="absolute top-2 left-2 bg-blue-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <RefreshCw className="h-3 w-3 animate-spin" /> Inverting...
            </div>
          )}
          {inversionComplete && (
            <div className="absolute top-2 left-2 bg-green-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <CheckCircle className="h-3 w-3" /> Inversion Complete
            </div>
          )}
        </div>

        {/* Results summary */}
        <div className="lg:col-span-2 bg-secondary/30 rounded-lg p-4">
          {!inversionComplete && !isProcessing && (
            <div className="text-center py-6">
              <Target className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
              <h3 className="text-sm font-medium text-foreground mb-1">Ready to Invert</h3>
              <p className="text-xs text-muted-foreground">Click "Run Inversion" to start</p>
            </div>
          )}
          {isProcessing && (
            <div className="text-center py-6">
              <RefreshCw className="h-8 w-8 text-primary mx-auto mb-2 animate-spin" />
              <h3 className="text-sm font-medium text-foreground mb-1">Inverting...</h3>
              <p className="text-xs text-muted-foreground">Running {pipelineStages[currentStageIdx]?.name || 'pipeline'}...</p>
            </div>
          )}
          {inversionComplete && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-400" />
                Inversion Results
              </h3>
              {scenario === 'magnetic_susceptibility' && (
                <>
                  <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded">
                    <span className="text-xs text-blue-400">3 susceptibility bodies recovered</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Max SI:</strong> 0.045</p>
                    <p><strong className="text-foreground">Depth:</strong> 50-100m</p>
                    <p><strong className="text-foreground">Action:</strong> Drill test S1</p>
                  </div>
                </>
              )}
              {scenario === 'gravity_density' && (
                <>
                  <div className="p-2 bg-orange-500/10 border border-orange-500/30 rounded">
                    <span className="text-xs text-orange-400">3 density bodies identified</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Max density:</strong> 3.2 g/cc</p>
                    <p><strong className="text-foreground">Ore body:</strong> D2 (high priority)</p>
                    <p><strong className="text-foreground">Action:</strong> Confirm with drilling</p>
                  </div>
                </>
              )}
              {scenario === 'resistivity' && (
                <>
                  <div className="p-2 bg-red-500/10 border border-red-500/30 rounded">
                    <span className="text-xs text-red-400">3 layers + 1 anomaly mapped</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p><strong className="text-foreground">Conductor:</strong> 10 ohm-m</p>
                    <p><strong className="text-foreground">Depth:</strong> 100-180m</p>
                    <p><strong className="text-foreground">Action:</strong> EM follow-up</p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 text-xs text-muted-foreground bg-secondary/30 rounded p-2">
        <strong className="text-foreground">Synthetic Demo:</strong> This visualization uses simulated data to demonstrate geophysical inversion workflows.
      </div>
    </div>
  );
}

export default function InversionPage() {
  const [inversionType, setInversionType] = useState('magnetic');
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleRun = async () => {
    setIsRunning(true);
    setProgress(0);
    for (let i = 0; i <= 100; i += 5) {
      await new Promise((r) => setTimeout(r, 200));
      setProgress(i);
    }
    setIsRunning(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Geophysical Inversion</h1>
          <p className="text-muted-foreground">Perform 3D potential field inversions</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="px-4 py-2 border border-input rounded-lg text-foreground hover:bg-secondary flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Import Survey
          </button>
          <button
            onClick={handleRun}
            disabled={isRunning}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {isRunning ? `Running... ${progress}%` : 'Run Inversion'}
          </button>
        </div>
      </div>

      {/* Interactive Inversion Demo */}
      <InteractiveInversionDemo />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Inversion Type</h2>
            <div className="space-y-2">
              {[
                { id: 'magnetic', label: 'Magnetic (Susceptibility)', icon: Magnet },
                { id: 'gravity', label: 'Gravity (Density)', icon: Activity },
                { id: 'em', label: 'EM (Conductivity)', icon: Activity },
                { id: 'dc', label: 'DC Resistivity', icon: Activity },
                { id: 'ip', label: 'IP (Chargeability)', icon: Activity },
              ].map((type) => (
                <label key={type.id} className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="inversionType"
                    value={type.id}
                    checked={inversionType === type.id}
                    onChange={(e) => setInversionType(e.target.value)}
                    className="text-primary"
                  />
                  <span className="text-sm text-foreground">{type.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Mesh Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Cell Size X (m)</label>
                <input type="number" defaultValue={25} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Cell Size Y (m)</label>
                <input type="number" defaultValue={25} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Cell Size Z (m)</label>
                <input type="number" defaultValue={10} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Mesh Type</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Regular</option>
                  <option>Octree (Adaptive)</option>
                  <option>Tensor</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-4">
            <h2 className="text-sm font-semibold text-foreground mb-3">Inversion Parameters</h2>
            <div className="space-y-3">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Max Iterations</label>
                <input type="number" defaultValue={50} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Target Misfit</label>
                <input type="number" defaultValue={1.0} step={0.1} className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm" />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Regularization</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Tikhonov (Smooth)</option>
                  <option>Minimum Support</option>
                  <option>Total Variation</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">Depth Weighting</label>
                <select className="w-full px-3 py-2 bg-background border border-input rounded-lg text-foreground text-sm">
                  <option>Standard (z^-1.5)</option>
                  <option>Sensitivity-based</option>
                  <option>None</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Survey Data</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Observations</p>
                <p className="text-2xl font-bold text-foreground">12,450</p>
              </div>
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Survey Area</p>
                <p className="text-2xl font-bold text-foreground">25 km</p>
              </div>
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Line Spacing</p>
                <p className="text-2xl font-bold text-foreground">100m</p>
              </div>
            </div>
          </div>

          {isRunning && (
            <div className="bg-card border border-border rounded-xl p-5">
              <h2 className="text-lg font-semibold text-foreground mb-4">Inversion Progress</h2>
              <div className="space-y-4">
                <div className="w-full bg-secondary rounded-full h-3">
                  <div
                    className="bg-primary h-3 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="grid grid-cols-4 gap-4 text-sm text-center">
                  <div>
                    <p className="text-muted-foreground">Iteration</p>
                    <p className="font-medium text-foreground">{Math.ceil(progress / 12.5)}/8</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Data Misfit</p>
                    <p className="font-medium text-foreground">{(15.2 - progress * 0.14).toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Model Norm</p>
                    <p className="font-medium text-foreground">{(0.85 - progress * 0.006).toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Est. Time</p>
                    <p className="font-medium text-foreground">{Math.ceil((100 - progress) * 0.2)}s</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Convergence History</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockConvergenceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="iteration" stroke="#9ca3af" fontSize={12} />
                  <YAxis stroke="#9ca3af" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1f2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Line type="monotone" dataKey="misfit" stroke="#3b82f6" strokeWidth={2} name="Data Misfit" />
                  <Line type="monotone" dataKey="regularization" stroke="#10b981" strokeWidth={2} name="Model Norm" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex items-center justify-center gap-6 mt-4 text-sm">
              <span className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-blue-500"></span>
                Data Misfit
              </span>
              <span className="flex items-center gap-2">
                <span className="w-3 h-0.5 bg-green-500"></span>
                Model Norm
              </span>
            </div>
          </div>

          <div className="bg-card border border-border rounded-xl p-5">
            <h2 className="text-lg font-semibold text-foreground mb-4">Results Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Final Misfit</p>
                <p className="text-xl font-bold text-foreground">0.95</p>
              </div>
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Iterations</p>
                <p className="text-xl font-bold text-foreground">8</p>
              </div>
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Model Cells</p>
                <p className="text-xl font-bold text-foreground">125,000</p>
              </div>
              <div className="p-4 bg-background rounded-lg">
                <p className="text-sm text-muted-foreground">Compute Time</p>
                <p className="text-xl font-bold text-foreground">4m 32s</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
