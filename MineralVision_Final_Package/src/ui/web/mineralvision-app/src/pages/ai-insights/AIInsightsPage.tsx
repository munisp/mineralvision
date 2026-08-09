import { useState, useEffect, useRef } from 'react';
import {
  Brain,
  AlertTriangle,
  TrendingUp,
  Search,
  MapPin,
  Calendar,
  ChevronRight,
  RefreshCw,
  Filter,
  Download,
  Eye,
  Layers,
  Target,
  Zap,
  CheckCircle,
  Clock,
  ArrowRight,
  Mountain,
  Droplets,
  Gem,
  Play,
  Image,
  Cpu,
  Scan,
} from 'lucide-react';

// Types for V-JEPA insights
interface InsightCard {
  id: string;
  type: 'anomaly' | 'change' | 'similarity';
  category: 'gold' | 'lithium' | 'alteration' | 'general';
  confidence: 'high' | 'medium' | 'low';
  what: string;
  why: string;
  nextAction: string;
  location: { lat: number; lng: number; name: string };
  timestamp: string;
  thumbnailUrl?: string;
  anomalyScore?: number;
  similarityScore?: number;
  changePercentage?: number;
  status: 'new' | 'reviewed' | 'actioned';
}

// Mock data for demonstration
const mockInsights: InsightCard[] = [
  {
    id: '1',
    type: 'anomaly',
    category: 'gold',
    confidence: 'high',
    what: 'Unusual alteration texture detected in drone imagery',
    why: 'This area shows texture patterns 94% similar to known gold-bearing alteration zones. The spectral signature indicates silicification with possible sulfide association.',
    nextAction: 'Recommend rock chip sampling along the 200m ridge line. Priority: High. Suggested samples: 5-8 at 25m intervals.',
    location: { lat: 9.0820, lng: 8.6753, name: 'Prospect Ridge A' },
    timestamp: '2025-12-21T10:30:00Z',
    anomalyScore: 94,
    status: 'new',
  },
  {
    id: '2',
    type: 'change',
    category: 'lithium',
    confidence: 'medium',
    what: 'Brine pond expansion detected over 3-month period',
    why: 'Satellite time-series shows 12% increase in evaporation pond area. Surface crust texture evolution suggests increasing lithium concentration in the northern sector.',
    nextAction: 'Recommend brine sampling at 3 new locations in expanded area. Schedule re-fly with thermal sensor to map temperature gradients.',
    location: { lat: -23.4567, lng: -67.8901, name: 'Salar Norte Block 7' },
    timestamp: '2025-12-20T14:15:00Z',
    changePercentage: 12,
    status: 'new',
  },
  {
    id: '3',
    type: 'similarity',
    category: 'alteration',
    confidence: 'high',
    what: 'Alteration signature matches epithermal Au target zones',
    why: 'Top 5 nearest neighbors are all from verified epithermal gold systems. Spectral texture indicates argillic-phyllic alteration with iron-oxide staining consistent with a preserved high-level system.',
    nextAction: 'Confirm with rock chip sampling and portable XRF across the alteration halo. If confirmed, prioritize this 15-hectare zone for grid-based soil geochemistry. Recommend 3 verification traverses.',
    location: { lat: 6.5244, lng: 3.3792, name: 'Tenement Block E-12' },
    timestamp: '2025-12-21T08:45:00Z',
    similarityScore: 89,
    status: 'reviewed',
  },
  {
    id: '4',
    type: 'anomaly',
    category: 'alteration',
    confidence: 'medium',
    what: 'Potential gossanous iron-staining pattern identified',
    why: 'Texture anomaly in satellite imagery correlates with known oxidation indicators above sulfide bodies. Pattern is 78th percentile unusual compared to baseline unaltered bedrock.',
    nextAction: 'Urgent: Ground-truth flagged locations with portable XRF. If Fe-staining is confirmed with anomalous base metals, extend alteration mapping and prioritize follow-up geochemistry.',
    location: { lat: 6.4521, lng: 3.4012, name: 'Tenement Block D-4' },
    timestamp: '2025-12-19T16:20:00Z',
    anomalyScore: 78,
    status: 'actioned',
  },
  {
    id: '5',
    type: 'anomaly',
    category: 'gold',
    confidence: 'low',
    what: 'Linear feature detected - possible structural control',
    why: 'Drone imagery shows subtle linear texture that may indicate fault/shear zone. Needs verification - confidence is low due to vegetation cover and sensor noise.',
    nextAction: 'Recommend ground-truthing traverse along the 500m lineament. If structure confirmed, extend soil sampling grid perpendicular to strike.',
    location: { lat: 9.1234, lng: 8.7890, name: 'Grid Line 15' },
    timestamp: '2025-12-18T11:00:00Z',
    anomalyScore: 62,
    status: 'new',
  },
];

// Analysis job status
interface AnalysisJob {
  id: string;
  name: string;
  type: 'anomaly_scan' | 'change_detection' | 'similarity_search';
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: string;
  estimatedCompletion?: string;
}

const mockJobs: AnalysisJob[] = [
  {
    id: 'job-1',
    name: 'Full site anomaly scan - Prospect A',
    type: 'anomaly_scan',
    status: 'running',
    progress: 67,
    startTime: '2025-12-21T12:00:00Z',
    estimatedCompletion: '2025-12-21T12:45:00Z',
  },
  {
    id: 'job-2',
    name: 'Monthly change detection - Salar Norte',
    type: 'change_detection',
    status: 'completed',
    progress: 100,
    startTime: '2025-12-21T10:00:00Z',
  },
];

function ConfidenceBadge({ confidence }: { confidence: 'high' | 'medium' | 'low' }) {
  const styles = {
    high: 'bg-green-500/20 text-green-400 border-green-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${styles[confidence]}`}>
      {confidence.charAt(0).toUpperCase() + confidence.slice(1)} Confidence
    </span>
  );
}

function CategoryIcon({ category }: { category: string }) {
  const icons = {
    gold: <Gem className="h-5 w-5 text-yellow-400" />,
    lithium: <Droplets className="h-5 w-5 text-blue-400" />,
    alteration: <Layers className="h-5 w-5 text-orange-400" />,
    general: <Mountain className="h-5 w-5 text-gray-400" />,
  };
  return icons[category as keyof typeof icons] || icons.general;
}

function TypeBadge({ type }: { type: 'anomaly' | 'change' | 'similarity' }) {
  const styles = {
    anomaly: 'bg-red-500/20 text-red-400',
    change: 'bg-purple-500/20 text-purple-400',
    similarity: 'bg-blue-500/20 text-blue-400',
  };
  const labels = {
    anomaly: 'Anomaly',
    change: 'Change',
    similarity: 'Similar',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded ${styles[type]}`}>
      {labels[type]}
    </span>
  );
}

function StatusBadge({ status }: { status: 'new' | 'reviewed' | 'actioned' }) {
  const styles = {
    new: 'bg-blue-500/20 text-blue-400',
    reviewed: 'bg-yellow-500/20 text-yellow-400',
    actioned: 'bg-green-500/20 text-green-400',
  };

  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded ${styles[status]}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}

function InsightCardComponent({ insight, onViewDetails }: { insight: InsightCard; onViewDetails: (id: string) => void }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <CategoryIcon category={insight.category} />
          <TypeBadge type={insight.type} />
          <ConfidenceBadge confidence={insight.confidence} />
        </div>
        <StatusBadge status={insight.status} />
      </div>

      {/* WHAT - The observation */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-primary mb-1 flex items-center gap-1">
          <Eye className="h-4 w-4" />
          What is happening
        </h3>
        <p className="text-foreground font-medium">{insight.what}</p>
      </div>

      {/* WHY - The evidence */}
      <div className="mb-4 bg-secondary/30 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-muted-foreground mb-1 flex items-center gap-1">
          <Brain className="h-4 w-4" />
          Why (Evidence)
        </h3>
        <p className="text-sm text-muted-foreground">{insight.why}</p>
        
        {/* Score indicators */}
        <div className="mt-2 flex gap-4">
          {insight.anomalyScore && (
            <div className="flex items-center gap-1">
              <AlertTriangle className="h-4 w-4 text-red-400" />
              <span className="text-xs text-muted-foreground">
                Anomaly: <span className="text-foreground font-medium">{insight.anomalyScore}%</span>
              </span>
            </div>
          )}
          {insight.similarityScore && (
            <div className="flex items-center gap-1">
              <Search className="h-4 w-4 text-blue-400" />
              <span className="text-xs text-muted-foreground">
                Similarity: <span className="text-foreground font-medium">{insight.similarityScore}%</span>
              </span>
            </div>
          )}
          {insight.changePercentage && (
            <div className="flex items-center gap-1">
              <TrendingUp className="h-4 w-4 text-purple-400" />
              <span className="text-xs text-muted-foreground">
                Change: <span className="text-foreground font-medium">+{insight.changePercentage}%</span>
              </span>
            </div>
          )}
        </div>
      </div>

      {/* NEXT ACTION - Recommended action */}
      <div className="mb-4 bg-primary/10 border border-primary/20 rounded-lg p-3">
        <h3 className="text-sm font-semibold text-primary mb-1 flex items-center gap-1">
          <Target className="h-4 w-4" />
          Recommended Action
        </h3>
        <p className="text-sm text-foreground">{insight.nextAction}</p>
      </div>

      {/* Footer - Location and time */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <MapPin className="h-3 w-3" />
            {insight.location.name}
          </span>
          <span className="flex items-center gap-1">
            <Calendar className="h-3 w-3" />
            {new Date(insight.timestamp).toLocaleDateString()}
          </span>
        </div>
        <button
          onClick={() => onViewDetails(insight.id)}
          className="flex items-center gap-1 text-primary hover:text-primary/80 transition-colors"
        >
          View Details
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function JobStatusCard({ job }: { job: AnalysisJob }) {
  const statusIcons = {
    queued: <Clock className="h-4 w-4 text-muted-foreground" />,
    running: <RefreshCw className="h-4 w-4 text-blue-400 animate-spin" />,
    completed: <CheckCircle className="h-4 w-4 text-green-400" />,
    failed: <AlertTriangle className="h-4 w-4 text-red-400" />,
  };

  return (
    <div className="bg-card border border-border rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {statusIcons[job.status]}
          <span className="text-sm font-medium text-foreground">{job.name}</span>
        </div>
        <span className="text-xs text-muted-foreground capitalize">{job.status}</span>
      </div>
      {job.status === 'running' && (
        <div className="w-full bg-secondary rounded-full h-2">
          <div
            className="bg-primary h-2 rounded-full transition-all duration-300"
            style={{ width: `${job.progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

// Interactive Analysis Demo Component
interface AnalysisPipelineStage {
  name: string;
  icon: React.ReactNode;
  status: 'pending' | 'running' | 'completed';
  duration?: number;
}

type AnalysisScenario = 'gold_anomaly' | 'lithium_change' | 'alteration_similarity';

// Anomaly detection points for gold scenario
const goldAnomalyPoints = [
  { id: 'A1', x: 150, y: 120, score: 94, type: 'alteration' },
  { id: 'A2', x: 280, y: 180, score: 87, type: 'structure' },
  { id: 'A3', x: 420, y: 140, score: 82, type: 'gossan' },
  { id: 'A4', x: 200, y: 260, score: 78, type: 'texture' },
  { id: 'A5', x: 350, y: 220, score: 72, type: 'color' },
];

// Change detection zones for lithium scenario
const lithiumChangeZones = [
  { id: 'C1', cx: 200, cy: 150, rx: 60, ry: 40, change: 15, type: 'expansion' },
  { id: 'C2', cx: 380, cy: 180, rx: 50, ry: 35, change: 12, type: 'crust' },
  { id: 'C3', cx: 280, cy: 260, rx: 70, ry: 45, change: 8, type: 'evaporation' },
];

// Similarity matches for alteration scenario
const alterationSimilarityMatches = [
  { id: 'S1', x: 120, y: 100, similarity: 92, match: 'epithermal_au' },
  { id: 'S2', x: 300, y: 140, similarity: 89, match: 'epithermal_au' },
  { id: 'S3', x: 450, y: 180, similarity: 85, match: 'argillic_alteration' },
  { id: 'S4', x: 180, y: 240, similarity: 78, match: 'iron_oxide_staining' },
  { id: 'S5', x: 380, y: 280, similarity: 75, match: 'silicified_zone' },
];

function AnalysisDemoOverlay({ scenario, show }: { scenario: AnalysisScenario; show: boolean }) {
  if (!show) return null;

  const getAnomalyColor = (score: number) => {
    if (score >= 90) return '#ef4444';
    if (score >= 80) return '#f59e0b';
    return '#eab308';
  };

  if (scenario === 'gold_anomaly') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
        {/* Survey boundary */}
        <polygon points="30,30 610,25 620,330 40,340" fill="none" stroke="#f59e0b" strokeWidth="2" strokeDasharray="8,4" className="animate-pulse" />
        
        {/* Scan grid */}
        <g stroke="#4b5563" strokeWidth="0.5" opacity="0.2">
          {[0, 1, 2, 3, 4].map(i => <line key={`h${i}`} x1="30" y1={30 + i * 75} x2="620" y2={25 + i * 75} />)}
          {[0, 1, 2, 3, 4, 5].map(i => <line key={`v${i}`} x1={30 + i * 120} y1="25" x2={35 + i * 120} y2="340" />)}
        </g>

        {/* Anomaly points */}
        {goldAnomalyPoints.map((point, idx) => (
          <g key={point.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <circle cx={point.x} cy={point.y} r="25" fill={getAnomalyColor(point.score)} opacity="0.2" />
            <circle cx={point.x} cy={point.y} r="15" fill={getAnomalyColor(point.score)} opacity="0.4" />
            <circle cx={point.x} cy={point.y} r="8" fill={getAnomalyColor(point.score)} stroke="white" strokeWidth="2" />
            <text x={point.x} y={point.y + 4} fill="white" fontSize="8" textAnchor="middle" fontWeight="bold">{point.score}</text>
            <rect x={point.x - 20} y={point.y + 18} width="40" height="14" fill="#1f2937" opacity="0.9" rx="2" />
            <text x={point.x} y={point.y + 28} fill="#9ca3af" fontSize="8" textAnchor="middle">{point.type}</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(500, 20)">
          <rect x="0" y="0" width="120" height="70" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="16" fill="white" fontSize="10" fontWeight="bold">Anomaly Score</text>
          <circle cx="20" cy="32" r="5" fill="#ef4444" />
          <text x="32" y="36" fill="#9ca3af" fontSize="9">High (90+)</text>
          <circle cx="20" cy="48" r="5" fill="#f59e0b" />
          <text x="32" y="52" fill="#9ca3af" fontSize="9">Medium (80-89)</text>
          <circle cx="20" cy="64" r="5" fill="#eab308" />
          <text x="32" y="68" fill="#9ca3af" fontSize="9">Low (&lt;80)</text>
        </g>

        <rect x="10" y="320" width="180" height="30" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="340" fill="#ef4444" fontSize="11" fontWeight="bold">Anomaly Detection (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'lithium_change') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
        {/* Salar boundary */}
        <ellipse cx="320" cy="180" rx="280" ry="150" fill="none" stroke="#3b82f6" strokeWidth="2" strokeDasharray="8,4" />
        
        {/* Change zones */}
        {lithiumChangeZones.map((zone, idx) => (
          <g key={zone.id} className="animate-fade-in" style={{ animationDelay: `${idx * 200}ms` }}>
            <ellipse cx={zone.cx} cy={zone.cy} rx={zone.rx} ry={zone.ry} fill="#8b5cf6" opacity="0.3" stroke="#8b5cf6" strokeWidth="2" />
            <text x={zone.cx} y={zone.cy - 5} fill="white" fontSize="12" textAnchor="middle" fontWeight="bold">+{zone.change}%</text>
            <text x={zone.cx} y={zone.cy + 10} fill="#c4b5fd" fontSize="9" textAnchor="middle">{zone.type}</text>
          </g>
        ))}

        {/* Time indicator */}
        <g transform="translate(30, 30)">
          <rect x="0" y="0" width="150" height="50" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="20" fill="white" fontSize="10" fontWeight="bold">Change Detection</text>
          <text x="10" y="35" fill="#9ca3af" fontSize="9">Baseline: Oct 2025</text>
          <text x="10" y="47" fill="#9ca3af" fontSize="9">Current: Dec 2025</text>
        </g>

        {/* Legend */}
        <g transform="translate(500, 20)">
          <rect x="0" y="0" width="120" height="50" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="16" fill="white" fontSize="10" fontWeight="bold">Change Zones</text>
          <ellipse cx="20" cy="35" rx="10" ry="6" fill="#8b5cf6" opacity="0.5" />
          <text x="36" y="39" fill="#9ca3af" fontSize="9">Area expansion</text>
        </g>

        <rect x="10" y="320" width="180" height="30" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="340" fill="#8b5cf6" fontSize="11" fontWeight="bold">Change Detection (Demo)</text>
      </svg>
    );
  }

  if (scenario === 'alteration_similarity') {
    return (
      <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
        {/* Field boundary */}
        <polygon points="40,40 600,30 620,320 50,340" fill="none" stroke="#22c55e" strokeWidth="2" strokeDasharray="8,4" className="animate-pulse" />
        
        {/* Similarity matches */}
        {alterationSimilarityMatches.map((match, idx) => (
          <g key={match.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <rect x={match.x - 30} y={match.y - 20} width="60" height="40" fill="#22c55e" opacity="0.2" stroke="#22c55e" strokeWidth="2" rx="4" />
            <text x={match.x} y={match.y - 5} fill="white" fontSize="11" textAnchor="middle" fontWeight="bold">{match.similarity}%</text>
            <text x={match.x} y={match.y + 10} fill="#86efac" fontSize="8" textAnchor="middle">{match.match.replace(/_/g, ' ')}</text>
          </g>
        ))}

        {/* Legend */}
        <g transform="translate(500, 20)">
          <rect x="0" y="0" width="120" height="50" fill="#1f2937" opacity="0.95" rx="4" />
          <text x="10" y="16" fill="white" fontSize="10" fontWeight="bold">Similarity Match</text>
          <rect x="15" y="28" width="15" height="15" fill="#22c55e" opacity="0.3" stroke="#22c55e" rx="2" />
          <text x="36" y="39" fill="#9ca3af" fontSize="9">Similar to reference</text>
        </g>

        <rect x="10" y="320" width="180" height="30" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="20" y="340" fill="#22c55e" fontSize="11" fontWeight="bold">Similarity Search (Demo)</text>
      </svg>
    );
  }

  return null;
}

function InteractiveAnalysisDemo() {
  const [scenario, setScenario] = useState<AnalysisScenario>('gold_anomaly');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [showOverlay, setShowOverlay] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(0);
  const [pipelineStages, setPipelineStages] = useState<AnalysisPipelineStage[]>([]);
  const analysisRef = useRef<boolean>(false);

  const getInitialStages = (): AnalysisPipelineStage[] => [
    { name: 'Load Data', icon: <Image className="h-4 w-4" />, status: 'pending' },
    { name: 'Preprocess', icon: <Cpu className="h-4 w-4" />, status: 'pending' },
    { name: 'V-JEPA Encode', icon: <Brain className="h-4 w-4" />, status: 'pending' },
    { name: 'Feature Extract', icon: <Scan className="h-4 w-4" />, status: 'pending' },
    { name: 'Analysis', icon: <Target className="h-4 w-4" />, status: 'pending' },
    { name: 'Generate Insights', icon: <Zap className="h-4 w-4" />, status: 'pending' },
  ];

  const stageDurations = [100, 80, 200, 150, 180, 120];

  const runAnalysis = async () => {
    if (analysisRef.current) return;
    analysisRef.current = true;
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());

    for (let i = 0; i < 6; i++) {
      setCurrentStageIdx(i);
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'running' } : s));
      await new Promise(r => setTimeout(r, stageDurations[i]));
      setPipelineStages(prev => prev.map((s, idx) => idx === i ? { ...s, status: 'completed', duration: stageDurations[i] } : s));
      if (i === 3) setShowOverlay(true);
    }

    setIsAnalyzing(false);
    setAnalysisComplete(true);
    analysisRef.current = false;
  };

  useEffect(() => {
    setAnalysisComplete(false);
    setShowOverlay(false);
    setPipelineStages(getInitialStages());
  }, [scenario]);

  const scenarioLabels = {
    gold_anomaly: 'Gold Anomaly Detection',
    lithium_change: 'Lithium Change Detection',
    alteration_similarity: 'Alteration Similarity Search',
  };

  const scenarioDescriptions = {
    gold_anomaly: 'Detect unusual alteration textures and geological anomalies in drone imagery',
    lithium_change: 'Monitor brine pond evolution and surface changes over time',
    alteration_similarity: 'Find areas similar to known hydrothermal alteration zones',
  };

  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Play className="h-5 w-5 text-primary" />
            Interactive Analysis Demo
          </h2>
          <p className="text-sm text-muted-foreground">Select a scenario and run the V-JEPA analysis pipeline</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value as AnalysisScenario)}
            disabled={isAnalyzing}
            className="bg-secondary border border-border rounded-lg px-3 py-2 text-sm text-foreground"
          >
            <option value="gold_anomaly">Gold Anomaly Detection</option>
            <option value="lithium_change">Lithium Change Detection</option>
            <option value="alteration_similarity">Alteration Similarity Search</option>
          </select>
          <button
            onClick={runAnalysis}
            disabled={isAnalyzing}
            className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
          >
            {isAnalyzing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
            {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Pipeline stages */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 mb-4">
        {pipelineStages.map((stage, idx) => (
          <div key={stage.name} className="flex items-center">
            <div className={`flex flex-col items-center p-2 rounded-lg border min-w-[80px] transition-all duration-300 ${
              stage.status === 'completed' ? 'bg-green-500/10 border-green-500/30' :
              stage.status === 'running' ? 'bg-blue-500/10 border-blue-500/30' :
              'bg-secondary/50 border-border'
            }`}>
              <div className={`p-1.5 rounded-full mb-1 ${
                stage.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                stage.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                'bg-secondary text-muted-foreground'
              }`}>
                {stage.status === 'running' ? <RefreshCw className="h-3 w-3 animate-spin" /> :
                 stage.status === 'completed' ? <CheckCircle className="h-3 w-3" /> : stage.icon}
              </div>
              <span className="text-xs font-medium text-foreground text-center">{stage.name}</span>
              {stage.status === 'completed' && stage.duration && (
                <span className="text-xs text-green-400">{stage.duration}ms</span>
              )}
            </div>
            {idx < pipelineStages.length - 1 && <ChevronRight className="h-3 w-3 text-muted-foreground mx-1" />}
          </div>
        ))}
      </div>

      {/* Visualization panel */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg overflow-hidden">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              {scenario === 'gold_anomaly' && <Gem className="h-12 w-12 text-yellow-600/50 mx-auto mb-2" />}
              {scenario === 'lithium_change' && <Droplets className="h-12 w-12 text-blue-600/50 mx-auto mb-2" />}
              {scenario === 'alteration_similarity' && <Layers className="h-12 w-12 text-orange-600/50 mx-auto mb-2" />}
              <span className="text-gray-500 text-sm">{scenarioLabels[scenario]}</span>
              <div className="mt-1 text-xs text-gray-600">{scenarioDescriptions[scenario]}</div>
            </div>
          </div>
          <AnalysisDemoOverlay scenario={scenario} show={showOverlay} />
          {isAnalyzing && (
            <div className="absolute top-2 left-2 bg-blue-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <RefreshCw className="h-3 w-3 animate-spin" /> Processing...
            </div>
          )}
          {analysisComplete && (
            <div className="absolute top-2 left-2 bg-green-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
              <CheckCircle className="h-3 w-3" /> Analysis Complete
            </div>
          )}
        </div>

        {/* Results summary */}
        <div className="bg-secondary/30 rounded-lg p-4">
          {!analysisComplete && !isAnalyzing && (
            <div className="text-center py-8">
              <Brain className="h-10 w-10 text-muted-foreground mx-auto mb-3" />
              <h3 className="text-sm font-medium text-foreground mb-1">Ready to Analyze</h3>
              <p className="text-xs text-muted-foreground">Click "Run Analysis" to start the V-JEPA pipeline</p>
            </div>
          )}
          {isAnalyzing && (
            <div className="text-center py-8">
              <RefreshCw className="h-10 w-10 text-primary mx-auto mb-3 animate-spin" />
              <h3 className="text-sm font-medium text-foreground mb-1">Analyzing...</h3>
              <p className="text-xs text-muted-foreground">Running {pipelineStages[currentStageIdx]?.name || 'pipeline'}...</p>
            </div>
          )}
          {analysisComplete && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-400" />
                Analysis Results
              </h3>
              {scenario === 'gold_anomaly' && (
                <>
                  <div className="p-2 bg-red-500/10 border border-red-500/30 rounded">
                    <span className="text-xs text-red-400">5 anomalies detected</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p className="mb-1"><strong className="text-foreground">Highest score:</strong> 94% (alteration texture)</p>
                    <p><strong className="text-foreground">Action:</strong> Recommend rock chip sampling at A1, A2</p>
                  </div>
                </>
              )}
              {scenario === 'lithium_change' && (
                <>
                  <div className="p-2 bg-purple-500/10 border border-purple-500/30 rounded">
                    <span className="text-xs text-purple-400">3 change zones detected</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p className="mb-1"><strong className="text-foreground">Max change:</strong> +15% (pond expansion)</p>
                    <p><strong className="text-foreground">Action:</strong> Schedule brine sampling at C1</p>
                  </div>
                </>
              )}
              {scenario === 'alteration_similarity' && (
                <>
                  <div className="p-2 bg-green-500/10 border border-green-500/30 rounded">
                    <span className="text-xs text-green-400">5 similar zones found</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <p className="mb-1"><strong className="text-foreground">Best match:</strong> 92% (epithermal Au)</p>
                    <p><strong className="text-foreground">Action:</strong> Verify with rock chip sampling at S1, S2</p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="mt-3 text-xs text-muted-foreground bg-secondary/30 rounded p-2">
        <strong className="text-foreground">Synthetic Demo:</strong> This visualization uses simulated data to demonstrate the V-JEPA analysis pipeline.
      </div>
    </div>
  );
}

export default function AIInsightsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');

  const filteredInsights = mockInsights.filter((insight) => {
    if (selectedCategory !== 'all' && insight.category !== selectedCategory) return false;
    if (selectedType !== 'all' && insight.type !== selectedType) return false;
    if (selectedStatus !== 'all' && insight.status !== selectedStatus) return false;
    return true;
  });

  const handleViewDetails = (id: string) => {
    console.log('View details for insight:', id);
    // In production, this would navigate to a detail view or open a modal
  };

  const stats = {
    total: mockInsights.length,
    new: mockInsights.filter((i) => i.status === 'new').length,
    highConfidence: mockInsights.filter((i) => i.confidence === 'high').length,
    actioned: mockInsights.filter((i) => i.status === 'actioned').length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Brain className="h-7 w-7 text-primary" />
            AI Insights (V-JEPA)
          </h1>
          <p className="text-muted-foreground mt-1">
            Automated analysis of mining and exploration imagery using deep learning
          </p>
        </div>
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-4 py-2 bg-secondary text-foreground rounded-lg hover:bg-secondary/80 transition-colors">
            <Download className="h-4 w-4" />
            Export
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors">
            <Zap className="h-4 w-4" />
            New Analysis
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground text-sm">Total Insights</span>
            <Layers className="h-5 w-5 text-muted-foreground" />
          </div>
          <p className="text-2xl font-bold text-foreground mt-1">{stats.total}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground text-sm">New Findings</span>
            <AlertTriangle className="h-5 w-5 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-blue-400 mt-1">{stats.new}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground text-sm">High Confidence</span>
            <CheckCircle className="h-5 w-5 text-green-400" />
          </div>
          <p className="text-2xl font-bold text-green-400 mt-1">{stats.highConfidence}</p>
        </div>
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground text-sm">Actioned</span>
            <Target className="h-5 w-5 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-purple-400 mt-1">{stats.actioned}</p>
        </div>
      </div>

      {/* Interactive Analysis Demo */}
      <InteractiveAnalysisDemo />

      {/* Running Jobs */}
      {mockJobs.some((j) => j.status === 'running' || j.status === 'queued') && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h2 className="text-lg font-semibold text-foreground mb-3 flex items-center gap-2">
            <RefreshCw className="h-5 w-5 text-primary" />
            Active Analysis Jobs
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {mockJobs
              .filter((j) => j.status === 'running' || j.status === 'queued')
              .map((job) => (
                <JobStatusCard key={job.id} job={job} />
              ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 bg-card border border-border rounded-lg p-4">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Filters:</span>
        </div>
        
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="bg-secondary border border-border rounded-lg px-3 py-1.5 text-sm text-foreground"
        >
          <option value="all">All Categories</option>
          <option value="gold">Gold Exploration</option>
          <option value="lithium">Lithium Exploration</option>
          <option value="alteration">Alteration Analysis</option>
        </select>

        <select
          value={selectedType}
          onChange={(e) => setSelectedType(e.target.value)}
          className="bg-secondary border border-border rounded-lg px-3 py-1.5 text-sm text-foreground"
        >
          <option value="all">All Types</option>
          <option value="anomaly">Anomalies</option>
          <option value="change">Changes</option>
          <option value="similarity">Similarities</option>
        </select>

        <select
          value={selectedStatus}
          onChange={(e) => setSelectedStatus(e.target.value)}
          className="bg-secondary border border-border rounded-lg px-3 py-1.5 text-sm text-foreground"
        >
          <option value="all">All Status</option>
          <option value="new">New</option>
          <option value="reviewed">Reviewed</option>
          <option value="actioned">Actioned</option>
        </select>

        <span className="text-sm text-muted-foreground ml-auto">
          Showing {filteredInsights.length} of {mockInsights.length} insights
        </span>
      </div>

      {/* Insights Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredInsights.map((insight) => (
          <InsightCardComponent
            key={insight.id}
            insight={insight}
            onViewDetails={handleViewDetails}
          />
        ))}
      </div>

      {filteredInsights.length === 0 && (
        <div className="text-center py-12 bg-card border border-border rounded-lg">
          <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">No insights found</h3>
          <p className="text-muted-foreground">
            Try adjusting your filters or run a new analysis
          </p>
        </div>
      )}

      {/* How It Works Section */}
      <div className="bg-card border border-border rounded-lg p-6">
        <h2 className="text-lg font-semibold text-foreground mb-4">How V-JEPA Analysis Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <AlertTriangle className="h-5 w-5 text-red-400" />
            </div>
            <div>
              <h3 className="font-medium text-foreground">Anomaly Detection</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Identifies areas that look different from baseline imagery. High anomaly scores indicate unusual textures that may warrant field verification.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-purple-500/20 flex items-center justify-center">
              <TrendingUp className="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <h3 className="font-medium text-foreground">Change Detection</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Monitors evolution over time by comparing imagery from different dates. Useful for tracking site development, brine evolution, or vegetation changes.
              </p>
            </div>
          </div>
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-500/20 flex items-center justify-center">
              <Search className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <h3 className="font-medium text-foreground">Similarity Search</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Finds areas that look similar to known examples. Use this to find more of what you've already identified - alteration zones, regolith types, or geological features.
              </p>
            </div>
          </div>
        </div>
        <div className="mt-4 p-3 bg-secondary/30 rounded-lg">
          <p className="text-sm text-muted-foreground flex items-start gap-2">
            <ArrowRight className="h-4 w-4 mt-0.5 flex-shrink-0 text-primary" />
            <span>
              <strong className="text-foreground">Important:</strong> AI findings are for triage and prioritization. 
              All insights should be verified through field sampling, laboratory analysis, or expert review before making operational decisions.
            </span>
          </p>
        </div>
      </div>
    </div>
  );
}
