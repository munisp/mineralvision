import { useState, useEffect, useRef } from 'react';
import {
  Eye,
  Target,
  Layers,
  Brain,
  Cpu,
  Activity,
  Download,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  MapPin,
  Zap,
  Crosshair,
  Scan,
  Mountain,
  Gem,
  Film,
  ChevronRight,
  Thermometer,
  Beaker,
  Box,
} from 'lucide-react';

// ============================================================================
// SCENARIO TYPES AND DATA
// ============================================================================

type ScenarioType = 'gold_mine' | 'outcrop_survey';

interface Detection {
  id: string;
  className: string;
  confidence: number;
  bbox: [number, number, number, number];
  color: string;
}

interface PipelineStage {
  name: string;
  icon: React.ReactNode;
  status: 'pending' | 'running' | 'completed';
  duration?: number;
}

// Gold Mine Scenario Data
// Gold mine detections for future use with DetectionOverlay
const _goldMineDetections: Detection[] = [
  { id: 'gm-1', className: 'open_pit_boundary', confidence: 0.96, bbox: [50, 60, 350, 280], color: '#ef4444' },
  { id: 'gm-2', className: 'haul_road', confidence: 0.94, bbox: [360, 180, 580, 220], color: '#f59e0b' },
  { id: 'gm-3', className: 'ore_stockpile', confidence: 0.91, bbox: [400, 80, 520, 160], color: '#22c55e' },
  { id: 'gm-4', className: 'waste_dump', confidence: 0.89, bbox: [100, 290, 250, 340], color: '#6b7280' },
  { id: 'gm-5', className: 'gossan_zone', confidence: 0.87, bbox: [180, 100, 280, 180], color: '#eab308' },
  { id: 'gm-6', className: 'heavy_equipment', confidence: 0.93, bbox: [300, 200, 350, 250], color: '#8b5cf6' },
];
void _goldMineDetections; // Suppress unused variable warning

const goldMineAnalysis = {
  title: 'Gold Mine Analysis',
  mainFinding: {
    label: 'Active Gold Mining Operation',
    detected: true,
    confidence: 0.94,
  },
  classification: {
    mineType: 'Open Pit',
    activityStage: 'Production',
    estimatedScale: 'Medium (5-20 hectares)',
    operationType: 'Semi-mechanized',
  },
  geologicalIndicators: [
    { name: 'Gossan/Iron Cap', present: true, confidence: 0.87, significance: 'High - indicates sulfide oxidation' },
    { name: 'Quartz Veining', present: true, confidence: 0.82, significance: 'Moderate - potential gold host' },
    { name: 'Alteration Halo', present: true, confidence: 0.79, significance: 'High - hydrothermal activity' },
  ],
  environmentalRisk: {
    level: 'medium',
    factors: [
      'Sediment plume visible in drainage (turbidity risk)',
      'No visible tailings containment structure',
      'Vegetation clearing extends beyond pit boundary',
    ],
  },
  safetyFlags: [
    'Steep pit walls without benching (collapse risk)',
    'No visible safety barriers around pit edge',
    'Workers observed without PPE',
  ],
  recommendations: [
    'Ground-truth gossan zone: collect 5-8 rock chips at 25m intervals along ridge',
    'Run soil geochemistry grid (50m spacing) over alteration halo',
    'Plan IP/resistivity survey perpendicular to suspected shear zone',
    'Verify tailings management: check for downstream contamination',
    'Schedule repeat drone survey in 14 days for change detection',
    'Document coordinates for regulatory compliance reporting',
  ],
  evidence: 'Bench geometry + haul roads + stockpiles indicate active open pit operation. Bright oxidized zones (gossan) consistent with hematite/limonite after sulfide. Milky drainage plume suggests suspended solids from processing.',
};

// Outcrop Survey Assessment Scenario Data
const outcropSurveyAnalysis = {
  title: 'Outcrop & Alteration Assessment',
  mainFinding: {
    label: 'Prospectivity Index for Epithermal Au',
    score: 78,
    rating: 'Good',
  },
  regolithProfile: {
    textureClass: 'Saprolite over bedrock',
    drainageClass: 'Well-drained',
    slope: '2-5%',
    depth: '45+ cm to bedrock',
    color: 'Dark brown, Fe-stained (10YR 3/3)',
  },
  chemicalProperties: {
    pH: { value: 5.8, optimal: '5.5-6.5', status: 'optimal' },
    gold: { value: 0.8, optimal: '>0.5 ppm Au', status: 'optimal' },
    arsenic: { value: 3.2, optimal: '>2 ppm As', status: 'optimal' },
    copper: { value: 12.5, optimal: '10-25 ppm Cu', status: 'optimal' },
    silver: { value: 0.18, optimal: '>0.15 ppm Ag', status: 'optimal' },
    lead: { value: 18, optimal: '>15 ppm Pb', status: 'optimal' },
    zinc: { value: 145, optimal: '>120 ppm Zn', status: 'optimal' },
  },
  physicalProperties: {
    oxidation: { value: 'Strong (gossanous)', status: 'good' },
    veining: { value: 'Quartz veins present', status: 'good' },
    alteration: { value: 'Argillic-phyllic', status: 'good' },
  },
  limitingFactors: [
    { factor: 'Slight slope may complicate drill pad placement', severity: 'low' },
    { factor: 'Monitor seasonal drainage across survey lines', severity: 'low' },
  ],
  riskFactors: {
    access: { risk: 'Low', confidence: 0.85 },
    permitting: { risk: 'Low', confidence: 0.82 },
    structuralComplexity: { risk: 'Moderate', confidence: 0.78 },
  },
  recommendations: [
    'Proceed with detailed mapping - surface indicators are favorable',
    'Collect rock chip samples (5-8 per outcrop) for assay',
    'Run infill soil geochemistry at 25m spacing over the anomaly',
    'Plan IP/resistivity survey across the alteration halo',
    'Trench across the quartz vein trend to expose fresh bedrock',
    'Prioritize target for first-pass scout drilling',
    'Log all samples in the QAQC-tracked drillhole database',
  ],
  explorationTarget: {
    estimate: '0.8-1.2 g/t Au (conceptual)',
    confidence: 0.75,
    factors: 'Based on geochemistry, alteration mapping, and regional analogues',
  },
};

// ============================================================================
// COMPONENTS
// ============================================================================

function PipelineVisualization({ stages }: { stages: PipelineStage[] }) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-2">
      {stages.map((stage, idx) => (
        <div key={stage.name} className="flex items-center">
          <div
            className={`flex flex-col items-center p-3 rounded-lg border min-w-[100px] transition-all duration-300 ${
              stage.status === 'completed'
                ? 'bg-green-500/10 border-green-500/30'
                : stage.status === 'running'
                ? 'bg-blue-500/10 border-blue-500/30'
                : 'bg-secondary/50 border-border'
            }`}
          >
            <div
              className={`p-2 rounded-full mb-2 transition-all duration-300 ${
                stage.status === 'completed'
                  ? 'bg-green-500/20 text-green-400'
                  : stage.status === 'running'
                  ? 'bg-blue-500/20 text-blue-400'
                  : 'bg-secondary text-muted-foreground'
              }`}
            >
              {stage.status === 'running' ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : stage.status === 'completed' ? (
                <CheckCircle className="h-4 w-4" />
              ) : (
                stage.icon
              )}
            </div>
            <span className="text-xs font-medium text-foreground text-center">
              {stage.name}
            </span>
            {stage.status === 'completed' && stage.duration && (
              <span className="text-xs text-green-400 mt-1">{stage.duration}ms</span>
            )}
          </div>
          {idx < stages.length - 1 && (
            <ChevronRight className="h-4 w-4 text-muted-foreground mx-1 flex-shrink-0" />
          )}
        </div>
      ))}
    </div>
  );
}

// DetectionOverlay for future use with bounding box visualization
function _DetectionOverlay({ detections, show }: { detections: Detection[]; show: boolean }) {
  if (!show) return null;
  
  return (
    <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
      {detections.map((det, idx) => {
        const [x1, y1, x2, y2] = det.bbox;
        return (
          <g key={det.id} className="animate-fade-in" style={{ animationDelay: `${idx * 150}ms` }}>
            <rect
              x={x1} y={y1} width={x2 - x1} height={y2 - y1}
              fill="none" stroke={det.color} strokeWidth="2"
              className="animate-pulse"
            />
            <rect x={x1} y={y1 - 22} width={Math.max(100, det.className.length * 7 + 40)} height="20" fill={det.color} opacity="0.9" rx="2" />
            <text x={x1 + 4} y={y1 - 7} fill="white" fontSize="11" fontWeight="500">
              {det.className.replace(/_/g, ' ')} {(det.confidence * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
}
void _DetectionOverlay; // Suppress unused function warning

// Gold Mine Survey Overlay with geological features
function GoldMineSurveyOverlay({ show }: { show: boolean }) {
  if (!show) return null;

  // Drill targets with priority scores
  const drillTargets = [
    { id: 'DT1', x: 200, y: 140, priority: 'High', score: 92, type: 'Primary' },
    { id: 'DT2', x: 280, y: 180, priority: 'High', score: 88, type: 'Primary' },
    { id: 'DT3', x: 150, y: 200, priority: 'Medium', score: 75, type: 'Secondary' },
    { id: 'DT4', x: 350, y: 120, priority: 'Medium', score: 72, type: 'Secondary' },
    { id: 'DT5', x: 420, y: 200, priority: 'Low', score: 65, type: 'Exploration' },
  ];

  // Geological structures
  const structures = [
    { type: 'fault', points: '80,50 180,150 220,280', color: '#ef4444' },
    { type: 'shear_zone', points: '300,40 350,180 380,320', color: '#f97316' },
    { type: 'vein', points: '150,100 250,160 200,220', color: '#eab308' },
  ];

  // Alteration zones
  const alterationZones = [
    { cx: 220, cy: 160, rx: 80, ry: 60, type: 'silicification', color: '#fbbf24' },
    { cx: 320, cy: 140, rx: 60, ry: 45, type: 'argillic', color: '#a78bfa' },
    { cx: 180, cy: 220, rx: 50, ry: 35, type: 'propylitic', color: '#34d399' },
  ];

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High': return '#ef4444';
      case 'Medium': return '#f59e0b';
      case 'Low': return '#6b7280';
      default: return '#6b7280';
    }
  };

  return (
    <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
      {/* Concession boundary */}
      <polygon
        points="30,30 610,25 620,330 40,340"
        fill="none"
        stroke="#f59e0b"
        strokeWidth="3"
        strokeDasharray="10,5"
        className="animate-pulse"
      />

      {/* Exploration grid */}
      <g stroke="#4b5563" strokeWidth="0.5" opacity="0.25">
        {[0, 1, 2, 3, 4, 5].map(i => (
          <line key={`h${i}`} x1="30" y1={30 + i * 60} x2="620" y2={25 + i * 60} />
        ))}
        {[0, 1, 2, 3, 4, 5, 6].map(i => (
          <line key={`v${i}`} x1={30 + i * 100} y1="25" x2={35 + i * 100} y2="340" />
        ))}
      </g>

      {/* Alteration zones (halos) */}
      {alterationZones.map((zone, idx) => (
        <g key={`alt-${idx}`} className="animate-fade-in" style={{ animationDelay: `${idx * 200}ms` }}>
          <ellipse
            cx={zone.cx} cy={zone.cy} rx={zone.rx} ry={zone.ry}
            fill={zone.color} opacity="0.2"
            stroke={zone.color} strokeWidth="1" strokeDasharray="4,2"
          />
          <text x={zone.cx} y={zone.cy + zone.ry + 12} fill={zone.color} fontSize="8" textAnchor="middle">
            {zone.type}
          </text>
        </g>
      ))}

      {/* Geological structures */}
      {structures.map((struct, idx) => (
        <g key={`struct-${idx}`} className="animate-fade-in" style={{ animationDelay: `${300 + idx * 150}ms` }}>
          <polyline
            points={struct.points}
            fill="none"
            stroke={struct.color}
            strokeWidth="3"
            strokeLinecap="round"
            opacity="0.8"
          />
          {struct.type === 'fault' && (
            <>
              <circle cx="80" cy="50" r="4" fill={struct.color} />
              <circle cx="220" cy="280" r="4" fill={struct.color} />
            </>
          )}
        </g>
      ))}

      {/* Pit outline */}
      <polygon
        points="120,80 320,70 380,200 340,280 150,290 100,180"
        fill="#1f2937"
        fillOpacity="0.3"
        stroke="#ef4444"
        strokeWidth="2"
      />
      <text x="230" y="180" fill="#ef4444" fontSize="10" textAnchor="middle" fontWeight="bold">
        OPEN PIT
      </text>

      {/* Drill targets */}
      {drillTargets.map((target, idx) => (
        <g key={target.id} className="animate-fade-in" style={{ animationDelay: `${500 + idx * 100}ms` }}>
          {/* Target ring */}
          <circle
            cx={target.x} cy={target.y} r="20"
            fill="none"
            stroke={getPriorityColor(target.priority)}
            strokeWidth="2"
            strokeDasharray="4,2"
          />
          {/* Crosshair */}
          <line x1={target.x - 12} y1={target.y} x2={target.x + 12} y2={target.y} stroke={getPriorityColor(target.priority)} strokeWidth="2" />
          <line x1={target.x} y1={target.y - 12} x2={target.x} y2={target.y + 12} stroke={getPriorityColor(target.priority)} strokeWidth="2" />
          {/* Center dot */}
          <circle cx={target.x} cy={target.y} r="4" fill={getPriorityColor(target.priority)} />
          {/* Label */}
          <rect x={target.x - 18} y={target.y + 22} width="36" height="16" fill="#1f2937" opacity="0.9" rx="2" />
          <text x={target.x} y={target.y + 34} fill="white" fontSize="9" textAnchor="middle" fontWeight="bold">
            {target.id}
          </text>
          {/* Score badge */}
          <circle cx={target.x + 18} cy={target.y - 18} r="10" fill={getPriorityColor(target.priority)} />
          <text x={target.x + 18} y={target.y - 14} fill="white" fontSize="8" textAnchor="middle" fontWeight="bold">
            {target.score}
          </text>
        </g>
      ))}

      {/* Gossan zone highlight */}
      <ellipse cx="200" cy="130" rx="50" ry="30" fill="#eab308" opacity="0.25" stroke="#eab308" strokeWidth="2" />
      <text x="200" y="100" fill="#eab308" fontSize="9" textAnchor="middle" fontWeight="bold">GOSSAN</text>

      {/* Legend */}
      <g transform="translate(480, 20)">
        <rect x="0" y="0" width="140" height="100" fill="#1f2937" opacity="0.95" rx="4" />
        <text x="10" y="16" fill="white" fontSize="10" fontWeight="bold">Exploration Targets</text>
        
        <circle cx="20" cy="32" r="5" fill="#ef4444" />
        <text x="32" y="36" fill="#9ca3af" fontSize="9">High Priority</text>
        
        <circle cx="20" cy="48" r="5" fill="#f59e0b" />
        <text x="32" y="52" fill="#9ca3af" fontSize="9">Medium Priority</text>
        
        <circle cx="20" cy="64" r="5" fill="#6b7280" />
        <text x="32" y="68" fill="#9ca3af" fontSize="9">Low Priority</text>

        <line x1="15" y1="80" x2="25" y2="80" stroke="#ef4444" strokeWidth="2" />
        <text x="32" y="84" fill="#9ca3af" fontSize="9">Fault/Structure</text>

        <ellipse cx="20" cy="94" rx="8" ry="4" fill="#fbbf24" opacity="0.5" />
        <text x="32" y="98" fill="#9ca3af" fontSize="9">Alteration Zone</text>
      </g>

      {/* Title label */}
      <rect x="10" y="320" width="220" height="30" fill="#1f2937" opacity="0.9" rx="4" />
      <text x="20" y="340" fill="#f59e0b" fontSize="11" fontWeight="bold">Geological Survey Map (Demo)</text>
    </svg>
  );
}

// Outcrop Survey Overlay for Outcrop Survey Assessment
function OutcropSurveyOverlay({ show }: { show: boolean }) {
  if (!show) return null;

  // Sample points with prospectivity scores
  const samplePoints = [
    { id: 'S1', x: 120, y: 80, score: 85, label: 'Sample 1' },
    { id: 'S2', x: 280, y: 60, score: 78, label: 'Sample 2' },
    { id: 'S3', x: 450, y: 90, score: 82, label: 'Sample 3' },
    { id: 'S4', x: 100, y: 180, score: 72, label: 'Sample 4' },
    { id: 'S5', x: 320, y: 160, score: 88, label: 'Sample 5' },
    { id: 'S6', x: 500, y: 170, score: 75, label: 'Sample 6' },
    { id: 'S7', x: 150, y: 280, score: 68, label: 'Sample 7' },
    { id: 'S8', x: 350, y: 260, score: 80, label: 'Sample 8' },
    { id: 'S9', x: 520, y: 290, score: 73, label: 'Sample 9' },
  ];

  const getScoreColor = (score: number) => {
    if (score >= 80) return '#22c55e'; // green
    if (score >= 70) return '#eab308'; // yellow
    return '#ef4444'; // red
  };

  return (
    <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 640 360" preserveAspectRatio="none">
      {/* Survey boundary polygon */}
      <polygon
        points="40,40 600,30 620,320 50,340"
        fill="none"
        stroke="#22c55e"
        strokeWidth="3"
        strokeDasharray="8,4"
        className="animate-pulse"
      />
      
      {/* Prospectivity heatmap zones (simplified gradient zones) */}
      <ellipse cx="320" cy="160" rx="180" ry="100" fill="#22c55e" opacity="0.15" />
      <ellipse cx="280" cy="140" rx="120" ry="70" fill="#22c55e" opacity="0.2" />
      <ellipse cx="150" cy="280" rx="80" ry="50" fill="#eab308" opacity="0.2" />
      <ellipse cx="520" cy="280" rx="70" ry="45" fill="#eab308" opacity="0.15" />
      
      {/* Grid lines for survey pattern */}
      <g stroke="#4b5563" strokeWidth="0.5" opacity="0.3">
        <line x1="40" y1="120" x2="620" y2="110" />
        <line x1="45" y1="200" x2="615" y2="195" />
        <line x1="50" y1="280" x2="610" y2="280" />
        <line x1="160" y1="35" x2="150" y2="340" />
        <line x1="320" y1="32" x2="320" y2="335" />
        <line x1="480" y1="30" x2="490" y2="330" />
      </g>

      {/* Sample points */}
      {samplePoints.map((point, idx) => (
        <g key={point.id} className="animate-fade-in" style={{ animationDelay: `${idx * 100}ms` }}>
          {/* Outer ring */}
          <circle
            cx={point.x}
            cy={point.y}
            r="18"
            fill={getScoreColor(point.score)}
            opacity="0.3"
          />
          {/* Inner circle */}
          <circle
            cx={point.x}
            cy={point.y}
            r="10"
            fill={getScoreColor(point.score)}
            stroke="white"
            strokeWidth="2"
          />
          {/* Score label */}
          <text
            x={point.x}
            y={point.y + 4}
            fill="white"
            fontSize="9"
            fontWeight="bold"
            textAnchor="middle"
          >
            {point.score}
          </text>
          {/* Sample ID label */}
          <rect
            x={point.x - 12}
            y={point.y + 14}
            width="24"
            height="14"
            fill="#1f2937"
            opacity="0.9"
            rx="2"
          />
          <text
            x={point.x}
            y={point.y + 24}
            fill="#9ca3af"
            fontSize="8"
            textAnchor="middle"
          >
            {point.id}
          </text>
        </g>
      ))}

      {/* Legend */}
      <g transform="translate(500, 20)">
        <rect x="0" y="0" width="120" height="70" fill="#1f2937" opacity="0.9" rx="4" />
        <text x="10" y="16" fill="white" fontSize="10" fontWeight="bold">Prospectivity Index</text>
        <circle cx="20" cy="32" r="6" fill="#22c55e" />
        <text x="32" y="36" fill="#9ca3af" fontSize="9">High (80+)</text>
        <circle cx="20" cy="48" r="6" fill="#eab308" />
        <text x="32" y="52" fill="#9ca3af" fontSize="9">Moderate (70-79)</text>
        <circle cx="20" cy="64" r="6" fill="#ef4444" />
        <text x="32" y="68" fill="#9ca3af" fontSize="9">Low (&lt;70)</text>
      </g>

      {/* Title label */}
      <rect x="10" y="320" width="200" height="30" fill="#1f2937" opacity="0.9" rx="4" />
      <text x="20" y="340" fill="#22c55e" fontSize="11" fontWeight="bold">Outcrop Survey Grid (Demo)</text>
    </svg>
  );
}

function GoldMineResultCard({ analysis, show }: { analysis: typeof goldMineAnalysis; show: boolean }) {
  if (!show) return null;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Main Finding */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          {analysis.title}
        </h3>
        <div className={`p-3 rounded-lg ${analysis.mainFinding.detected ? 'bg-yellow-500/10 border border-yellow-500/30' : 'bg-green-500/10 border border-green-500/30'}`}>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-foreground">{analysis.mainFinding.label}</span>
            <span className="text-lg font-bold text-yellow-400">CONFIRMED</span>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
              <div className="h-full bg-yellow-500 rounded-full transition-all duration-1000" style={{ width: `${analysis.mainFinding.confidence * 100}%` }} />
            </div>
            <span className="text-xs text-muted-foreground">{(analysis.mainFinding.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* Classification */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Gem className="h-4 w-4 text-yellow-400" />
          Site Classification
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(analysis.classification).map(([key, value]) => (
            <div key={key} className="p-2 bg-secondary/30 rounded-lg">
              <span className="text-xs text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
              <p className="text-sm font-medium text-foreground">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Geological Indicators */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Mountain className="h-4 w-4 text-orange-400" />
          Geological Indicators
        </h3>
        <div className="space-y-2">
          {analysis.geologicalIndicators.map((ind, idx) => (
            <div key={idx} className="p-2 bg-secondary/30 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">{ind.name}</span>
                <span className="text-xs text-green-400">{(ind.confidence * 100).toFixed(0)}%</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">{ind.significance}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Environmental Risk */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-yellow-400" />
          Environmental Risk: <span className="text-yellow-400 uppercase">{analysis.environmentalRisk.level}</span>
        </h3>
        <div className="space-y-1">
          {analysis.environmentalRisk.factors.map((factor, idx) => (
            <div key={idx} className="text-xs text-yellow-400 bg-yellow-500/10 px-2 py-1 rounded">{factor}</div>
          ))}
        </div>
      </div>

      {/* Safety Flags */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-red-400" />
          Safety Concerns
        </h3>
        <div className="space-y-1">
          {analysis.safetyFlags.map((flag, idx) => (
            <div key={idx} className="text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded">{flag}</div>
          ))}
        </div>
      </div>

      {/* Evidence */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Eye className="h-4 w-4 text-blue-400" />
          Visual Evidence
        </h3>
        <p className="text-xs text-muted-foreground italic">{analysis.evidence}</p>
      </div>

      {/* Recommendations */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Target className="h-4 w-4 text-primary" />
          Recommended Actions
        </h3>
        <div className="space-y-1">
          {analysis.recommendations.map((rec, idx) => (
            <div key={idx} className="text-xs text-foreground bg-primary/10 px-2 py-1.5 rounded flex items-start gap-1">
              <span className="text-primary font-bold">{idx + 1}.</span> {rec}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function OutcropSurveyResultCard({ analysis, show }: { analysis: typeof outcropSurveyAnalysis; show: boolean }) {
  if (!show) return null;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'optimal': return 'text-green-400';
      case 'good': return 'text-green-400';
      case 'marginal': return 'text-yellow-400';
      case 'poor': return 'text-red-400';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Main Finding - Prospectivity Score */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Mountain className="h-4 w-4 text-green-400" />
          {analysis.title}
        </h3>
        <div className="bg-green-500/10 border border-green-500/30 p-4 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-foreground">{analysis.mainFinding.label}</span>
            <div className="text-right">
              <span className="text-3xl font-bold text-green-400">{analysis.mainFinding.score}</span>
              <span className="text-lg text-green-400">/100</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-3 bg-secondary rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full" style={{ width: `${analysis.mainFinding.score}%` }} />
            </div>
            <span className="text-sm font-bold text-green-400">{analysis.mainFinding.rating}</span>
          </div>
        </div>
      </div>

      {/* Regolith Profile */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Layers className="h-4 w-4 text-amber-600" />
          Regolith Profile
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(analysis.regolithProfile).map(([key, value]) => (
            <div key={key} className="p-2 bg-secondary/30 rounded-lg">
              <span className="text-xs text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
              <p className="text-sm font-medium text-foreground">{value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Chemical Properties */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Beaker className="h-4 w-4 text-blue-400" />
          Chemical Properties
        </h3>
        <div className="space-y-2">
          {Object.entries(analysis.chemicalProperties).map(([key, data]) => (
            <div key={key} className="flex items-center justify-between p-2 bg-secondary/30 rounded-lg">
              <span className="text-xs text-muted-foreground uppercase">{key}</span>
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-foreground">{data.value}{key === 'pH' ? '' : ' ppm'}</span>
                <span className={`text-xs ${getStatusColor(data.status)}`}>({data.optimal})</span>
                <CheckCircle className={`h-3 w-3 ${getStatusColor(data.status)}`} />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Physical Properties */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Thermometer className="h-4 w-4 text-orange-400" />
          Physical Properties
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(analysis.physicalProperties).map(([key, data]) => (
            <div key={key} className="p-2 bg-secondary/30 rounded-lg text-center">
              <span className="text-xs text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
              <p className={`text-sm font-medium ${getStatusColor(data.status)}`}>{data.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Exploration Risk */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-yellow-400" />
          Exploration Risk Assessment
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {Object.entries(analysis.riskFactors).map(([factor, data]) => (
            <div key={factor} className={`p-2 rounded-lg text-center ${data.risk === 'Low' ? 'bg-green-500/10' : data.risk === 'Moderate' ? 'bg-yellow-500/10' : 'bg-red-500/10'}`}>
              <span className="text-xs text-muted-foreground capitalize">{factor}</span>
              <p className={`text-sm font-bold ${data.risk === 'Low' ? 'text-green-400' : data.risk === 'Moderate' ? 'text-yellow-400' : 'text-red-400'}`}>{data.risk}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Limiting Factors */}
      {analysis.limitingFactors.length > 0 && (
        <div className="bg-card border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-yellow-400" />
            Limiting Factors
          </h3>
          <div className="space-y-1">
            {analysis.limitingFactors.map((lf, idx) => (
              <div key={idx} className={`text-xs px-2 py-1 rounded ${lf.severity === 'low' ? 'text-yellow-400 bg-yellow-500/10' : 'text-red-400 bg-red-500/10'}`}>
                {lf.factor}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Exploration Target */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <TrendingUp className="h-4 w-4 text-green-400" />
          Exploration Target
        </h3>
        <div className="bg-green-500/10 border border-green-500/30 p-3 rounded-lg">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Conceptual Grade</span>
            <span className="text-xl font-bold text-green-400">{analysis.explorationTarget.estimate}</span>
          </div>
          <p className="text-xs text-muted-foreground mt-2">{analysis.explorationTarget.factors}</p>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
          <Target className="h-4 w-4 text-primary" />
          Exploration Recommendations
        </h3>
        <div className="space-y-1">
          {analysis.recommendations.map((rec, idx) => (
            <div key={idx} className="text-xs text-foreground bg-primary/10 px-2 py-1.5 rounded flex items-start gap-1">
              <span className="text-primary font-bold">{idx + 1}.</span> {rec}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// MAIN PAGE COMPONENT
// ============================================================================

export default function Molmo2Page() {
  const [scenario, setScenario] = useState<ScenarioType>('gold_mine');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [showDetections, setShowDetections] = useState(false);
  const [currentStageIdx, setCurrentStageIdx] = useState(-1);
  const analysisRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getInitialStages = (): PipelineStage[] => [
    { name: 'Load Image', icon: <Film className="h-4 w-4" />, status: 'pending' },
    { name: 'YOLO11', icon: <Scan className="h-4 w-4" />, status: 'pending' },
    { name: 'RF-DETR', icon: <Box className="h-4 w-4" />, status: 'pending' },
    { name: 'Ensemble', icon: <Layers className="h-4 w-4" />, status: 'pending' },
    { name: 'SAM3', icon: <Crosshair className="h-4 w-4" />, status: 'pending' },
    { name: 'V-JEPA', icon: <Activity className="h-4 w-4" />, status: 'pending' },
    { name: 'Molmo2', icon: <Brain className="h-4 w-4" />, status: 'pending' },
  ];

  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(getInitialStages());

  const stageDurations = [120, 45, 52, 18, 89, 134, 267];

  const runAnalysis = () => {
    if (isAnalyzing) return;
    
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setShowDetections(false);
    setCurrentStageIdx(0);
    setPipelineStages(getInitialStages());

    let stageIdx = 0;
    
    const processStage = () => {
      if (stageIdx >= pipelineStages.length) {
        setIsAnalyzing(false);
        setAnalysisComplete(true);
        setShowDetections(true);
        return;
      }

      // Set current stage to running
      setPipelineStages(prev => prev.map((s, i) => ({
        ...s,
        status: i === stageIdx ? 'running' : i < stageIdx ? 'completed' : 'pending',
        duration: i < stageIdx ? stageDurations[i] : undefined,
      })));

      // After duration, complete this stage and move to next
      analysisRef.current = setTimeout(() => {
        setPipelineStages(prev => prev.map((s, i) => ({
          ...s,
          status: i <= stageIdx ? 'completed' : 'pending',
          duration: i <= stageIdx ? stageDurations[i] : undefined,
        })));
        
        // Show detections after ensemble stage
        if (stageIdx === 3) {
          setShowDetections(true);
        }
        
        stageIdx++;
        setCurrentStageIdx(stageIdx);
        processStage();
      }, stageDurations[stageIdx] * 3); // Slow down for visibility
    };

    processStage();
  };

  const resetAnalysis = () => {
    if (analysisRef.current) {
      clearTimeout(analysisRef.current);
    }
    setIsAnalyzing(false);
    setAnalysisComplete(false);
    setShowDetections(false);
    setCurrentStageIdx(-1);
    setPipelineStages(getInitialStages());
  };

  // Reset when scenario changes
  useEffect(() => {
    resetAnalysis();
  }, [scenario]);

  const totalDuration = stageDurations.reduce((a, b) => a + b, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
            <Brain className="h-7 w-7 text-primary" />
            Molmo2 AI Analysis
          </h1>
          <p className="text-muted-foreground mt-1">
            Select a scenario and click Analyze to run the AI pipeline
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {/* Scenario Selector */}
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value as ScenarioType)}
            className="px-4 py-2 bg-secondary text-foreground rounded-lg border border-border text-sm font-medium"
            disabled={isAnalyzing}
          >
            <option value="gold_mine">Gold Mine Analysis</option>
            <option value="outcrop_survey">Outcrop Survey Assessment</option>
          </select>
          
          <button
            onClick={runAnalysis}
            disabled={isAnalyzing}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
              isAnalyzing
                ? 'bg-secondary text-muted-foreground cursor-not-allowed'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}
          >
            {isAnalyzing ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4" />
                Analyze
              </>
            )}
          </button>

          {analysisComplete && (
            <button
              onClick={() => {
                const data = scenario === 'gold_mine' ? goldMineAnalysis : outcropSurveyAnalysis;
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${scenario}_analysis.json`;
                a.click();
              }}
              className="flex items-center gap-2 px-4 py-2 bg-secondary text-foreground rounded-lg hover:bg-secondary/80"
            >
              <Download className="h-4 w-4" />
              Export JSON
            </button>
          )}
        </div>
      </div>

      {/* Pipeline Visualization */}
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
            <Cpu className="h-5 w-5 text-primary" />
            Ensemble Pipeline
          </h2>
          {analysisComplete && (
            <span className="text-xs text-green-400 flex items-center gap-1">
              <CheckCircle className="h-3 w-3" />
              Completed in {totalDuration}ms
            </span>
          )}
        </div>
        <PipelineVisualization stages={pipelineStages} />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Image/Video Panel */}
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          <div className="relative aspect-video bg-gradient-to-br from-gray-800 to-gray-900">
            {/* Scenario-specific background */}
            <div className="absolute inset-0 flex items-center justify-center">
              {scenario === 'gold_mine' ? (
                <div className="text-center">
                  <Gem className="h-16 w-16 text-yellow-600/50 mx-auto mb-2" />
                  <span className="text-gray-500 text-sm">Gold Mine - Drone Survey</span>
                  <div className="mt-1 text-xs text-gray-600">Site: Prospect Ridge Alpha</div>
                  <div className="text-xs text-gray-600">Coordinates: 9.0820N, 8.6753E</div>
                </div>
              ) : (
                <div className="text-center">
                  <Mountain className="h-16 w-16 text-green-600/50 mx-auto mb-2" />
                  <span className="text-gray-500 text-sm">Gold Prospect - Outcrop Survey</span>
                  <div className="mt-1 text-xs text-gray-600">Site: Tenement Block E-12</div>
                  <div className="text-xs text-gray-600">Area: 15 hectares</div>
                </div>
              )}
            </div>

            {/* Gold Mine Survey Overlay - geological features and drill targets */}
            {scenario === 'gold_mine' && (
              <GoldMineSurveyOverlay show={showDetections} />
            )}

            {/* Outcrop Survey Overlay - Only for outcrop survey */}
            {scenario === 'outcrop_survey' && (
              <OutcropSurveyOverlay show={showDetections} />
            )}

            {/* Status overlay */}
            {isAnalyzing && (
              <div className="absolute top-2 left-2 bg-blue-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
                <RefreshCw className="h-3 w-3 animate-spin" />
                Processing...
              </div>
            )}
            {analysisComplete && (
              <div className="absolute top-2 left-2 bg-green-500/80 px-3 py-1 rounded text-xs text-white flex items-center gap-2">
                <CheckCircle className="h-3 w-3" />
                Analysis Complete
              </div>
            )}
          </div>

          {/* Geological survey summary */}
          {showDetections && scenario === 'gold_mine' && (
            <div className="p-3 bg-secondary/30 border-t border-border">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Crosshair className="h-3 w-3 text-red-400" />
                  5 drill targets identified
                </span>
                <span className="flex items-center gap-1">
                  <Layers className="h-3 w-3 text-yellow-400" />
                  3 alteration zones mapped
                </span>
              </div>
            </div>
          )}

          {/* Outcrop survey summary */}
          {showDetections && scenario === 'outcrop_survey' && (
            <div className="p-3 bg-secondary/30 border-t border-border">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <MapPin className="h-3 w-3 text-green-400" />
                  9 sample points analyzed
                </span>
                <span className="flex items-center gap-1">
                  <Layers className="h-3 w-3 text-green-400" />
                  Prospectivity interpolation applied
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="max-h-[600px] overflow-y-auto pr-2">
          {!analysisComplete && !isAnalyzing && (
            <div className="bg-card border border-border rounded-lg p-8 text-center">
              <Brain className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium text-foreground mb-2">Ready to Analyze</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Select a scenario and click "Analyze" to run the Molmo2 AI pipeline
              </p>
              <div className="text-xs text-muted-foreground">
                {scenario === 'gold_mine' 
                  ? 'This will analyze drone imagery for gold mining indicators, geological features, and environmental risks.'
                  : 'This will assess lithology, alteration, and geochemical indicators across the outcrop survey grid.'}
              </div>
            </div>
          )}

          {isAnalyzing && (
            <div className="bg-card border border-border rounded-lg p-8 text-center">
              <RefreshCw className="h-12 w-12 text-primary mx-auto mb-4 animate-spin" />
              <h3 className="text-lg font-medium text-foreground mb-2">Analyzing...</h3>
              <p className="text-sm text-muted-foreground">
                Running {pipelineStages[currentStageIdx]?.name || 'pipeline'}...
              </p>
            </div>
          )}

          {analysisComplete && scenario === 'gold_mine' && (
            <GoldMineResultCard analysis={goldMineAnalysis} show={true} />
          )}

          {analysisComplete && scenario === 'outcrop_survey' && (
            <OutcropSurveyResultCard analysis={outcropSurveyAnalysis} show={true} />
          )}
        </div>
      </div>

      {/* Footer info */}
      <div className="bg-secondary/30 border border-border rounded-lg p-3 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">AI Pipeline:</span> This analysis uses the Molmo2 ensemble pipeline integrating YOLO11, RF-DETR, SAM3, and V-JEPA models for comprehensive geological and lithological assessment. Connect to backend services for real-time inference.
      </div>
    </div>
  );
}
