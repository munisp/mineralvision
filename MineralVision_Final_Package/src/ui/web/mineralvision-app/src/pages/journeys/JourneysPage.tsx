import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Tooltip,
  Paper,
  Divider,
} from '@mui/material';
import {
  PlayArrow,
  CheckCircle,
  Error,
  HourglassEmpty,
  Refresh,
  Info,
  Timeline,
  Schedule,
  Security,
  Storage,
  Cloud,
  Memory,
  Sensors,
  Psychology,
  Visibility,
  ThreeDRotation,
  VerifiedUser,
  Explore,
} from '@mui/icons-material';

interface JourneyStep {
  id: string;
  name: string;
  step_type: string;
  endpoint?: string;
  module?: string;
  timeout_seconds: number;
  requires_approval: boolean;
  kafka_topic?: string;
  permission_check?: string;
}

interface Journey {
  id: string;
  name: string;
  description: string;
  category: string;
  steps: JourneyStep[];
  ui_entry_point: string;
  required_permissions: string[];
  estimated_duration_minutes: number;
  tags: string[];
}

interface WorkflowRun {
  workflow_id: string;
  run_id: string;
  journey_id: string;
  status: string;
  current_step?: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

interface MiddlewareStatus {
  status: Record<string, string>;
  connected_count: number;
  total_count: number;
}

const categoryIcons: Record<string, React.ReactNode> = {
  project_management: <Storage />,
  data_ingestion: <Cloud />,
  qaqc: <VerifiedUser />,
  geostatistics: <Timeline />,
  geophysics: <Memory />,
  sensor_fusion: <Sensors />,
  ml_predictions: <Psychology />,
  vision_ai: <Visibility />,
  digital_twin: <ThreeDRotation />,
  compliance: <Security />,
  autonomous: <Explore />,
};

const categoryColors: Record<string, string> = {
  project_management: '#2196f3',
  data_ingestion: '#4caf50',
  qaqc: '#ff9800',
  geostatistics: '#9c27b0',
  geophysics: '#f44336',
  sensor_fusion: '#00bcd4',
  ml_predictions: '#e91e63',
  vision_ai: '#673ab7',
  digital_twin: '#3f51b5',
  compliance: '#795548',
  autonomous: '#607d8b',
};

const JourneysPage: React.FC = () => {
  const [journeys, setJourneys] = useState<Journey[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedJourney, setSelectedJourney] = useState<Journey | null>(null);
  const [activeRuns, setActiveRuns] = useState<WorkflowRun[]>([]);
  const [middlewareStatus, setMiddlewareStatus] = useState<MiddlewareStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [startDialogOpen, setStartDialogOpen] = useState(false);
  const [projectId, setProjectId] = useState('');
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJourneys();
    fetchMiddlewareStatus();
    fetchActiveRuns();
    const interval = setInterval(fetchActiveRuns, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchJourneys = async () => {
    try {
      const response = await fetch('/api/journeys/');
      if (response.ok) {
        const data = await response.json();
        setJourneys(data.journeys);
        setCategories(data.categories);
      } else {
        // API returned error, use mock data
        console.warn('API returned error, using mock data');
        setJourneys(getMockJourneys());
        setCategories(['project_management', 'data_ingestion', 'qaqc', 'geostatistics', 'geophysics', 'sensor_fusion', 'ml_predictions', 'vision_ai', 'digital_twin', 'compliance', 'autonomous']);
      }
    } catch (err) {
      console.error('Failed to fetch journeys:', err);
      // Use mock data for demo
      setJourneys(getMockJourneys());
      setCategories(['project_management', 'data_ingestion', 'qaqc', 'geostatistics', 'geophysics', 'sensor_fusion', 'ml_predictions', 'vision_ai', 'digital_twin', 'compliance', 'autonomous']);
    } finally {
      setLoading(false);
    }
  };

  const fetchMiddlewareStatus = async () => {
    const mockStatus = {
      status: {
        kafka: 'disconnected',
        fluvio: 'disconnected',
        redis: 'disconnected',
        keycloak: 'disconnected',
        permify: 'disconnected',
        dapr: 'disconnected',
        tigerbeetle: 'disconnected',
        lakehouse: 'disconnected',
      },
      connected_count: 0,
      total_count: 8,
    };
    try {
      const response = await fetch('/api/journeys/middleware/status');
      if (response.ok) {
        const data = await response.json();
        setMiddlewareStatus(data);
      } else {
        console.warn('Middleware status API returned error, using mock data');
        setMiddlewareStatus(mockStatus);
      }
    } catch (err) {
      console.error('Failed to fetch middleware status:', err);
      setMiddlewareStatus(mockStatus);
    }
  };

  const fetchActiveRuns = async () => {
    try {
      const response = await fetch('/api/journeys/runs?status=running');
      if (response.ok) {
        const data = await response.json();
        setActiveRuns(data.runs);
      }
    } catch (err) {
      console.error('Failed to fetch active runs:', err);
    }
  };

  const handleStartJourney = async () => {
    if (!selectedJourney || !projectId) return;
    
    setStarting(true);
    setError(null);
    
    try {
      const response = await fetch('/api/journeys/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          journey_id: selectedJourney.id,
          project_id: projectId,
          inputs: {},
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setActiveRuns(prev => [...prev, data]);
        setStartDialogOpen(false);
        setProjectId('');
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start journey');
      }
    } catch (err) {
      setError('Failed to connect to server');
    } finally {
      setStarting(false);
    }
  };

  const filteredJourneys = selectedCategory === 'all'
    ? journeys
    : journeys.filter(j => j.category === selectedCategory);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle color="success" />;
      case 'failed': return <Error color="error" />;
      case 'running': return <CircularProgress size={20} />;
      case 'waiting_approval': return <HourglassEmpty color="warning" />;
      default: return <HourglassEmpty />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          User Journeys
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Tooltip title="Refresh">
            <IconButton onClick={() => { fetchJourneys(); fetchActiveRuns(); }}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Middleware Status */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>Middleware Status</Typography>
        <Grid container spacing={1}>
          {middlewareStatus && Object.entries(middlewareStatus.status).map(([name, status]) => (
            <Grid size="auto" key={name}>
              <Chip
                label={name}
                color={status === 'connected' ? 'success' : 'default'}
                size="small"
                icon={status === 'connected' ? <CheckCircle /> : <Error />}
              />
            </Grid>
          ))}
        </Grid>
        {middlewareStatus && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            {middlewareStatus.connected_count}/{middlewareStatus.total_count} middleware components connected
          </Typography>
        )}
      </Paper>

      {/* Active Runs */}
      {activeRuns.length > 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>Active Workflow Runs</Typography>
          <List dense>
            {activeRuns.map(run => (
              <ListItem key={run.workflow_id}>
                <ListItemIcon>{getStatusIcon(run.status)}</ListItemIcon>
                <ListItemText
                  primary={run.journey_id}
                  secondary={`Started: ${run.started_at} | Step: ${run.current_step || 'N/A'}`}
                />
                <Chip label={run.status} size="small" />
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Category Tabs */}
      <Tabs
        value={selectedCategory}
        onChange={(_, value) => setSelectedCategory(value)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ mb: 3 }}
      >
        <Tab value="all" label="All Journeys" />
        {categories.map(cat => (
          <Tab
            key={cat}
            value={cat}
            label={cat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            icon={categoryIcons[cat] as React.ReactElement | undefined}
            iconPosition="start"
          />
        ))}
      </Tabs>

      {/* Journey Cards */}
      {loading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {filteredJourneys.map(journey => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={journey.id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  borderLeft: `4px solid ${categoryColors[journey.category] || '#666'}`,
                  '&:hover': { boxShadow: 4 },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    {categoryIcons[journey.category]}
                    <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                      {journey.category.replace(/_/g, ' ').toUpperCase()}
                    </Typography>
                  </Box>
                  <Typography variant="h6" gutterBottom>
                    {journey.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {journey.description}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 2 }}>
                    {journey.tags.slice(0, 3).map(tag => (
                      <Chip key={tag} label={tag} size="small" variant="outlined" />
                    ))}
                  </Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Schedule fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="caption">
                        {journey.estimated_duration_minutes} min
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Timeline fontSize="small" sx={{ mr: 0.5 }} />
                      <Typography variant="caption">
                        {journey.steps.length} steps
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
                <Divider />
                <Box sx={{ p: 1, display: 'flex', justifyContent: 'space-between' }}>
                  <Button
                    size="small"
                    startIcon={<Info />}
                    onClick={() => setSelectedJourney(journey)}
                  >
                    Details
                  </Button>
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<PlayArrow />}
                    onClick={() => {
                      setSelectedJourney(journey);
                      setStartDialogOpen(true);
                    }}
                  >
                    Start
                  </Button>
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Journey Details Dialog */}
      <Dialog
        open={!!selectedJourney && !startDialogOpen}
        onClose={() => setSelectedJourney(null)}
        maxWidth="md"
        fullWidth
      >
        {selectedJourney && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {categoryIcons[selectedJourney.category]}
                {selectedJourney.name}
              </Box>
            </DialogTitle>
            <DialogContent>
              <Typography variant="body1" paragraph>
                {selectedJourney.description}
              </Typography>
              
              <Typography variant="subtitle2" gutterBottom>Required Permissions</Typography>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 2 }}>
                {selectedJourney.required_permissions.map(perm => (
                  <Chip key={perm} label={perm} size="small" icon={<Security />} />
                ))}
              </Box>

              <Typography variant="subtitle2" gutterBottom>Journey Steps</Typography>
              <Stepper orientation="vertical">
                {selectedJourney.steps.map((step) => (
                  <Step key={step.id} active>
                    <StepLabel>
                      <Typography variant="subtitle2">{step.name}</Typography>
                    </StepLabel>
                    <StepContent>
                      <Typography variant="body2" color="text.secondary">
                        Type: {step.step_type.replace(/_/g, ' ')}
                      </Typography>
                      {step.endpoint && (
                        <Typography variant="caption" display="block">
                          Endpoint: {step.endpoint}
                        </Typography>
                      )}
                      {step.module && (
                        <Typography variant="caption" display="block">
                          Module: {step.module}
                        </Typography>
                      )}
                      {step.requires_approval && (
                        <Chip label="Requires Approval" size="small" color="warning" sx={{ mt: 1 }} />
                      )}
                    </StepContent>
                  </Step>
                ))}
              </Stepper>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setSelectedJourney(null)}>Close</Button>
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={() => setStartDialogOpen(true)}
              >
                Start Journey
              </Button>
            </DialogActions>
          </>
        )}
      </Dialog>

      {/* Start Journey Dialog */}
      <Dialog
        open={startDialogOpen}
        onClose={() => setStartDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Start Journey: {selectedJourney?.name}</DialogTitle>
        <DialogContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>
          )}
          <TextField
            fullWidth
            label="Project ID"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="Enter project ID"
            sx={{ mt: 2 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            This journey will execute {selectedJourney?.steps.length} steps and take approximately {selectedJourney?.estimated_duration_minutes} minutes.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStartDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleStartJourney}
            disabled={!projectId || starting}
            startIcon={starting ? <CircularProgress size={16} /> : <PlayArrow />}
          >
            {starting ? 'Starting...' : 'Start'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Mock data for demo when API is not available
const getMockJourneys = (): Journey[] => [
  {
    id: 'journey-001',
    name: 'Create Exploration Project',
    description: 'Create a new mineral exploration project and onboard team members',
    category: 'project_management',
    ui_entry_point: '/projects',
    required_permissions: ['projects:create', 'users:invite'],
    estimated_duration_minutes: 10,
    tags: ['project', 'onboarding', 'team'],
    steps: [
      { id: 'step-001-1', name: 'Create Project', step_type: 'api_call', endpoint: '/api/projects', timeout_seconds: 300, requires_approval: false },
      { id: 'step-001-2', name: 'Invite Team Members', step_type: 'api_call', endpoint: '/api/users/invite', timeout_seconds: 300, requires_approval: false },
      { id: 'step-001-3', name: 'Record Audit Trail', step_type: 'blockchain_record', endpoint: '/api/blockchain/record', timeout_seconds: 300, requires_approval: false },
    ],
  },
  {
    id: 'journey-002',
    name: 'Upload Drillhole Data',
    description: 'Upload and validate drillhole collar, survey, and assay data',
    category: 'data_ingestion',
    ui_entry_point: '/geology/drillholes',
    required_permissions: ['drillholes:create', 'upload:write'],
    estimated_duration_minutes: 15,
    tags: ['drillholes', 'upload', 'validation'],
    steps: [
      { id: 'step-002-1', name: 'Upload Collar File', step_type: 'data_ingestion', endpoint: '/api/upload', timeout_seconds: 300, requires_approval: false },
      { id: 'step-002-2', name: 'Validate Data Format', step_type: 'api_call', endpoint: '/api/drillholes/validate', timeout_seconds: 300, requires_approval: false },
      { id: 'step-002-3', name: 'Import Drillholes', step_type: 'api_call', endpoint: '/api/drillholes/import', timeout_seconds: 300, requires_approval: false },
      { id: 'step-002-4', name: 'Store to Lakehouse', step_type: 'api_call', endpoint: '/api/drillholes/persist', timeout_seconds: 300, requires_approval: false },
    ],
  },
  {
    id: 'journey-018',
    name: 'Gold Prospectivity Mapping',
    description: 'Generate gold prospectivity map using multi-modal ML features',
    category: 'ml_predictions',
    ui_entry_point: '/ai-insights',
    required_permissions: ['predictive_modeling:execute'],
    estimated_duration_minutes: 45,
    tags: ['gold', 'prospectivity', 'ml'],
    steps: [
      { id: 'step-018-1', name: 'Load Feature Data', step_type: 'api_call', endpoint: '/api/predictive-modeling/features', timeout_seconds: 300, requires_approval: false },
      { id: 'step-018-2', name: 'Run Gold Exploration Model', step_type: 'ml_inference', module: 'src.api.ml.gold_exploration', timeout_seconds: 1800, requires_approval: false, kafka_topic: 'mineralvision.ml.gold.inference.completed' },
      { id: 'step-018-3', name: 'Run Prospectivity Workflow', step_type: 'ml_inference', module: 'src.api.ml.prospectivity_workflow', timeout_seconds: 1800, requires_approval: false },
      { id: 'step-018-4', name: 'Generate Prospectivity Map', step_type: 'visualization', endpoint: '/api/visualization/prospectivity', timeout_seconds: 300, requires_approval: false },
    ],
  },
  {
    id: 'journey-023',
    name: 'Molmo2 Drone Video Analysis',
    description: 'Analyze drone video footage using Molmo2 ensemble pipeline',
    category: 'vision_ai',
    ui_entry_point: '/molmo2',
    required_permissions: ['molmo:execute'],
    estimated_duration_minutes: 20,
    tags: ['molmo2', 'drone', 'video', 'analysis'],
    steps: [
      { id: 'step-023-1', name: 'Upload Video', step_type: 'data_ingestion', endpoint: '/api/upload', timeout_seconds: 300, requires_approval: false },
      { id: 'step-023-2', name: 'Run Ensemble Pipeline', step_type: 'ml_inference', module: 'src.api.molmo.ensemble_pipeline', timeout_seconds: 1800, requires_approval: false, kafka_topic: 'mineralvision.molmo.ensemble.completed' },
      { id: 'step-023-3', name: 'Run Drone Video Analysis', step_type: 'ml_inference', module: 'src.api.molmo.drone_video_analysis', timeout_seconds: 1800, requires_approval: false },
      { id: 'step-023-4', name: 'Store Findings', step_type: 'api_call', endpoint: '/api/molmo/findings', timeout_seconds: 300, requires_approval: false },
    ],
  },
  {
    id: 'journey-027',
    name: 'Digital Twin Session',
    description: 'Start a digital twin session with real-time streaming and 3D visualization',
    category: 'digital_twin',
    ui_entry_point: '/visualization/3d',
    required_permissions: ['digital_twin:execute'],
    estimated_duration_minutes: 0,
    tags: ['digital_twin', '3d', 'realtime'],
    steps: [
      { id: 'step-027-1', name: 'Initialize Digital Twin', step_type: 'api_call', endpoint: '/api/digital-twin/initialize', timeout_seconds: 300, requires_approval: false, kafka_topic: 'mineralvision.digital_twin.session.started' },
      { id: 'step-027-2', name: 'Start Real-time Streaming', step_type: 'api_call', endpoint: '/api/digital-twin/stream/start', timeout_seconds: 300, requires_approval: false },
      { id: 'step-027-3', name: 'Load 3D Visualization', step_type: 'visualization', module: 'src.api.digital_twin.visualization_3d', timeout_seconds: 300, requires_approval: false },
    ],
  },
  {
    id: 'journey-030',
    name: 'Autonomous Exploration Recommendation',
    description: 'Generate autonomous exploration recommendations based on fused data',
    category: 'autonomous',
    ui_entry_point: '/ai-insights',
    required_permissions: ['autonomous_exploration:execute'],
    estimated_duration_minutes: 30,
    tags: ['autonomous', 'exploration', 'recommendation'],
    steps: [
      { id: 'step-030-1', name: 'Load Fused Data', step_type: 'api_call', endpoint: '/api/sensor-fusion/fused', timeout_seconds: 300, requires_approval: false },
      { id: 'step-030-2', name: 'Run Autonomous Exploration', step_type: 'api_call', endpoint: '/api/autonomous-exploration/recommend', timeout_seconds: 1800, requires_approval: false, kafka_topic: 'mineralvision.autonomous.recommendation.generated' },
      { id: 'step-030-3', name: 'Human Review', step_type: 'human_approval', timeout_seconds: 86400, requires_approval: true, permission_check: 'autonomous_exploration:approve' },
      { id: 'step-030-4', name: 'Generate Survey Plan', step_type: 'report_generation', endpoint: '/api/reports', timeout_seconds: 300, requires_approval: false },
      { id: 'step-030-5', name: 'Update Digital Twin', step_type: 'api_call', endpoint: '/api/digital-twin/plan', timeout_seconds: 300, requires_approval: false },
    ],
  },
];

export default JourneysPage;
