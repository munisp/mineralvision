import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Tabs,
  Tab,
  Button,
  Chip,
  LinearProgress,
  Alert,
  AlertTitle,
  IconButton,
  Tooltip,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Stepper,
  Step,
  StepLabel,
  Drawer,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  Badge,
} from '@mui/material';
import {
  Satellite as SatelliteIcon,
  Terrain as TerrainIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Map as MapIcon,
  Science as ScienceIcon,
  Notifications as NotificationsIcon,
  Layers as LayersIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Add as AddIcon,
  Visibility as VisibilityIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  CompareArrows as CompareIcon,
  CalendarMonth as CalendarIcon,
  Timeline as TimelineIcon,
  Close as CloseIcon,
  LocationOn as LocationIcon,
  Speed as SpeedIcon,
  Diamond as DiamondIcon,
  Bolt as BoltIcon,
  Analytics as AnalyticsIcon,
  Explore as ExploreIcon,
  Assessment as AssessmentIcon,
  BubbleChart as BubbleChartIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ScatterChart,
  Scatter,
} from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

// Mock data for mineral exploration sites
const mockSites = [
  { id: 'site_001', name: 'Kaduna Gold Prospect', mineral: 'gold', area: 450, prospectivity: 0.82, prospectivityChange: 0.05, status: 'high_potential', alerts: 1, discoveryDate: '2023-06-15', phase: 'Advanced Exploration', lastSurvey: '2024-01-15' },
  { id: 'site_002', name: 'Jos Lithium Zone', mineral: 'lithium', area: 320, prospectivity: 0.68, prospectivityChange: 0.02, status: 'moderate', alerts: 0, discoveryDate: '2023-09-20', phase: 'Resource Definition', lastSurvey: '2024-01-12' },
  { id: 'site_003', name: 'Nasarawa REE Deposit', mineral: 'rare_earth', area: 280, prospectivity: 0.55, prospectivityChange: -0.08, status: 'early_stage', alerts: 2, discoveryDate: '2024-01-05', phase: 'Initial Assessment', lastSurvey: '2024-01-18' },
  { id: 'site_004', name: 'Zamfara Gold Belt', mineral: 'gold', area: 680, prospectivity: 0.75, prospectivityChange: 0.03, status: 'high_potential', alerts: 0, discoveryDate: '2022-11-10', phase: 'Feasibility Study', lastSurvey: '2024-01-10' },
];

// Time series data for prospectivity scores
const prospectivityTimeSeries = [
  { date: 'Nov 20', kaduna: 0.77, jos: 0.66, nasarawa: 0.63, zamfara: 0.72 },
  { date: 'Nov 27', kaduna: 0.78, jos: 0.66, nasarawa: 0.61, zamfara: 0.73 },
  { date: 'Dec 04', kaduna: 0.79, jos: 0.67, nasarawa: 0.59, zamfara: 0.73 },
  { date: 'Dec 11', kaduna: 0.80, jos: 0.67, nasarawa: 0.57, zamfara: 0.74 },
  { date: 'Dec 18', kaduna: 0.80, jos: 0.68, nasarawa: 0.56, zamfara: 0.74 },
  { date: 'Dec 25', kaduna: 0.81, jos: 0.68, nasarawa: 0.55, zamfara: 0.75 },
  { date: 'Jan 01', kaduna: 0.81, jos: 0.68, nasarawa: 0.55, zamfara: 0.75 },
  { date: 'Jan 08', kaduna: 0.82, jos: 0.68, nasarawa: 0.55, zamfara: 0.75 },
  { date: 'Jan 15', kaduna: 0.82, jos: 0.68, nasarawa: 0.55, zamfara: 0.75 },
];

// Geochemical analysis data
const geochemicalData = [
  { element: 'Au (ppb)', kaduna: 850, jos: 120, nasarawa: 45, zamfara: 720, threshold: 100 },
  { element: 'Li (ppm)', kaduna: 45, jos: 1850, nasarawa: 320, zamfara: 38, threshold: 500 },
  { element: 'Ce (ppm)', kaduna: 85, jos: 120, nasarawa: 2450, zamfara: 92, threshold: 1000 },
  { element: 'Cu (ppm)', kaduna: 320, jos: 180, nasarawa: 95, zamfara: 280, threshold: 200 },
  { element: 'Ag (ppm)', kaduna: 12, jos: 3, nasarawa: 2, zamfara: 9, threshold: 5 },
];

// Geophysical survey data
const geophysicalData = [
  { survey: 'Magnetics', kaduna: 92, jos: 78, nasarawa: 65, zamfara: 88 },
  { survey: 'Gravity', kaduna: 85, jos: 82, nasarawa: 58, zamfara: 80 },
  { survey: 'EM', kaduna: 88, jos: 75, nasarawa: 62, zamfara: 85 },
  { survey: 'IP/Resistivity', kaduna: 90, jos: 70, nasarawa: 55, zamfara: 82 },
  { survey: 'Seismic', kaduna: 78, jos: 85, nasarawa: 48, zamfara: 75 },
];

// Drill results data
const drillResults = [
  { hole: 'KGP-001', site: 'Kaduna', depth: 245, grade: 4.2, width: 12.5, status: 'completed' },
  { hole: 'KGP-002', site: 'Kaduna', depth: 180, grade: 3.8, width: 8.2, status: 'completed' },
  { hole: 'JLZ-001', site: 'Jos', depth: 120, grade: 1.2, width: 25.0, status: 'completed' },
  { hole: 'JLZ-002', site: 'Jos', depth: 95, grade: 1.5, width: 18.5, status: 'in_progress' },
  { hole: 'ZGB-001', site: 'Zamfara', depth: 310, grade: 5.1, width: 15.2, status: 'completed' },
];

// Resource estimates
const resourceEstimates = [
  { site: 'Kaduna Gold', category: 'Indicated', tonnage: 2.4, grade: 3.8, contained: 293, unit: 'koz Au' },
  { site: 'Kaduna Gold', category: 'Inferred', tonnage: 1.8, grade: 3.2, contained: 185, unit: 'koz Au' },
  { site: 'Jos Lithium', category: 'Indicated', tonnage: 8.5, grade: 1.35, contained: 115, unit: 'kt Li2O' },
  { site: 'Jos Lithium', category: 'Inferred', tonnage: 5.2, grade: 1.20, contained: 62, unit: 'kt Li2O' },
  { site: 'Zamfara Gold', category: 'Indicated', tonnage: 3.8, grade: 4.5, contained: 550, unit: 'koz Au' },
];

// Alerts for mineral exploration
const mockAlerts = [
  { id: 'alert_001', type: 'geochemical', severity: 'high', site: 'Kaduna Gold Prospect', message: 'High-grade Au intercept in KGP-003: 8.5 g/t over 6.2m', time: '4 hours ago', rule: 'Grade Threshold Alert (>5 g/t)', recommendation: 'Prioritize infill drilling in this zone.' },
  { id: 'alert_002', type: 'geophysical', severity: 'medium', site: 'Nasarawa REE Deposit', message: 'Magnetic anomaly detected in NE sector', time: '1 day ago', rule: 'Anomaly Detection', recommendation: 'Schedule ground truthing survey.' },
  { id: 'alert_003', type: 'environmental', severity: 'low', site: 'Jos Lithium Zone', message: 'Water table monitoring shows seasonal variation', time: '2 days ago', rule: 'Environmental Monitoring', recommendation: 'Update baseline water study.' },
];

// Target generation data
const targetZones = [
  { zone: 'Priority 1', area: 45, confidence: 95, color: '#4caf50' },
  { zone: 'Priority 2', area: 82, confidence: 80, color: '#8bc34a' },
  { zone: 'Priority 3', area: 125, confidence: 65, color: '#ffeb3b' },
  { zone: 'Priority 4', area: 98, confidence: 50, color: '#ff9800' },
  { zone: 'Background', area: 380, confidence: 20, color: '#9e9e9e' },
];

// Anomaly scatter data
const anomalyData = [
  { x: 120, y: 850, z: 92, name: 'Target A', mineral: 'gold' },
  { x: 180, y: 720, z: 88, name: 'Target B', mineral: 'gold' },
  { x: 250, y: 1850, z: 78, name: 'Target C', mineral: 'lithium' },
  { x: 320, y: 2450, z: 65, name: 'Target D', mineral: 'rare_earth' },
  { x: 150, y: 550, z: 75, name: 'Target E', mineral: 'gold' },
];

const MineralMonitoringPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedSite, setSelectedSite] = useState('all');
  const [loading, setLoading] = useState(false);
  const [targetDialogOpen, setTargetDialogOpen] = useState(false);
  const [targetStep, setTargetStep] = useState(0);
  const [selectedMineral, setSelectedMineral] = useState('all');
  const [compareMode, setCompareMode] = useState(false);
  const [dateRange, setDateRange] = useState<number[]>([0, 90]);
  const [detailDrawerOpen, setDetailDrawerOpen] = useState(false);
  const [selectedSiteDetail, setSelectedSiteDetail] = useState<typeof mockSites[0] | null>(null);
  const [alertFilter, setAlertFilter] = useState('all');

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleRefresh = () => {
    setLoading(true);
    setTimeout(() => setLoading(false), 1500);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'high_potential': return 'success';
      case 'moderate': return 'warning';
      case 'early_stage': return 'info';
      default: return 'default';
    }
  };

  const getProspectivityColor = (score: number) => {
    if (score >= 0.7) return '#4caf50';
    if (score >= 0.5) return '#ff9800';
    return '#f44336';
  };

  const getMineralColor = (mineral: string) => {
    switch (mineral) {
      case 'gold': return '#ffd700';
      case 'lithium': return '#00bcd4';
      case 'rare_earth': return '#9c27b0';
      default: return '#757575';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const openSiteDetail = (site: typeof mockSites[0]) => {
    setSelectedSiteDetail(site);
    setDetailDrawerOpen(true);
  };

  const totalArea = mockSites.reduce((sum, s) => sum + s.area, 0);
  const avgProspectivity = mockSites.reduce((sum, s) => sum + s.prospectivity, 0) / mockSites.length;
  const totalAlerts = mockAlerts.length;
  const highPotentialSites = mockSites.filter(s => s.status === 'high_potential').length;

  return (
    <Box sx={{ p: 3, bgcolor: '#f5f7fa', minHeight: '100vh' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, fontWeight: 600 }}>
            <DiamondIcon fontSize="large" sx={{ color: '#ffd700' }} />
            Mineral Exploration Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            AI-powered prospectivity mapping, geochemical analysis, and target generation
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Mineral</InputLabel>
            <Select value={selectedMineral} onChange={(e) => setSelectedMineral(e.target.value)} label="Mineral">
              <MenuItem value="all">All Minerals</MenuItem>
              <MenuItem value="gold"><Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#ffd700' }} />Gold</Box></MenuItem>
              <MenuItem value="lithium"><Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#00bcd4' }} />Lithium</Box></MenuItem>
              <MenuItem value="rare_earth"><Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}><Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#9c27b0' }} />Rare Earth</Box></MenuItem>
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Site</InputLabel>
            <Select value={selectedSite} onChange={(e) => setSelectedSite(e.target.value)} label="Site">
              <MenuItem value="all">All Sites ({mockSites.length})</MenuItem>
              {mockSites.map(s => (
                <MenuItem key={s.id} value={s.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: getMineralColor(s.mineral) }} />
                    {s.name}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Tooltip title="Last updated: Jan 18, 2024">
            <IconButton onClick={handleRefresh} color="primary" sx={{ bgcolor: 'white' }}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2, borderRadius: 1 }} />}

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #ffd700 0%, #ff8f00 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(255, 215, 0, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" fontWeight="bold">{mockSites.length}</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Active Sites</Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7 }}>{totalArea.toFixed(0)} km² total</Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <ExploreIcon sx={{ fontSize: 32 }} />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(17, 153, 142, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" fontWeight="bold">{(avgProspectivity * 100).toFixed(0)}%</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Avg Prospectivity</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                    <TrendingUpIcon sx={{ fontSize: 16 }} />
                    <Typography variant="caption">+2.5% from last month</Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <AnalyticsIcon sx={{ fontSize: 32 }} />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(102, 126, 234, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" fontWeight="bold">{highPotentialSites}</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>High Potential</Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7 }}>Ready for drilling</Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <DiamondIcon sx={{ fontSize: 32 }} />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(240, 147, 251, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Badge badgeContent={1} color="error" sx={{ '& .MuiBadge-badge': { bgcolor: '#fff', color: '#f44336' } }}>
                    <Typography variant="h3" fontWeight="bold">{totalAlerts}</Typography>
                  </Badge>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Active Alerts</Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7 }}>1 high priority</Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <WarningIcon sx={{ fontSize: 32 }} />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Content Tabs */}
      <Card sx={{ borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider', '& .MuiTab-root': { minHeight: 64, textTransform: 'none', fontWeight: 500 } }}>
          <Tab icon={<SatelliteIcon />} label="Prospectivity" iconPosition="start" />
          <Tab icon={<ScienceIcon />} label="Geochemistry" iconPosition="start" />
          <Tab icon={<LayersIcon />} label="Geophysics" iconPosition="start" />
          <Tab icon={<AssessmentIcon />} label="Drilling & Resources" iconPosition="start" />
          <Tab icon={<NotificationsIcon />} label={<Badge badgeContent={totalAlerts} color="error" sx={{ '& .MuiBadge-badge': { right: -12 } }}>Alerts</Badge>} iconPosition="start" />
        </Tabs>

        {/* Prospectivity Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">AI Prospectivity Analysis</Typography>
              <Typography variant="body2" color="text.secondary">Machine learning-based mineral potential mapping</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button startIcon={<AddIcon />} variant="contained" onClick={() => setTargetDialogOpen(true)} sx={{ borderRadius: 2 }}>Generate Targets</Button>
              <Tooltip title="Compare sites">
                <ToggleButton value="compare" selected={compareMode} onChange={() => setCompareMode(!compareMode)} size="small">
                  <CompareIcon />
                </ToggleButton>
              </Tooltip>
            </Box>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" fontWeight="600">Prospectivity Map - All Sites</Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Chip icon={<CalendarIcon />} label="Jan 18, 2024" size="small" />
                      <Button size="small" startIcon={<DownloadIcon />}>Export</Button>
                    </Box>
                  </Box>
                  
                  <Box sx={{ height: 400, bgcolor: '#1a1a2e', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
                    <svg width="100%" height="100%" viewBox="0 0 800 400">
                      {/* Kaduna Gold - High Potential */}
                      <ellipse cx="180" cy="120" rx="80" ry="60" fill="url(#goldGradient)" stroke="#ffd700" strokeWidth="3" opacity="0.9" />
                      <text x="180" y="115" fill="white" fontSize="12" fontWeight="bold" textAnchor="middle">Kaduna Gold</text>
                      <text x="180" y="130" fill="white" fontSize="10" textAnchor="middle">82% Prospectivity</text>
                      <circle cx="220" cy="90" r="12" fill="rgba(76,175,80,0.8)" stroke="#4caf50" strokeWidth="2">
                        <animate attributeName="r" values="12;16;12" dur="2s" repeatCount="indefinite"/>
                      </circle>
                      <text x="220" y="94" fill="white" fontSize="9" textAnchor="middle">!</text>
                      
                      {/* Jos Lithium - Moderate */}
                      <ellipse cx="450" cy="100" rx="70" ry="50" fill="url(#lithiumGradient)" stroke="#00bcd4" strokeWidth="3" opacity="0.9" />
                      <text x="450" y="95" fill="white" fontSize="12" fontWeight="bold" textAnchor="middle">Jos Lithium</text>
                      <text x="450" y="110" fill="white" fontSize="10" textAnchor="middle">68% Prospectivity</text>
                      
                      {/* Nasarawa REE - Early Stage */}
                      <ellipse cx="650" cy="180" rx="65" ry="45" fill="url(#reeGradient)" stroke="#9c27b0" strokeWidth="3" opacity="0.9" />
                      <text x="650" y="175" fill="white" fontSize="12" fontWeight="bold" textAnchor="middle">Nasarawa REE</text>
                      <text x="650" y="190" fill="white" fontSize="10" textAnchor="middle">55% Prospectivity</text>
                      <circle cx="690" cy="150" r="12" fill="rgba(255,152,0,0.8)" stroke="#ff9800" strokeWidth="2">
                        <animate attributeName="r" values="12;16;12" dur="2s" repeatCount="indefinite"/>
                      </circle>
                      
                      {/* Zamfara Gold - High Potential */}
                      <ellipse cx="300" cy="280" rx="90" ry="65" fill="url(#goldGradient2)" stroke="#ffd700" strokeWidth="3" opacity="0.9" />
                      <text x="300" y="275" fill="white" fontSize="12" fontWeight="bold" textAnchor="middle">Zamfara Gold</text>
                      <text x="300" y="290" fill="white" fontSize="10" textAnchor="middle">75% Prospectivity</text>
                      
                      {/* Target zones */}
                      <circle cx="200" cy="140" r="25" fill="none" stroke="#4caf50" strokeWidth="2" strokeDasharray="5,5">
                        <animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>
                      </circle>
                      <circle cx="320" cy="260" r="30" fill="none" stroke="#4caf50" strokeWidth="2" strokeDasharray="5,5">
                        <animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>
                      </circle>
                      
                      <defs>
                        <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#ffd700"/><stop offset="100%" stopColor="#ff8f00"/>
                        </linearGradient>
                        <linearGradient id="goldGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#ffb300"/><stop offset="100%" stopColor="#ff6f00"/>
                        </linearGradient>
                        <linearGradient id="lithiumGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#00bcd4"/><stop offset="100%" stopColor="#0097a7"/>
                        </linearGradient>
                        <linearGradient id="reeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#9c27b0"/><stop offset="100%" stopColor="#7b1fa2"/>
                        </linearGradient>
                      </defs>
                    </svg>
                    
                    <Box sx={{ position: 'absolute', right: 16, top: 16, background: 'rgba(255,255,255,0.95)', p: 1.5, borderRadius: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
                      <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>Prospectivity Scale</Typography>
                      {[
                        { range: '80-100%', color: '#4caf50', label: 'Very High' },
                        { range: '60-80%', color: '#8bc34a', label: 'High' },
                        { range: '40-60%', color: '#ff9800', label: 'Moderate' },
                        { range: '20-40%', color: '#ff5722', label: 'Low' },
                        { range: '0-20%', color: '#9e9e9e', label: 'Background' },
                      ].map((item) => (
                        <Box key={item.range} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Box sx={{ width: 24, height: 12, bgcolor: item.color, borderRadius: 0.5 }} />
                          <Typography variant="caption">{item.range}</Typography>
                          <Typography variant="caption" color="text.secondary">({item.label})</Typography>
                        </Box>
                      ))}
                    </Box>
                    
                    <Box sx={{ position: 'absolute', left: 16, bottom: 16, display: 'flex', gap: 1 }}>
                      <Chip icon={<LayersIcon />} label="Layers" size="small" sx={{ bgcolor: 'white' }} />
                      <Chip icon={<LocationIcon />} label="Zoom to Fit" size="small" sx={{ bgcolor: 'white' }} />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight="600">Prospectivity Trend Analysis</Typography>
                      <Typography variant="caption" color="text.secondary">90-day prospectivity score history</Typography>
                    </Box>
                  </Box>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={prospectivityTimeSeries}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0.4, 0.9]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 12 }} />
                      <RechartsTooltip contentStyle={{ borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }} formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
                      <Legend />
                      <Line type="monotone" dataKey="kaduna" name="Kaduna Gold" stroke="#ffd700" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="jos" name="Jos Lithium" stroke="#00bcd4" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="nasarawa" name="Nasarawa REE" stroke="#9c27b0" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="zamfara" name="Zamfara Gold" stroke="#ff8f00" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                  
                  <Box sx={{ mt: 2, p: 2, bgcolor: '#e8f5e9', borderRadius: 2, display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                    <TrendingUpIcon color="success" />
                    <Box>
                      <Typography variant="body2" fontWeight="500">Trend Alert: Kaduna Gold Prospect</Typography>
                      <Typography variant="caption" color="text.secondary">Prospectivity increased from 77% to 82% (+5%) over 90 days. New drill results support resource upgrade.</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Target Zone Distribution</Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie data={targetZones} dataKey="area" nameKey="zone" cx="50%" cy="50%" outerRadius={80} label={({ zone, confidence }) => `${zone}: ${confidence}%`}>
                        {targetZones.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} />))}
                      </Pie>
                      <RechartsTooltip formatter={(value: number) => `${value} km²`} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Site Summary</Typography>
                  {mockSites.map((site) => (
                    <Box key={site.id} sx={{ mb: 2, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, cursor: 'pointer', transition: 'all 0.2s', '&:hover': { bgcolor: '#e0e0e0', transform: 'translateX(4px)' } }} onClick={() => openSiteDetail(site)}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: getMineralColor(site.mineral) }} />
                          <Typography variant="body2" fontWeight="500">{site.name}</Typography>
                          {site.alerts > 0 && <Badge badgeContent={site.alerts} color="error" sx={{ '& .MuiBadge-badge': { fontSize: 10 } }} />}
                        </Box>
                        <Chip size="small" label={site.status.replace('_', ' ')} color={getStatusColor(site.status) as 'success' | 'warning' | 'info' | 'default'} />
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress variant="determinate" value={site.prospectivity * 100} sx={{ flexGrow: 1, height: 10, borderRadius: 5, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { bgcolor: getProspectivityColor(site.prospectivity), borderRadius: 5 } }} />
                        <Typography variant="body2" fontWeight="600" sx={{ minWidth: 45 }}>{(site.prospectivity * 100).toFixed(0)}%</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', color: site.prospectivityChange >= 0 ? 'success.main' : 'error.main' }}>
                          {site.prospectivityChange >= 0 ? <TrendingUpIcon sx={{ fontSize: 16 }} /> : <TrendingDownIcon sx={{ fontSize: 16 }} />}
                          <Typography variant="caption">{site.prospectivityChange >= 0 ? '+' : ''}{(site.prospectivityChange * 100).toFixed(0)}%</Typography>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">{site.area} km²</Typography>
                        <Typography variant="caption" color="text.secondary">{site.mineral.replace('_', ' ')}</Typography>
                        <Typography variant="caption" color="text.secondary">{site.phase}</Typography>
                      </Box>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Geochemistry Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Geochemical Analysis</Typography>
              <Typography variant="body2" color="text.secondary">Multi-element assay results and anomaly detection</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button startIcon={<DownloadIcon />} variant="outlined" sx={{ borderRadius: 2 }}>Export Data</Button>
              <Button startIcon={<AddIcon />} variant="contained" sx={{ borderRadius: 2 }}>Add Samples</Button>
            </Box>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Element Concentrations by Site</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={geochemicalData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="element" type="category" width={80} tick={{ fontSize: 11 }} />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="kaduna" name="Kaduna" fill="#ffd700" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="jos" name="Jos" fill="#00bcd4" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="nasarawa" name="Nasarawa" fill="#9c27b0" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="zamfara" name="Zamfara" fill="#ff8f00" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Anomaly Detection - Multi-Element Plot</Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name="Easting" unit="m" />
                      <YAxis type="number" dataKey="y" name="Concentration" />
                      <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter name="Targets" data={anomalyData} fill="#4caf50">
                        {anomalyData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getMineralColor(entry.mineral)} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2, bgcolor: '#fff8e1' }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Anomaly Summary</Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">Gold Anomalies</Typography>
                    <Typography variant="h4" fontWeight="bold" sx={{ color: '#ffd700' }}>12</Typography>
                    <Typography variant="caption" color="text.secondary">&gt;100 ppb Au threshold</Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">Lithium Anomalies</Typography>
                    <Typography variant="h4" fontWeight="bold" sx={{ color: '#00bcd4' }}>5</Typography>
                    <Typography variant="caption" color="text.secondary">&gt;500 ppm Li threshold</Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="text.secondary">REE Anomalies</Typography>
                    <Typography variant="h4" fontWeight="bold" sx={{ color: '#9c27b0' }}>3</Typography>
                    <Typography variant="caption" color="text.secondary">&gt;1000 ppm TREE threshold</Typography>
                  </Box>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Sample Statistics</Typography>
                  <Grid container spacing={1}>
                    {[{ label: 'Total Samples', value: '2,450' }, { label: 'This Month', value: '185' }, { label: 'Pending Analysis', value: '42' }, { label: 'QA/QC Pass Rate', value: '98.5%' }].map((item) => (
                      <Grid item xs={6} key={item.label}>
                        <Box sx={{ p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, textAlign: 'center' }}>
                          <Typography variant="h5" fontWeight="bold">{item.value}</Typography>
                          <Typography variant="caption" color="text.secondary">{item.label}</Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Geophysics Tab */}
        <TabPanel value={tabValue} index={2}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Geophysical Surveys</Typography>
              <Typography variant="body2" color="text.secondary">Integrated geophysical data analysis and inversion</Typography>
            </Box>
            <Button startIcon={<AddIcon />} variant="contained" sx={{ borderRadius: 2 }}>New Survey</Button>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Survey Coverage by Site</Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={geophysicalData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="survey" tick={{ fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Radar name="Kaduna" dataKey="kaduna" stroke="#ffd700" fill="#ffd700" fillOpacity={0.3} />
                      <Radar name="Jos" dataKey="jos" stroke="#00bcd4" fill="#00bcd4" fillOpacity={0.3} />
                      <Radar name="Nasarawa" dataKey="nasarawa" stroke="#9c27b0" fill="#9c27b0" fillOpacity={0.3} />
                      <Radar name="Zamfara" dataKey="zamfara" stroke="#ff8f00" fill="#ff8f00" fillOpacity={0.3} />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Survey Status</Typography>
                  {[{ type: 'Airborne Magnetics', status: 'Completed', coverage: '100%', color: 'success' }, { type: 'Ground Gravity', status: 'In Progress', coverage: '75%', color: 'warning' }, { type: 'IP/Resistivity', status: 'Planned', coverage: '0%', color: 'info' }, { type: 'Drone EM', status: 'Completed', coverage: '100%', color: 'success' }].map((survey, idx) => (
                    <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1.5, borderBottom: '1px solid #eee' }}>
                      <Box>
                        <Typography variant="body2" fontWeight="500">{survey.type}</Typography>
                        <Typography variant="caption" color="text.secondary">{survey.coverage} coverage</Typography>
                      </Box>
                      <Chip size="small" label={survey.status} color={survey.color as 'success' | 'warning' | 'info'} />
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Drilling & Resources Tab */}
        <TabPanel value={tabValue} index={3}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Drilling & Resource Estimates</Typography>
              <Typography variant="body2" color="text.secondary">Drill results and mineral resource calculations</Typography>
            </Box>
            <Button startIcon={<AddIcon />} variant="contained" sx={{ borderRadius: 2 }}>Log Drill Hole</Button>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Recent Drill Results</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                          <TableCell sx={{ fontWeight: 600 }}>Hole ID</TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>Site</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Depth (m)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Grade</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Width (m)</TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {drillResults.map((hole) => (
                          <TableRow key={hole.hole} hover>
                            <TableCell><Typography variant="body2" fontWeight="500">{hole.hole}</Typography></TableCell>
                            <TableCell>{hole.site}</TableCell>
                            <TableCell align="right">{hole.depth}</TableCell>
                            <TableCell align="right"><Typography fontWeight="600" color={hole.grade > 3 ? 'success.main' : 'text.primary'}>{hole.grade} g/t</Typography></TableCell>
                            <TableCell align="right">{hole.width}</TableCell>
                            <TableCell><Chip size="small" label={hole.status.replace('_', ' ')} color={hole.status === 'completed' ? 'success' : 'warning'} /></TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Resource Estimates</Typography>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                          <TableCell sx={{ fontWeight: 600 }}>Site</TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>Category</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Tonnage (Mt)</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Grade</TableCell>
                          <TableCell align="right" sx={{ fontWeight: 600 }}>Contained</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {resourceEstimates.map((res, idx) => (
                          <TableRow key={idx} hover>
                            <TableCell><Typography variant="body2" fontWeight="500">{res.site}</Typography></TableCell>
                            <TableCell><Chip size="small" label={res.category} variant="outlined" /></TableCell>
                            <TableCell align="right">{res.tonnage}</TableCell>
                            <TableCell align="right">{res.grade}</TableCell>
                            <TableCell align="right"><Typography fontWeight="600">{res.contained} {res.unit}</Typography></TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2, bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Total Resources</Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">Gold (Indicated + Inferred)</Typography>
                    <Typography variant="h3" fontWeight="bold" sx={{ color: '#ffd700' }}>1.03 Moz</Typography>
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">Lithium (Indicated + Inferred)</Typography>
                    <Typography variant="h3" fontWeight="bold" sx={{ color: '#00bcd4' }}>177 kt Li2O</Typography>
                  </Box>
                  <Divider sx={{ my: 2 }} />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Box><Typography variant="caption" color="text.secondary">Drill Meters</Typography><Typography variant="h6" fontWeight="bold">12,450m</Typography></Box>
                    <Box><Typography variant="caption" color="text.secondary">Holes</Typography><Typography variant="h6" fontWeight="bold">85</Typography></Box>
                  </Box>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Drilling Progress</Typography>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">Kaduna Program</Typography>
                      <Typography variant="body2" fontWeight="600">75%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={75} sx={{ height: 8, borderRadius: 4, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { bgcolor: '#ffd700' } }} />
                  </Box>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">Jos Program</Typography>
                      <Typography variant="body2" fontWeight="600">45%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={45} sx={{ height: 8, borderRadius: 4, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { bgcolor: '#00bcd4' } }} />
                  </Box>
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="body2">Zamfara Program</Typography>
                      <Typography variant="body2" fontWeight="600">90%</Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={90} sx={{ height: 8, borderRadius: 4, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { bgcolor: '#ff8f00' } }} />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Alerts Tab */}
        <TabPanel value={tabValue} index={4}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Exploration Alerts</Typography>
              <Typography variant="body2" color="text.secondary">Monitor discoveries and anomalies</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <ToggleButtonGroup value={alertFilter} exclusive onChange={(_, v) => v && setAlertFilter(v)} size="small">
                <ToggleButton value="all">All ({mockAlerts.length})</ToggleButton>
                <ToggleButton value="active">Active</ToggleButton>
                <ToggleButton value="acknowledged">Acknowledged</ToggleButton>
              </ToggleButtonGroup>
              <Button variant="outlined" sx={{ borderRadius: 2 }}>Configure Rules</Button>
            </Box>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              {mockAlerts.map((alert) => (
                <Card key={alert.id} variant="outlined" sx={{ mb: 2, borderRadius: 2, borderLeft: `4px solid ${alert.severity === 'high' ? '#4caf50' : alert.severity === 'medium' ? '#ff9800' : '#2196f3'}` }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <Box sx={{ display: 'flex', gap: 2 }}>
                        <Avatar sx={{ bgcolor: alert.severity === 'high' ? '#e8f5e9' : alert.severity === 'medium' ? '#fff3e0' : '#e3f2fd', color: alert.severity === 'high' ? '#4caf50' : alert.severity === 'medium' ? '#ff9800' : '#2196f3' }}>
                          {alert.type === 'geochemical' ? <ScienceIcon /> : alert.type === 'geophysical' ? <LayersIcon /> : <TerrainIcon />}
                        </Avatar>
                        <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                            <Typography variant="subtitle1" fontWeight="600">{alert.site}</Typography>
                            <Chip size="small" label={alert.severity} color={alert.severity === 'high' ? 'success' : getSeverityColor(alert.severity) as 'warning' | 'info' | 'default'} />
                          </Box>
                          <Typography variant="body1">{alert.message}</Typography>
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>Rule: {alert.rule} - {alert.time}</Typography>
                          <Box sx={{ mt: 2, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 1 }}>
                            <Typography variant="caption" fontWeight="600" color="text.secondary">RECOMMENDATION</Typography>
                            <Typography variant="body2">{alert.recommendation}</Typography>
                          </Box>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button size="small" variant="outlined" sx={{ borderRadius: 1 }}>Acknowledge</Button>
                        <Button size="small" variant="contained" color="success" sx={{ borderRadius: 1 }}>Investigate</Button>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              ))}
            </Grid>

            <Grid item xs={12} md={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Alert Summary</Typography>
                  <Grid container spacing={1}>
                    {[{ label: 'Discovery', count: 1, color: '#4caf50' }, { label: 'Anomaly', count: 1, color: '#ff9800' }, { label: 'Environmental', count: 1, color: '#2196f3' }, { label: 'Safety', count: 0, color: '#f44336' }].map((item) => (
                      <Grid item xs={6} key={item.label}>
                        <Box sx={{ p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, textAlign: 'center', borderLeft: `3px solid ${item.color}` }}>
                          <Typography variant="h4" fontWeight="bold">{item.count}</Typography>
                          <Typography variant="caption" color="text.secondary">{item.label}</Typography>
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Active Alert Rules</Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>6 rules configured</Typography>
                  {[{ name: 'High-Grade Intercept', trigger: '>5 g/t Au over 3m', enabled: true }, { name: 'Geochemical Anomaly', trigger: '>3x background', enabled: true }, { name: 'Geophysical Anomaly', trigger: 'Conductivity >100 S/m', enabled: true }, { name: 'Resource Upgrade', trigger: 'Indicated conversion', enabled: true }, { name: 'Environmental Alert', trigger: 'Water quality change', enabled: false }].map((rule, idx) => (
                    <Box key={idx} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 1, borderBottom: '1px solid #eee' }}>
                      <Box>
                        <Typography variant="body2" fontWeight="500">{rule.name}</Typography>
                        <Typography variant="caption" color="text.secondary">{rule.trigger}</Typography>
                      </Box>
                      <Chip size="small" label={rule.enabled ? 'Active' : 'Disabled'} color={rule.enabled ? 'success' : 'default'} variant="outlined" />
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Site Detail Drawer */}
      <Drawer anchor="right" open={detailDrawerOpen} onClose={() => setDetailDrawerOpen(false)} PaperProps={{ sx: { width: 400, p: 3 } }}>
        {selectedSiteDetail && (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" fontWeight="600">Site Details</Typography>
              <IconButton onClick={() => setDetailDrawerOpen(false)}><CloseIcon /></IconButton>
            </Box>
            <Box sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: getMineralColor(selectedSiteDetail.mineral) }} />
                <Typography variant="h5" fontWeight="bold">{selectedSiteDetail.name}</Typography>
              </Box>
              <Chip size="small" label={selectedSiteDetail.status.replace('_', ' ')} color={getStatusColor(selectedSiteDetail.status) as 'success' | 'warning' | 'info' | 'default'} sx={{ mt: 1 }} />
            </Box>
            <Divider sx={{ mb: 3 }} />
            <List dense>
              <ListItem><ListItemIcon><MapIcon /></ListItemIcon><ListItemText primary="Area" secondary={`${selectedSiteDetail.area} km²`} /></ListItem>
              <ListItem><ListItemIcon><DiamondIcon /></ListItemIcon><ListItemText primary="Mineral Type" secondary={selectedSiteDetail.mineral.replace('_', ' ')} /></ListItem>
              <ListItem><ListItemIcon><AnalyticsIcon /></ListItemIcon><ListItemText primary="Prospectivity" secondary={`${(selectedSiteDetail.prospectivity * 100).toFixed(0)}%`} /></ListItem>
              <ListItem><ListItemIcon><TimelineIcon /></ListItemIcon><ListItemText primary="Change (90d)" secondary={`${selectedSiteDetail.prospectivityChange >= 0 ? '+' : ''}${(selectedSiteDetail.prospectivityChange * 100).toFixed(0)}%`} /></ListItem>
              <ListItem><ListItemIcon><CalendarIcon /></ListItemIcon><ListItemText primary="Discovery Date" secondary={selectedSiteDetail.discoveryDate} /></ListItem>
              <ListItem><ListItemIcon><SpeedIcon /></ListItemIcon><ListItemText primary="Phase" secondary={selectedSiteDetail.phase} /></ListItem>
              <ListItem><ListItemIcon><SatelliteIcon /></ListItemIcon><ListItemText primary="Last Survey" secondary={selectedSiteDetail.lastSurvey} /></ListItem>
            </List>
            <Divider sx={{ my: 3 }} />
            <Typography variant="subtitle2" fontWeight="600" gutterBottom>Quick Actions</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button variant="outlined" startIcon={<ScienceIcon />} fullWidth>View Geochemistry</Button>
              <Button variant="outlined" startIcon={<LayersIcon />} fullWidth>View Geophysics</Button>
              <Button variant="outlined" startIcon={<DownloadIcon />} fullWidth>Export Site Data</Button>
            </Box>
          </>
        )}
      </Drawer>

      {/* Target Generation Dialog */}
      <Dialog open={targetDialogOpen} onClose={() => { setTargetDialogOpen(false); setTargetStep(0); }} maxWidth="md" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" fontWeight="600">Generate Exploration Targets</Typography>
          <IconButton onClick={() => { setTargetDialogOpen(false); setTargetStep(0); }}><CloseIcon /></IconButton>
        </DialogTitle>
        <DialogContent>
          <Stepper activeStep={targetStep} sx={{ mb: 4, mt: 2 }}>
            <Step><StepLabel>Select Site</StepLabel></Step>
            <Step><StepLabel>Configure Model</StepLabel></Step>
            <Step><StepLabel>Set Parameters</StepLabel></Step>
            <Step><StepLabel>Generate & Review</StepLabel></Step>
          </Stepper>
          
          {targetStep === 0 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Select the site for target generation</Typography>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel>Site</InputLabel>
                <Select label="Site" defaultValue="">
                  {mockSites.map(s => (
                    <MenuItem key={s.id} value={s.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: getMineralColor(s.mineral) }} />
                        <Box>
                          <Typography variant="body2">{s.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{s.area} km² - {(s.prospectivity * 100).toFixed(0)}% prospectivity</Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          )}
          
          {targetStep === 1 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Select the ML model for target generation</Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                {[{ type: 'Random Forest', desc: 'Ensemble learning for multi-layer analysis', icon: <BubbleChartIcon /> }, { type: 'Neural Network', desc: 'Deep learning for complex patterns', icon: <AnalyticsIcon /> }, { type: 'Gradient Boosting', desc: 'High accuracy for structured data', icon: <TrendingUpIcon /> }, { type: 'Ensemble', desc: 'Combined model predictions', icon: <LayersIcon /> }].map((item) => (
                  <Grid item xs={6} key={item.type}>
                    <Card variant="outlined" sx={{ p: 2, cursor: 'pointer', '&:hover': { bgcolor: '#f5f5f5', borderColor: 'primary.main' }, transition: 'all 0.2s' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>{item.icon}<Typography variant="subtitle1" fontWeight="600">{item.type}</Typography></Box>
                      <Typography variant="body2" color="text.secondary">{item.desc}</Typography>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}
          
          {targetStep === 2 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Configure target generation parameters</Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}><TextField fullWidth label="Minimum Confidence (%)" type="number" defaultValue="60" /></Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel>Target Priority</InputLabel>
                    <Select label="Target Priority" defaultValue={3}>
                      <MenuItem value={1}>Priority 1 Only</MenuItem>
                      <MenuItem value={2}>Priority 1-2</MenuItem>
                      <MenuItem value={3}>Priority 1-3</MenuItem>
                      <MenuItem value={4}>All Priorities</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}><TextField fullWidth label="Min Area (km²)" type="number" defaultValue="5" /></Grid>
                <Grid item xs={6}><TextField fullWidth label="Max Targets" type="number" defaultValue="20" /></Grid>
              </Grid>
            </Box>
          )}
          
          {targetStep === 3 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Review and generate targets</Typography>
              <Box sx={{ height: 200, bgcolor: '#f5f5f5', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2 }}>
                <Typography color="text.secondary">Target Preview Map</Typography>
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Export Format</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip label="GeoJSON" variant="outlined" />
                  <Chip label="Shapefile" variant="outlined" />
                  <Chip label="KML" variant="outlined" />
                  <Chip label="CSV" variant="outlined" />
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => { setTargetDialogOpen(false); setTargetStep(0); }}>Cancel</Button>
          {targetStep > 0 && <Button onClick={() => setTargetStep(targetStep - 1)}>Back</Button>}
          {targetStep < 3 ? (
            <Button variant="contained" onClick={() => setTargetStep(targetStep + 1)}>Next</Button>
          ) : (
            <Button variant="contained" color="success" onClick={() => { setTargetDialogOpen(false); setTargetStep(0); }}>Generate Targets</Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default MineralMonitoringPage;
