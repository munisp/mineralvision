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
  Grass as GrassIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Map as MapIcon,
  Agriculture as AgricultureIcon,
  Notifications as NotificationsIcon,
  CloudQueue as CloudIcon,
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
  Layers as LayersIcon,
  Speed as SpeedIcon,
  WbSunny as SunIcon,
  Opacity as OpacityIcon,
  Air as WindIcon,
  FilterDrama as CloudyIcon,
  Thunderstorm as StormIcon,
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

const mockFields = [
  { id: 'field_001', name: 'Block A - Oil Palm', crop: 'oil_palm', area: 25.5, ndvi: 0.72, ndviChange: 0.03, status: 'healthy', alerts: 0, plantingDate: '2020-03-15', growthStage: 'Mature', lastImagery: '2024-01-18' },
  { id: 'field_002', name: 'Block B - Cocoa', crop: 'cocoa', area: 18.3, ndvi: 0.58, ndviChange: -0.05, status: 'moderate', alerts: 1, plantingDate: '2021-06-20', growthStage: 'Productive', lastImagery: '2024-01-18' },
  { id: 'field_003', name: 'Block C - Ginger', crop: 'ginger', area: 12.8, ndvi: 0.45, ndviChange: -0.15, status: 'stressed', alerts: 2, plantingDate: '2023-09-01', growthStage: 'Vegetative', lastImagery: '2024-01-18' },
  { id: 'field_004', name: 'Block D - Oil Palm', crop: 'oil_palm', area: 30.2, ndvi: 0.68, ndviChange: 0.01, status: 'healthy', alerts: 0, plantingDate: '2019-11-10', growthStage: 'Mature', lastImagery: '2024-01-18' },
];

const ndviTimeSeries = [
  { date: 'Dec 20', blockA: 0.69, blockB: 0.63, blockC: 0.60, blockD: 0.67 },
  { date: 'Dec 23', blockA: 0.70, blockB: 0.62, blockC: 0.58, blockD: 0.67 },
  { date: 'Dec 26', blockA: 0.71, blockB: 0.61, blockC: 0.55, blockD: 0.68 },
  { date: 'Dec 29', blockA: 0.71, blockB: 0.60, blockC: 0.52, blockD: 0.68 },
  { date: 'Jan 01', blockA: 0.72, blockB: 0.59, blockC: 0.49, blockD: 0.68 },
  { date: 'Jan 04', blockA: 0.72, blockB: 0.58, blockC: 0.47, blockD: 0.68 },
  { date: 'Jan 07', blockA: 0.72, blockB: 0.58, blockC: 0.46, blockD: 0.68 },
  { date: 'Jan 10', blockA: 0.72, blockB: 0.58, blockC: 0.45, blockD: 0.68 },
  { date: 'Jan 13', blockA: 0.72, blockB: 0.58, blockC: 0.45, blockD: 0.68 },
  { date: 'Jan 18', blockA: 0.72, blockB: 0.58, blockC: 0.45, blockD: 0.68 },
];

const mockWeatherForecast = [
  { date: 'Today', temp: 32, tempMin: 24, rain: 0, humidity: 75, wind: 8, icon: 'sunny' },
  { date: 'Tomorrow', temp: 30, tempMin: 23, rain: 15, humidity: 82, wind: 12, icon: 'rain' },
  { date: 'Wed', temp: 28, tempMin: 22, rain: 45, humidity: 90, wind: 18, icon: 'storm' },
  { date: 'Thu', temp: 29, tempMin: 23, rain: 20, humidity: 85, wind: 10, icon: 'cloudy' },
  { date: 'Fri', temp: 31, tempMin: 24, rain: 5, humidity: 78, wind: 8, icon: 'partly' },
  { date: 'Sat', temp: 33, tempMin: 25, rain: 0, humidity: 70, wind: 6, icon: 'sunny' },
  { date: 'Sun', temp: 32, tempMin: 24, rain: 10, humidity: 76, wind: 9, icon: 'rain' },
];

const weatherChartData = [
  { day: 'Mon', temp: 31, rain: 5, et0: 5.2 },
  { day: 'Tue', temp: 32, rain: 0, et0: 5.8 },
  { day: 'Wed', temp: 30, rain: 25, et0: 4.2 },
  { day: 'Thu', temp: 28, rain: 45, et0: 3.5 },
  { day: 'Fri', temp: 29, rain: 15, et0: 4.0 },
  { day: 'Sat', temp: 31, rain: 5, et0: 5.0 },
  { day: 'Sun', temp: 32, rain: 0, et0: 5.5 },
];

const waterBalanceData = [
  { day: 'Mon', precipitation: 5, et0: 5.2, cumulative: -0.2 },
  { day: 'Tue', precipitation: 5, et0: 11.0, cumulative: -6.0 },
  { day: 'Wed', precipitation: 30, et0: 14.5, cumulative: 9.5 },
  { day: 'Thu', precipitation: 75, et0: 18.0, cumulative: 51.0 },
  { day: 'Fri', precipitation: 90, et0: 22.0, cumulative: 62.0 },
  { day: 'Sat', precipitation: 95, et0: 27.0, cumulative: 63.0 },
  { day: 'Sun', precipitation: 95, et0: 32.5, cumulative: 57.5 },
];

const mockAlerts = [
  { id: 'alert_001', type: 'vegetation', severity: 'high', field: 'Block C - Ginger', message: 'NDVI dropped 15% in 7 days', time: '2 hours ago', rule: 'Rapid NDVI Decline (>10% in 7 days)', recommendation: 'Inspect field for pest damage or water stress.' },
  { id: 'alert_002', type: 'weather', severity: 'medium', field: 'All Fields', message: 'Heavy rain expected Wednesday', time: '5 hours ago', rule: 'Heavy Rain Warning (>40mm)', recommendation: 'Delay fertilizer application.' },
  { id: 'alert_003', type: 'vegetation', severity: 'medium', field: 'Block B - Cocoa', message: 'Moderate stress detected in NE corner', time: '1 day ago', rule: 'Zone Stress Detection', recommendation: 'Check irrigation coverage.' },
];

const mockVRAMaps = [
  { id: 'vra_001', type: 'Nitrogen', field: 'Block A - Oil Palm', created: '2024-01-15', savings: '12%', savingsKg: 245, savingsCost: 1225, zones: 5 },
  { id: 'vra_002', type: 'Sowing', field: 'Block C - Ginger', created: '2024-01-10', savings: '8%', savingsKg: 180, savingsCost: 540, zones: 3 },
  { id: 'vra_003', type: 'P&K', field: 'Block B - Cocoa', created: '2024-01-05', savings: '15%', savingsKg: 320, savingsCost: 1920, zones: 5 },
];

const vraZoneDistribution = [
  { zone: 'Very Low', area: 8.5, rate: 60, color: '#ef5350' },
  { zone: 'Low', area: 15.2, rate: 90, color: '#ff9800' },
  { zone: 'Medium', area: 28.4, rate: 120, color: '#ffeb3b' },
  { zone: 'High', area: 22.1, rate: 150, color: '#8bc34a' },
  { zone: 'Very High', area: 12.6, rate: 180, color: '#4caf50' },
];

const fieldHealthRadar = [
  { metric: 'NDVI', blockA: 90, blockB: 72, blockC: 56, blockD: 85 },
  { metric: 'Moisture', blockA: 85, blockB: 78, blockC: 45, blockD: 82 },
  { metric: 'Canopy', blockA: 88, blockB: 70, blockC: 52, blockD: 84 },
  { metric: 'Vigor', blockA: 92, blockB: 68, blockC: 48, blockD: 86 },
  { metric: 'Stress', blockA: 95, blockB: 65, blockC: 40, blockD: 88 },
];

const CropMonitoringPage: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [selectedField, setSelectedField] = useState('all');
  const [loading, setLoading] = useState(false);
  const [vraDialogOpen, setVraDialogOpen] = useState(false);
  const [vraStep, setVraStep] = useState(0);
  const [selectedIndex, setSelectedIndex] = useState('NDVI');
  const [compareMode, setCompareMode] = useState(false);
  const [dateRange, setDateRange] = useState<number[]>([0, 30]);
  const [detailDrawerOpen, setDetailDrawerOpen] = useState(false);
  const [selectedFieldDetail, setSelectedFieldDetail] = useState<typeof mockFields[0] | null>(null);
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
      case 'healthy': return 'success';
      case 'moderate': return 'warning';
      case 'stressed': return 'error';
      default: return 'default';
    }
  };

  const getNDVIColor = (ndvi: number) => {
    if (ndvi >= 0.6) return '#4caf50';
    if (ndvi >= 0.4) return '#ff9800';
    return '#f44336';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getWeatherIcon = (icon: string) => {
    switch (icon) {
      case 'sunny': return <SunIcon sx={{ fontSize: 40, color: '#ff9800' }} />;
      case 'rain': return <OpacityIcon sx={{ fontSize: 40, color: '#2196f3' }} />;
      case 'storm': return <StormIcon sx={{ fontSize: 40, color: '#5c6bc0' }} />;
      case 'cloudy': return <CloudyIcon sx={{ fontSize: 40, color: '#78909c' }} />;
      default: return <CloudIcon sx={{ fontSize: 40, color: '#90a4ae' }} />;
    }
  };

  const openFieldDetail = (field: typeof mockFields[0]) => {
    setSelectedFieldDetail(field);
    setDetailDrawerOpen(true);
  };

  const totalArea = mockFields.reduce((sum, f) => sum + f.area, 0);
  const avgNDVI = mockFields.reduce((sum, f) => sum + f.ndvi, 0) / mockFields.length;
  const totalAlerts = mockAlerts.length;

  return (
    <Box sx={{ p: 3, bgcolor: '#f5f7fa', minHeight: '100vh' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1, fontWeight: 600 }}>
            <AgricultureIcon fontSize="large" sx={{ color: '#2e7d32' }} />
            Crop Monitoring Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            EOS-style vegetation monitoring, weather integration, and VRA mapping
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Field</InputLabel>
            <Select value={selectedField} onChange={(e) => setSelectedField(e.target.value)} label="Field">
              <MenuItem value="all">All Fields ({mockFields.length})</MenuItem>
              {mockFields.map(f => (
                <MenuItem key={f.id} value={f.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: getNDVIColor(f.ndvi) }} />
                    {f.name}
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

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(102, 126, 234, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" fontWeight="bold">{mockFields.length}</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Total Fields</Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7 }}>{totalArea.toFixed(1)} ha total</Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <MapIcon sx={{ fontSize: 32 }} />
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
                  <Typography variant="h3" fontWeight="bold">{avgNDVI.toFixed(2)}</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Avg NDVI</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                    <TrendingDownIcon sx={{ fontSize: 16 }} />
                    <Typography variant="caption">-0.02 from last week</Typography>
                  </Box>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <GrassIcon sx={{ fontSize: 32 }} />
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
        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white', borderRadius: 3, boxShadow: '0 8px 32px rgba(79, 172, 254, 0.3)', transition: 'transform 0.2s', '&:hover': { transform: 'translateY(-4px)' } }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Box>
                  <Typography variant="h3" fontWeight="bold">30C</Typography>
                  <Typography variant="body1" sx={{ opacity: 0.9 }}>Current Temp</Typography>
                  <Typography variant="caption" sx={{ opacity: 0.7 }}>Humidity: 75%</Typography>
                </Box>
                <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)', width: 56, height: 56 }}>
                  <SunIcon sx={{ fontSize: 32 }} />
                </Avatar>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card sx={{ borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.08)' }}>
        <Tabs value={tabValue} onChange={handleTabChange} sx={{ borderBottom: 1, borderColor: 'divider', '& .MuiTab-root': { minHeight: 64, textTransform: 'none', fontWeight: 500 } }}>
          <Tab icon={<SatelliteIcon />} label="Vegetation Indices" iconPosition="start" />
          <Tab icon={<MapIcon />} label="Field Management" iconPosition="start" />
          <Tab icon={<CloudIcon />} label="Weather" iconPosition="start" />
          <Tab icon={<AgricultureIcon />} label="VRA Maps" iconPosition="start" />
          <Tab icon={<NotificationsIcon />} label={<Badge badgeContent={totalAlerts} color="error" sx={{ '& .MuiBadge-badge': { right: -12 } }}>Alerts</Badge>} iconPosition="start" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Vegetation Health Analysis</Typography>
              <Typography variant="body2" color="text.secondary">Monitor crop health using satellite-derived indices</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <ToggleButtonGroup value={selectedIndex} exclusive onChange={(_, v) => v && setSelectedIndex(v)} size="small">
                {['NDVI', 'NDRE', 'SAVI', 'EVI'].map((idx) => (
                  <ToggleButton key={idx} value={idx} sx={{ px: 2 }}>{idx}</ToggleButton>
                ))}
              </ToggleButtonGroup>
              <Tooltip title="Compare two dates">
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
                    <Typography variant="subtitle1" fontWeight="600">{selectedIndex} Map - All Fields</Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Chip icon={<CalendarIcon />} label="Jan 18, 2024" size="small" />
                      <Button size="small" startIcon={<DownloadIcon />}>Export</Button>
                    </Box>
                  </Box>
                  
                  <Box sx={{ height: 400, bgcolor: '#1a1a2e', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
                    <svg width="100%" height="100%" viewBox="0 0 800 400">
                      <polygon points="50,50 200,30 220,150 180,200 40,180" fill="url(#greenGradient)" stroke="#fff" strokeWidth="2" opacity="0.9" />
                      <text x="110" y="110" fill="white" fontSize="12" fontWeight="bold">Block A</text>
                      <text x="110" y="125" fill="white" fontSize="10">NDVI: 0.72</text>
                      
                      <polygon points="240,40 400,50 420,180 380,220 230,200 220,150" fill="url(#yellowGradient)" stroke="#fff" strokeWidth="2" opacity="0.9" />
                      <text x="310" y="120" fill="white" fontSize="12" fontWeight="bold">Block B</text>
                      <text x="310" y="135" fill="white" fontSize="10">NDVI: 0.58</text>
                      
                      <polygon points="450,60 580,40 620,160 590,220 440,200" fill="url(#redGradient)" stroke="#fff" strokeWidth="2" opacity="0.9" />
                      <text x="520" y="120" fill="white" fontSize="12" fontWeight="bold">Block C</text>
                      <text x="520" y="135" fill="white" fontSize="10">NDVI: 0.45</text>
                      <circle cx="560" cy="100" r="15" fill="rgba(244,67,54,0.5)" stroke="#f44336" strokeWidth="2">
                        <animate attributeName="r" values="15;20;15" dur="2s" repeatCount="indefinite"/>
                      </circle>
                      <text x="560" y="105" fill="white" fontSize="10" textAnchor="middle">!</text>
                      
                      <polygon points="640,50 780,70 760,200 700,240 620,220 620,160" fill="url(#greenGradient2)" stroke="#fff" strokeWidth="2" opacity="0.9" />
                      <text x="700" y="130" fill="white" fontSize="12" fontWeight="bold">Block D</text>
                      <text x="700" y="145" fill="white" fontSize="10">NDVI: 0.68</text>
                      
                      <ellipse cx="540" cy="150" rx="40" ry="30" fill="none" stroke="#f44336" strokeWidth="2" strokeDasharray="5,5">
                        <animate attributeName="stroke-opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>
                      </ellipse>
                      
                      <defs>
                        <linearGradient id="greenGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#4caf50"/><stop offset="100%" stopColor="#2e7d32"/>
                        </linearGradient>
                        <linearGradient id="greenGradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#66bb6a"/><stop offset="100%" stopColor="#388e3c"/>
                        </linearGradient>
                        <linearGradient id="yellowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#ff9800"/><stop offset="100%" stopColor="#f57c00"/>
                        </linearGradient>
                        <linearGradient id="redGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stopColor="#f44336"/><stop offset="100%" stopColor="#c62828"/>
                        </linearGradient>
                      </defs>
                    </svg>
                    
                    <Box sx={{ position: 'absolute', right: 16, top: 16, background: 'rgba(255,255,255,0.95)', p: 1.5, borderRadius: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
                      <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>{selectedIndex} Scale</Typography>
                      {[
                        { range: '0.8-1.0', color: '#1b5e20', label: 'Excellent' },
                        { range: '0.6-0.8', color: '#4caf50', label: 'Healthy' },
                        { range: '0.4-0.6', color: '#ff9800', label: 'Moderate' },
                        { range: '0.2-0.4', color: '#ff5722', label: 'Stressed' },
                        { range: '0.0-0.2', color: '#f44336', label: 'Critical' },
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
                  
                  <Box sx={{ mt: 2, px: 2 }}>
                    <Typography variant="caption" color="text.secondary">Imagery Date Range</Typography>
                    <Slider value={dateRange} onChange={(_, v) => setDateRange(v as number[])} valueLabelDisplay="auto" valueLabelFormat={(v) => `${30 - v} days ago`} min={0} max={30} marks={[{ value: 0, label: 'Today' }, { value: 15, label: '15 days' }, { value: 30, label: '30 days' }]} />
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box>
                      <Typography variant="subtitle1" fontWeight="600">{selectedIndex} Trend Analysis</Typography>
                      <Typography variant="caption" color="text.secondary">30-day vegetation index history</Typography>
                    </Box>
                  </Box>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={ndviTimeSeries}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                      <YAxis domain={[0.3, 0.8]} tick={{ fontSize: 12 }} />
                      <RechartsTooltip contentStyle={{ borderRadius: 8, boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }} />
                      <Legend />
                      <Line type="monotone" dataKey="blockA" name="Block A" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="blockB" name="Block B" stroke="#ff9800" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="blockC" name="Block C" stroke="#f44336" strokeWidth={2} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="blockD" name="Block D" stroke="#2196f3" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                  
                  <Box sx={{ mt: 2, p: 2, bgcolor: '#fff3e0', borderRadius: 2, display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                    <TrendingDownIcon color="warning" />
                    <Box>
                      <Typography variant="body2" fontWeight="500">Trend Alert: Block C - Ginger</Typography>
                      <Typography variant="caption" color="text.secondary">NDVI declined from 0.60 to 0.45 (-25%) over 30 days. Recommend field inspection.</Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Field Health Comparison</Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <RadarChart data={fieldHealthRadar}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Radar name="Block A" dataKey="blockA" stroke="#4caf50" fill="#4caf50" fillOpacity={0.3} />
                      <Radar name="Block C" dataKey="blockC" stroke="#f44336" fill="#f44336" fillOpacity={0.3} />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Field Health Summary</Typography>
                  {mockFields.map((field) => (
                    <Box key={field.id} sx={{ mb: 2, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, cursor: 'pointer', transition: 'all 0.2s', '&:hover': { bgcolor: '#e0e0e0', transform: 'translateX(4px)' } }} onClick={() => openFieldDetail(field)}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" fontWeight="500">{field.name}</Typography>
                          {field.alerts > 0 && <Badge badgeContent={field.alerts} color="error" sx={{ '& .MuiBadge-badge': { fontSize: 10 } }} />}
                        </Box>
                        <Chip size="small" label={field.status} color={getStatusColor(field.status) as 'success' | 'warning' | 'error' | 'default'} />
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress variant="determinate" value={field.ndvi * 100} sx={{ flexGrow: 1, height: 10, borderRadius: 5, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { bgcolor: getNDVIColor(field.ndvi), borderRadius: 5 } }} />
                        <Typography variant="body2" fontWeight="600" sx={{ minWidth: 45 }}>{field.ndvi.toFixed(2)}</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', color: field.ndviChange >= 0 ? 'success.main' : 'error.main' }}>
                          {field.ndviChange >= 0 ? <TrendingUpIcon sx={{ fontSize: 16 }} /> : <TrendingDownIcon sx={{ fontSize: 16 }} />}
                          <Typography variant="caption">{field.ndviChange >= 0 ? '+' : ''}{field.ndviChange.toFixed(2)}</Typography>
                        </Box>
                      </Box>
                      <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">{field.area} ha</Typography>
                        <Typography variant="caption" color="text.secondary">{field.crop.replace('_', ' ')}</Typography>
                        <Typography variant="caption" color="text.secondary">{field.growthStage}</Typography>
                      </Box>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Field Inventory</Typography>
              <Typography variant="body2" color="text.secondary">Manage field boundaries and metadata</Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button startIcon={<AddIcon />} variant="contained" sx={{ borderRadius: 2 }}>Add Field</Button>
              <Button startIcon={<DownloadIcon />} variant="outlined" sx={{ borderRadius: 2 }}>Export GeoJSON</Button>
            </Box>
          </Box>
          
          <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
            <Table>
              <TableHead>
                <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                  <TableCell sx={{ fontWeight: 600 }}>Field Name</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Crop Type</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600 }}>Area (ha)</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600 }}>NDVI</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Status</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Growth Stage</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>Alerts</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {mockFields.map((field) => (
                  <TableRow key={field.id} hover sx={{ cursor: 'pointer' }} onClick={() => openFieldDetail(field)}>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: getNDVIColor(field.ndvi) }} />
                        <Typography variant="body2" fontWeight="500">{field.name}</Typography>
                      </Box>
                    </TableCell>
                    <TableCell><Chip size="small" label={field.crop.replace('_', ' ')} variant="outlined" /></TableCell>
                    <TableCell align="right">{field.area.toFixed(1)}</TableCell>
                    <TableCell align="right">
                      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                        {field.ndvi.toFixed(2)}
                        <Box sx={{ display: 'flex', alignItems: 'center', color: field.ndviChange >= 0 ? 'success.main' : 'error.main' }}>
                          {field.ndviChange >= 0 ? <TrendingUpIcon sx={{ fontSize: 14 }} /> : <TrendingDownIcon sx={{ fontSize: 14 }} />}
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell><Chip size="small" label={field.status} color={getStatusColor(field.status) as 'success' | 'warning' | 'error' | 'default'} /></TableCell>
                    <TableCell>{field.growthStage}</TableCell>
                    <TableCell align="center">{field.alerts > 0 ? <Chip size="small" label={field.alerts} color="error" /> : <CheckCircleIcon color="success" fontSize="small" />}</TableCell>
                    <TableCell align="center">
                      <Tooltip title="View Details">
                        <IconButton size="small" onClick={(e) => { e.stopPropagation(); openFieldDetail(field); }}><VisibilityIcon /></IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid item xs={12} md={4}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">Total Area</Typography>
                  <Typography variant="h4" fontWeight="bold">{totalArea.toFixed(1)} ha</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">Crop Distribution</Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                    <Chip label="Oil Palm: 55.7 ha" size="small" />
                    <Chip label="Cocoa: 18.3 ha" size="small" />
                    <Chip label="Ginger: 12.8 ha" size="small" />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={4}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle2" color="text.secondary">Last Imagery Update</Typography>
                  <Typography variant="h6">Jan 18, 2024</Typography>
                  <Typography variant="caption" color="text.secondary">Next: Jan 23, 2024</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Weather Intelligence</Typography>
              <Typography variant="body2" color="text.secondary">14-day forecast and water balance</Typography>
            </Box>
            <Chip icon={<LocationIcon />} label="Farm: 6.5N, 3.4E" />
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>7-Day Forecast</Typography>
                  <Grid container spacing={1}>
                    {mockWeatherForecast.map((day, index) => (
                      <Grid item xs={12/7} key={index}>
                        <Card variant="outlined" sx={{ textAlign: 'center', p: 1.5, borderRadius: 2, bgcolor: index === 0 ? '#e3f2fd' : 'transparent', border: index === 0 ? '2px solid #2196f3' : undefined }}>
                          <Typography variant="subtitle2" color={index === 0 ? 'primary' : 'text.secondary'} fontWeight={index === 0 ? 600 : 400}>{day.date}</Typography>
                          <Box sx={{ my: 1 }}>{getWeatherIcon(day.icon)}</Box>
                          <Typography variant="h6" fontWeight="bold">{day.temp}C</Typography>
                          <Typography variant="caption" color="text.secondary" display="block">{day.tempMin}C min</Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5, mt: 1 }}>
                            <OpacityIcon sx={{ fontSize: 14, color: '#2196f3' }} />
                            <Typography variant="caption" fontWeight={day.rain > 20 ? 600 : 400} color={day.rain > 20 ? 'primary' : 'text.secondary'}>{day.rain}mm</Typography>
                          </Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                            <WindIcon sx={{ fontSize: 14, color: '#78909c' }} />
                            <Typography variant="caption" color="text.secondary">{day.wind} km/h</Typography>
                          </Box>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Temperature & Precipitation</Typography>
                  <ResponsiveContainer width="100%" height={250}>
                    <ComposedChart data={weatherChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                      <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
                      <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
                      <RechartsTooltip contentStyle={{ borderRadius: 8 }} />
                      <Legend />
                      <Bar yAxisId="right" dataKey="rain" name="Rainfall (mm)" fill="#2196f3" radius={[4, 4, 0, 0]} />
                      <Line yAxisId="left" type="monotone" dataKey="temp" name="Temperature (C)" stroke="#ff5722" strokeWidth={2} dot={{ r: 4 }} />
                      <Line yAxisId="left" type="monotone" dataKey="et0" name="ET0 (mm)" stroke="#4caf50" strokeWidth={2} dot={{ r: 4 }} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Cumulative Water Balance</Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={waterBalanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} />
                      <RechartsTooltip contentStyle={{ borderRadius: 8 }} />
                      <Legend />
                      <Area type="monotone" dataKey="precipitation" name="Cumulative Rain" stackId="1" stroke="#2196f3" fill="#bbdefb" />
                      <Area type="monotone" dataKey="et0" name="Cumulative ET0" stackId="2" stroke="#ff9800" fill="#ffe0b2" />
                      <Line type="monotone" dataKey="cumulative" name="Net Balance" stroke="#4caf50" strokeWidth={3} dot={{ r: 4 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Water Balance Summary</Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" color="text.secondary">Precipitation (7 days)</Typography>
                    <Typography variant="h4" fontWeight="bold" color="primary">95 mm</Typography>
                  </Box>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" color="text.secondary">Evapotranspiration (ET0)</Typography>
                    <Typography variant="h4" fontWeight="bold" color="warning.main">42 mm</Typography>
                  </Box>
                  <Divider sx={{ my: 2 }} />
                  <Box>
                    <Typography variant="body2" color="text.secondary">Net Water Balance</Typography>
                    <Typography variant="h3" fontWeight="bold" color="success.main">+53 mm</Typography>
                    <Chip icon={<CheckCircleIcon />} label="No irrigation needed" color="success" size="small" sx={{ mt: 1 }} />
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Weather Risks</Typography>
                  <Alert severity="warning" sx={{ mb: 2, borderRadius: 2 }}>
                    <AlertTitle sx={{ fontWeight: 600 }}>Heavy Rain Expected</AlertTitle>
                    45mm rainfall forecast for Wednesday. Delay fertilizer application.
                  </Alert>
                  <Alert severity="info" sx={{ borderRadius: 2 }}>
                    <AlertTitle sx={{ fontWeight: 600 }}>Growing Degree Days</AlertTitle>
                    Accumulated GDD: 1,250C. Oil palm on track.
                  </Alert>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Variable Rate Application Maps</Typography>
              <Typography variant="body2" color="text.secondary">Precision agriculture maps for optimized input</Typography>
            </Box>
            <Button startIcon={<AddIcon />} variant="contained" onClick={() => setVraDialogOpen(true)} sx={{ borderRadius: 2 }}>Create VRA Map</Button>
          </Box>

          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="subtitle1" fontWeight="600">VRA Zone Map - Block A Nitrogen</Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button size="small" startIcon={<DownloadIcon />}>ISO-XML</Button>
                      <Button size="small" startIcon={<DownloadIcon />}>Shapefile</Button>
                    </Box>
                  </Box>
                  
                  <Box sx={{ height: 350, bgcolor: '#1a1a2e', borderRadius: 2, position: 'relative', overflow: 'hidden' }}>
                    <svg width="100%" height="100%" viewBox="0 0 800 350">
                      <polygon points="50,50 150,30 180,120 140,180 40,160" fill="#ef5350" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="150,30 280,40 300,100 180,120" fill="#ff9800" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="280,40 450,50 480,150 420,200 300,180 300,100" fill="#ffeb3b" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="450,50 600,40 650,160 580,220 480,200 480,150" fill="#8bc34a" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="600,40 750,60 730,180 650,220 650,160" fill="#4caf50" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="40,160 140,180 180,280 100,300 30,260" fill="#ff9800" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="140,180 300,180 320,280 180,300 180,280" fill="#ffeb3b" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="300,180 480,200 500,300 320,320 320,280" fill="#8bc34a" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <polygon points="480,200 650,220 680,320 500,340 500,300" fill="#4caf50" opacity="0.8" stroke="#fff" strokeWidth="1" />
                      <text x="100" y="100" fill="white" fontSize="11" fontWeight="bold">60 kg/ha</text>
                      <text x="220" y="80" fill="white" fontSize="11" fontWeight="bold">90 kg/ha</text>
                      <text x="380" y="120" fill="black" fontSize="11" fontWeight="bold">120 kg/ha</text>
                      <text x="540" y="130" fill="white" fontSize="11" fontWeight="bold">150 kg/ha</text>
                      <text x="680" y="120" fill="white" fontSize="11" fontWeight="bold">180 kg/ha</text>
                    </svg>
                    
                    <Box sx={{ position: 'absolute', right: 16, top: 16, background: 'rgba(255,255,255,0.95)', p: 1.5, borderRadius: 2, boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
                      <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>Application Zones</Typography>
                      {vraZoneDistribution.map((zone) => (
                        <Box key={zone.zone} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                          <Box sx={{ width: 20, height: 12, bgcolor: zone.color, borderRadius: 0.5 }} />
                          <Typography variant="caption">{zone.rate} kg/ha</Typography>
                          <Typography variant="caption" color="text.secondary">({zone.area} ha)</Typography>
                        </Box>
                      ))}
                    </Box>
                  </Box>
                </CardContent>
              </Card>
              
              <Card variant="outlined" sx={{ borderRadius: 2, mt: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Zone Distribution & Savings</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                          <Pie data={vraZoneDistribution} dataKey="area" nameKey="zone" cx="50%" cy="50%" outerRadius={80} label={({ zone, area }) => `${zone}: ${area}ha`}>
                            {vraZoneDistribution.map((entry, index) => (<Cell key={`cell-${index}`} fill={entry.color} />))}
                          </Pie>
                          <RechartsTooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={vraZoneDistribution} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" />
                          <YAxis dataKey="zone" type="category" width={80} tick={{ fontSize: 11 }} />
                          <RechartsTooltip />
                          <Bar dataKey="rate" name="Rate (kg/ha)" fill="#2196f3" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <Card variant="outlined" sx={{ borderRadius: 2, mb: 2, bgcolor: '#e8f5e9' }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Total VRA Savings</Typography>
                  <Typography variant="h3" fontWeight="bold" color="success.main">$3,685</Typography>
                  <Typography variant="body2" color="text.secondary">745 kg product saved this season</Typography>
                  <Divider sx={{ my: 2 }} />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Box><Typography variant="caption" color="text.secondary">Avg Savings</Typography><Typography variant="h6" fontWeight="bold">11.7%</Typography></Box>
                    <Box><Typography variant="caption" color="text.secondary">Maps Created</Typography><Typography variant="h6" fontWeight="bold">3</Typography></Box>
                    <Box><Typography variant="caption" color="text.secondary">Area Covered</Typography><Typography variant="h6" fontWeight="bold">56.6 ha</Typography></Box>
                  </Box>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ borderRadius: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" fontWeight="600" gutterBottom>Recent VRA Maps</Typography>
                  {mockVRAMaps.map((map) => (
                    <Box key={map.id} sx={{ mb: 2, p: 1.5, bgcolor: '#f5f5f5', borderRadius: 2, transition: 'all 0.2s', '&:hover': { bgcolor: '#e0e0e0' } }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2" fontWeight="600">{map.type} Map</Typography>
                        <Chip size="small" label={`${map.savings} saved`} color="success" />
                      </Box>
                      <Typography variant="body2" color="text.secondary">{map.field}</Typography>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">{map.created} - {map.zones} zones</Typography>
                        <Typography variant="caption" fontWeight="500" color="success.main">${map.savingsCost} saved</Typography>
                      </Box>
                      <Box sx={{ mt: 1.5, display: 'flex', gap: 1 }}>
                        <Button size="small" startIcon={<DownloadIcon />} variant="outlined" sx={{ borderRadius: 1 }}>Export</Button>
                        <Button size="small" startIcon={<VisibilityIcon />} variant="outlined" sx={{ borderRadius: 1 }}>View</Button>
                      </Box>
                    </Box>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Box>
              <Typography variant="h6" fontWeight="600">Alert Management</Typography>
              <Typography variant="body2" color="text.secondary">Monitor and respond to field alerts</Typography>
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
                <Card key={alert.id} variant="outlined" sx={{ mb: 2, borderRadius: 2, borderLeft: `4px solid ${alert.severity === 'high' ? '#f44336' : alert.severity === 'medium' ? '#ff9800' : '#2196f3'}` }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <Box sx={{ display: 'flex', gap: 2 }}>
                        <Avatar sx={{ bgcolor: alert.severity === 'high' ? '#ffebee' : alert.severity === 'medium' ? '#fff3e0' : '#e3f2fd', color: alert.severity === 'high' ? '#f44336' : alert.severity === 'medium' ? '#ff9800' : '#2196f3' }}>
                          {alert.type === 'vegetation' ? <GrassIcon /> : <CloudIcon />}
                        </Avatar>
                        <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                            <Typography variant="subtitle1" fontWeight="600">{alert.field}</Typography>
                            <Chip size="small" label={alert.severity} color={getSeverityColor(alert.severity) as 'error' | 'warning' | 'info' | 'default'} />
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
                        <Button size="small" variant="contained" color="success" sx={{ borderRadius: 1 }}>Resolve</Button>
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
                    {[{ label: 'Critical', count: 0, color: '#d32f2f' }, { label: 'High', count: 1, color: '#f44336' }, { label: 'Medium', count: 2, color: '#ff9800' }, { label: 'Low', count: 0, color: '#2196f3' }].map((item) => (
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
                  <Typography variant="body2" color="text.secondary" gutterBottom>8 rules configured</Typography>
                  {[{ name: 'Rapid NDVI Decline', trigger: '>10% drop in 7 days', enabled: true }, { name: 'Low NDVI Threshold', trigger: 'NDVI < 0.4', enabled: true }, { name: 'Heavy Rain Warning', trigger: '>40mm forecast', enabled: true }, { name: 'Drought Risk', trigger: '<10mm in 14 days', enabled: true }, { name: 'Harvest Timing', trigger: 'GDD threshold', enabled: false }].map((rule, idx) => (
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

      <Drawer anchor="right" open={detailDrawerOpen} onClose={() => setDetailDrawerOpen(false)} PaperProps={{ sx: { width: 400, p: 3 } }}>
        {selectedFieldDetail && (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h6" fontWeight="600">Field Details</Typography>
              <IconButton onClick={() => setDetailDrawerOpen(false)}><CloseIcon /></IconButton>
            </Box>
            <Box sx={{ mb: 3 }}>
              <Typography variant="h5" fontWeight="bold">{selectedFieldDetail.name}</Typography>
              <Chip size="small" label={selectedFieldDetail.status} color={getStatusColor(selectedFieldDetail.status) as 'success' | 'warning' | 'error' | 'default'} sx={{ mt: 1 }} />
            </Box>
            <Divider sx={{ mb: 3 }} />
            <List dense>
              <ListItem><ListItemIcon><MapIcon /></ListItemIcon><ListItemText primary="Area" secondary={`${selectedFieldDetail.area} hectares`} /></ListItem>
              <ListItem><ListItemIcon><GrassIcon /></ListItemIcon><ListItemText primary="Crop Type" secondary={selectedFieldDetail.crop.replace('_', ' ')} /></ListItem>
              <ListItem><ListItemIcon><SatelliteIcon /></ListItemIcon><ListItemText primary="Current NDVI" secondary={selectedFieldDetail.ndvi.toFixed(2)} /></ListItem>
              <ListItem><ListItemIcon><TimelineIcon /></ListItemIcon><ListItemText primary="NDVI Change (7d)" secondary={`${selectedFieldDetail.ndviChange >= 0 ? '+' : ''}${selectedFieldDetail.ndviChange.toFixed(2)}`} /></ListItem>
              <ListItem><ListItemIcon><CalendarIcon /></ListItemIcon><ListItemText primary="Planting Date" secondary={selectedFieldDetail.plantingDate} /></ListItem>
              <ListItem><ListItemIcon><SpeedIcon /></ListItemIcon><ListItemText primary="Growth Stage" secondary={selectedFieldDetail.growthStage} /></ListItem>
              <ListItem><ListItemIcon><SatelliteIcon /></ListItemIcon><ListItemText primary="Last Imagery" secondary={selectedFieldDetail.lastImagery} /></ListItem>
            </List>
            <Divider sx={{ my: 3 }} />
            <Typography variant="subtitle2" fontWeight="600" gutterBottom>Quick Actions</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Button variant="outlined" startIcon={<SatelliteIcon />} fullWidth>View NDVI History</Button>
              <Button variant="outlined" startIcon={<AgricultureIcon />} fullWidth>Create VRA Map</Button>
              <Button variant="outlined" startIcon={<DownloadIcon />} fullWidth>Export Field Data</Button>
            </Box>
          </>
        )}
      </Drawer>

      <Dialog open={vraDialogOpen} onClose={() => { setVraDialogOpen(false); setVraStep(0); }} maxWidth="md" fullWidth>
        <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" fontWeight="600">Create VRA Map</Typography>
          <IconButton onClick={() => { setVraDialogOpen(false); setVraStep(0); }}><CloseIcon /></IconButton>
        </DialogTitle>
        <DialogContent>
          <Stepper activeStep={vraStep} sx={{ mb: 4, mt: 2 }}>
            <Step><StepLabel>Select Field</StepLabel></Step>
            <Step><StepLabel>Choose Map Type</StepLabel></Step>
            <Step><StepLabel>Configure Zones</StepLabel></Step>
            <Step><StepLabel>Preview & Export</StepLabel></Step>
          </Stepper>
          
          {vraStep === 0 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Select the field for VRA map generation</Typography>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel>Field</InputLabel>
                <Select label="Field" defaultValue="">
                  {mockFields.map(f => (
                    <MenuItem key={f.id} value={f.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: getNDVIColor(f.ndvi) }} />
                        <Box>
                          <Typography variant="body2">{f.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{f.area} ha - NDVI: {f.ndvi}</Typography>
                        </Box>
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          )}
          
          {vraStep === 1 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Select the type of VRA map to create</Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                {[{ type: 'Nitrogen', desc: 'Optimize N fertilizer based on NDVI zones', icon: <GrassIcon /> }, { type: 'Sowing', desc: 'Variable seeding rates by productivity zone', icon: <AgricultureIcon /> }, { type: 'P&K', desc: 'Phosphorus and potassium application', icon: <SatelliteIcon /> }, { type: 'Lime', desc: 'Soil pH correction application', icon: <MapIcon /> }].map((item) => (
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
          
          {vraStep === 2 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Configure zone parameters</Typography>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}><TextField fullWidth label="Target Yield (t/ha)" type="number" defaultValue="25" /></Grid>
                <Grid item xs={6}>
                  <FormControl fullWidth>
                    <InputLabel>Number of Zones</InputLabel>
                    <Select label="Number of Zones" defaultValue={5}>
                      <MenuItem value={3}>3 Zones</MenuItem>
                      <MenuItem value={5}>5 Zones</MenuItem>
                      <MenuItem value={7}>7 Zones</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={6}><TextField fullWidth label="Min Rate (kg/ha)" type="number" defaultValue="60" /></Grid>
                <Grid item xs={6}><TextField fullWidth label="Max Rate (kg/ha)" type="number" defaultValue="180" /></Grid>
              </Grid>
            </Box>
          )}
          
          {vraStep === 3 && (
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>Preview and export your VRA map</Typography>
              <Box sx={{ height: 200, bgcolor: '#f5f5f5', borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2 }}>
                <Typography color="text.secondary">VRA Map Preview</Typography>
              </Box>
              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>Export Format</Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip label="ISO-XML" variant="outlined" />
                  <Chip label="Shapefile" variant="outlined" />
                  <Chip label="GeoJSON" variant="outlined" />
                  <Chip label="CSV" variant="outlined" />
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => { setVraDialogOpen(false); setVraStep(0); }}>Cancel</Button>
          {vraStep > 0 && <Button onClick={() => setVraStep(vraStep - 1)}>Back</Button>}
          {vraStep < 3 ? (
            <Button variant="contained" onClick={() => setVraStep(vraStep + 1)}>Next</Button>
          ) : (
            <Button variant="contained" color="success" onClick={() => { setVraDialogOpen(false); setVraStep(0); }}>Create & Export</Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CropMonitoringPage;
