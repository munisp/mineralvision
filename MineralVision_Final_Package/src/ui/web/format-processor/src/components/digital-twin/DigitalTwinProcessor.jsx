import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Grid, 
  Paper, 
  Tabs, 
  Tab, 
  Box, 
  Button, 
  TextField, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Divider,
  CircularProgress,
  Snackbar,
  Alert
} from '@mui/material';
import { 
  Timeline, 
  TimelineItem, 
  TimelineSeparator, 
  TimelineConnector, 
  TimelineContent, 
  TimelineDot 
} from '@mui/lab';
import { 
  Map as MapIcon, 
  Science as ScienceIcon, 
  Assessment as AssessmentIcon, 
  Construction as ConstructionIcon,
  Nature as NatureIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { API_BASE_URL } from '../../services/api';
import axios from 'axios';
import MineralVision3DEngine from '../visualization/MineralVision3DEngine';

// Styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  height: '100%',
  display: 'flex',
  flexDirection: 'column'
}));

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`digital-twin-tabpanel-${index}`}
      aria-labelledby={`digital-twin-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

const DigitalTwinProcessor = () => {
  // State
  const [activeTab, setActiveTab] = useState(0);
  const [entities, setEntities] = useState([]);
  const [simulations, setSimulations] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [selectedSimulation, setSelectedSimulation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Form states
  const [newDepositForm, setNewDepositForm] = useState({
    name: '',
    mineralType: '',
    probability: 0.5,
    volumeEstimate: 0,
    depth: 0,
    latitude: 0,
    longitude: 0
  });
  
  const [newExtractionSimForm, setNewExtractionSimForm] = useState({
    name: '',
    description: '',
    depositId: '',
    extractionRate: 1000,
    extractionDuration: 365,
    extractionMethod: 'open_pit',
    efficiency: 0.8,
    mineralPrice: 100
  });
  
  const [newEnvironmentalSimForm, setNewEnvironmentalSimForm] = useState({
    name: '',
    description: '',
    extractionSimulationId: '',
    areaId: '',
    environmentalFactors: {
      biodiversityIndex: 0.5,
      waterSensitivity: 0.5,
      protectedSpecies: false,
      proximityToWaterBodies: 0
    }
  });
  
  // Load entities and simulations on component mount
  useEffect(() => {
    fetchEntities();
    fetchSimulations();
  }, []);
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  // Fetch entities from API
  const fetchEntities = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/digital-twin/entities`);
      setEntities(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch entities');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Fetch simulations from API
  const fetchSimulations = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/digital-twin/simulations`);
      setSimulations(response.data);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch simulations');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Handle entity selection
  const handleEntitySelect = (entity) => {
    setSelectedEntity(entity);
  };
  
  // Handle simulation selection
  const handleSimulationSelect = (simulation) => {
    setSelectedSimulation(simulation);
  };
  
  // Create new mineral deposit
  const createMineralDeposit = async () => {
    try {
      setLoading(true);
      const payload = {
        name: newDepositForm.name,
        metadata: { entity_type: 'MineralDeposit' },
        mineral_type: newDepositForm.mineralType,
        probability: parseFloat(newDepositForm.probability),
        volume_estimate: parseFloat(newDepositForm.volumeEstimate),
        depth: parseFloat(newDepositForm.depth),
        latitude: parseFloat(newDepositForm.latitude),
        longitude: parseFloat(newDepositForm.longitude)
      };
      
      const response = await axios.post(`${API_BASE_URL}/digital-twin/entities/mineral-deposit`, payload);
      setEntities([...entities, response.data]);
      setSuccess('Mineral deposit created successfully');
      setNewDepositForm({
        name: '',
        mineralType: '',
        probability: 0.5,
        volumeEstimate: 0,
        depth: 0,
        latitude: 0,
        longitude: 0
      });
      setLoading(false);
    } catch (err) {
      setError('Failed to create mineral deposit');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Create new extraction simulation
  const createExtractionSimulation = async () => {
    try {
      setLoading(true);
      const payload = {
        name: newExtractionSimForm.name,
        description: newExtractionSimForm.description,
        deposit_id: newExtractionSimForm.depositId,
        extraction_rate: parseFloat(newExtractionSimForm.extractionRate),
        extraction_duration: parseInt(newExtractionSimForm.extractionDuration),
        extraction_method: newExtractionSimForm.extractionMethod,
        efficiency: parseFloat(newExtractionSimForm.efficiency),
        mineral_price: parseFloat(newExtractionSimForm.mineralPrice)
      };
      
      const response = await axios.post(`${API_BASE_URL}/digital-twin/simulations/extraction`, payload);
      setSimulations([...simulations, response.data]);
      setSuccess('Extraction simulation created successfully');
      setNewExtractionSimForm({
        name: '',
        description: '',
        depositId: '',
        extractionRate: 1000,
        extractionDuration: 365,
        extractionMethod: 'open_pit',
        efficiency: 0.8,
        mineralPrice: 100
      });
      setLoading(false);
    } catch (err) {
      setError('Failed to create extraction simulation');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Create new environmental impact simulation
  const createEnvironmentalSimulation = async () => {
    try {
      setLoading(true);
      const payload = {
        name: newEnvironmentalSimForm.name,
        description: newEnvironmentalSimForm.description,
        extraction_simulation_id: newEnvironmentalSimForm.extractionSimulationId,
        area_id: newEnvironmentalSimForm.areaId,
        environmental_factors: newEnvironmentalSimForm.environmentalFactors
      };
      
      const response = await axios.post(`${API_BASE_URL}/digital-twin/simulations/environmental`, payload);
      setSimulations([...simulations, response.data]);
      setSuccess('Environmental simulation created successfully');
      setNewEnvironmentalSimForm({
        name: '',
        description: '',
        extractionSimulationId: '',
        areaId: '',
        environmentalFactors: {
          biodiversityIndex: 0.5,
          waterSensitivity: 0.5,
          protectedSpecies: false,
          proximityToWaterBodies: 0
        }
      });
      setLoading(false);
    } catch (err) {
      setError('Failed to create environmental simulation');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Run simulation
  const runSimulation = async (simulationId) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_BASE_URL}/digital-twin/simulations/${simulationId}/run`);
      
      // Update the simulation in the list
      const updatedSimulations = simulations.map(sim => 
        sim.simulation_id === simulationId ? response.data : sim
      );
      
      setSimulations(updatedSimulations);
      
      // If this is the selected simulation, update it
      if (selectedSimulation && selectedSimulation.simulation_id === simulationId) {
        setSelectedSimulation(response.data);
      }
      
      setSuccess('Simulation run successfully');
      setLoading(false);
    } catch (err) {
      setError('Failed to run simulation');
      setLoading(false);
      console.error(err);
    }
  };
  
  // Handle form changes
  const handleDepositFormChange = (e) => {
    const { name, value } = e.target;
    setNewDepositForm({
      ...newDepositForm,
      [name]: value
    });
  };
  
  const handleExtractionSimFormChange = (e) => {
    const { name, value } = e.target;
    setNewExtractionSimForm({
      ...newExtractionSimForm,
      [name]: value
    });
  };
  
  const handleEnvironmentalSimFormChange = (e) => {
    const { name, value } = e.target;
    setNewEnvironmentalSimForm({
      ...newEnvironmentalSimForm,
      [name]: value
    });
  };
  
  const handleEnvironmentalFactorsChange = (e) => {
    const { name, value } = e.target;
    setNewEnvironmentalSimForm({
      ...newEnvironmentalSimForm,
      environmentalFactors: {
        ...newEnvironmentalSimForm.environmentalFactors,
        [name]: value
      }
    });
  };
  
  // Clear alerts
  const handleClearAlert = () => {
    setError(null);
    setSuccess(null);
  };
  
  // Render entity list
  const renderEntityList = () => {
    if (entities.length === 0) {
      return <Typography variant="body1">No entities found</Typography>;
    }
    
    return (
      <Grid container spacing={2}>
        {entities.map(entity => (
          <Grid item xs={12} sm={6} md={4} key={entity.entity_id}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 2, 
                cursor: 'pointer',
                bgcolor: selectedEntity && selectedEntity.entity_id === entity.entity_id ? 'primary.light' : 'background.paper',
                color: selectedEntity && selectedEntity.entity_id === entity.entity_id ? 'primary.contrastText' : 'text.primary'
              }}
              onClick={() => handleEntitySelect(entity)}
            >
              <Typography variant="h6">{entity.name}</Typography>
              <Typography variant="body2">
                {entity.metadata.entity_type || 'Entity'}
              </Typography>
              {entity.properties.mineral_type && (
                <Typography variant="body2">
                  Mineral: {entity.properties.mineral_type}
                </Typography>
              )}
              {entity.properties.probability && (
                <Typography variant="body2">
                  Probability: {(entity.properties.probability * 100).toFixed(1)}%
                </Typography>
              )}
            </Paper>
          </Grid>
        ))}
      </Grid>
    );
  };
  
  // Render simulation list
  const renderSimulationList = () => {
    if (simulations.length === 0) {
      return <Typography variant="body1">No simulations found</Typography>;
    }
    
    return (
      <Grid container spacing={2}>
        {simulations.map(simulation => (
          <Grid item xs={12} sm={6} md={4} key={simulation.simulation_id}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 2, 
                cursor: 'pointer',
                bgcolor: selectedSimulation && selectedSimulation.simulation_id === simulation.simulation_id ? 'primary.light' : 'background.paper',
                color: selectedSimulation && selectedSimulation.simulation_id === simulation.simulation_id ? 'primary.contrastText' : 'text.primary'
              }}
              onClick={() => handleSimulationSelect(simulation)}
            >
              <Typography variant="h6">{simulation.name}</Typography>
              <Typography variant="body2">
                {simulation.description}
              </Typography>
              <Typography variant="body2">
                Status: {simulation.status}
              </Typography>
              <Box sx={{ mt: 1 }}>
                <Button 
                  variant="contained" 
                  size="small"
                  disabled={simulation.status === 'running'}
                  onClick={(e) => {
                    e.stopPropagation();
                    runSimulation(simulation.simulation_id);
                  }}
                >
                  Run Simulation
                </Button>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    );
  };
  
  // Render entity details
  const renderEntityDetails = () => {
    if (!selectedEntity) {
      return <Typography variant="body1">Select an entity to view details</Typography>;
    }
    
    return (
      <Box>
        <Typography variant="h5">{selectedEntity.name}</Typography>
        <Typography variant="subtitle1">
          {selectedEntity.metadata.entity_type || 'Entity'}
        </Typography>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6">Properties</Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {Object.entries(selectedEntity.properties).map(([key, value]) => (
            <Grid item xs={6} key={key}>
              <Typography variant="body2">
                <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
              </Typography>
            </Grid>
          ))}
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6">Relationships</Typography>
        {Object.keys(selectedEntity.relationships).length === 0 ? (
          <Typography variant="body2">No relationships</Typography>
        ) : (
          <Timeline position="alternate">
            {Object.entries(selectedEntity.relationships).map(([relationType, relations]) => (
              relations.map((relation, index) => (
                <TimelineItem key={`${relationType}-${index}`}>
                  <TimelineSeparator>
                    <TimelineDot color="primary" />
                    {index < relations.length - 1 && <TimelineConnector />}
                  </TimelineSeparator>
                  <TimelineContent>
                    <Typography variant="body1">{relationType}</Typography>
                    <Typography variant="body2">Target: {relation.target_id}</Typography>
                  </TimelineContent>
                </TimelineItem>
              ))
            ))}
          </Timeline>
        )}
        
        {selectedEntity.geometry && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6">Visualization</Typography>
            <Box sx={{ height: 400, mt: 2 }}>
              <MineralVision3DEngine 
                data={[selectedEntity]} 
                type="entity"
              />
            </Box>
          </>
        )}
      </Box>
    );
  };
  
  // Render simulation details
  const renderSimulationDetails = () => {
    if (!selectedSimulation) {
      return <Typography variant="body1">Select a simulation to view details</Typography>;
    }
    
    return (
      <Box>
        <Typography variant="h5">{selectedSimulation.name}</Typography>
        <Typography variant="subtitle1">
          {selectedSimulation.description}
        </Typography>
        <Typography variant="subtitle2" color={
          selectedSimulation.status === 'completed' ? 'success.main' :
          selectedSimulation.status === 'running' ? 'info.main' :
          selectedSimulation.status === 'failed' ? 'error.main' :
          'text.secondary'
        }>
          Status: {selectedSimulation.status}
        </Typography>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6">Parameters</Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {Object.entries(selectedSimulation.parameters).map(([key, value]) => (
            <Grid item xs={6} key={key}>
              <Typography variant="body2">
                <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
              </Typography>
            </Grid>
          ))}
        </Grid>
        
        <Divider sx={{ my: 2 }} />
        
        <Typography variant="h6">Results</Typography>
        {Object.keys(selectedSimulation.results).length === 0 ? (
          <Typography variant="body2">No results yet. Run the simulation to generate results.</Typography>
        ) : (
          <Grid container spacing={2} sx={{ mt: 1 }}>
            {Object.entries(selectedSimulation.results).map(([key, value]) => (
              <Grid item xs={6} key={key}>
                <Typography variant="body2">
                  <strong>{key}:</strong> {typeof value === 'object' ? JSON.stringify(value) : value}
                </Typography>
              </Grid>
            ))}
          </Grid>
        )}
        
        {selectedSimulation.results && selectedSimulation.results.profit !== undefined && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6">Financial Summary</Typography>
            <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="body2">Total Revenue</Typography>
                  <Typography variant="h6" color="success.main">
                    ${selectedSimulation.results.total_revenue.toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2">Total Cost</Typography>
                  <Typography variant="h6" color="error.main">
                    ${selectedSimulation.results.total_cost.toLocaleString()}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2">Profit</Typography>
                  <Typography variant="h6" color={selectedSimulation.results.profit >= 0 ? 'success.main' : 'error.main'}>
                    ${selectedSimulation.results.profit.toLocaleString()}
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          </>
        )}
        
        {selectedSimulation.results && selectedSimulation.results.impact_score !== undefined && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6">Environmental Impact</Typography>
            <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="body2">Impact Score</Typography>
                  <Typography variant="h6" color={
                    selectedSimulation.results.impact_score < 0.3 ? 'success.main' :
                    selectedSimulation.results.impact_score < 0.7 ? 'warning.main' :
                    'error.main'
                  }>
                    {selectedSimulation.results.impact_score.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2">Land Disturbance</Typography>
                  <Typography variant="h6">
                    {selectedSimulation.results.land_disturbance.toLocaleString()} m²
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2">Water Usage</Typography>
                  <Typography variant="h6">
                    {selectedSimulation.results.water_usage.toLocaleString()} m³
                  </Typography>
                </Grid>
              </Grid>
              
              {selectedSimulation.results.mitigation_recommendations && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2">Mitigation Recommendations:</Typography>
                  <ul>
                    {selectedSimulation.results.mitigation_recommendations.map((rec, index) => (
                      <li key={index}>
                        <Typography variant="body2">{rec}</Typography>
                      </li>
                    ))}
                  </ul>
                </Box>
              )}
            </Box>
          </>
        )}
        
        <Box sx={{ mt: 2 }}>
          <Button 
            variant="contained" 
            disabled={selectedSimulation.status === 'running'}
            onClick={() => runSimulation(selectedSimulation.simulation_id)}
          >
            Run Simulation
          </Button>
        </Box>
      </Box>
    );
  };
  
  // Render create entity form
  const renderCreateEntityForm = () => {
    return (
      <Box>
        <Typography variant="h6">Create Mineral Deposit</Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Name"
              name="name"
              value={newDepositForm.name}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Mineral Type"
              name="mineralType"
              value={newDepositForm.mineralType}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Probability"
              name="probability"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
              value={newDepositForm.probability}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Volume Estimate (m³)"
              name="volumeEstimate"
              type="number"
              value={newDepositForm.volumeEstimate}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Depth (m)"
              name="depth"
              type="number"
              value={newDepositForm.depth}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Latitude"
              name="latitude"
              type="number"
              inputProps={{ step: 0.000001 }}
              value={newDepositForm.latitude}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Longitude"
              name="longitude"
              type="number"
              inputProps={{ step: 0.000001 }}
              value={newDepositForm.longitude}
              onChange={handleDepositFormChange}
            />
          </Grid>
          <Grid item xs={12}>
            <Button 
              variant="contained" 
              onClick={createMineralDeposit}
              disabled={loading || !newDepositForm.name || !newDepositForm.mineralType}
            >
              Create Deposit
            </Button>
          </Grid>
        </Grid>
      </Box>
    );
  };
  
  // Render create simulation form
  const renderCreateSimulationForm = () => {
    const mineralDeposits = entities.filter(entity => 
      entity.metadata.entity_type === 'MineralDeposit' || 
      (entity.properties && entity.properties.mineral_type)
    );
    
    const explorationAreas = entities.filter(entity => 
      entity.metadata.entity_type === 'ExplorationArea' || 
      (entity.properties && entity.properties.status)
    );
    
    const extractionSimulations = simulations.filter(sim => 
      sim.parameters && sim.parameters.extraction_method
    );
    
    return (
      <Box>
        <Typography variant="h6">Create Extraction Simulation</Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Name"
              name="name"
              value={newExtractionSimForm.name}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={newExtractionSimForm.description}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Mineral Deposit</InputLabel>
              <Select
                name="depositId"
                value={newExtractionSimForm.depositId}
                onChange={handleExtractionSimFormChange}
                label="Mineral Deposit"
              >
                {mineralDeposits.map(deposit => (
                  <MenuItem key={deposit.entity_id} value={deposit.entity_id}>
                    {deposit.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Extraction Rate (tons/day)"
              name="extractionRate"
              type="number"
              value={newExtractionSimForm.extractionRate}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Extraction Duration (days)"
              name="extractionDuration"
              type="number"
              value={newExtractionSimForm.extractionDuration}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth>
              <InputLabel>Extraction Method</InputLabel>
              <Select
                name="extractionMethod"
                value={newExtractionSimForm.extractionMethod}
                onChange={handleExtractionSimFormChange}
                label="Extraction Method"
              >
                <MenuItem value="open_pit">Open Pit</MenuItem>
                <MenuItem value="underground">Underground</MenuItem>
                <MenuItem value="in_situ">In-Situ</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Efficiency"
              name="efficiency"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
              value={newExtractionSimForm.efficiency}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Mineral Price ($/ton)"
              name="mineralPrice"
              type="number"
              value={newExtractionSimForm.mineralPrice}
              onChange={handleExtractionSimFormChange}
            />
          </Grid>
          <Grid item xs={12}>
            <Button 
              variant="contained" 
              onClick={createExtractionSimulation}
              disabled={loading || !newExtractionSimForm.name || !newExtractionSimForm.depositId}
            >
              Create Extraction Simulation
            </Button>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 3 }} />
        
        <Typography variant="h6">Create Environmental Impact Simulation</Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Name"
              name="name"
              value={newEnvironmentalSimForm.name}
              onChange={handleEnvironmentalSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Description"
              name="description"
              value={newEnvironmentalSimForm.description}
              onChange={handleEnvironmentalSimFormChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Extraction Simulation</InputLabel>
              <Select
                name="extractionSimulationId"
                value={newEnvironmentalSimForm.extractionSimulationId}
                onChange={handleEnvironmentalSimFormChange}
                label="Extraction Simulation"
              >
                {extractionSimulations.map(sim => (
                  <MenuItem key={sim.simulation_id} value={sim.simulation_id}>
                    {sim.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Exploration Area</InputLabel>
              <Select
                name="areaId"
                value={newEnvironmentalSimForm.areaId}
                onChange={handleEnvironmentalSimFormChange}
                label="Exploration Area"
              >
                {explorationAreas.map(area => (
                  <MenuItem key={area.entity_id} value={area.entity_id}>
                    {area.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12}>
            <Typography variant="subtitle1">Environmental Factors</Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Biodiversity Index"
              name="biodiversityIndex"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
              value={newEnvironmentalSimForm.environmentalFactors.biodiversityIndex}
              onChange={handleEnvironmentalFactorsChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Water Sensitivity"
              name="waterSensitivity"
              type="number"
              inputProps={{ min: 0, max: 1, step: 0.01 }}
              value={newEnvironmentalSimForm.environmentalFactors.waterSensitivity}
              onChange={handleEnvironmentalFactorsChange}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <FormControl fullWidth>
              <InputLabel>Protected Species Present</InputLabel>
              <Select
                name="protectedSpecies"
                value={newEnvironmentalSimForm.environmentalFactors.protectedSpecies}
                onChange={handleEnvironmentalFactorsChange}
                label="Protected Species Present"
              >
                <MenuItem value={true}>Yes</MenuItem>
                <MenuItem value={false}>No</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Proximity to Water Bodies (km)"
              name="proximityToWaterBodies"
              type="number"
              value={newEnvironmentalSimForm.environmentalFactors.proximityToWaterBodies}
              onChange={handleEnvironmentalFactorsChange}
            />
          </Grid>
          <Grid item xs={12}>
            <Button 
              variant="contained" 
              onClick={createEnvironmentalSimulation}
              disabled={
                loading || 
                !newEnvironmentalSimForm.name || 
                !newEnvironmentalSimForm.extractionSimulationId ||
                !newEnvironmentalSimForm.areaId
              }
            >
              Create Environmental Simulation
            </Button>
          </Grid>
        </Grid>
      </Box>
    );
  };
  
  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Digital Twin System
      </Typography>
      <Typography variant="subtitle1" gutterBottom>
        Create and manage virtual replicas of exploration and mining operations
      </Typography>
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange} aria-label="digital twin tabs">
          <Tab icon={<MapIcon />} label="Entities" />
          <Tab icon={<ScienceIcon />} label="Simulations" />
          <Tab icon={<AssessmentIcon />} label="Create" />
        </Tabs>
      </Box>
      
      {/* Loading indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2, mb: 2 }}>
          <CircularProgress />
        </Box>
      )}
      
      {/* Error and success alerts */}
      <Snackbar open={!!error} autoHideDuration={6000} onClose={handleClearAlert}>
        <Alert onClose={handleClearAlert} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
      
      <Snackbar open={!!success} autoHideDuration={6000} onClose={handleClearAlert}>
        <Alert onClose={handleClearAlert} severity="success" sx={{ width: '100%' }}>
          {success}
        </Alert>
      </Snackbar>
      
      {/* Tab panels */}
      <TabPanel value={activeTab} index={0}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Entity List
              </Typography>
              {renderEntityList()}
            </StyledPaper>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Entity Details
              </Typography>
              {renderEntityDetails()}
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
      
      <TabPanel value={activeTab} index={1}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Simulation List
              </Typography>
              {renderSimulationList()}
            </StyledPaper>
          </Grid>
          <Grid item xs={12} md={6}>
            <StyledPaper>
              <Typography variant="h6" gutterBottom>
                Simulation Details
              </Typography>
              {renderSimulationDetails()}
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
      
      <TabPanel value={activeTab} index={2}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <StyledPaper>
              {renderCreateEntityForm()}
            </StyledPaper>
          </Grid>
          <Grid item xs={12}>
            <StyledPaper>
              {renderCreateSimulationForm()}
            </StyledPaper>
          </Grid>
        </Grid>
      </TabPanel>
    </Container>
  );
};

export default DigitalTwinProcessor;
