import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Row, 
  Col, 
  Card, 
  Button, 
  Form, 
  Table, 
  Tabs, 
  Tab, 
  Badge, 
  Spinner,
  Alert,
  Modal,
  ListGroup
} from 'react-bootstrap';
import { MapContainer, TileLayer, Marker, Popup, Polygon, useMap } from 'react-leaflet';
import L from 'leaflet';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { 
  faDrone, 
  faMapMarkerAlt, 
  faRoute, 
  faDrawPolygon, 
  faSync, 
  faPlus, 
  faPlay, 
  faCheck, 
  faTimes,
  faRobot,
  faLayerGroup,
  faChartLine
} from '@fortawesome/free-solid-svg-icons';
import axios from 'axios';
import { API_BASE_URL } from '../../services/api';
import './AutonomousExplorationProcessor.css';

// Custom map component that allows for map reference access
const MapComponent = ({ center, zoom, children, onMapReady }) => {
  const mapRef = useMap();
  
  useEffect(() => {
    if (onMapReady) {
      onMapReady(mapRef);
    }
  }, [mapRef, onMapReady]);
  
  return null;
};

const AutonomousExplorationProcessor = () => {
  // State for active tab
  const [activeTab, setActiveTab] = useState('drones');
  
  // State for drones
  const [drones, setDrones] = useState([]);
  const [droneStates, setDroneStates] = useState({});
  const [selectedDrone, setSelectedDrone] = useState(null);
  const [showDroneForm, setShowDroneForm] = useState(false);
  const [droneForm, setDroneForm] = useState({
    model: '',
    battery_capacity_mah: 5000,
    max_flight_time_minutes: 30,
    max_speed_mps: 10,
    max_payload_grams: 500,
    camera_resolution: '4K',
    sensors: [],
    communication_range_meters: 2000
  });
  
  // State for sampling points
  const [samplingPoints, setSamplingPoints] = useState([]);
  const [selectedSamplingPoints, setSelectedSamplingPoints] = useState([]);
  const [showSamplingPointForm, setShowSamplingPointForm] = useState(false);
  const [samplingPointForm, setSamplingPointForm] = useState({
    position: { lat: 0, lon: 0 },
    priority: 5,
    sample_type: 'soil',
    estimated_time_seconds: 300
  });
  
  // State for missions
  const [missions, setMissions] = useState([]);
  const [selectedMission, setSelectedMission] = useState(null);
  
  // State for exploration areas
  const [explorationAreas, setExplorationAreas] = useState([]);
  const [selectedArea, setSelectedArea] = useState(null);
  const [showAreaForm, setShowAreaForm] = useState(false);
  const [areaForm, setAreaForm] = useState({
    name: '',
    geometry: null,
    priority: 5
  });
  const [drawingArea, setDrawingArea] = useState(false);
  const [drawnPolygon, setDrawnPolygon] = useState([]);
  
  // State for advanced planning
  const [showSwarmForm, setShowSwarmForm] = useState(false);
  const [swarmForm, setSwarmForm] = useState({
    area_id: '',
    num_drones: 3
  });
  const [showAdaptiveForm, setShowAdaptiveForm] = useState(false);
  const [adaptiveForm, setAdaptiveForm] = useState({
    area_id: '',
    initial_results: {}
  });
  
  // State for map
  const [mapCenter, setMapCenter] = useState([0, 0]);
  const [mapZoom, setMapZoom] = useState(13);
  const [mapRef, setMapRef] = useState(null);
  
  // State for loading and alerts
  const [loading, setLoading] = useState(false);
  const [alert, setAlert] = useState({ show: false, variant: 'info', message: '' });
  
  // Load data on component mount
  useEffect(() => {
    fetchDrones();
    fetchSamplingPoints();
    fetchMissions();
    fetchExplorationAreas();
  }, []);
  
  // Fetch drones from API
  const fetchDrones = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/autonomous-exploration/drones`);
      setDrones(response.data);
      
      // Fetch state for each drone
      const states = {};
      for (const drone of response.data) {
        const stateResponse = await axios.get(`${API_BASE_URL}/autonomous-exploration/drones/${drone.drone_id}/state`);
        states[drone.drone_id] = stateResponse.data;
      }
      setDroneStates(states);
      
      // Set map center to first drone if available
      if (response.data.length > 0 && states[response.data[0].drone_id]) {
        const firstDroneState = states[response.data[0].drone_id];
        setMapCenter([firstDroneState.position.lat, firstDroneState.position.lon]);
      }
    } catch (error) {
      console.error('Error fetching drones:', error);
      showAlert('danger', 'Failed to fetch drones');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch sampling points from API
  const fetchSamplingPoints = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/autonomous-exploration/sampling-points`);
      setSamplingPoints(response.data);
    } catch (error) {
      console.error('Error fetching sampling points:', error);
      showAlert('danger', 'Failed to fetch sampling points');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch missions from API
  const fetchMissions = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/autonomous-exploration/missions`);
      setMissions(response.data);
    } catch (error) {
      console.error('Error fetching missions:', error);
      showAlert('danger', 'Failed to fetch missions');
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch exploration areas from API
  const fetchExplorationAreas = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/autonomous-exploration/exploration-areas`);
      setExplorationAreas(response.data);
    } catch (error) {
      console.error('Error fetching exploration areas:', error);
      showAlert('danger', 'Failed to fetch exploration areas');
    } finally {
      setLoading(false);
    }
  };
  
  // Register a new drone
  const registerDrone = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/drones`, droneForm);
      showAlert('success', `Drone registered with ID: ${response.data.drone_id}`);
      setShowDroneForm(false);
      resetDroneForm();
      fetchDrones();
    } catch (error) {
      console.error('Error registering drone:', error);
      showAlert('danger', 'Failed to register drone');
    } finally {
      setLoading(false);
    }
  };
  
  // Add a new sampling point
  const addSamplingPoint = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/sampling-points`, samplingPointForm);
      showAlert('success', `Sampling point added with ID: ${response.data.point_id}`);
      setShowSamplingPointForm(false);
      resetSamplingPointForm();
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error adding sampling point:', error);
      showAlert('danger', 'Failed to add sampling point');
    } finally {
      setLoading(false);
    }
  };
  
  // Create a mission plan
  const createMission = async () => {
    if (!selectedDrone || selectedSamplingPoints.length === 0) {
      showAlert('warning', 'Please select a drone and at least one sampling point');
      return;
    }
    
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/autonomous-exploration/missions`,
        selectedSamplingPoints,
        { params: { drone_id: selectedDrone.drone_id } }
      );
      showAlert('success', `Mission created with ID: ${response.data.mission_id}`);
      setSelectedSamplingPoints([]);
      fetchMissions();
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error creating mission:', error);
      showAlert('danger', 'Failed to create mission');
    } finally {
      setLoading(false);
    }
  };
  
  // Start a mission
  const startMission = async (missionId) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/missions/${missionId}/start`);
      if (response.data.success) {
        showAlert('success', 'Mission started successfully');
        fetchMissions();
        fetchDrones();
      } else {
        showAlert('warning', 'Failed to start mission');
      }
    } catch (error) {
      console.error('Error starting mission:', error);
      showAlert('danger', 'Failed to start mission');
    } finally {
      setLoading(false);
    }
  };
  
  // Complete a mission
  const completeMission = async (missionId, success = true) => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/autonomous-exploration/missions/${missionId}/complete`,
        {},
        { params: { success } }
      );
      if (response.data.success) {
        showAlert('success', `Mission ${success ? 'completed' : 'failed'} successfully`);
        fetchMissions();
        fetchDrones();
        fetchSamplingPoints();
      } else {
        showAlert('warning', 'Failed to update mission status');
      }
    } catch (error) {
      console.error('Error completing mission:', error);
      showAlert('danger', 'Failed to update mission status');
    } finally {
      setLoading(false);
    }
  };
  
  // Add a new exploration area
  const addExplorationArea = async () => {
    if (!drawnPolygon || drawnPolygon.length < 3) {
      showAlert('warning', 'Please draw a valid polygon on the map');
      return;
    }
    
    // Convert drawn polygon to GeoJSON format
    const coordinates = [...drawnPolygon, drawnPolygon[0]].map(point => [point[1], point[0]]);
    const geometry = {
      type: 'Polygon',
      coordinates: [coordinates]
    };
    
    const areaData = {
      ...areaForm,
      geometry
    };
    
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/exploration-areas`, areaData);
      showAlert('success', `Exploration area added with ID: ${response.data.area_id}`);
      setShowAreaForm(false);
      resetAreaForm();
      setDrawnPolygon([]);
      setDrawingArea(false);
      fetchExplorationAreas();
    } catch (error) {
      console.error('Error adding exploration area:', error);
      showAlert('danger', 'Failed to add exploration area');
    } finally {
      setLoading(false);
    }
  };
  
  // Generate sampling points for an area
  const generateSamplingPoints = async (areaId, density = 0.0001) => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/autonomous-exploration/exploration-areas/${areaId}/sampling-points`,
        {},
        { params: { density } }
      );
      showAlert('success', `Generated ${response.data.length} sampling points`);
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error generating sampling points:', error);
      showAlert('danger', 'Failed to generate sampling points');
    } finally {
      setLoading(false);
    }
  };
  
  // Optimize drone assignments
  const optimizeDroneAssignments = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/optimize-assignments`);
      const assignments = response.data.assignments;
      
      // Create missions for each assignment
      for (const [droneId, pointIds] of Object.entries(assignments)) {
        if (pointIds.length > 0) {
          await axios.post(
            `${API_BASE_URL}/autonomous-exploration/missions`,
            pointIds,
            { params: { drone_id: droneId } }
          );
        }
      }
      
      showAlert('success', `Created optimized missions for ${Object.keys(assignments).length} drones`);
      fetchMissions();
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error optimizing drone assignments:', error);
      showAlert('danger', 'Failed to optimize drone assignments');
    } finally {
      setLoading(false);
    }
  };
  
  // Generate swarm mission
  const generateSwarmMission = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/swarm-mission`, swarmForm);
      showAlert('success', `Generated ${response.data.length} swarm missions`);
      setShowSwarmForm(false);
      resetSwarmForm();
      fetchMissions();
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error generating swarm mission:', error);
      showAlert('danger', 'Failed to generate swarm mission');
    } finally {
      setLoading(false);
    }
  };
  
  // Generate adaptive sampling points
  const generateAdaptiveSampling = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/autonomous-exploration/adaptive-sampling`, adaptiveForm);
      showAlert('success', `Generated ${response.data.length} adaptive sampling points`);
      setShowAdaptiveForm(false);
      resetAdaptiveForm();
      fetchSamplingPoints();
    } catch (error) {
      console.error('Error generating adaptive sampling points:', error);
      showAlert('danger', 'Failed to generate adaptive sampling points');
    } finally {
      setLoading(false);
    }
  };
  
  // Map event handlers
  const handleMapClick = (e) => {
    if (drawingArea) {
      setDrawnPolygon([...drawnPolygon, [e.latlng.lat, e.latlng.lng]]);
    } else if (showSamplingPointForm) {
      setSamplingPointForm({
        ...samplingPointForm,
        position: { lat: e.latlng.lat, lon: e.latlng.lng }
      });
    }
  };
  
  const handleMapReady = (map) => {
    setMapRef(map);
    
    // Add click handler to map
    map.on('click', handleMapClick);
    
    // Cleanup on unmount
    return () => {
      map.off('click', handleMapClick);
    };
  };
  
  // Form reset functions
  const resetDroneForm = () => {
    setDroneForm({
      model: '',
      battery_capacity_mah: 5000,
      max_flight_time_minutes: 30,
      max_speed_mps: 10,
      max_payload_grams: 500,
      camera_resolution: '4K',
      sensors: [],
      communication_range_meters: 2000
    });
  };
  
  const resetSamplingPointForm = () => {
    setSamplingPointForm({
      position: { lat: 0, lon: 0 },
      priority: 5,
      sample_type: 'soil',
      estimated_time_seconds: 300
    });
  };
  
  const resetAreaForm = () => {
    setAreaForm({
      name: '',
      geometry: null,
      priority: 5
    });
  };
  
  const resetSwarmForm = () => {
    setSwarmForm({
      area_id: '',
      num_drones: 3
    });
  };
  
  const resetAdaptiveForm = () => {
    setAdaptiveForm({
      area_id: '',
      initial_results: {}
    });
  };
  
  // Helper functions
  const showAlert = (variant, message) => {
    setAlert({ show: true, variant, message });
    setTimeout(() => {
      setAlert({ show: false, variant: 'info', message: '' });
    }, 5000);
  };
  
  const getStatusBadge = (status) => {
    let variant = 'secondary';
    
    switch (status) {
      case 'idle':
      case 'pending':
        variant = 'secondary';
        break;
      case 'mission':
      case 'in_progress':
      case 'planned':
        variant = 'primary';
        break;
      case 'completed':
        variant = 'success';
        break;
      case 'returning':
        variant = 'info';
        break;
      case 'charging':
        variant = 'warning';
        break;
      case 'error':
      case 'failed':
        variant = 'danger';
        break;
      default:
        variant = 'secondary';
    }
    
    return <Badge bg={variant}>{status.toUpperCase()}</Badge>;
  };
  
  const getBatteryStatusClass = (percentage) => {
    if (percentage >= 70) return 'battery-high';
    if (percentage >= 30) return 'battery-medium';
    return 'battery-low';
  };
  
  // Render functions for tabs
  const renderDronesTab = () => {
    return (
      <div className="tab-content">
        <div className="d-flex justify-content-between mb-3">
          <h3>Drone Fleet Management</h3>
          <Button variant="primary" onClick={() => setShowDroneForm(true)}>
            <FontAwesomeIcon icon={faPlus} /> Register Drone
          </Button>
        </div>
        
        {drones.length === 0 ? (
          <Alert variant="info">
            No drones registered. Add a drone to get started.
          </Alert>
        ) : (
          <Row>
            {drones.map(drone => {
              const state = droneStates[drone.drone_id] || {
                status: 'unknown',
                battery_percentage: 0,
                position: { lat: 0, lon: 0 }
              };
              
              return (
                <Col md={4} key={drone.drone_id} className="mb-4">
                  <Card className={`drone-card ${state.status === 'error' ? 'border-danger' : ''}`}>
                    <Card.Header className="d-flex justify-content-between align-items-center">
                      <div>
                        <FontAwesomeIcon icon={faDrone} className="me-2" />
                        {drone.model}
                      </div>
                      {getStatusBadge(state.status)}
                    </Card.Header>
                    <Card.Body>
                      <div className="d-flex justify-content-between mb-2">
                        <div>ID:</div>
                        <div className="text-truncate ms-2">{drone.drone_id.substring(0, 8)}...</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Battery:</div>
                        <div className={getBatteryStatusClass(state.battery_percentage)}>
                          {state.battery_percentage}%
                        </div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Max Flight Time:</div>
                        <div>{drone.max_flight_time_minutes} min</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Max Speed:</div>
                        <div>{drone.max_speed_mps} m/s</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Payload:</div>
                        <div>{drone.max_payload_grams} g</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Camera:</div>
                        <div>{drone.camera_resolution}</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Sensors:</div>
                        <div>{drone.sensors.join(', ')}</div>
                      </div>
                      <div className="d-flex justify-content-between mb-2">
                        <div>Comm Range:</div>
                        <div>{drone.communication_range_meters} m</div>
                      </div>
                      {state.current_mission_id && (
                        <div className="d-flex justify-content-between mb-2">
                          <div>Mission:</div>
                          <div className="text-truncate ms-2">{state.current_mission_id.substring(0, 8)}...</div>
                        </div>
                      )}
                    </Card.Body>
                    <Card.Footer>
                      <div className="d-flex justify-content-between">
                        <Button 
                          variant="outline-primary" 
                          size="sm"
                          onClick={() => {
                            setSelectedDrone(drone);
                            setMapCenter([state.position.lat, state.position.lon]);
                            setMapZoom(15);
                          }}
                        >
                          <FontAwesomeIcon icon={faMapMarkerAlt} /> Locate
                        </Button>
                        <Button 
                          variant="outline-success" 
                          size="sm"
                          disabled={state.status !== 'idle'}
                          onClick={() => {
                            setSelectedDrone(drone);
                            setActiveTab('missions');
                          }}
                        >
                          <FontAwesomeIcon icon={faRoute} /> Assign Mission
                        </Button>
                      </div>
                    </Card.Footer>
                  </Card>
                </Col>
              );
            })}
          </Row>
        )}
        
        {/* Drone Registration Modal */}
        <Modal show={showDroneForm} onHide={() => setShowDroneForm(false)} size="lg">
          <Modal.Header closeButton>
            <Modal.Title>Register New Drone</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Model</Form.Label>
                <Form.Control 
                  type="text" 
                  value={droneForm.model} 
                  onChange={(e) => setDroneForm({...droneForm, model: e.target.value})}
                  placeholder="Enter drone model"
                />
              </Form.Group>
              
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Battery Capacity (mAh)</Form.Label>
                    <Form.Control 
                      type="number" 
                      value={droneForm.battery_capacity_mah} 
                      onChange={(e) => setDroneForm({...droneForm, battery_capacity_mah: parseInt(e.target.value)})}
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Max Flight Time (minutes)</Form.Label>
                    <Form.Control 
                      type="number" 
                      value={droneForm.max_flight_time_minutes} 
                      onChange={(e) => setDroneForm({...droneForm, max_flight_time_minutes: parseInt(e.target.value)})}
                    />
                  </Form.Group>
                </Col>
              </Row>
              
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Max Speed (m/s)</Form.Label>
                    <Form.Control 
                      type="number" 
                      value={droneForm.max_speed_mps} 
                      onChange={(e) => setDroneForm({...droneForm, max_speed_mps: parseFloat(e.target.value)})}
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Max Payload (grams)</Form.Label>
                    <Form.Control 
                      type="number" 
                      value={droneForm.max_payload_grams} 
                      onChange={(e) => setDroneForm({...droneForm, max_payload_grams: parseInt(e.target.value)})}
                    />
                  </Form.Group>
                </Col>
              </Row>
              
              <Form.Group className="mb-3">
                <Form.Label>Camera Resolution</Form.Label>
                <Form.Control 
                  type="text" 
                  value={droneForm.camera_resolution} 
                  onChange={(e) => setDroneForm({...droneForm, camera_resolution: e.target.value})}
                  placeholder="e.g., 4K, 1080p"
                />
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Sensors (comma-separated)</Form.Label>
                <Form.Control 
                  type="text" 
                  value={droneForm.sensors.join(', ')} 
                  onChange={(e) => setDroneForm({...droneForm, sensors: e.target.value.split(',').map(s => s.trim())})}
                  placeholder="e.g., GPS, IMU, Magnetometer"
                />
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Communication Range (meters)</Form.Label>
                <Form.Control 
                  type="number" 
                  value={droneForm.communication_range_meters} 
                  onChange={(e) => setDroneForm({...droneForm, communication_range_meters: parseInt(e.target.value)})}
                />
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowDroneForm(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={registerDrone} disabled={loading}>
              {loading ? <Spinner animation="border" size="sm" /> : 'Register Drone'}
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    );
  };
  
  const renderSamplingPointsTab = () => {
    return (
      <div className="tab-content">
        <div className="d-flex justify-content-between mb-3">
          <h3>Sampling Points</h3>
          <Button variant="primary" onClick={() => {
            resetSamplingPointForm();
            setShowSamplingPointForm(true);
          }}>
            <FontAwesomeIcon icon={faPlus} /> Add Sampling Point
          </Button>
        </div>
        
        {samplingPoints.length === 0 ? (
          <Alert variant="info">
            No sampling points defined. Add sampling points or generate them from exploration areas.
          </Alert>
        ) : (
          <Table striped bordered hover responsive>
            <thead>
              <tr>
                <th>Select</th>
                <th>ID</th>
                <th>Position</th>
                <th>Type</th>
                <th>Priority</th>
                <th>Est. Time</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {samplingPoints.map(point => (
                <tr key={point.point_id} className={point.status === 'completed' ? 'table-success' : ''}>
                  <td>
                    <Form.Check 
                      type="checkbox" 
                      disabled={point.status !== 'pending'}
                      checked={selectedSamplingPoints.includes(point.point_id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedSamplingPoints([...selectedSamplingPoints, point.point_id]);
                        } else {
                          setSelectedSamplingPoints(selectedSamplingPoints.filter(id => id !== point.point_id));
                        }
                      }}
                    />
                  </td>
                  <td>{point.point_id.substring(0, 8)}...</td>
                  <td>
                    {point.position.lat.toFixed(6)}, {point.position.lon.toFixed(6)}
                  </td>
                  <td>{point.sample_type}</td>
                  <td>
                    <Badge bg={
                      point.priority >= 8 ? 'danger' :
                      point.priority >= 5 ? 'warning' :
                      'info'
                    }>
                      {point.priority}
                    </Badge>
                  </td>
                  <td>{Math.floor(point.estimated_time_seconds / 60)}:{(point.estimated_time_seconds % 60).toString().padStart(2, '0')}</td>
                  <td>{getStatusBadge(point.status)}</td>
                  <td>
                    <Button 
                      variant="outline-primary" 
                      size="sm"
                      onClick={() => {
                        setMapCenter([point.position.lat, point.position.lon]);
                        setMapZoom(18);
                      }}
                    >
                      <FontAwesomeIcon icon={faMapMarkerAlt} /> Locate
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
        
        {selectedSamplingPoints.length > 0 && (
          <div className="d-flex justify-content-between mb-3">
            <div>
              <Badge bg="primary">{selectedSamplingPoints.length} points selected</Badge>
            </div>
            <div>
              <Button 
                variant="success" 
                onClick={() => {
                  if (!selectedDrone) {
                    showAlert('warning', 'Please select a drone first');
                    setActiveTab('drones');
                  } else {
                    createMission();
                  }
                }}
              >
                <FontAwesomeIcon icon={faRoute} /> Create Mission
              </Button>
              <Button 
                variant="secondary" 
                className="ms-2"
                onClick={() => setSelectedSamplingPoints([])}
              >
                Clear Selection
              </Button>
            </div>
          </div>
        )}
        
        {/* Sampling Point Form Modal */}
        <Modal show={showSamplingPointForm} onHide={() => setShowSamplingPointForm(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Add Sampling Point</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p>Click on the map to set the sampling point location or enter coordinates manually.</p>
            
            <Form>
              <Row>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Latitude</Form.Label>
                    <Form.Control 
                      type="number" 
                      step="0.000001"
                      value={samplingPointForm.position.lat} 
                      onChange={(e) => setSamplingPointForm({
                        ...samplingPointForm, 
                        position: {
                          ...samplingPointForm.position,
                          lat: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </Form.Group>
                </Col>
                <Col md={6}>
                  <Form.Group className="mb-3">
                    <Form.Label>Longitude</Form.Label>
                    <Form.Control 
                      type="number" 
                      step="0.000001"
                      value={samplingPointForm.position.lon} 
                      onChange={(e) => setSamplingPointForm({
                        ...samplingPointForm, 
                        position: {
                          ...samplingPointForm.position,
                          lon: parseFloat(e.target.value)
                        }
                      })}
                    />
                  </Form.Group>
                </Col>
              </Row>
              
              <Form.Group className="mb-3">
                <Form.Label>Sample Type</Form.Label>
                <Form.Select 
                  value={samplingPointForm.sample_type} 
                  onChange={(e) => setSamplingPointForm({...samplingPointForm, sample_type: e.target.value})}
                >
                  <option value="soil">Soil</option>
                  <option value="rock">Rock</option>
                  <option value="water">Water</option>
                  <option value="vegetation">Vegetation</option>
                  <option value="air">Air</option>
                </Form.Select>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Priority (1-10)</Form.Label>
                <Form.Control 
                  type="range" 
                  min="1" 
                  max="10" 
                  value={samplingPointForm.priority} 
                  onChange={(e) => setSamplingPointForm({...samplingPointForm, priority: parseInt(e.target.value)})}
                />
                <div className="d-flex justify-content-between">
                  <span>Low</span>
                  <span>Medium</span>
                  <span>High</span>
                </div>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Estimated Time (seconds)</Form.Label>
                <Form.Control 
                  type="number" 
                  value={samplingPointForm.estimated_time_seconds} 
                  onChange={(e) => setSamplingPointForm({...samplingPointForm, estimated_time_seconds: parseInt(e.target.value)})}
                />
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowSamplingPointForm(false)}>
              Cancel
            </Button>
            <Button variant="primary" onClick={addSamplingPoint} disabled={loading}>
              {loading ? <Spinner animation="border" size="sm" /> : 'Add Sampling Point'}
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    );
  };
  
  const renderMissionsTab = () => {
    return (
      <div className="tab-content">
        <div className="d-flex justify-content-between mb-3">
          <h3>Mission Planning</h3>
          <div>
            <Button 
              variant="primary" 
              onClick={() => {
                setActiveTab('sampling-points');
                setSelectedSamplingPoints([]);
              }}
            >
              <FontAwesomeIcon icon={faPlus} /> Create Mission
            </Button>
            <Button 
              variant="outline-primary" 
              className="ms-2"
              onClick={optimizeDroneAssignments}
              disabled={loading}
            >
              <FontAwesomeIcon icon={faSync} /> Optimize Assignments
            </Button>
          </div>
        </div>
        
        {missions.length === 0 ? (
          <Alert variant="info">
            No missions planned. Create a mission by selecting sampling points and a drone.
          </Alert>
        ) : (
          <Table striped bordered hover responsive>
            <thead>
              <tr>
                <th>ID</th>
                <th>Drone</th>
                <th>Points</th>
                <th>Est. Duration</th>
                <th>Battery Req.</th>
                <th>Status</th>
                <th>Start Time</th>
                <th>End Time</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {missions.map(mission => (
                <tr key={mission.mission_id}>
                  <td>{mission.mission_id.substring(0, 8)}...</td>
                  <td>{mission.drone_id.substring(0, 8)}...</td>
                  <td>{mission.sampling_points.length}</td>
                  <td>{mission.estimated_duration_minutes.toFixed(1)} min</td>
                  <td>
                    <div className={getBatteryStatusClass(mission.battery_requirement_percentage)}>
                      {mission.battery_requirement_percentage.toFixed(1)}%
                    </div>
                  </td>
                  <td>{getStatusBadge(mission.status)}</td>
                  <td>{mission.start_time ? new Date(mission.start_time).toLocaleString() : '-'}</td>
                  <td>{mission.end_time ? new Date(mission.end_time).toLocaleString() : '-'}</td>
                  <td>
                    <div className="d-flex">
                      <Button 
                        variant="outline-primary" 
                        size="sm"
                        className="me-1"
                        onClick={() => {
                          setSelectedMission(mission);
                          
                          // Center map on first waypoint if available
                          if (mission.waypoints && mission.waypoints.length > 0) {
                            setMapCenter([mission.waypoints[0].lat, mission.waypoints[0].lon]);
                            setMapZoom(15);
                          }
                        }}
                      >
                        <FontAwesomeIcon icon={faMapMarkerAlt} />
                      </Button>
                      
                      {mission.status === 'planned' && (
                        <Button 
                          variant="outline-success" 
                          size="sm"
                          className="me-1"
                          onClick={() => startMission(mission.mission_id)}
                          disabled={loading}
                        >
                          <FontAwesomeIcon icon={faPlay} />
                        </Button>
                      )}
                      
                      {mission.status === 'in_progress' && (
                        <>
                          <Button 
                            variant="outline-success" 
                            size="sm"
                            className="me-1"
                            onClick={() => completeMission(mission.mission_id, true)}
                            disabled={loading}
                          >
                            <FontAwesomeIcon icon={faCheck} />
                          </Button>
                          <Button 
                            variant="outline-danger" 
                            size="sm"
                            onClick={() => completeMission(mission.mission_id, false)}
                            disabled={loading}
                          >
                            <FontAwesomeIcon icon={faTimes} />
                          </Button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    );
  };
  
  const renderExplorationAreasTab = () => {
    return (
      <div className="tab-content">
        <div className="d-flex justify-content-between mb-3">
          <h3>Exploration Areas</h3>
          <Button variant="primary" onClick={() => {
            resetAreaForm();
            setDrawnPolygon([]);
            setDrawingArea(true);
            setShowAreaForm(true);
          }}>
            <FontAwesomeIcon icon={faDrawPolygon} /> Define Area
          </Button>
        </div>
        
        {explorationAreas.length === 0 ? (
          <Alert variant="info">
            No exploration areas defined. Define an area by drawing a polygon on the map.
          </Alert>
        ) : (
          <Row>
            {explorationAreas.map(area => (
              <Col md={4} key={area.area_id} className="mb-4">
                <Card>
                  <Card.Header>
                    <div className="d-flex justify-content-between align-items-center">
                      <h5 className="mb-0">{area.name}</h5>
                      {getStatusBadge(area.exploration_status)}
                    </div>
                  </Card.Header>
                  <Card.Body>
                    <div className="d-flex justify-content-between mb-2">
                      <div>ID:</div>
                      <div className="text-truncate ms-2">{area.area_id.substring(0, 8)}...</div>
                    </div>
                    <div className="d-flex justify-content-between mb-2">
                      <div>Priority:</div>
                      <Badge bg={
                        area.priority >= 8 ? 'danger' :
                        area.priority >= 5 ? 'warning' :
                        'info'
                      }>
                        {area.priority}
                      </Badge>
                    </div>
                    <div className="d-flex justify-content-between mb-2">
                      <div>Completion:</div>
                      <div>{area.completion_percentage.toFixed(1)}%</div>
                    </div>
                    <div className="progress mb-3">
                      <div 
                        className="progress-bar" 
                        role="progressbar" 
                        style={{ width: `${area.completion_percentage}%` }}
                        aria-valuenow={area.completion_percentage} 
                        aria-valuemin="0" 
                        aria-valuemax="100"
                      ></div>
                    </div>
                  </Card.Body>
                  <Card.Footer>
                    <div className="d-flex justify-content-between">
                      <Button 
                        variant="outline-primary" 
                        size="sm"
                        onClick={() => {
                          setSelectedArea(area);
                          
                          // Center map on area
                          if (area.geometry && area.geometry.coordinates && area.geometry.coordinates[0]) {
                            const coords = area.geometry.coordinates[0];
                            const lats = coords.map(c => c[1]);
                            const lons = coords.map(c => c[0]);
                            const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
                            const centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;
                            setMapCenter([centerLat, centerLon]);
                            setMapZoom(13);
                          }
                        }}
                      >
                        <FontAwesomeIcon icon={faMapMarkerAlt} /> View
                      </Button>
                      <Button 
                        variant="outline-success" 
                        size="sm"
                        onClick={() => generateSamplingPoints(area.area_id)}
                        disabled={loading}
                      >
                        <FontAwesomeIcon icon={faPlus} /> Generate Points
                      </Button>
                    </div>
                  </Card.Footer>
                </Card>
              </Col>
            ))}
          </Row>
        )}
        
        {/* Area Form Modal */}
        <Modal show={showAreaForm} onHide={() => {
          setShowAreaForm(false);
          setDrawingArea(false);
        }}>
          <Modal.Header closeButton>
            <Modal.Title>Define Exploration Area</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <p>
              {drawingArea ? 
                'Click on the map to draw a polygon. Click at least 3 points to define the area.' :
                'Please complete the form below to define the exploration area.'
              }
            </p>
            
            {drawingArea && (
              <Alert variant="info">
                <FontAwesomeIcon icon={faDrawPolygon} /> Drawing mode is active. 
                {drawnPolygon.length > 0 && ` ${drawnPolygon.length} points drawn.`}
              </Alert>
            )}
            
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Area Name</Form.Label>
                <Form.Control 
                  type="text" 
                  value={areaForm.name} 
                  onChange={(e) => setAreaForm({...areaForm, name: e.target.value})}
                  placeholder="Enter a descriptive name"
                />
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Priority (1-10)</Form.Label>
                <Form.Control 
                  type="range" 
                  min="1" 
                  max="10" 
                  value={areaForm.priority} 
                  onChange={(e) => setAreaForm({...areaForm, priority: parseInt(e.target.value)})}
                />
                <div className="d-flex justify-content-between">
                  <span>Low</span>
                  <span>Medium</span>
                  <span>High</span>
                </div>
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button 
              variant="secondary" 
              onClick={() => {
                setShowAreaForm(false);
                setDrawingArea(false);
                setDrawnPolygon([]);
              }}
            >
              Cancel
            </Button>
            {drawingArea && drawnPolygon.length >= 3 ? (
              <Button 
                variant="success" 
                onClick={() => {
                  setDrawingArea(false);
                }}
              >
                Complete Polygon
              </Button>
            ) : !drawingArea ? (
              <Button 
                variant="primary" 
                onClick={addExplorationArea} 
                disabled={loading || !areaForm.name || drawnPolygon.length < 3}
              >
                {loading ? <Spinner animation="border" size="sm" /> : 'Add Exploration Area'}
              </Button>
            ) : null}
          </Modal.Footer>
        </Modal>
      </div>
    );
  };
  
  const renderAdvancedPlanningTab = () => {
    return (
      <div className="tab-content">
        <div className="mb-4">
          <h3>Advanced Mission Planning</h3>
          <p>
            Use advanced algorithms to optimize exploration missions and adapt to field conditions.
          </p>
        </div>
        
        <Row>
          <Col md={6} className="mb-4">
            <Card>
              <Card.Header>
                <h5 className="mb-0">
                  <FontAwesomeIcon icon={faRobot} className="me-2" />
                  Swarm Mission Planning
                </h5>
              </Card.Header>
              <Card.Body>
                <p>
                  Generate coordinated missions for multiple drones to efficiently explore an area.
                  Drones will work together to cover the area in the shortest time possible.
                </p>
                <Button 
                  variant="primary" 
                  onClick={() => {
                    resetSwarmForm();
                    setShowSwarmForm(true);
                  }}
                >
                  Generate Swarm Mission
                </Button>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6} className="mb-4">
            <Card>
              <Card.Header>
                <h5 className="mb-0">
                  <FontAwesomeIcon icon={faChartLine} className="me-2" />
                  Adaptive Sampling
                </h5>
              </Card.Header>
              <Card.Body>
                <p>
                  Generate new sampling points based on initial results to focus on high-value areas.
                  This adaptive approach optimizes resource usage and improves exploration outcomes.
                </p>
                <Button 
                  variant="primary" 
                  onClick={() => {
                    resetAdaptiveForm();
                    setShowAdaptiveForm(true);
                  }}
                >
                  Generate Adaptive Sampling
                </Button>
              </Card.Body>
            </Card>
          </Col>
          
          <Col md={6} className="mb-4">
            <Card>
              <Card.Header>
                <h5 className="mb-0">
                  <FontAwesomeIcon icon={faLayerGroup} className="me-2" />
                  Optimize Drone Assignments
                </h5>
              </Card.Header>
              <Card.Body>
                <p>
                  Automatically assign drones to sampling points using the Hungarian algorithm to minimize
                  total distance traveled and optimize battery usage.
                </p>
                <Button 
                  variant="primary" 
                  onClick={optimizeDroneAssignments}
                  disabled={loading}
                >
                  Optimize Assignments
                </Button>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        {/* Swarm Mission Modal */}
        <Modal show={showSwarmForm} onHide={() => setShowSwarmForm(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Generate Swarm Mission</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Exploration Area</Form.Label>
                <Form.Select 
                  value={swarmForm.area_id} 
                  onChange={(e) => setSwarmForm({...swarmForm, area_id: e.target.value})}
                >
                  <option value="">Select an area</option>
                  {explorationAreas.map(area => (
                    <option key={area.area_id} value={area.area_id}>
                      {area.name}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Number of Drones</Form.Label>
                <Form.Control 
                  type="number" 
                  min="2"
                  max="10"
                  value={swarmForm.num_drones} 
                  onChange={(e) => setSwarmForm({...swarmForm, num_drones: parseInt(e.target.value)})}
                />
                <Form.Text className="text-muted">
                  Recommended: 2-5 drones for optimal coordination
                </Form.Text>
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowSwarmForm(false)}>
              Cancel
            </Button>
            <Button 
              variant="primary" 
              onClick={generateSwarmMission} 
              disabled={loading || !swarmForm.area_id}
            >
              {loading ? <Spinner animation="border" size="sm" /> : 'Generate Swarm Mission'}
            </Button>
          </Modal.Footer>
        </Modal>
        
        {/* Adaptive Sampling Modal */}
        <Modal show={showAdaptiveForm} onHide={() => setShowAdaptiveForm(false)}>
          <Modal.Header closeButton>
            <Modal.Title>Generate Adaptive Sampling</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form>
              <Form.Group className="mb-3">
                <Form.Label>Exploration Area</Form.Label>
                <Form.Select 
                  value={adaptiveForm.area_id} 
                  onChange={(e) => setAdaptiveForm({...adaptiveForm, area_id: e.target.value})}
                >
                  <option value="">Select an area</option>
                  {explorationAreas.map(area => (
                    <option key={area.area_id} value={area.area_id}>
                      {area.name}
                    </option>
                  ))}
                </Form.Select>
              </Form.Group>
              
              <Form.Group className="mb-3">
                <Form.Label>Initial Results</Form.Label>
                <div className="mb-2">
                  <small className="text-muted">
                    Enter sampling point IDs and their result values (0.0-1.0) to guide adaptive sampling.
                  </small>
                </div>
                
                {Object.entries(adaptiveForm.initial_results).map(([pointId, value], index) => (
                  <Row key={index} className="mb-2">
                    <Col md={8}>
                      <Form.Control 
                        type="text" 
                        value={pointId} 
                        onChange={(e) => {
                          const newResults = {...adaptiveForm.initial_results};
                          const oldValue = newResults[pointId];
                          delete newResults[pointId];
                          newResults[e.target.value] = oldValue;
                          setAdaptiveForm({...adaptiveForm, initial_results: newResults});
                        }}
                        placeholder="Sampling Point ID"
                      />
                    </Col>
                    <Col md={3}>
                      <Form.Control 
                        type="number" 
                        min="0"
                        max="1"
                        step="0.1"
                        value={value} 
                        onChange={(e) => {
                          const newResults = {...adaptiveForm.initial_results};
                          newResults[pointId] = parseFloat(e.target.value);
                          setAdaptiveForm({...adaptiveForm, initial_results: newResults});
                        }}
                      />
                    </Col>
                    <Col md={1}>
                      <Button 
                        variant="outline-danger" 
                        size="sm"
                        onClick={() => {
                          const newResults = {...adaptiveForm.initial_results};
                          delete newResults[pointId];
                          setAdaptiveForm({...adaptiveForm, initial_results: newResults});
                        }}
                      >
                        <FontAwesomeIcon icon={faTimes} />
                      </Button>
                    </Col>
                  </Row>
                ))}
                
                <Button 
                  variant="outline-primary" 
                  size="sm"
                  onClick={() => {
                    const newResults = {...adaptiveForm.initial_results};
                    newResults[`point_${Object.keys(newResults).length + 1}`] = 0.5;
                    setAdaptiveForm({...adaptiveForm, initial_results: newResults});
                  }}
                >
                  <FontAwesomeIcon icon={faPlus} /> Add Result
                </Button>
              </Form.Group>
            </Form>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowAdaptiveForm(false)}>
              Cancel
            </Button>
            <Button 
              variant="primary" 
              onClick={generateAdaptiveSampling} 
              disabled={loading || !adaptiveForm.area_id || Object.keys(adaptiveForm.initial_results).length === 0}
            >
              {loading ? <Spinner animation="border" size="sm" /> : 'Generate Adaptive Sampling'}
            </Button>
          </Modal.Footer>
        </Modal>
      </div>
    );
  };
  
  const renderMapTab = () => {
    return (
      <div className="tab-content">
        <div className="d-flex justify-content-between mb-3">
          <h3>Exploration Map</h3>
          <div>
            {drawingArea && (
              <Button 
                variant="warning" 
                className="me-2"
                onClick={() => {
                  setDrawnPolygon([]);
                }}
              >
                Clear Drawing
              </Button>
            )}
            <Button 
              variant="primary" 
              onClick={() => {
                fetchDrones();
                fetchSamplingPoints();
                fetchMissions();
                fetchExplorationAreas();
              }}
              disabled={loading}
            >
              <FontAwesomeIcon icon={faSync} /> Refresh
            </Button>
          </div>
        </div>
        
        <div className="map-container">
          <MapContainer 
            center={mapCenter} 
            zoom={mapZoom} 
            style={{ height: '600px', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            
            <MapComponent onMapReady={handleMapReady} />
            
            {/* Render drones */}
            {Object.values(droneStates).map(state => (
              <Marker 
                key={state.drone_id}
                position={[state.position.lat, state.position.lon]}
                icon={L.divIcon({
                  html: `<div class="drone-marker ${state.status}"><i class="fa fa-drone"></i></div>`,
                  className: 'drone-marker-container',
                  iconSize: [30, 30]
                })}
              >
                <Popup>
                  <div>
                    <h6>Drone: {drones.find(d => d.drone_id === state.drone_id)?.model || 'Unknown'}</h6>
                    <p>Status: {state.status.toUpperCase()}</p>
                    <p>Battery: {state.battery_percentage}%</p>
                    {state.current_mission_id && (
                      <p>Mission: {state.current_mission_id.substring(0, 8)}...</p>
                    )}
                  </div>
                </Popup>
              </Marker>
            ))}
            
            {/* Render sampling points */}
            {samplingPoints.map(point => (
              <Marker 
                key={point.point_id}
                position={[point.position.lat, point.position.lon]}
                icon={L.divIcon({
                  html: `<div class="sampling-point-marker ${point.status}" data-priority="${point.priority}"></div>`,
                  className: 'sampling-point-marker-container',
                  iconSize: [20, 20]
                })}
              >
                <Popup>
                  <div>
                    <h6>Sampling Point: {point.point_id.substring(0, 8)}...</h6>
                    <p>Type: {point.sample_type}</p>
                    <p>Priority: {point.priority}</p>
                    <p>Status: {point.status.toUpperCase()}</p>
                    {point.assigned_drone_id && (
                      <p>Assigned to: {point.assigned_drone_id.substring(0, 8)}...</p>
                    )}
                  </div>
                </Popup>
              </Marker>
            ))}
            
            {/* Render mission routes */}
            {selectedMission && selectedMission.waypoints && selectedMission.waypoints.length > 0 && (
              <Polygon 
                positions={selectedMission.waypoints.map(wp => [wp.lat, wp.lon])}
                color="#0066cc"
                weight={3}
                dashArray="5, 5"
              />
            )}
            
            {/* Render exploration areas */}
            {explorationAreas.map(area => {
              if (area.geometry && area.geometry.coordinates && area.geometry.coordinates[0]) {
                const positions = area.geometry.coordinates[0].map(coord => [coord[1], coord[0]]);
                return (
                  <Polygon 
                    key={area.area_id}
                    positions={positions}
                    color={
                      area.exploration_status === 'completed' ? '#28a745' :
                      area.exploration_status === 'in_progress' ? '#0066cc' :
                      '#6c757d'
                    }
                    fillOpacity={0.2}
                  >
                    <Popup>
                      <div>
                        <h6>{area.name}</h6>
                        <p>Status: {area.exploration_status.toUpperCase()}</p>
                        <p>Completion: {area.completion_percentage.toFixed(1)}%</p>
                        <p>Priority: {area.priority}</p>
                      </div>
                    </Popup>
                  </Polygon>
                );
              }
              return null;
            })}
            
            {/* Render drawn polygon */}
            {drawnPolygon.length > 0 && (
              <Polygon 
                positions={drawnPolygon}
                color="#ff6600"
                fillOpacity={0.2}
              />
            )}
          </MapContainer>
        </div>
        
        <div className="map-legend mt-3">
          <h5>Map Legend</h5>
          <div className="d-flex flex-wrap">
            <div className="legend-item">
              <div className="drone-marker idle"></div>
              <span>Idle Drone</span>
            </div>
            <div className="legend-item">
              <div className="drone-marker mission"></div>
              <span>Active Drone</span>
            </div>
            <div className="legend-item">
              <div className="sampling-point-marker pending" data-priority="5"></div>
              <span>Pending Sample</span>
            </div>
            <div className="legend-item">
              <div className="sampling-point-marker assigned" data-priority="5"></div>
              <span>Assigned Sample</span>
            </div>
            <div className="legend-item">
              <div className="sampling-point-marker completed" data-priority="5"></div>
              <span>Completed Sample</span>
            </div>
            <div className="legend-item">
              <div className="area-marker pending"></div>
              <span>Pending Area</span>
            </div>
            <div className="legend-item">
              <div className="area-marker in-progress"></div>
              <span>Active Area</span>
            </div>
            <div className="legend-item">
              <div className="area-marker completed"></div>
              <span>Completed Area</span>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <Container fluid className="autonomous-exploration-processor">
      <Row className="mb-4">
        <Col>
          <h2 className="page-title">
            <FontAwesomeIcon icon={faRobot} className="me-2" />
            Autonomous Exploration System
          </h2>
          <p className="page-description">
            Plan, coordinate, and monitor autonomous drone missions for mineral exploration.
          </p>
        </Col>
      </Row>
      
      {alert.show && (
        <Alert variant={alert.variant} onClose={() => setAlert({...alert, show: false})} dismissible>
          {alert.message}
        </Alert>
      )}
      
      <Row>
        <Col md={12}>
          <Tabs
            activeKey={activeTab}
            onSelect={(k) => setActiveTab(k)}
            className="mb-4"
          >
            <Tab eventKey="drones" title={<><FontAwesomeIcon icon={faDrone} /> Drones</>}>
              {renderDronesTab()}
            </Tab>
            <Tab eventKey="sampling-points" title={<><FontAwesomeIcon icon={faMapMarkerAlt} /> Sampling Points</>}>
              {renderSamplingPointsTab()}
            </Tab>
            <Tab eventKey="missions" title={<><FontAwesomeIcon icon={faRoute} /> Missions</>}>
              {renderMissionsTab()}
            </Tab>
            <Tab eventKey="exploration-areas" title={<><FontAwesomeIcon icon={faDrawPolygon} /> Exploration Areas</>}>
              {renderExplorationAreasTab()}
            </Tab>
            <Tab eventKey="advanced-planning" title={<><FontAwesomeIcon icon={faRobot} /> Advanced Planning</>}>
              {renderAdvancedPlanningTab()}
            </Tab>
            <Tab eventKey="map" title={<><FontAwesomeIcon icon={faMapMarkerAlt} /> Map</>}>
              {renderMapTab()}
            </Tab>
          </Tabs>
        </Col>
      </Row>
    </Container>
  );
};

export default AutonomousExplorationProcessor;
