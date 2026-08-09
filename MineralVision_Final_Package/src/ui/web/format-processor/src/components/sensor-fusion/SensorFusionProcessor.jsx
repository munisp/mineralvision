import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, ProgressBar, Alert, Tabs, Tab, Table } from 'react-bootstrap';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { API_BASE_URL } from '../../services/api';
import ProgressiveLoader from '../common/ProgressiveLoader';
import { MineralVision3DEngine } from '../visualization/MineralVision3DEngine';

const SensorFusionProcessor = () => {
  // State for uploaded sensor data
  const [sensorData, setSensorData] = useState([]);
  const [selectedSensorData, setSelectedSensorData] = useState([]);
  const [fusionResults, setFusionResults] = useState([]);
  const [selectedFusionResult, setSelectedFusionResult] = useState(null);
  
  // State for uploads
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState(null);
  
  // State for fusion
  const [fusionAlgorithm, setFusionAlgorithm] = useState('weighted_average');
  const [fusionParameters, setFusionParameters] = useState({});
  const [fusionName, setFusionName] = useState('');
  const [fusionInProgress, setFusionInProgress] = useState(false);
  const [fusionError, setFusionError] = useState(null);
  
  // State for visualization
  const [visualizationData, setVisualizationData] = useState(null);
  const [visualizationLoading, setVisualizationLoading] = useState(false);
  
  // Tabs
  const [activeTab, setActiveTab] = useState('upload');
  
  // Fetch sensor data and fusion results on component mount
  useEffect(() => {
    fetchSensorData();
    fetchFusionResults();
  }, []);
  
  // Fetch sensor data
  const fetchSensorData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/sensor-fusion/data`);
      setSensorData(response.data);
    } catch (error) {
      console.error('Error fetching sensor data:', error);
    }
  };
  
  // Fetch fusion results
  const fetchFusionResults = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/sensor-fusion/fusion`);
      setFusionResults(response.data);
    } catch (error) {
      console.error('Error fetching fusion results:', error);
    }
  };
  
  // Handle file drop for sensor data upload
  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;
    
    setUploading(true);
    setUploadProgress(0);
    setUploadError(null);
    
    // Get file and sensor type
    const file = acceptedFiles[0];
    const sensorType = determineSensorType(file.name);
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Upload file
      const response = await axios.post(
        `${API_BASE_URL}/sensor-fusion/upload/${sensorType}`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
          },
        }
      );
      
      // Update sensor data list
      setSensorData([...sensorData, response.data]);
      setUploading(false);
      setUploadProgress(100);
      
      // Switch to fusion tab
      setActiveTab('fusion');
    } catch (error) {
      setUploadError(error.response?.data?.detail || 'Error uploading file');
      setUploading(false);
    }
  };
  
  // Determine sensor type from file name
  const determineSensorType = (fileName) => {
    const lowerFileName = fileName.toLowerCase();
    
    if (lowerFileName.includes('hyper') || lowerFileName.includes('spectral') || lowerFileName.endsWith('.hdr')) {
      return 'hyperspectral';
    } else if (lowerFileName.includes('lidar') || lowerFileName.includes('las') || lowerFileName.endsWith('.las') || lowerFileName.endsWith('.laz')) {
      return 'lidar';
    } else if (lowerFileName.includes('mag') || lowerFileName.includes('magnetic') || lowerFileName.endsWith('.mag')) {
      return 'magnetometry';
    }
    
    // Default to hyperspectral if can't determine
    return 'hyperspectral';
  };
  
  // Dropzone configuration
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.tif', '.tiff', '.hdr'],
      'application/octet-stream': ['.las', '.laz', '.mag'],
      'text/csv': ['.csv', '.txt'],
    },
    multiple: false,
  });
  
  // Handle sensor data selection
  const handleSensorDataSelection = (dataId) => {
    if (selectedSensorData.includes(dataId)) {
      setSelectedSensorData(selectedSensorData.filter(id => id !== dataId));
    } else {
      setSelectedSensorData([...selectedSensorData, dataId]);
    }
  };
  
  // Handle fusion algorithm change
  const handleFusionAlgorithmChange = (e) => {
    setFusionAlgorithm(e.target.value);
    
    // Set default parameters based on algorithm
    if (e.target.value === 'weighted_average') {
      setFusionParameters({
        normalize_values: true,
        fill_missing: true,
        smooth_result: false,
        smooth_sigma: 1.0,
      });
    } else if (e.target.value === 'bayesian') {
      setFusionParameters({
        normalize_values: true,
        prior_variance: 1.0,
      });
    }
  };
  
  // Handle fusion parameter change
  const handleFusionParameterChange = (paramName, value) => {
    setFusionParameters({
      ...fusionParameters,
      [paramName]: value,
    });
  };
  
  // Handle fusion
  const handleFusion = async () => {
    if (selectedSensorData.length < 2) {
      setFusionError('At least two sensor data sources must be selected for fusion');
      return;
    }
    
    setFusionInProgress(true);
    setFusionError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/sensor-fusion/fuse`, {
        sensor_data_ids: selectedSensorData,
        algorithm: fusionAlgorithm,
        parameters: fusionParameters,
        name: fusionName || `Fusion_${new Date().toISOString()}`,
      });
      
      // Update fusion results
      setFusionResults([...fusionResults, response.data]);
      setFusionInProgress(false);
      
      // Select the new fusion result
      setSelectedFusionResult(response.data.result_id);
      
      // Switch to results tab
      setActiveTab('results');
    } catch (error) {
      setFusionError(error.response?.data?.detail || 'Error performing fusion');
      setFusionInProgress(false);
    }
  };
  
  // Handle fusion result selection
  const handleFusionResultSelection = (resultId) => {
    setSelectedFusionResult(resultId);
    loadVisualizationData(resultId);
  };
  
  // Load visualization data
  const loadVisualizationData = async (resultId) => {
    setVisualizationLoading(true);
    
    try {
      // Get fusion result data
      const response = await axios.get(`${API_BASE_URL}/sensor-fusion/export/${resultId}?format=json`);
      setVisualizationData(response.data);
      setVisualizationLoading(false);
    } catch (error) {
      console.error('Error loading visualization data:', error);
      setVisualizationLoading(false);
    }
  };
  
  // Export fusion result
  const handleExport = async (resultId, format) => {
    try {
      window.open(`${API_BASE_URL}/sensor-fusion/export/${resultId}?format=${format}`, '_blank');
    } catch (error) {
      console.error('Error exporting fusion result:', error);
    }
  };
  
  return (
    <Container fluid className="sensor-fusion-processor">
      <h2>Sensor Fusion Processor</h2>
      
      <Tabs activeKey={activeTab} onSelect={(k) => setActiveTab(k)} className="mb-4">
        <Tab eventKey="upload" title="Upload Data">
          <Card className="mb-4">
            <Card.Header>Upload Sensor Data</Card.Header>
            <Card.Body>
              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'active' : ''}`}
                style={{
                  border: '2px dashed #cccccc',
                  borderRadius: '4px',
                  padding: '20px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  backgroundColor: isDragActive ? '#f8f9fa' : 'white',
                }}
              >
                <input {...getInputProps()} />
                {isDragActive ? (
                  <p>Drop the files here...</p>
                ) : (
                  <p>Drag and drop sensor data files here, or click to select files</p>
                )}
                <p className="text-muted">
                  Supported formats: GeoTIFF, LAS/LAZ, CSV, HDR (ENVI), MAG
                </p>
              </div>
              
              {uploading && (
                <div className="mt-3">
                  <ProgressBar now={uploadProgress} label={`${uploadProgress}%`} />
                </div>
              )}
              
              {uploadError && (
                <Alert variant="danger" className="mt-3">
                  {uploadError}
                </Alert>
              )}
            </Card.Body>
          </Card>
          
          <Card>
            <Card.Header>Available Sensor Data</Card.Header>
            <Card.Body>
              {sensorData.length === 0 ? (
                <p>No sensor data available. Upload data to begin.</p>
              ) : (
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Type</th>
                      <th>File Name</th>
                      <th>Timestamp</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sensorData.map((data) => (
                      <tr key={data.data_id}>
                        <td>{data.data_id.substring(0, 8)}...</td>
                        <td>{data.sensor_type}</td>
                        <td>{data.file_name}</td>
                        <td>{new Date(data.timestamp).toLocaleString()}</td>
                        <td>
                          <Button
                            variant="outline-primary"
                            size="sm"
                            onClick={() => {
                              handleSensorDataSelection(data.data_id);
                              setActiveTab('fusion');
                            }}
                          >
                            Select for Fusion
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="fusion" title="Fusion">
          <Card className="mb-4">
            <Card.Header>Selected Sensor Data</Card.Header>
            <Card.Body>
              {selectedSensorData.length === 0 ? (
                <p>No sensor data selected. Select data from the Upload tab.</p>
              ) : (
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Type</th>
                      <th>File Name</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedSensorData.map((dataId) => {
                      const data = sensorData.find((d) => d.data_id === dataId);
                      if (!data) return null;
                      
                      return (
                        <tr key={data.data_id}>
                          <td>{data.data_id.substring(0, 8)}...</td>
                          <td>{data.sensor_type}</td>
                          <td>{data.file_name}</td>
                          <td>
                            <Button
                              variant="outline-danger"
                              size="sm"
                              onClick={() => handleSensorDataSelection(data.data_id)}
                            >
                              Remove
                            </Button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
          
          <Card>
            <Card.Header>Fusion Configuration</Card.Header>
            <Card.Body>
              <Form>
                <Form.Group className="mb-3">
                  <Form.Label>Fusion Name</Form.Label>
                  <Form.Control
                    type="text"
                    placeholder="Enter a name for this fusion"
                    value={fusionName}
                    onChange={(e) => setFusionName(e.target.value)}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Fusion Algorithm</Form.Label>
                  <Form.Select
                    value={fusionAlgorithm}
                    onChange={handleFusionAlgorithmChange}
                  >
                    <option value="weighted_average">Weighted Average</option>
                    <option value="bayesian">Bayesian Fusion</option>
                  </Form.Select>
                </Form.Group>
                
                <h5>Algorithm Parameters</h5>
                
                {fusionAlgorithm === 'weighted_average' && (
                  <>
                    <Form.Group className="mb-3">
                      <Form.Check
                        type="checkbox"
                        label="Normalize Values"
                        checked={fusionParameters.normalize_values || false}
                        onChange={(e) =>
                          handleFusionParameterChange('normalize_values', e.target.checked)
                        }
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Check
                        type="checkbox"
                        label="Fill Missing Values"
                        checked={fusionParameters.fill_missing || false}
                        onChange={(e) =>
                          handleFusionParameterChange('fill_missing', e.target.checked)
                        }
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Check
                        type="checkbox"
                        label="Smooth Result"
                        checked={fusionParameters.smooth_result || false}
                        onChange={(e) =>
                          handleFusionParameterChange('smooth_result', e.target.checked)
                        }
                      />
                    </Form.Group>
                    
                    {fusionParameters.smooth_result && (
                      <Form.Group className="mb-3">
                        <Form.Label>Smoothing Sigma</Form.Label>
                        <Form.Control
                          type="number"
                          min="0.1"
                          step="0.1"
                          value={fusionParameters.smooth_sigma || 1.0}
                          onChange={(e) =>
                            handleFusionParameterChange(
                              'smooth_sigma',
                              parseFloat(e.target.value)
                            )
                          }
                        />
                      </Form.Group>
                    )}
                  </>
                )}
                
                {fusionAlgorithm === 'bayesian' && (
                  <>
                    <Form.Group className="mb-3">
                      <Form.Check
                        type="checkbox"
                        label="Normalize Values"
                        checked={fusionParameters.normalize_values || false}
                        onChange={(e) =>
                          handleFusionParameterChange('normalize_values', e.target.checked)
                        }
                      />
                    </Form.Group>
                    
                    <Form.Group className="mb-3">
                      <Form.Label>Prior Variance</Form.Label>
                      <Form.Control
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={fusionParameters.prior_variance || 1.0}
                        onChange={(e) =>
                          handleFusionParameterChange(
                            'prior_variance',
                            parseFloat(e.target.value)
                          )
                        }
                      />
                    </Form.Group>
                  </>
                )}
                
                <Button
                  variant="primary"
                  onClick={handleFusion}
                  disabled={
                    selectedSensorData.length < 2 || fusionInProgress
                  }
                >
                  {fusionInProgress ? 'Processing...' : 'Perform Fusion'}
                </Button>
                
                {fusionError && (
                  <Alert variant="danger" className="mt-3">
                    {fusionError}
                  </Alert>
                )}
              </Form>
            </Card.Body>
          </Card>
        </Tab>
        
        <Tab eventKey="results" title="Results">
          <Row>
            <Col md={4}>
              <Card>
                <Card.Header>Fusion Results</Card.Header>
                <Card.Body>
                  {fusionResults.length === 0 ? (
                    <p>No fusion results available. Perform fusion to see results.</p>
                  ) : (
                    <Table striped bordered hover>
                      <thead>
                        <tr>
                          <th>Name</th>
                          <th>Algorithm</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {fusionResults.map((result) => (
                          <tr
                            key={result.result_id}
                            className={
                              selectedFusionResult === result.result_id
                                ? 'table-primary'
                                : ''
                            }
                          >
                            <td>{result.name || 'Unnamed'}</td>
                            <td>{result.algorithm}</td>
                            <td>
                              <Button
                                variant="outline-primary"
                                size="sm"
                                onClick={() =>
                                  handleFusionResultSelection(result.result_id)
                                }
                              >
                                View
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  )}
                </Card.Body>
              </Card>
              
              {selectedFusionResult && (
                <Card className="mt-3">
                  <Card.Header>Export Options</Card.Header>
                  <Card.Body>
                    <div className="d-grid gap-2">
                      <Button
                        variant="outline-primary"
                        onClick={() =>
                          handleExport(selectedFusionResult, 'geotiff')
                        }
                      >
                        Export as GeoTIFF
                      </Button>
                      <Button
                        variant="outline-primary"
                        onClick={() => handleExport(selectedFusionResult, 'csv')}
                      >
                        Export as CSV
                      </Button>
                      <Button
                        variant="outline-primary"
                        onClick={() =>
                          handleExport(selectedFusionResult, 'json')
                        }
                      >
                        Export as JSON
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              )}
            </Col>
            
            <Col md={8}>
              <Card>
                <Card.Header>Visualization</Card.Header>
                <Card.Body>
                  {!selectedFusionResult ? (
                    <p>Select a fusion result to visualize</p>
                  ) : visualizationLoading ? (
                    <ProgressiveLoader />
                  ) : visualizationData ? (
                    <div style={{ height: '500px' }}>
                      <MineralVision3DEngine
                        data={visualizationData}
                        options={{
                          showAxes: true,
                          showGrid: true,
                          colorMap: 'viridis',
                        }}
                      />
                    </div>
                  ) : (
                    <p>Error loading visualization data</p>
                  )}
                </Card.Body>
              </Card>
              
              {selectedFusionResult && visualizationData && (
                <Card className="mt-3">
                  <Card.Header>Metadata</Card.Header>
                  <Card.Body>
                    <pre
                      style={{
                        maxHeight: '200px',
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify(visualizationData.metadata, null, 2)}
                    </pre>
                  </Card.Body>
                </Card>
              )}
            </Col>
          </Row>
        </Tab>
      </Tabs>
    </Container>
  );
};

export default SensorFusionProcessor;
