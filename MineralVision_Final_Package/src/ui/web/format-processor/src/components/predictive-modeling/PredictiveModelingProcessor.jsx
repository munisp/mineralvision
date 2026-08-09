import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, Spinner, Table, ProgressBar } from 'react-bootstrap';
import { Line } from 'react-chartjs-2';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import { MapContainer, TileLayer, Marker, Popup, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import 'chart.js/auto';

import { API_BASE_URL } from '../../services/api';

const PredictiveModelingProcessor = () => {
  // State for model training
  const [trainingData, setTrainingData] = useState({
    geologicalData: null,
    geophysicalData: null,
    geochemicalData: null,
    remoteSensingData: null,
    historicalData: null,
    modelName: '',
    hiddenDims: [256, 128, 64],
    uncertaintyEstimation: true
  });
  
  // State for file uploads
  const [uploadedFiles, setUploadedFiles] = useState({
    geological: [],
    geophysical: [],
    geochemical: [],
    remoteSensing: [],
    historical: []
  });
  
  // State for prediction
  const [predictionInput, setPredictionInput] = useState({
    features: [],
    withUncertainty: true
  });
  const [predictionResult, setPredictionResult] = useState(null);
  
  // State for model list
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  
  // State for training job
  const [trainingJob, setTrainingJob] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  
  // State for UI
  const [activeTab, setActiveTab] = useState('train');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // State for visualization
  const [mapData, setMapData] = useState(null);
  const [predictionMap, setPredictionMap] = useState(null);
  
  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, []);
  
  // Poll training status if a job is active
  useEffect(() => {
    let interval;
    if (trainingJob && trainingJob.status !== 'completed' && trainingJob.status !== 'failed') {
      interval = setInterval(() => {
        fetchTrainingStatus(trainingJob.job_id);
      }, 5000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [trainingJob]);
  
  // Fetch available models
  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/predictive-modeling/models`);
      setModels(response.data);
      if (response.data.length > 0) {
        setSelectedModel(response.data[0]);
      }
    } catch (err) {
      setError('Failed to fetch models: ' + err.message);
    }
  };
  
  // Fetch training job status
  const fetchTrainingStatus = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/predictive-modeling/training-status/${jobId}`);
      setTrainingStatus(response.data);
      
      // Update training job if status changed
      if (response.data.status !== trainingJob.status) {
        setTrainingJob(response.data);
        
        // Show success message if completed
        if (response.data.status === 'completed') {
          setSuccess('Model training completed successfully!');
          fetchModels(); // Refresh model list
        }
        
        // Show error if failed
        if (response.data.status === 'failed') {
          setError('Model training failed: ' + (response.data.error || 'Unknown error'));
        }
      }
    } catch (err) {
      setError('Failed to fetch training status: ' + err.message);
    }
  };
  
  // Handle file upload
  const handleFileUpload = async (files, dataType) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', files[0]);
      formData.append('data_type', dataType);
      formData.append('description', `Uploaded ${dataType} data`);
      
      const response = await axios.post(`${API_BASE_URL}/predictive-modeling/upload-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      // Update uploaded files
      setUploadedFiles(prev => ({
        ...prev,
        [dataType]: [...prev[dataType], {
          name: files[0].name,
          path: response.data.file_path,
          size: files[0].size,
          type: files[0].type
        }]
      }));
      
      // Update training data
      setTrainingData(prev => ({
        ...prev,
        [`${dataType}Data`]: response.data.file_path
      }));
      
      setSuccess(`${dataType} data uploaded successfully!`);
    } catch (err) {
      setError('Failed to upload file: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle model training
  const handleTrainModel = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predictive-modeling/train`, {
        geological_data: trainingData.geologicalData,
        geophysical_data: trainingData.geophysicalData,
        geochemical_data: trainingData.geochemicalData,
        remote_sensing_data: trainingData.remoteSensingData,
        historical_data: trainingData.historicalData,
        model_name: trainingData.modelName,
        hidden_dims: trainingData.hiddenDims,
        uncertainty_estimation: trainingData.uncertaintyEstimation
      });
      
      setTrainingJob(response.data);
      setSuccess('Model training started successfully!');
    } catch (err) {
      setError('Failed to start model training: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle prediction
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predictive-modeling/predict`, {
        features: predictionInput.features,
        with_uncertainty: predictionInput.withUncertainty
      });
      
      setPredictionResult(response.data);
      setSuccess('Prediction completed successfully!');
      
      // Generate visualization data (simplified example)
      generateVisualization(response.data);
    } catch (err) {
      setError('Failed to make prediction: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Generate visualization data (simplified example)
  const generateVisualization = (prediction) => {
    // In a real implementation, this would use actual geospatial data
    // This is just a placeholder for demonstration
    
    // Generate sample map data
    const sampleMapData = {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: {
            name: 'Prediction Area',
            prediction: prediction.prediction,
            uncertainty: prediction.uncertainty
          },
          geometry: {
            type: 'Polygon',
            coordinates: [
              [
                [-74.0, 40.7],
                [-74.1, 40.7],
                [-74.1, 40.8],
                [-74.0, 40.8],
                [-74.0, 40.7]
              ]
            ]
          }
        }
      ]
    };
    
    setMapData(sampleMapData);
    
    // Generate prediction map (heatmap-like data)
    const samplePredictionMap = {
      labels: ['Area 1', 'Area 2', 'Area 3', 'Area 4', 'Area 5'],
      datasets: [
        {
          label: 'Mineral Deposit Probability',
          data: [
            prediction.prediction * 0.9,
            prediction.prediction * 1.1,
            prediction.prediction * 0.8,
            prediction.prediction * 1.2,
            prediction.prediction * 1.0
          ],
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        },
        {
          label: 'Uncertainty',
          data: prediction.uncertainty ? [
            prediction.uncertainty * 0.9,
            prediction.uncertainty * 1.1,
            prediction.uncertainty * 0.8,
            prediction.uncertainty * 1.2,
            prediction.uncertainty * 1.0
          ] : [],
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }
      ]
    };
    
    setPredictionMap(samplePredictionMap);
  };
  
  // Dropzone setup for file uploads
  const { getRootProps: getGeologicalRootProps, getInputProps: getGeologicalInputProps } = 
    useDropzone({ onDrop: files => handleFileUpload(files, 'geological') });
  
  const { getRootProps: getGeophysicalRootProps, getInputProps: getGeophysicalInputProps } = 
    useDropzone({ onDrop: files => handleFileUpload(files, 'geophysical') });
  
  const { getRootProps: getGeochemicalRootProps, getInputProps: getGeochemicalInputProps } = 
    useDropzone({ onDrop: files => handleFileUpload(files, 'geochemical') });
  
  const { getRootProps: getRemoteSensingRootProps, getInputProps: getRemoteSensingInputProps } = 
    useDropzone({ onDrop: files => handleFileUpload(files, 'remoteSensing') });
  
  const { getRootProps: getHistoricalRootProps, getInputProps: getHistoricalInputProps } = 
    useDropzone({ onDrop: files => handleFileUpload(files, 'historical') });
  
  // Render file upload dropzone
  const renderDropzone = (getRootProps, getInputProps, fileType, uploadedFiles) => (
    <div {...getRootProps()} className="dropzone">
      <input {...getInputProps()} />
      <p>Drag & drop {fileType} data files here, or click to select files</p>
      {uploadedFiles.length > 0 && (
        <div className="mt-2">
          <h6>Uploaded Files:</h6>
          <ul>
            {uploadedFiles.map((file, index) => (
              <li key={index}>{file.name}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
  
  // Render training form
  const renderTrainingForm = () => (
    <Card>
      <Card.Header>Train Predictive Model</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>Model Name</Form.Label>
            <Form.Control 
              type="text" 
              placeholder="Enter model name" 
              value={trainingData.modelName}
              onChange={e => setTrainingData({...trainingData, modelName: e.target.value})}
            />
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Geological Data</Form.Label>
            {renderDropzone(
              getGeologicalRootProps, 
              getGeologicalInputProps, 
              'geological', 
              uploadedFiles.geological
            )}
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Geophysical Data</Form.Label>
            {renderDropzone(
              getGeophysicalRootProps, 
              getGeophysicalInputProps, 
              'geophysical', 
              uploadedFiles.geophysical
            )}
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Geochemical Data</Form.Label>
            {renderDropzone(
              getGeochemicalRootProps, 
              getGeochemicalInputProps, 
              'geochemical', 
              uploadedFiles.geochemical
            )}
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Remote Sensing Data</Form.Label>
            {renderDropzone(
              getRemoteSensingRootProps, 
              getRemoteSensingInputProps, 
              'remoteSensing', 
              uploadedFiles.remoteSensing
            )}
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Historical Mining Data</Form.Label>
            {renderDropzone(
              getHistoricalRootProps, 
              getHistoricalInputProps, 
              'historical', 
              uploadedFiles.historical
            )}
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Check 
              type="checkbox" 
              label="Enable Uncertainty Estimation" 
              checked={trainingData.uncertaintyEstimation}
              onChange={e => setTrainingData({...trainingData, uncertaintyEstimation: e.target.checked})}
            />
          </Form.Group>
          
          <Button 
            variant="primary" 
            onClick={handleTrainModel}
            disabled={loading || !trainingData.modelName}
          >
            {loading ? <Spinner animation="border" size="sm" /> : 'Train Model'}
          </Button>
        </Form>
        
        {trainingJob && (
          <div className="mt-4">
            <h5>Training Job Status</h5>
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Job ID</td>
                  <td>{trainingJob.job_id}</td>
                </tr>
                <tr>
                  <td>Status</td>
                  <td>{trainingJob.status}</td>
                </tr>
                <tr>
                  <td>Model Name</td>
                  <td>{trainingJob.model_name}</td>
                </tr>
                {trainingStatus && trainingStatus.metrics && (
                  <>
                    <tr>
                      <td>Test Loss</td>
                      <td>{trainingStatus.metrics.test_loss}</td>
                    </tr>
                    <tr>
                      <td>Test Accuracy</td>
                      <td>{trainingStatus.metrics.test_acc}</td>
                    </tr>
                  </>
                )}
              </tbody>
            </Table>
            
            {trainingJob.status === 'running' && (
              <ProgressBar animated now={45} label="Training in progress..." />
            )}
          </div>
        )}
      </Card.Body>
    </Card>
  );
  
  // Render prediction form
  const renderPredictionForm = () => (
    <Card>
      <Card.Header>Make Predictions</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>Select Model</Form.Label>
            <Form.Select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
            >
              {models.map((model, index) => (
                <option key={index} value={model}>{model}</option>
              ))}
            </Form.Select>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Input Features (comma-separated values)</Form.Label>
            <Form.Control 
              as="textarea" 
              rows={3} 
              placeholder="Enter feature values separated by commas" 
              onChange={e => setPredictionInput({
                ...predictionInput, 
                features: e.target.value.split(',').map(v => parseFloat(v.trim()))
              })}
            />
            <Form.Text className="text-muted">
              Enter numerical values separated by commas (e.g., 0.5, 1.2, 3.7, ...)
            </Form.Text>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Check 
              type="checkbox" 
              label="Include Uncertainty Estimation" 
              checked={predictionInput.withUncertainty}
              onChange={e => setPredictionInput({...predictionInput, withUncertainty: e.target.checked})}
            />
          </Form.Group>
          
          <Button 
            variant="primary" 
            onClick={handlePredict}
            disabled={loading || predictionInput.features.length === 0 || models.length === 0}
          >
            {loading ? <Spinner animation="border" size="sm" /> : 'Make Prediction'}
          </Button>
        </Form>
        
        {predictionResult && (
          <div className="mt-4">
            <h5>Prediction Results</h5>
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Prediction ID</td>
                  <td>{predictionResult.prediction_id}</td>
                </tr>
                <tr>
                  <td>Mineral Deposit Probability</td>
                  <td>{(predictionResult.prediction * 100).toFixed(2)}%</td>
                </tr>
                {predictionResult.uncertainty !== null && (
                  <tr>
                    <td>Uncertainty (Variance)</td>
                    <td>{predictionResult.uncertainty.toFixed(4)}</td>
                  </tr>
                )}
              </tbody>
            </Table>
            
            {predictionMap && (
              <div className="mt-4">
                <h5>Prediction Visualization</h5>
                <Line data={predictionMap} />
              </div>
            )}
            
            {mapData && (
              <div className="mt-4">
                <h5>Spatial Prediction</h5>
                <div style={{ height: '400px' }}>
                  <MapContainer 
                    center={[40.75, -74.05]} 
                    zoom={11} 
                    style={{ height: '100%', width: '100%' }}
                  >
                    <TileLayer
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                      attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    />
                    {mapData && (
                      <GeoJSON 
                        data={mapData} 
                        style={feature => ({
                          fillColor: `rgba(255, ${Math.floor(255 * (1 - feature.properties.prediction))}, 0, 0.5)`,
                          weight: 2,
                          opacity: 1,
                          color: 'white',
                          dashArray: '3',
                          fillOpacity: 0.7
                        })}
                      />
                    )}
                  </MapContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </Card.Body>
    </Card>
  );
  
  return (
    <Container fluid className="mt-4">
      <h2>AI-Powered Predictive Modeling</h2>
      <p className="lead">
        Predict mineral deposits using deep learning models with uncertainty quantification.
      </p>
      
      {error && <Alert variant="danger">{error}</Alert>}
      {success && <Alert variant="success">{success}</Alert>}
      
      <Row className="mb-3">
        <Col>
          <Button 
            variant={activeTab === 'train' ? 'primary' : 'outline-primary'} 
            onClick={() => setActiveTab('train')}
            className="me-2"
          >
            Train Model
          </Button>
          <Button 
            variant={activeTab === 'predict' ? 'primary' : 'outline-primary'} 
            onClick={() => setActiveTab('predict')}
          >
            Make Predictions
          </Button>
        </Col>
      </Row>
      
      <Row>
        <Col>
          {activeTab === 'train' ? renderTrainingForm() : renderPredictionForm()}
        </Col>
      </Row>
    </Container>
  );
};

export default PredictiveModelingProcessor;
