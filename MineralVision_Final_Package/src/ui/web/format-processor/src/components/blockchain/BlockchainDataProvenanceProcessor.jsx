import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Form, Button, Alert, Spinner, Table, Nav, Tab } from 'react-bootstrap';
import { MapContainer, TileLayer, GeoJSON, Popup } from 'react-leaflet';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import 'leaflet/dist/leaflet.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck, faTimes, faInfoCircle, faExchangeAlt, faFileUpload, faSearch } from '@fortawesome/free-solid-svg-icons';

import { API_BASE_URL } from '../../services/api';

const BlockchainDataProvenanceProcessor = () => {
  // State for data registration
  const [dataRegistration, setDataRegistration] = useState({
    file: null,
    dataType: '',
    metadata: {},
    offlineMode: false
  });
  
  // State for data verification
  const [dataVerification, setDataVerification] = useState({
    file: null,
    dataId: '',
    ipfsHash: ''
  });
  
  // State for mineral rights
  const [mineralRight, setMineralRight] = useState({
    geographicBoundary: null,
    validUntil: '',
    mineralTypes: [],
    metadata: {},
    offlineMode: false
  });
  
  // State for mineral right transfer
  const [mineralRightTransfer, setMineralRightTransfer] = useState({
    rightId: '',
    newOwnerAddress: '',
    offlineMode: false
  });
  
  // State for data update
  const [dataUpdate, setDataUpdate] = useState({
    file: null,
    dataId: '',
    metadataUpdates: {},
    offlineMode: false
  });
  
  // State for UI
  const [activeTab, setActiveTab] = useState('register');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // State for results
  const [registrationResult, setRegistrationResult] = useState(null);
  const [verificationResult, setVerificationResult] = useState(null);
  const [mineralRightResult, setMineralRightResult] = useState(null);
  const [transferResult, setTransferResult] = useState(null);
  const [updateResult, setUpdateResult] = useState(null);
  
  // State for blockchain status
  const [blockchainStatus, setBlockchainStatus] = useState(null);
  
  // Fetch blockchain status on component mount
  useEffect(() => {
    fetchBlockchainStatus();
  }, []);
  
  // Fetch blockchain status
  const fetchBlockchainStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/blockchain/status`);
      setBlockchainStatus(response.data);
    } catch (err) {
      setError('Failed to fetch blockchain status: ' + err.message);
    }
  };
  
  // Handle file drop for data registration
  const onDropRegistration = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setDataRegistration({
        ...dataRegistration,
        file: acceptedFiles[0]
      });
    }
  };
  
  // Handle file drop for data verification
  const onDropVerification = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setDataVerification({
        ...dataVerification,
        file: acceptedFiles[0]
      });
    }
  };
  
  // Handle file drop for data update
  const onDropUpdate = (acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setDataUpdate({
        ...dataUpdate,
        file: acceptedFiles[0]
      });
    }
  };
  
  // Dropzone setup
  const { getRootProps: getRegistrationRootProps, getInputProps: getRegistrationInputProps } = 
    useDropzone({ onDrop: onDropRegistration });
  
  const { getRootProps: getVerificationRootProps, getInputProps: getVerificationInputProps } = 
    useDropzone({ onDrop: onDropVerification });
  
  const { getRootProps: getUpdateRootProps, getInputProps: getUpdateInputProps } = 
    useDropzone({ onDrop: onDropUpdate });
  
  // Handle data registration
  const handleRegisterData = async () => {
    if (!dataRegistration.file) {
      setError('Please select a file to register');
      return;
    }
    
    if (!dataRegistration.dataType) {
      setError('Please enter a data type');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const formData = new FormData();
      formData.append('file', dataRegistration.file);
      formData.append('data_type', dataRegistration.dataType);
      formData.append('metadata', JSON.stringify(dataRegistration.metadata));
      formData.append('offline_mode', dataRegistration.offlineMode);
      
      const response = await axios.post(`${API_BASE_URL}/blockchain/register-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setRegistrationResult(response.data);
      setSuccess('Data registered successfully!');
    } catch (err) {
      setError('Failed to register data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle data verification
  const handleVerifyData = async () => {
    if (!dataVerification.file) {
      setError('Please select a file to verify');
      return;
    }
    
    if (!dataVerification.dataId && !dataVerification.ipfsHash) {
      setError('Please enter either a data ID or IPFS hash');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const formData = new FormData();
      formData.append('file', dataVerification.file);
      
      if (dataVerification.dataId) {
        formData.append('data_id', dataVerification.dataId);
      }
      
      if (dataVerification.ipfsHash) {
        formData.append('ipfs_hash', dataVerification.ipfsHash);
      }
      
      const response = await axios.post(`${API_BASE_URL}/blockchain/verify-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setVerificationResult(response.data);
      setSuccess('Data verification completed!');
    } catch (err) {
      setError('Failed to verify data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle mineral right registration
  const handleRegisterMineralRight = async () => {
    if (!mineralRight.geographicBoundary) {
      setError('Please define a geographic boundary');
      return;
    }
    
    if (!mineralRight.validUntil) {
      setError('Please enter a valid until date');
      return;
    }
    
    if (mineralRight.mineralTypes.length === 0) {
      setError('Please enter at least one mineral type');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/blockchain/register-mineral-right`, {
        geographic_boundary: mineralRight.geographicBoundary,
        valid_until: mineralRight.validUntil,
        mineral_types: mineralRight.mineralTypes,
        metadata: mineralRight.metadata,
        offline_mode: mineralRight.offlineMode
      });
      
      setMineralRightResult(response.data);
      setSuccess('Mineral right registered successfully!');
    } catch (err) {
      setError('Failed to register mineral right: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle mineral right transfer
  const handleTransferMineralRight = async () => {
    if (!mineralRightTransfer.rightId) {
      setError('Please enter a right ID');
      return;
    }
    
    if (!mineralRightTransfer.newOwnerAddress) {
      setError('Please enter a new owner address');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/blockchain/transfer-mineral-right`, {
        right_id: mineralRightTransfer.rightId,
        new_owner_address: mineralRightTransfer.newOwnerAddress,
        offline_mode: mineralRightTransfer.offlineMode
      });
      
      setTransferResult(response.data);
      setSuccess('Mineral right transferred successfully!');
    } catch (err) {
      setError('Failed to transfer mineral right: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle data update
  const handleUpdateData = async () => {
    if (!dataUpdate.file) {
      setError('Please select a file to update');
      return;
    }
    
    if (!dataUpdate.dataId) {
      setError('Please enter a data ID');
      return;
    }
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      const formData = new FormData();
      formData.append('file', dataUpdate.file);
      formData.append('data_id', dataUpdate.dataId);
      formData.append('metadata_updates', JSON.stringify(dataUpdate.metadataUpdates));
      formData.append('offline_mode', dataUpdate.offlineMode);
      
      const response = await axios.post(`${API_BASE_URL}/blockchain/update-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setUpdateResult(response.data);
      setSuccess('Data updated successfully!');
    } catch (err) {
      setError('Failed to update data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };
  
  // Handle geographic boundary drawing
  const handleMapClick = (e) => {
    // In a real implementation, this would use a drawing tool
    // This is a simplified example
    const sampleBoundary = {
      type: 'Feature',
      properties: {
        name: 'Sample Boundary'
      },
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [e.latlng.lng - 0.01, e.latlng.lat - 0.01],
            [e.latlng.lng + 0.01, e.latlng.lat - 0.01],
            [e.latlng.lng + 0.01, e.latlng.lat + 0.01],
            [e.latlng.lng - 0.01, e.latlng.lat + 0.01],
            [e.latlng.lng - 0.01, e.latlng.lat - 0.01]
          ]
        ]
      }
    };
    
    setMineralRight({
      ...mineralRight,
      geographicBoundary: sampleBoundary
    });
  };
  
  // Render blockchain status
  const renderBlockchainStatus = () => (
    <Card className="mb-4">
      <Card.Header>Blockchain System Status</Card.Header>
      <Card.Body>
        {blockchainStatus ? (
          <Table striped bordered hover>
            <tbody>
              <tr>
                <td>Ethereum Connection</td>
                <td>
                  {blockchainStatus.ethereum_available ? (
                    <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Available</span>
                  ) : (
                    <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Unavailable</span>
                  )}
                </td>
              </tr>
              <tr>
                <td>IPFS Connection</td>
                <td>
                  {blockchainStatus.ipfs_available ? (
                    <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Available</span>
                  ) : (
                    <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Unavailable</span>
                  )}
                </td>
              </tr>
              <tr>
                <td>Smart Contract</td>
                <td>
                  {blockchainStatus.contract_available ? (
                    <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Available</span>
                  ) : (
                    <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Unavailable</span>
                  )}
                </td>
              </tr>
              <tr>
                <td>Ethereum Account</td>
                <td>
                  {blockchainStatus.account_available ? (
                    <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Available</span>
                  ) : (
                    <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Unavailable</span>
                  )}
                </td>
              </tr>
              <tr>
                <td>Local Storage Path</td>
                <td>{blockchainStatus.local_storage_path}</td>
              </tr>
            </tbody>
          </Table>
        ) : (
          <p>Loading blockchain status...</p>
        )}
      </Card.Body>
    </Card>
  );
  
  // Render data registration form
  const renderDataRegistrationForm = () => (
    <Card>
      <Card.Header>Register Data on Blockchain</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>Data File</Form.Label>
            <div {...getRegistrationRootProps()} className="dropzone p-3 border rounded mb-2">
              <input {...getRegistrationInputProps()} />
              <p className="mb-0">Drag & drop a file here, or click to select a file</p>
              {dataRegistration.file && (
                <p className="mt-2 mb-0"><strong>Selected file:</strong> {dataRegistration.file.name}</p>
              )}
            </div>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Data Type</Form.Label>
            <Form.Control 
              type="text" 
              placeholder="e.g., geological, geophysical, geochemical" 
              value={dataRegistration.dataType}
              onChange={e => setDataRegistration({...dataRegistration, dataType: e.target.value})}
            />
            <Form.Text className="text-muted">
              Specify the type of data being registered
            </Form.Text>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Metadata (Optional)</Form.Label>
            <Form.Control 
              as="textarea" 
              rows={3} 
              placeholder="Enter metadata as JSON" 
              value={dataRegistration.metadata ? JSON.stringify(dataRegistration.metadata, null, 2) : ''}
              onChange={e => {
                try {
                  const parsedMetadata = e.target.value ? JSON.parse(e.target.value) : {};
                  setDataRegistration({...dataRegistration, metadata: parsedMetadata});
                } catch (err) {
                  // Allow invalid JSON during typing
                }
              }}
            />
            <Form.Text className="text-muted">
              Additional information about the data in JSON format
            </Form.Text>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Check 
              type="checkbox" 
              label="Offline Mode" 
              checked={dataRegistration.offlineMode}
              onChange={e => setDataRegistration({...dataRegistration, offlineMode: e.target.checked})}
            />
            <Form.Text className="text-muted">
              Enable if blockchain or IPFS services are unavailable
            </Form.Text>
          </Form.Group>
          
          <Button 
            variant="primary" 
            onClick={handleRegisterData}
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Registering...
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faFileUpload} className="me-2" />
                Register Data
              </>
            )}
          </Button>
        </Form>
        
        {registrationResult && (
          <div className="mt-4">
            <h5>Registration Result</h5>
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Data ID</td>
                  <td>{registrationResult.data_id}</td>
                </tr>
                <tr>
                  <td>Data Hash</td>
                  <td>{registrationResult.data_hash}</td>
                </tr>
                <tr>
                  <td>Timestamp</td>
                  <td>{registrationResult.timestamp}</td>
                </tr>
                {registrationResult.ipfs_hash && (
                  <tr>
                    <td>IPFS Hash</td>
                    <td>{registrationResult.ipfs_hash}</td>
                  </tr>
                )}
                {registrationResult.transaction_hash && (
                  <tr>
                    <td>Transaction Hash</td>
                    <td>{registrationResult.transaction_hash}</td>
                  </tr>
                )}
                {registrationResult.block_number && (
                  <tr>
                    <td>Block Number</td>
                    <td>{registrationResult.block_number}</td>
                  </tr>
                )}
                {registrationResult.local_file_path && (
                  <tr>
                    <td>Local File Path</td>
                    <td>{registrationResult.local_file_path}</td>
                  </tr>
                )}
              </tbody>
            </Table>
          </div>
        )}
      </Card.Body>
    </Card>
  );
  
  // Render data verification form
  const renderDataVerificationForm = () => (
    <Card>
      <Card.Header>Verify Data Integrity and Provenance</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>Data File</Form.Label>
            <div {...getVerificationRootProps()} className="dropzone p-3 border rounded mb-2">
              <input {...getVerificationInputProps()} />
              <p className="mb-0">Drag & drop a file here, or click to select a file</p>
              {dataVerification.file && (
                <p className="mt-2 mb-0"><strong>Selected file:</strong> {dataVerification.file.name}</p>
              )}
            </div>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Data ID (Optional if IPFS Hash provided)</Form.Label>
            <Form.Control 
              type="text" 
              placeholder="Enter data ID" 
              value={dataVerification.dataId}
              onChange={e => setDataVerification({...dataVerification, dataId: e.target.value})}
            />
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>IPFS Hash (Optional if Data ID provided)</Form.Label>
            <Form.Control 
              type="text" 
              placeholder="Enter IPFS hash" 
              value={dataVerification.ipfsHash}
              onChange={e => setDataVerification({...dataVerification, ipfsHash: e.target.value})}
            />
          </Form.Group>
          
          <Button 
            variant="primary" 
            onClick={handleVerifyData}
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Verifying...
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faSearch} className="me-2" />
                Verify Data
              </>
            )}
          </Button>
        </Form>
        
        {verificationResult && (
          <div className="mt-4">
            <h5>Verification Result</h5>
            <div className="mb-3">
              <strong>Verification Status: </strong>
              {verificationResult.verified ? (
                <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Verified</span>
              ) : (
                <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Not Verified</span>
              )}
            </div>
            
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Calculated Hash</td>
                  <td>{verificationResult.calculated_hash}</td>
                </tr>
                {verificationResult.ipfs_verified !== undefined && (
                  <tr>
                    <td>IPFS Verification</td>
                    <td>
                      {verificationResult.ipfs_verified ? (
                        <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Verified</span>
                      ) : (
                        <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Not Verified</span>
                      )}
                    </td>
                  </tr>
                )}
                {verificationResult.blockchain_verified !== undefined && (
                  <tr>
                    <td>Blockchain Verification</td>
                    <td>
                      {verificationResult.blockchain_verified ? (
                        <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Verified</span>
                      ) : (
                        <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Not Verified</span>
                      )}
                    </td>
                  </tr>
                )}
                {verificationResult.local_verified !== undefined && (
                  <tr>
                    <td>Local Storage Verification</td>
                    <td>
                      {verificationResult.local_verified ? (
                        <span className="text-success"><FontAwesomeIcon icon={faCheck} /> Verified</span>
                      ) : (
                        <span className="text-danger"><FontAwesomeIcon icon={faTimes} /> Not Verified</span>
                      )}
                    </td>
                  </tr>
                )}
              </tbody>
            </Table>
            
            {verificationResult.provenance && (
              <div className="mt-3">
                <h5>Provenance Information</h5>
                <Table striped bordered hover>
                  <tbody>
                    {Object.entries(verificationResult.provenance).map(([key, value]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td>{typeof value === 'object' ? JSON.stringify(value) : value}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            )}
            
            {verificationResult.provenance_history && verificationResult.provenance_history.length > 0 && (
              <div className="mt-3">
                <h5>Provenance History</h5>
                <Table striped bordered hover>
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Owner</th>
                      <th>Operation</th>
                      <th>IPFS Hash</th>
                    </tr>
                  </thead>
                  <tbody>
                    {verificationResult.provenance_history.map((record, index) => (
                      <tr key={index}>
                        <td>{record.timestamp}</td>
                        <td>{record.owner}</td>
                        <td>{record.operation}</td>
                        <td>{record.ipfs_hash}</td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </div>
            )}
          </div>
        )}
      </Card.Body>
    </Card>
  );
  
  // Render mineral rights form
  const renderMineralRightsForm = () => (
    <Card>
      <Card.Header>Mineral Rights Management</Card.Header>
      <Card.Body>
        <Tab.Container defaultActiveKey="register">
          <Nav variant="tabs" className="mb-3">
            <Nav.Item>
              <Nav.Link eventKey="register">Register Mineral Right</Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link eventKey="transfer">Transfer Mineral Right</Nav.Link>
            </Nav.Item>
          </Nav>
          
          <Tab.Content>
            <Tab.Pane eventKey="register">
              <Form className="mt-3">
                <Form.Group className="mb-3">
                  <Form.Label>Geographic Boundary</Form.Label>
                  <div style={{ height: '300px' }} className="mb-2 border rounded">
                    <MapContainer 
                      center={[51.505, -0.09]} 
                      zoom={13} 
                      style={{ height: '100%', width: '100%' }}
                      onClick={handleMapClick}
                    >
                      <TileLayer
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                      />
                      {mineralRight.geographicBoundary && (
                        <GeoJSON 
                          data={mineralRight.geographicBoundary}
                          style={{ color: '#ff7800', weight: 2, opacity: 0.65 }}
                        >
                          <Popup>
                            Selected boundary for mineral rights
                          </Popup>
                        </GeoJSON>
                      )}
                    </MapContainer>
                  </div>
                  <Form.Text className="text-muted">
                    Click on the map to define a boundary (simplified for demo)
                  </Form.Text>
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Valid Until</Form.Label>
                  <Form.Control 
                    type="date" 
                    value={mineralRight.validUntil}
                    onChange={e => setMineralRight({...mineralRight, validUntil: e.target.value})}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Mineral Types</Form.Label>
                  <Form.Control 
                    type="text" 
                    placeholder="Enter mineral types (comma-separated)" 
                    value={mineralRight.mineralTypes.join(', ')}
                    onChange={e => setMineralRight({
                      ...mineralRight, 
                      mineralTypes: e.target.value.split(',').map(type => type.trim()).filter(type => type)
                    })}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>Metadata (Optional)</Form.Label>
                  <Form.Control 
                    as="textarea" 
                    rows={3} 
                    placeholder="Enter metadata as JSON" 
                    value={mineralRight.metadata ? JSON.stringify(mineralRight.metadata, null, 2) : ''}
                    onChange={e => {
                      try {
                        const parsedMetadata = e.target.value ? JSON.parse(e.target.value) : {};
                        setMineralRight({...mineralRight, metadata: parsedMetadata});
                      } catch (err) {
                        // Allow invalid JSON during typing
                      }
                    }}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Check 
                    type="checkbox" 
                    label="Offline Mode" 
                    checked={mineralRight.offlineMode}
                    onChange={e => setMineralRight({...mineralRight, offlineMode: e.target.checked})}
                  />
                </Form.Group>
                
                <Button 
                  variant="primary" 
                  onClick={handleRegisterMineralRight}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Spinner animation="border" size="sm" className="me-2" />
                      Registering...
                    </>
                  ) : (
                    <>
                      <FontAwesomeIcon icon={faFileUpload} className="me-2" />
                      Register Mineral Right
                    </>
                  )}
                </Button>
              </Form>
              
              {mineralRightResult && (
                <div className="mt-4">
                  <h5>Registration Result</h5>
                  <Table striped bordered hover>
                    <tbody>
                      <tr>
                        <td>Right ID</td>
                        <td>{mineralRightResult.right_id}</td>
                      </tr>
                      {mineralRightResult.transaction_hash && (
                        <tr>
                          <td>Transaction Hash</td>
                          <td>{mineralRightResult.transaction_hash}</td>
                        </tr>
                      )}
                      {mineralRightResult.block_number && (
                        <tr>
                          <td>Block Number</td>
                          <td>{mineralRightResult.block_number}</td>
                        </tr>
                      )}
                      {mineralRightResult.local_file_path && (
                        <tr>
                          <td>Local File Path</td>
                          <td>{mineralRightResult.local_file_path}</td>
                        </tr>
                      )}
                    </tbody>
                  </Table>
                </div>
              )}
            </Tab.Pane>
            
            <Tab.Pane eventKey="transfer">
              <Form className="mt-3">
                <Form.Group className="mb-3">
                  <Form.Label>Right ID</Form.Label>
                  <Form.Control 
                    type="text" 
                    placeholder="Enter mineral right ID" 
                    value={mineralRightTransfer.rightId}
                    onChange={e => setMineralRightTransfer({...mineralRightTransfer, rightId: e.target.value})}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Label>New Owner Address</Form.Label>
                  <Form.Control 
                    type="text" 
                    placeholder="Enter Ethereum address of new owner" 
                    value={mineralRightTransfer.newOwnerAddress}
                    onChange={e => setMineralRightTransfer({...mineralRightTransfer, newOwnerAddress: e.target.value})}
                  />
                </Form.Group>
                
                <Form.Group className="mb-3">
                  <Form.Check 
                    type="checkbox" 
                    label="Offline Mode" 
                    checked={mineralRightTransfer.offlineMode}
                    onChange={e => setMineralRightTransfer({...mineralRightTransfer, offlineMode: e.target.checked})}
                  />
                </Form.Group>
                
                <Button 
                  variant="primary" 
                  onClick={handleTransferMineralRight}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Spinner animation="border" size="sm" className="me-2" />
                      Transferring...
                    </>
                  ) : (
                    <>
                      <FontAwesomeIcon icon={faExchangeAlt} className="me-2" />
                      Transfer Mineral Right
                    </>
                  )}
                </Button>
              </Form>
              
              {transferResult && (
                <div className="mt-4">
                  <h5>Transfer Result</h5>
                  <Table striped bordered hover>
                    <tbody>
                      <tr>
                        <td>Right ID</td>
                        <td>{transferResult.right_id}</td>
                      </tr>
                      <tr>
                        <td>New Owner</td>
                        <td>{transferResult.new_owner}</td>
                      </tr>
                      <tr>
                        <td>Timestamp</td>
                        <td>{transferResult.timestamp}</td>
                      </tr>
                      {transferResult.transaction_hash && (
                        <tr>
                          <td>Transaction Hash</td>
                          <td>{transferResult.transaction_hash}</td>
                        </tr>
                      )}
                      {transferResult.block_number && (
                        <tr>
                          <td>Block Number</td>
                          <td>{transferResult.block_number}</td>
                        </tr>
                      )}
                    </tbody>
                  </Table>
                </div>
              )}
            </Tab.Pane>
          </Tab.Content>
        </Tab.Container>
      </Card.Body>
    </Card>
  );
  
  // Render data update form
  const renderDataUpdateForm = () => (
    <Card>
      <Card.Header>Update Data</Card.Header>
      <Card.Body>
        <Form>
          <Form.Group className="mb-3">
            <Form.Label>Data ID</Form.Label>
            <Form.Control 
              type="text" 
              placeholder="Enter data ID to update" 
              value={dataUpdate.dataId}
              onChange={e => setDataUpdate({...dataUpdate, dataId: e.target.value})}
            />
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Updated Data File</Form.Label>
            <div {...getUpdateRootProps()} className="dropzone p-3 border rounded mb-2">
              <input {...getUpdateInputProps()} />
              <p className="mb-0">Drag & drop a file here, or click to select a file</p>
              {dataUpdate.file && (
                <p className="mt-2 mb-0"><strong>Selected file:</strong> {dataUpdate.file.name}</p>
              )}
            </div>
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Label>Metadata Updates (Optional)</Form.Label>
            <Form.Control 
              as="textarea" 
              rows={3} 
              placeholder="Enter metadata updates as JSON" 
              value={dataUpdate.metadataUpdates ? JSON.stringify(dataUpdate.metadataUpdates, null, 2) : ''}
              onChange={e => {
                try {
                  const parsedMetadata = e.target.value ? JSON.parse(e.target.value) : {};
                  setDataUpdate({...dataUpdate, metadataUpdates: parsedMetadata});
                } catch (err) {
                  // Allow invalid JSON during typing
                }
              }}
            />
          </Form.Group>
          
          <Form.Group className="mb-3">
            <Form.Check 
              type="checkbox" 
              label="Offline Mode" 
              checked={dataUpdate.offlineMode}
              onChange={e => setDataUpdate({...dataUpdate, offlineMode: e.target.checked})}
            />
          </Form.Group>
          
          <Button 
            variant="primary" 
            onClick={handleUpdateData}
            disabled={loading}
          >
            {loading ? (
              <>
                <Spinner animation="border" size="sm" className="me-2" />
                Updating...
              </>
            ) : (
              <>
                <FontAwesomeIcon icon={faFileUpload} className="me-2" />
                Update Data
              </>
            )}
          </Button>
        </Form>
        
        {updateResult && (
          <div className="mt-4">
            <h5>Update Result</h5>
            <Table striped bordered hover>
              <tbody>
                <tr>
                  <td>Data ID</td>
                  <td>{updateResult.data_id}</td>
                </tr>
                <tr>
                  <td>Data Hash</td>
                  <td>{updateResult.data_hash}</td>
                </tr>
                <tr>
                  <td>Update Timestamp</td>
                  <td>{updateResult.update_timestamp}</td>
                </tr>
                {updateResult.ipfs_hash && (
                  <tr>
                    <td>IPFS Hash</td>
                    <td>{updateResult.ipfs_hash}</td>
                  </tr>
                )}
                {updateResult.transaction_hash && (
                  <tr>
                    <td>Transaction Hash</td>
                    <td>{updateResult.transaction_hash}</td>
                  </tr>
                )}
                {updateResult.block_number && (
                  <tr>
                    <td>Block Number</td>
                    <td>{updateResult.block_number}</td>
                  </tr>
                )}
                {updateResult.local_file_path && (
                  <tr>
                    <td>Local File Path</td>
                    <td>{updateResult.local_file_path}</td>
                  </tr>
                )}
              </tbody>
            </Table>
          </div>
        )}
      </Card.Body>
    </Card>
  );
  
  return (
    <Container fluid className="mt-4">
      <h2>Blockchain Data Provenance System</h2>
      <p className="lead">
        Secure and immutable record of data collection, processing, and analysis
      </p>
      
      {error && <Alert variant="danger">{error}</Alert>}
      {success && <Alert variant="success">{success}</Alert>}
      
      {renderBlockchainStatus()}
      
      <Row className="mb-3">
        <Col>
          <Nav variant="tabs">
            <Nav.Item>
              <Nav.Link 
                active={activeTab === 'register'} 
                onClick={() => setActiveTab('register')}
              >
                Register Data
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link 
                active={activeTab === 'verify'} 
                onClick={() => setActiveTab('verify')}
              >
                Verify Data
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link 
                active={activeTab === 'mineral'} 
                onClick={() => setActiveTab('mineral')}
              >
                Mineral Rights
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link 
                active={activeTab === 'update'} 
                onClick={() => setActiveTab('update')}
              >
                Update Data
              </Nav.Link>
            </Nav.Item>
          </Nav>
        </Col>
      </Row>
      
      <Row>
        <Col>
          {activeTab === 'register' && renderDataRegistrationForm()}
          {activeTab === 'verify' && renderDataVerificationForm()}
          {activeTab === 'mineral' && renderMineralRightsForm()}
          {activeTab === 'update' && renderDataUpdateForm()}
        </Col>
      </Row>
    </Container>
  );
};

export default BlockchainDataProvenanceProcessor;
