import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Tabs, 
  Tab, 
  Box, 
  Paper, 
  Button, 
  TextField, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Alert
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  Map as MapIcon,
  CalendarToday as CalendarIcon,
  AttachFile as AttachFileIcon,
  Save as SaveIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { MapContainer, TileLayer, Marker, Popup, GeoJSON } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { format } from 'date-fns';
import api from '../../services/api';

// Fix Leaflet icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`indigenous-knowledge-tabpanel-${index}`}
      aria-labelledby={`indigenous-knowledge-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const IndigenousKnowledgeProcessor = () => {
  // State for tab management
  const [tabValue, setTabValue] = useState(0);

  // State for knowledge holders
  const [knowledgeHolders, setKnowledgeHolders] = useState([]);
  const [selectedHolder, setSelectedHolder] = useState(null);
  const [holderFormOpen, setHolderFormOpen] = useState(false);
  const [holderForm, setHolderForm] = useState({
    name: '',
    community: '',
    role: '',
    contactInfo: {}
  });

  // State for traditional knowledge
  const [traditionalKnowledge, setTraditionalKnowledge] = useState([]);
  const [selectedKnowledge, setSelectedKnowledge] = useState(null);
  const [knowledgeFormOpen, setKnowledgeFormOpen] = useState(false);
  const [knowledgeForm, setKnowledgeForm] = useState({
    title: '',
    description: '',
    knowledgeType: 'general',
    holderId: '',
    community: '',
    accessLevel: 'public',
    geometry: null,
    siteType: '',
    significance: '',
    protectionLevel: 'standard',
    seasonalRestrictions: [],
    resourceType: '',
    seasonalUse: {},
    currentUse: true,
    sustainabilityPractices: []
  });

  // State for consultations
  const [consultations, setConsultations] = useState([]);
  const [selectedConsultation, setSelectedConsultation] = useState(null);
  const [consultationFormOpen, setConsultationFormOpen] = useState(false);
  const [consultationForm, setConsultationForm] = useState({
    title: '',
    description: '',
    community: '',
    consultationDate: new Date(),
    status: 'planned'
  });

  // State for agreements
  const [agreements, setAgreements] = useState([]);
  const [selectedAgreement, setSelectedAgreement] = useState(null);
  const [agreementFormOpen, setAgreementFormOpen] = useState(false);
  const [agreementForm, setAgreementForm] = useState({
    title: '',
    description: '',
    community: '',
    startDate: new Date(),
    endDate: null,
    status: 'draft',
    agreementType: ''
  });

  // State for map
  const [mapCenter, setMapCenter] = useState([0, 0]);
  const [mapZoom, setMapZoom] = useState(2);
  const [drawingMode, setDrawingMode] = useState(false);
  const [drawnItems, setDrawnItems] = useState(new L.FeatureGroup());

  // State for notifications
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Load data on component mount
  useEffect(() => {
    fetchKnowledgeHolders();
    fetchTraditionalKnowledge();
    fetchConsultations();
    fetchAgreements();
  }, []);

  // Tab change handler
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Knowledge Holders API calls
  const fetchKnowledgeHolders = async () => {
    try {
      const response = await api.get('/indigenous-knowledge/holders');
      setKnowledgeHolders(response.data.holders || []);
    } catch (error) {
      showNotification('Error fetching knowledge holders', 'error');
    }
  };

  const createKnowledgeHolder = async () => {
    try {
      const formData = new FormData();
      formData.append('name', holderForm.name);
      formData.append('community', holderForm.community);
      formData.append('role', holderForm.role);
      formData.append('contact_info', JSON.stringify(holderForm.contactInfo));

      const response = await api.post('/indigenous-knowledge/holders', formData);
      setKnowledgeHolders([...knowledgeHolders, response.data.holder]);
      setHolderFormOpen(false);
      resetHolderForm();
      showNotification('Knowledge holder created successfully', 'success');
    } catch (error) {
      showNotification('Error creating knowledge holder', 'error');
    }
  };

  const updateKnowledgeHolder = async () => {
    try {
      const formData = new FormData();
      formData.append('name', holderForm.name);
      formData.append('community', holderForm.community);
      formData.append('role', holderForm.role);
      formData.append('contact_info', JSON.stringify(holderForm.contactInfo));

      const response = await api.put(`/indigenous-knowledge/holders/${selectedHolder.holder_id}`, formData);
      
      const updatedHolders = knowledgeHolders.map(holder => 
        holder.holder_id === selectedHolder.holder_id ? response.data.holder : holder
      );
      
      setKnowledgeHolders(updatedHolders);
      setHolderFormOpen(false);
      resetHolderForm();
      setSelectedHolder(null);
      showNotification('Knowledge holder updated successfully', 'success');
    } catch (error) {
      showNotification('Error updating knowledge holder', 'error');
    }
  };

  // Traditional Knowledge API calls
  const fetchTraditionalKnowledge = async () => {
    try {
      const response = await api.get('/indigenous-knowledge/knowledge');
      setTraditionalKnowledge(response.data.knowledge || []);
    } catch (error) {
      showNotification('Error fetching traditional knowledge', 'error');
    }
  };

  const createTraditionalKnowledge = async () => {
    try {
      const formData = new FormData();
      formData.append('title', knowledgeForm.title);
      formData.append('description', knowledgeForm.description);
      formData.append('knowledge_type', knowledgeForm.knowledgeType);
      formData.append('holder_id', knowledgeForm.holderId);
      formData.append('community', knowledgeForm.community);
      formData.append('access_level', knowledgeForm.accessLevel);
      
      if (knowledgeForm.geometry) {
        formData.append('geometry', JSON.stringify(knowledgeForm.geometry));
      }

      // Add type-specific fields
      if (knowledgeForm.knowledgeType === 'cultural_heritage_site') {
        formData.append('site_type', knowledgeForm.siteType);
        formData.append('significance', knowledgeForm.significance);
        formData.append('protection_level', knowledgeForm.protectionLevel);
        formData.append('seasonal_restrictions', JSON.stringify(knowledgeForm.seasonalRestrictions));
      } else if (knowledgeForm.knowledgeType === 'resource_area') {
        formData.append('resource_type', knowledgeForm.resourceType);
        formData.append('seasonal_use', JSON.stringify(knowledgeForm.seasonalUse));
        formData.append('current_use', knowledgeForm.currentUse);
        formData.append('sustainability_practices', JSON.stringify(knowledgeForm.sustainabilityPractices));
      }

      const response = await api.post('/indigenous-knowledge/knowledge', formData);
      setTraditionalKnowledge([...traditionalKnowledge, response.data.knowledge]);
      setKnowledgeFormOpen(false);
      resetKnowledgeForm();
      showNotification('Traditional knowledge created successfully', 'success');
    } catch (error) {
      showNotification('Error creating traditional knowledge', 'error');
    }
  };

  // Consultation API calls
  const fetchConsultations = async () => {
    try {
      const response = await api.get('/indigenous-knowledge/consultations');
      setConsultations(response.data.consultations || []);
    } catch (error) {
      showNotification('Error fetching consultations', 'error');
    }
  };

  const createConsultation = async () => {
    try {
      const formData = new FormData();
      formData.append('title', consultationForm.title);
      formData.append('description', consultationForm.description);
      formData.append('community', consultationForm.community);
      formData.append('consultation_date', format(consultationForm.consultationDate, 'yyyy-MM-dd'));
      formData.append('status', consultationForm.status);

      const response = await api.post('/indigenous-knowledge/consultations', formData);
      setConsultations([...consultations, response.data.consultation]);
      setConsultationFormOpen(false);
      resetConsultationForm();
      showNotification('Consultation created successfully', 'success');
    } catch (error) {
      showNotification('Error creating consultation', 'error');
    }
  };

  // Agreement API calls
  const fetchAgreements = async () => {
    try {
      const response = await api.get('/indigenous-knowledge/agreements');
      setAgreements(response.data.agreements || []);
    } catch (error) {
      showNotification('Error fetching agreements', 'error');
    }
  };

  const createAgreement = async () => {
    try {
      const formData = new FormData();
      formData.append('title', agreementForm.title);
      formData.append('description', agreementForm.description);
      formData.append('community', agreementForm.community);
      formData.append('start_date', format(agreementForm.startDate, 'yyyy-MM-dd'));
      
      if (agreementForm.endDate) {
        formData.append('end_date', format(agreementForm.endDate, 'yyyy-MM-dd'));
      }
      
      formData.append('status', agreementForm.status);
      formData.append('agreement_type', agreementForm.agreementType);

      const response = await api.post('/indigenous-knowledge/agreements', formData);
      setAgreements([...agreements, response.data.agreement]);
      setAgreementFormOpen(false);
      resetAgreementForm();
      showNotification('Agreement created successfully', 'success');
    } catch (error) {
      showNotification('Error creating agreement', 'error');
    }
  };

  // Form reset functions
  const resetHolderForm = () => {
    setHolderForm({
      name: '',
      community: '',
      role: '',
      contactInfo: {}
    });
  };

  const resetKnowledgeForm = () => {
    setKnowledgeForm({
      title: '',
      description: '',
      knowledgeType: 'general',
      holderId: '',
      community: '',
      accessLevel: 'public',
      geometry: null,
      siteType: '',
      significance: '',
      protectionLevel: 'standard',
      seasonalRestrictions: [],
      resourceType: '',
      seasonalUse: {},
      currentUse: true,
      sustainabilityPractices: []
    });
  };

  const resetConsultationForm = () => {
    setConsultationForm({
      title: '',
      description: '',
      community: '',
      consultationDate: new Date(),
      status: 'planned'
    });
  };

  const resetAgreementForm = () => {
    setAgreementForm({
      title: '',
      description: '',
      community: '',
      startDate: new Date(),
      endDate: null,
      status: 'draft',
      agreementType: ''
    });
  };

  // Edit functions
  const editKnowledgeHolder = (holder) => {
    setSelectedHolder(holder);
    setHolderForm({
      name: holder.name,
      community: holder.community,
      role: holder.role,
      contactInfo: holder.contact_info
    });
    setHolderFormOpen(true);
  };

  // Notification helper
  const showNotification = (message, severity) => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  const handleCloseNotification = () => {
    setNotification({
      ...notification,
      open: false
    });
  };

  return (
    <Container maxWidth="lg">
      <Paper elevation={3} sx={{ p: 3, mt: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Indigenous Knowledge Integration
        </Typography>
        <Typography variant="body1" paragraph>
          This module enables respectful incorporation of traditional knowledge into the mineral exploration process,
          supporting collaborative mapping, cultural heritage site identification, benefit-sharing tracking, and consultation management.
        </Typography>

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="indigenous knowledge tabs">
            <Tab label="Knowledge Holders" />
            <Tab label="Traditional Knowledge" />
            <Tab label="Consultations" />
            <Tab label="Benefit Sharing" />
            <Tab label="Map View" />
          </Tabs>
        </Box>

        {/* Knowledge Holders Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="h6">Knowledge Holders</Typography>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              onClick={() => {
                resetHolderForm();
                setSelectedHolder(null);
                setHolderFormOpen(true);
              }}
            >
              Add Knowledge Holder
            </Button>
          </Box>

          <List>
            {knowledgeHolders.map((holder) => (
              <Paper key={holder.holder_id} elevation={2} sx={{ mb: 2, p: 2 }}>
                <ListItem>
                  <ListItemText
                    primary={holder.name}
                    secondary={
                      <>
                        <Typography component="span" variant="body2" color="text.primary">
                          {holder.community}
                        </Typography>
                        {` — ${holder.role || 'No role specified'}`}
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" aria-label="edit" onClick={() => editKnowledgeHolder(holder)}>
                      <EditIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                {holder.knowledge_contributions && holder.knowledge_contributions.length > 0 && (
                  <Box sx={{ ml: 2, mt: 1 }}>
                    <Typography variant="subtitle2">Contributions:</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {holder.knowledge_contributions.map((contribution, index) => (
                        <Chip key={index} label={`Contribution ${index + 1}`} size="small" />
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>
            ))}
            {knowledgeHolders.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                No knowledge holders found. Add a new knowledge holder to get started.
              </Typography>
            )}
          </List>

          {/* Knowledge Holder Form Dialog */}
          <Dialog open={holderFormOpen} onClose={() => setHolderFormOpen(false)} maxWidth="md" fullWidth>
            <DialogTitle>{selectedHolder ? 'Edit Knowledge Holder' : 'Add Knowledge Holder'}</DialogTitle>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Name"
                    value={holderForm.name}
                    onChange={(e) => setHolderForm({ ...holderForm, name: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Community/Nation"
                    value={holderForm.community}
                    onChange={(e) => setHolderForm({ ...holderForm, community: e.target.value })}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Role"
                    value={holderForm.role}
                    onChange={(e) => setHolderForm({ ...holderForm, role: e.target.value })}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Contact Information (Email)"
                    value={holderForm.contactInfo.email || ''}
                    onChange={(e) => setHolderForm({ 
                      ...holderForm, 
                      contactInfo: { ...holderForm.contactInfo, email: e.target.value } 
                    })}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Contact Information (Phone)"
                    value={holderForm.contactInfo.phone || ''}
                    onChange={(e) => setHolderForm({ 
                      ...holderForm, 
                      contactInfo: { ...holderForm.contactInfo, phone: e.target.value } 
                    })}
                  />
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setHolderFormOpen(false)}>Cancel</Button>
              <Button 
                onClick={selectedHolder ? updateKnowledgeHolder : createKnowledgeHolder} 
                variant="contained"
              >
                {selectedHolder ? 'Update' : 'Create'}
              </Button>
            </DialogActions>
          </Dialog>
        </TabPanel>

        {/* Traditional Knowledge Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="h6">Traditional Knowledge</Typography>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              onClick={() => {
                resetKnowledgeForm();
                setSelectedKnowledge(null);
                setKnowledgeFormOpen(true);
              }}
            >
              Add Traditional Knowledge
            </Button>
          </Box>

          <List>
            {traditionalKnowledge.map((knowledge) => (
              <Paper key={knowledge.knowledge_id} elevation={2} sx={{ mb: 2, p: 2 }}>
                <ListItem>
                  <ListItemText
                    primary={knowledge.title}
                    secondary={
                      <>
                        <Typography component="span" variant="body2" color="text.primary">
                          {knowledge.knowledge_type.replace('_', ' ').toUpperCase()}
                        </Typography>
                        {` — ${knowledge.description || 'No description'}`}
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" aria-label="view on map" onClick={() => {
                      setTabValue(4); // Switch to map tab
                      if (knowledge.geometry) {
                        // Center map on knowledge geometry
                        const coords = knowledge.geometry.type === 'Point' 
                          ? [knowledge.geometry.coordinates[1], knowledge.geometry.coordinates[0]]
                          : [knowledge.geometry.coordinates[0][0][1], knowledge.geometry.coordinates[0][0][0]];
                        setMapCenter(coords);
                        setMapZoom(12);
                      }
                    }}>
                      <MapIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Box sx={{ ml: 2, mt: 1 }}>
                  <Typography variant="subtitle2">Community: {knowledge.community || 'Not specified'}</Typography>
                  <Typography variant="subtitle2">Access Level: {knowledge.access_level}</Typography>
                  
                  {knowledge.knowledge_type === 'cultural_heritage_site' && (
                    <>
                      <Typography variant="subtitle2">Site Type: {knowledge.site_type || 'Not specified'}</Typography>
                      <Typography variant="subtitle2">Protection Level: {knowledge.protection_level}</Typography>
                    </>
                  )}
                  
                  {knowledge.knowledge_type === 'resource_area' && (
                    <>
                      <Typography variant="subtitle2">Resource Type: {knowledge.resource_type || 'Not specified'}</Typography>
                      <Typography variant="subtitle2">Current Use: {knowledge.current_use ? 'Yes' : 'No'}</Typography>
                    </>
                  )}
                </Box>
              </Paper>
            ))}
            {traditionalKnowledge.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                No traditional knowledge found. Add new knowledge to get started.
              </Typography>
            )}
          </List>

          {/* Traditional Knowledge Form Dialog */}
          <Dialog open={knowledgeFormOpen} onClose={() => setKnowledgeFormOpen(false)} maxWidth="md" fullWidth>
            <DialogTitle>Add Traditional Knowledge</DialogTitle>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Title"
                    value={knowledgeForm.title}
                    onChange={(e) => setKnowledgeForm({ ...knowledgeForm, title: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Description"
                    value={knowledgeForm.description}
                    onChange={(e) => setKnowledgeForm({ ...knowledgeForm, description: e.target.value })}
                    multiline
                    rows={3}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Knowledge Type</InputLabel>
                    <Select
                      value={knowledgeForm.knowledgeType}
                      label="Knowledge Type"
                      onChange={(e) => setKnowledgeForm({ ...knowledgeForm, knowledgeType: e.target.value })}
                    >
                      <MenuItem value="general">General Knowledge</MenuItem>
                      <MenuItem value="cultural_heritage_site">Cultural Heritage Site</MenuItem>
                      <MenuItem value="resource_area">Resource Area</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Knowledge Holder</InputLabel>
                    <Select
                      value={knowledgeForm.holderId}
                      label="Knowledge Holder"
                      onChange={(e) => setKnowledgeForm({ ...knowledgeForm, holderId: e.target.value })}
                    >
                      <MenuItem value="">None</MenuItem>
                      {knowledgeHolders.map((holder) => (
                        <MenuItem key={holder.holder_id} value={holder.holder_id}>
                          {holder.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Community/Nation"
                    value={knowledgeForm.community}
                    onChange={(e) => setKnowledgeForm({ ...knowledgeForm, community: e.target.value })}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Access Level</InputLabel>
                    <Select
                      value={knowledgeForm.accessLevel}
                      label="Access Level"
                      onChange={(e) => setKnowledgeForm({ ...knowledgeForm, accessLevel: e.target.value })}
                    >
                      <MenuItem value="public">Public</MenuItem>
                      <MenuItem value="restricted">Restricted</MenuItem>
                      <MenuItem value="confidential">Confidential</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                {/* Type-specific fields */}
                {knowledgeForm.knowledgeType === 'cultural_heritage_site' && (
                  <>
                    <Grid item xs={12} md={6}>
                      <TextField
                        fullWidth
                        label="Site Type"
                        value={knowledgeForm.siteType}
                        onChange={(e) => setKnowledgeForm({ ...knowledgeForm, siteType: e.target.value })}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <TextField
                        fullWidth
                        label="Significance"
                        value={knowledgeForm.significance}
                        onChange={(e) => setKnowledgeForm({ ...knowledgeForm, significance: e.target.value })}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth>
                        <InputLabel>Protection Level</InputLabel>
                        <Select
                          value={knowledgeForm.protectionLevel}
                          label="Protection Level"
                          onChange={(e) => setKnowledgeForm({ ...knowledgeForm, protectionLevel: e.target.value })}
                        >
                          <MenuItem value="standard">Standard</MenuItem>
                          <MenuItem value="high">High</MenuItem>
                          <MenuItem value="critical">Critical</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </>
                )}

                {knowledgeForm.knowledgeType === 'resource_area' && (
                  <>
                    <Grid item xs={12} md={6}>
                      <TextField
                        fullWidth
                        label="Resource Type"
                        value={knowledgeForm.resourceType}
                        onChange={(e) => setKnowledgeForm({ ...knowledgeForm, resourceType: e.target.value })}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth>
                        <InputLabel>Current Use</InputLabel>
                        <Select
                          value={knowledgeForm.currentUse}
                          label="Current Use"
                          onChange={(e) => setKnowledgeForm({ ...knowledgeForm, currentUse: e.target.value })}
                        >
                          <MenuItem value={true}>Yes</MenuItem>
                          <MenuItem value={false}>No</MenuItem>
                        </Select>
                      </FormControl>
                    </Grid>
                  </>
                )}

                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Location
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Use the Map tab to define the location for this knowledge entry.
                  </Typography>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setKnowledgeFormOpen(false)}>Cancel</Button>
              <Button 
                onClick={createTraditionalKnowledge} 
                variant="contained"
              >
                Create
              </Button>
            </DialogActions>
          </Dialog>
        </TabPanel>

        {/* Consultations Tab */}
        <TabPanel value={tabValue} index={2}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="h6">Consultations</Typography>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              onClick={() => {
                resetConsultationForm();
                setSelectedConsultation(null);
                setConsultationFormOpen(true);
              }}
            >
              Add Consultation
            </Button>
          </Box>

          <List>
            {consultations.map((consultation) => (
              <Paper key={consultation.consultation_id} elevation={2} sx={{ mb: 2, p: 2 }}>
                <ListItem>
                  <ListItemText
                    primary={consultation.title}
                    secondary={
                      <>
                        <Typography component="span" variant="body2" color="text.primary">
                          {consultation.community}
                        </Typography>
                        {` — ${consultation.description || 'No description'}`}
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" aria-label="calendar">
                      <CalendarIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Box sx={{ ml: 2, mt: 1 }}>
                  <Typography variant="subtitle2">
                    Date: {new Date(consultation.consultation_date).toLocaleDateString()}
                  </Typography>
                  <Typography variant="subtitle2">
                    Status: <Chip size="small" label={consultation.status.toUpperCase()} color={
                      consultation.status === 'completed' ? 'success' : 
                      consultation.status === 'cancelled' ? 'error' : 'primary'
                    } />
                  </Typography>
                  
                  {consultation.participants && consultation.participants.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="subtitle2">Participants:</Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {consultation.participants.map((participant, index) => (
                          <Chip key={index} label={participant.name} size="small" />
                        ))}
                      </Box>
                    </Box>
                  )}
                  
                  {consultation.outcomes && consultation.outcomes.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="subtitle2">Outcomes:</Typography>
                      <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                        {consultation.outcomes.map((outcome, index) => (
                          <li key={index}>{outcome}</li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              </Paper>
            ))}
            {consultations.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                No consultations found. Add a new consultation to get started.
              </Typography>
            )}
          </List>

          {/* Consultation Form Dialog */}
          <Dialog open={consultationFormOpen} onClose={() => setConsultationFormOpen(false)} maxWidth="md" fullWidth>
            <DialogTitle>Add Consultation</DialogTitle>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Title"
                    value={consultationForm.title}
                    onChange={(e) => setConsultationForm({ ...consultationForm, title: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Description"
                    value={consultationForm.description}
                    onChange={(e) => setConsultationForm({ ...consultationForm, description: e.target.value })}
                    multiline
                    rows={3}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Community/Nation"
                    value={consultationForm.community}
                    onChange={(e) => setConsultationForm({ ...consultationForm, community: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <LocalizationProvider dateAdapter={AdapterDateFns}>
                    <DatePicker
                      label="Consultation Date"
                      value={consultationForm.consultationDate}
                      onChange={(newDate) => setConsultationForm({ ...consultationForm, consultationDate: newDate })}
                      renderInput={(params) => <TextField {...params} fullWidth />}
                    />
                  </LocalizationProvider>
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Status</InputLabel>
                    <Select
                      value={consultationForm.status}
                      label="Status"
                      onChange={(e) => setConsultationForm({ ...consultationForm, status: e.target.value })}
                    >
                      <MenuItem value="planned">Planned</MenuItem>
                      <MenuItem value="completed">Completed</MenuItem>
                      <MenuItem value="cancelled">Cancelled</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setConsultationFormOpen(false)}>Cancel</Button>
              <Button 
                onClick={createConsultation} 
                variant="contained"
              >
                Create
              </Button>
            </DialogActions>
          </Dialog>
        </TabPanel>

        {/* Benefit Sharing Tab */}
        <TabPanel value={tabValue} index={3}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="h6">Benefit Sharing Agreements</Typography>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              onClick={() => {
                resetAgreementForm();
                setSelectedAgreement(null);
                setAgreementFormOpen(true);
              }}
            >
              Add Agreement
            </Button>
          </Box>

          <List>
            {agreements.map((agreement) => (
              <Paper key={agreement.agreement_id} elevation={2} sx={{ mb: 2, p: 2 }}>
                <ListItem>
                  <ListItemText
                    primary={agreement.title}
                    secondary={
                      <>
                        <Typography component="span" variant="body2" color="text.primary">
                          {agreement.community}
                        </Typography>
                        {` — ${agreement.description || 'No description'}`}
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton edge="end" aria-label="attachments">
                      <AttachFileIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
                <Box sx={{ ml: 2, mt: 1 }}>
                  <Typography variant="subtitle2">
                    Type: {agreement.agreement_type || 'Not specified'}
                  </Typography>
                  <Typography variant="subtitle2">
                    Status: <Chip size="small" label={agreement.status.toUpperCase()} color={
                      agreement.status === 'active' ? 'success' : 
                      agreement.status === 'terminated' ? 'error' : 
                      agreement.status === 'expired' ? 'warning' : 'default'
                    } />
                  </Typography>
                  <Typography variant="subtitle2">
                    Period: {agreement.start_date ? new Date(agreement.start_date).toLocaleDateString() : 'Not specified'} 
                    {agreement.end_date ? ` to ${new Date(agreement.end_date).toLocaleDateString()}` : ' (No end date)'}
                  </Typography>
                  
                  {agreement.benefits && agreement.benefits.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="subtitle2">Benefits:</Typography>
                      <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                        {agreement.benefits.map((benefit, index) => (
                          <li key={index}>
                            {benefit.type}: {benefit.description}
                            {benefit.value ? ` (Value: ${benefit.value})` : ''}
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                  
                  {agreement.commitments && agreement.commitments.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="subtitle2">Commitments:</Typography>
                      <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                        {agreement.commitments.map((commitment, index) => (
                          <li key={index}>
                            {commitment.description} (Responsible: {commitment.responsible})
                          </li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              </Paper>
            ))}
            {agreements.length === 0 && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
                No benefit sharing agreements found. Add a new agreement to get started.
              </Typography>
            )}
          </List>

          {/* Agreement Form Dialog */}
          <Dialog open={agreementFormOpen} onClose={() => setAgreementFormOpen(false)} maxWidth="md" fullWidth>
            <DialogTitle>Add Benefit Sharing Agreement</DialogTitle>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Title"
                    value={agreementForm.title}
                    onChange={(e) => setAgreementForm({ ...agreementForm, title: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Description"
                    value={agreementForm.description}
                    onChange={(e) => setAgreementForm({ ...agreementForm, description: e.target.value })}
                    multiline
                    rows={3}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Community/Nation"
                    value={agreementForm.community}
                    onChange={(e) => setAgreementForm({ ...agreementForm, community: e.target.value })}
                    required
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Agreement Type"
                    value={agreementForm.agreementType}
                    onChange={(e) => setAgreementForm({ ...agreementForm, agreementType: e.target.value })}
                    placeholder="e.g., Financial, Employment, Education"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <LocalizationProvider dateAdapter={AdapterDateFns}>
                    <DatePicker
                      label="Start Date"
                      value={agreementForm.startDate}
                      onChange={(newDate) => setAgreementForm({ ...agreementForm, startDate: newDate })}
                      renderInput={(params) => <TextField {...params} fullWidth />}
                    />
                  </LocalizationProvider>
                </Grid>
                <Grid item xs={12} md={6}>
                  <LocalizationProvider dateAdapter={AdapterDateFns}>
                    <DatePicker
                      label="End Date (Optional)"
                      value={agreementForm.endDate}
                      onChange={(newDate) => setAgreementForm({ ...agreementForm, endDate: newDate })}
                      renderInput={(params) => <TextField {...params} fullWidth />}
                    />
                  </LocalizationProvider>
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Status</InputLabel>
                    <Select
                      value={agreementForm.status}
                      label="Status"
                      onChange={(e) => setAgreementForm({ ...agreementForm, status: e.target.value })}
                    >
                      <MenuItem value="draft">Draft</MenuItem>
                      <MenuItem value="active">Active</MenuItem>
                      <MenuItem value="expired">Expired</MenuItem>
                      <MenuItem value="terminated">Terminated</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setAgreementFormOpen(false)}>Cancel</Button>
              <Button 
                onClick={createAgreement} 
                variant="contained"
              >
                Create
              </Button>
            </DialogActions>
          </Dialog>
        </TabPanel>

        {/* Map View Tab */}
        <TabPanel value={tabValue} index={4}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="h6">Map View</Typography>
            <Box>
              <Button 
                variant="outlined" 
                startIcon={<RefreshIcon />}
                onClick={fetchTraditionalKnowledge}
                sx={{ mr: 1 }}
              >
                Refresh Data
              </Button>
              <Button 
                variant="contained" 
                color={drawingMode ? "secondary" : "primary"}
                onClick={() => setDrawingMode(!drawingMode)}
              >
                {drawingMode ? "Cancel Drawing" : "Draw on Map"}
              </Button>
            </Box>
          </Box>

          <Paper elevation={3} sx={{ height: 500, width: '100%' }}>
            <MapContainer 
              center={mapCenter} 
              zoom={mapZoom} 
              style={{ height: '100%', width: '100%' }}
              whenCreated={(mapInstance) => {
                // Initialize drawing controls
                // This would be implemented with Leaflet.draw in a real application
              }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              
              {/* Render traditional knowledge on the map */}
              {traditionalKnowledge.map((knowledge) => {
                if (knowledge.geometry) {
                  if (knowledge.geometry.type === 'Point') {
                    const position = [knowledge.geometry.coordinates[1], knowledge.geometry.coordinates[0]];
                    return (
                      <Marker key={knowledge.knowledge_id} position={position}>
                        <Popup>
                          <div>
                            <h3>{knowledge.title}</h3>
                            <p>{knowledge.description}</p>
                            <p>Type: {knowledge.knowledge_type.replace('_', ' ')}</p>
                            <p>Community: {knowledge.community}</p>
                          </div>
                        </Popup>
                      </Marker>
                    );
                  } else {
                    // For polygons and other geometry types
                    return (
                      <GeoJSON 
                        key={knowledge.knowledge_id} 
                        data={knowledge.geometry}
                        style={() => ({
                          color: knowledge.knowledge_type === 'cultural_heritage_site' ? '#ff0000' : '#0000ff',
                          weight: 2,
                          opacity: 0.7,
                          fillOpacity: 0.4
                        })}
                      >
                        <Popup>
                          <div>
                            <h3>{knowledge.title}</h3>
                            <p>{knowledge.description}</p>
                            <p>Type: {knowledge.knowledge_type.replace('_', ' ')}</p>
                            <p>Community: {knowledge.community}</p>
                          </div>
                        </Popup>
                      </GeoJSON>
                    );
                  }
                }
                return null;
              })}
            </MapContainer>
          </Paper>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Map Legend
            </Typography>
            <Grid container spacing={2}>
              <Grid item>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: 16, height: 16, bgcolor: '#ff0000', mr: 1 }} />
                  <Typography variant="body2">Cultural Heritage Site</Typography>
                </Box>
              </Grid>
              <Grid item>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: 16, height: 16, bgcolor: '#0000ff', mr: 1 }} />
                  <Typography variant="body2">Resource Area</Typography>
                </Box>
              </Grid>
              <Grid item>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Box sx={{ width: 16, height: 16, bgcolor: '#00ff00', mr: 1 }} />
                  <Typography variant="body2">General Knowledge</Typography>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </TabPanel>

        {/* Notification Snackbar */}
        <Snackbar 
          open={notification.open} 
          autoHideDuration={6000} 
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert onClose={handleCloseNotification} severity={notification.severity} sx={{ width: '100%' }}>
            {notification.message}
          </Alert>
        </Snackbar>
      </Paper>
    </Container>
  );
};

export default IndigenousKnowledgeProcessor;
