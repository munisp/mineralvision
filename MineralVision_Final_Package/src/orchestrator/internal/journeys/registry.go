// Package journeys provides the journey registry and manifest definitions
package journeys

import (
	"fmt"
	"time"
)

// StepType defines the type of journey step
type StepType string

const (
	StepTypeAPICall          StepType = "api_call"
	StepTypeMLInference      StepType = "ml_inference"
	StepTypeSensorFusion     StepType = "sensor_fusion"
	StepTypeDataIngestion    StepType = "data_ingestion"
	StepTypeReportGeneration StepType = "report_generation"
	StepTypeVisualization    StepType = "visualization"
	StepTypeBlockchainRecord StepType = "blockchain_record"
	StepTypeHumanApproval    StepType = "human_approval"
	StepTypeEventPublish     StepType = "event_publish"
	StepTypeLedgerWrite      StepType = "ledger_write"
)

// Step represents a single step in a journey
type Step struct {
	ID               string            `json:"id"`
	Name             string            `json:"name"`
	StepType         StepType          `json:"step_type"`
	Endpoint         string            `json:"endpoint,omitempty"`
	Module           string            `json:"module,omitempty"`
	Method           string            `json:"method,omitempty"`
	InputMapping     map[string]string `json:"input_mapping,omitempty"`
	OutputMapping    map[string]string `json:"output_mapping,omitempty"`
	TimeoutSeconds   int               `json:"timeout_seconds"`
	RetryCount       int               `json:"retry_count"`
	RequiresApproval bool              `json:"requires_approval"`
	KafkaTopic       string            `json:"kafka_topic,omitempty"`
	FluvioTopic      string            `json:"fluvio_topic,omitempty"`
	PermissionCheck  string            `json:"permission_check,omitempty"`
	LedgerEntryType  string            `json:"ledger_entry_type,omitempty"`
}

// Journey represents a complete user journey
type Journey struct {
	ID                       string   `json:"id"`
	Name                     string   `json:"name"`
	Description              string   `json:"description"`
	Category                 string   `json:"category"`
	Steps                    []Step   `json:"steps"`
	UIEntryPoint             string   `json:"ui_entry_point"`
	RequiredPermissions      []string `json:"required_permissions"`
	EstimatedDurationMinutes int      `json:"estimated_duration_minutes"`
	Tags                     []string `json:"tags"`
}

// Registry holds all available journeys
type Registry struct {
	journeys map[string]*Journey
}

// NewRegistry creates a new journey registry with built-in journeys
func NewRegistry() *Registry {
	r := &Registry{
		journeys: make(map[string]*Journey),
	}
	r.loadBuiltinJourneys()
	return r
}

// Get returns a journey by ID
func (r *Registry) Get(id string) *Journey {
	return r.journeys[id]
}

// List returns all journeys, optionally filtered by category
func (r *Registry) List(category string) []*Journey {
	var result []*Journey
	for _, j := range r.journeys {
		if category == "" || j.Category == category {
			result = append(result, j)
		}
	}
	return result
}

// Count returns the number of journeys
func (r *Registry) Count() int {
	return len(r.journeys)
}

// Register adds a journey to the registry
func (r *Registry) Register(j *Journey) {
	r.journeys[j.ID] = j
}

// GenerateWorkflowID generates a unique workflow ID for a journey
func GenerateWorkflowID(journeyID string) string {
	return fmt.Sprintf("journey-%s-%d", journeyID, time.Now().UnixNano())
}

// loadBuiltinJourneys loads the 30 built-in user journeys
func (r *Registry) loadBuiltinJourneys() {
	journeys := []*Journey{
		// Category: Project Management (1-3)
		{
			ID:                       "journey-001",
			Name:                     "Create Exploration Project",
			Description:              "Create a new mineral exploration project and onboard team members",
			Category:                 "project_management",
			UIEntryPoint:             "/projects",
			RequiredPermissions:      []string{"projects:create", "users:invite"},
			EstimatedDurationMinutes: 10,
			Tags:                     []string{"project", "onboarding", "team"},
			Steps: []Step{
				{ID: "step-001-1", Name: "Create Project", StepType: StepTypeAPICall, Endpoint: "/api/projects", Method: "POST", KafkaTopic: "mineralvision.projects.created", PermissionCheck: "projects:create", LedgerEntryType: "project_created", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-001-2", Name: "Invite Team Members", StepType: StepTypeAPICall, Endpoint: "/api/users/invite", Method: "POST", KafkaTopic: "mineralvision.users.invited", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-001-3", Name: "Record Audit Trail", StepType: StepTypeBlockchainRecord, Endpoint: "/api/blockchain/record", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		// Category: Data Ingestion (2-5)
		{
			ID:                       "journey-002",
			Name:                     "Upload Drillhole Data",
			Description:              "Upload and validate drillhole collar, survey, and assay data",
			Category:                 "data_ingestion",
			UIEntryPoint:             "/geology/drillholes",
			RequiredPermissions:      []string{"drillholes:create", "upload:write"},
			EstimatedDurationMinutes: 15,
			Tags:                     []string{"drillholes", "upload", "validation"},
			Steps: []Step{
				{ID: "step-002-1", Name: "Upload Collar File", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", KafkaTopic: "mineralvision.upload.started", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-002-2", Name: "Validate Data Format", StepType: StepTypeAPICall, Endpoint: "/api/drillholes/validate", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-002-3", Name: "Import Drillholes", StepType: StepTypeAPICall, Endpoint: "/api/drillholes/import", Method: "POST", KafkaTopic: "mineralvision.drillholes.imported", LedgerEntryType: "drillholes_imported", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-002-4", Name: "Store to Lakehouse", StepType: StepTypeAPICall, Endpoint: "/api/drillholes/persist", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-003",
			Name:                     "Upload Lab Samples",
			Description:              "Upload laboratory sample results and link to drillholes",
			Category:                 "data_ingestion",
			UIEntryPoint:             "/geology/samples",
			RequiredPermissions:      []string{"samples:create", "upload:write"},
			EstimatedDurationMinutes: 10,
			Tags:                     []string{"samples", "lab", "assays"},
			Steps: []Step{
				{ID: "step-003-1", Name: "Upload Sample File", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", KafkaTopic: "mineralvision.upload.started", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-003-2", Name: "Parse LIMS Format", StepType: StepTypeDataIngestion, Module: "src.api.ingestion.lims_ingestion", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-003-3", Name: "Import Samples", StepType: StepTypeAPICall, Endpoint: "/api/samples", Method: "POST", KafkaTopic: "mineralvision.samples.imported", LedgerEntryType: "samples_imported", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-004",
			Name:                     "Ingest GNSS Survey Data",
			Description:              "Import GNSS field survey tracks and validate positional accuracy",
			Category:                 "data_ingestion",
			UIEntryPoint:             "/gnss",
			RequiredPermissions:      []string{"gnss:write"},
			EstimatedDurationMinutes: 8,
			Tags:                     []string{"gnss", "survey", "positioning"},
			Steps: []Step{
				{ID: "step-004-1", Name: "Upload GNSS Data", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-004-2", Name: "Parse GNSS Format", StepType: StepTypeDataIngestion, Module: "src.api.ingestion.gnss_ingestion", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-004-3", Name: "Validate Accuracy", StepType: StepTypeAPICall, Endpoint: "/api/gnss/validate", Method: "POST", FluvioTopic: "mineralvision.gnss.validated", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-004-4", Name: "Store Survey", StepType: StepTypeAPICall, Endpoint: "/api/gnss/surveys", Method: "POST", LedgerEntryType: "gnss_survey_stored", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-005",
			Name:                     "Ingest LiDAR Point Cloud",
			Description:              "Import and process LiDAR point cloud data for terrain modeling",
			Category:                 "data_ingestion",
			UIEntryPoint:             "/sensors/sensor-fusion",
			RequiredPermissions:      []string{"sensor_fusion:write"},
			EstimatedDurationMinutes: 20,
			Tags:                     []string{"lidar", "terrain", "point_cloud"},
			Steps: []Step{
				{ID: "step-005-1", Name: "Upload LiDAR Data", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-005-2", Name: "Parse LiDAR Format", StepType: StepTypeDataIngestion, Module: "src.api.ingestion.lidar_ingestion", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-005-3", Name: "Process Point Cloud", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.lidar_adapter", FluvioTopic: "mineralvision.sensor.lidar.processed", TimeoutSeconds: 600, RetryCount: 3},
				{ID: "step-005-4", Name: "Generate DEM", StepType: StepTypeAPICall, Endpoint: "/api/sensor-fusion/lidar/dem", Method: "POST", KafkaTopic: "mineralvision.sensor.dem.generated", TimeoutSeconds: 600, RetryCount: 3},
			},
		},

		// Category: QA/QC (6-7)
		{
			ID:                       "journey-006",
			Name:                     "Run QA/QC Analysis",
			Description:              "Execute quality assurance checks on assay data and flag outliers",
			Category:                 "qaqc",
			UIEntryPoint:             "/geology/qaqc",
			RequiredPermissions:      []string{"qaqc:execute"},
			EstimatedDurationMinutes: 10,
			Tags:                     []string{"qaqc", "validation", "outliers"},
			Steps: []Step{
				{ID: "step-006-1", Name: "Fetch Sample Data", StepType: StepTypeAPICall, Endpoint: "/api/samples", Method: "GET", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-006-2", Name: "Run QA/QC Checks", StepType: StepTypeAPICall, Endpoint: "/api/qaqc/analyze", Method: "POST", KafkaTopic: "mineralvision.qaqc.completed", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-006-3", Name: "Flag Outliers", StepType: StepTypeAPICall, Endpoint: "/api/qaqc/outliers", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-006-4", Name: "Generate QA/QC Report", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-007",
			Name:                     "Review QA/QC Exceptions",
			Description:              "Review flagged QA/QC exceptions and approve/reject corrections",
			Category:                 "qaqc",
			UIEntryPoint:             "/geology/qaqc",
			RequiredPermissions:      []string{"qaqc:approve"},
			EstimatedDurationMinutes: 15,
			Tags:                     []string{"qaqc", "review", "approval"},
			Steps: []Step{
				{ID: "step-007-1", Name: "List Pending Exceptions", StepType: StepTypeAPICall, Endpoint: "/api/qaqc/exceptions", Method: "GET", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-007-2", Name: "Human Review", StepType: StepTypeHumanApproval, RequiresApproval: true, PermissionCheck: "qaqc:approve", TimeoutSeconds: 86400, RetryCount: 1},
				{ID: "step-007-3", Name: "Apply Corrections", StepType: StepTypeAPICall, Endpoint: "/api/qaqc/corrections", Method: "POST", KafkaTopic: "mineralvision.qaqc.corrections.applied", LedgerEntryType: "qaqc_corrections", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		// Category: Geostatistics (8-11)
		{
			ID:                       "journey-008",
			Name:                     "Run Variography Analysis",
			Description:              "Compute experimental variograms and fit theoretical models",
			Category:                 "geostatistics",
			UIEntryPoint:             "/geostatistics/variography",
			RequiredPermissions:      []string{"geostatistics:execute"},
			EstimatedDurationMinutes: 15,
			Tags:                     []string{"variography", "geostatistics", "spatial"},
			Steps: []Step{
				{ID: "step-008-1", Name: "Select Domain", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/domains", Method: "GET", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-008-2", Name: "Compute Experimental Variogram", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/variography/experimental", Method: "POST", KafkaTopic: "mineralvision.geostatistics.variogram.computed", TimeoutSeconds: 600, RetryCount: 3},
				{ID: "step-008-3", Name: "Fit Theoretical Model", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/variography/fit", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-008-4", Name: "Store Variogram Model", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/variography/save", Method: "POST", LedgerEntryType: "variogram_model_saved", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-009",
			Name:                     "Run Kriging Estimation",
			Description:              "Execute kriging interpolation to generate grade estimates",
			Category:                 "geostatistics",
			UIEntryPoint:             "/geostatistics/kriging",
			RequiredPermissions:      []string{"geostatistics:execute"},
			EstimatedDurationMinutes: 30,
			Tags:                     []string{"kriging", "estimation", "grades"},
			Steps: []Step{
				{ID: "step-009-1", Name: "Load Variogram Model", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/variography", Method: "GET", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-009-2", Name: "Configure Kriging Parameters", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/kriging/configure", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-009-3", Name: "Execute Kriging", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/kriging/run", Method: "POST", KafkaTopic: "mineralvision.geostatistics.kriging.completed", TimeoutSeconds: 1800, RetryCount: 3},
				{ID: "step-009-4", Name: "Store Results to Lakehouse", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/kriging/persist", Method: "POST", LedgerEntryType: "kriging_results_stored", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-010",
			Name:                     "Build Block Model",
			Description:              "Generate a 3D block model from kriged estimates",
			Category:                 "geostatistics",
			UIEntryPoint:             "/geostatistics/block-model",
			RequiredPermissions:      []string{"geostatistics:execute", "visualization:write"},
			EstimatedDurationMinutes: 45,
			Tags:                     []string{"block_model", "3d", "resource"},
			Steps: []Step{
				{ID: "step-010-1", Name: "Define Block Model Grid", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/block-model/grid", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-010-2", Name: "Populate Blocks", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/block-model/populate", Method: "POST", KafkaTopic: "mineralvision.geostatistics.blockmodel.populated", TimeoutSeconds: 3600, RetryCount: 3},
				{ID: "step-010-3", Name: "Calculate Resources", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/block-model/resources", Method: "POST", TimeoutSeconds: 600, RetryCount: 3},
				{ID: "step-010-4", Name: "Export Block Model", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/block-model/export", Method: "POST", LedgerEntryType: "block_model_exported", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-010-5", Name: "Record Provenance", StepType: StepTypeBlockchainRecord, Endpoint: "/api/blockchain/record", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		{
			ID:                       "journey-011",
			Name:                     "Generate Grade Shells",
			Description:              "Create grade shell surfaces for resource classification",
			Category:                 "geostatistics",
			UIEntryPoint:             "/geostatistics/grade-shells",
			RequiredPermissions:      []string{"geostatistics:execute"},
			EstimatedDurationMinutes: 20,
			Tags:                     []string{"grade_shells", "surfaces", "classification"},
			Steps: []Step{
				{ID: "step-011-1", Name: "Load Block Model", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/block-model", Method: "GET", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-011-2", Name: "Define Cut-off Grades", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/grade-shells/cutoffs", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
				{ID: "step-011-3", Name: "Generate Shells", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/grade-shells/generate", Method: "POST", KafkaTopic: "mineralvision.geostatistics.shells.generated", TimeoutSeconds: 600, RetryCount: 3},
				{ID: "step-011-4", Name: "Export Surfaces", StepType: StepTypeAPICall, Endpoint: "/api/geostatistics/grade-shells/export", Method: "POST", TimeoutSeconds: 300, RetryCount: 3},
			},
		},

		// Additional journeys (12-30) - abbreviated for space
		{ID: "journey-012", Name: "Run Geophysics Inversion", Description: "Execute geophysical inversion on survey data", Category: "geophysics", UIEntryPoint: "/geophysics/inversion", RequiredPermissions: []string{"geophysics:execute"}, EstimatedDurationMinutes: 60, Tags: []string{"inversion", "geophysics", "modeling"}, Steps: []Step{{ID: "step-012-1", Name: "Load Survey Data", StepType: StepTypeAPICall, Endpoint: "/api/inversion/surveys", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-012-2", Name: "Configure Inversion", StepType: StepTypeAPICall, Endpoint: "/api/inversion/configure", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-012-3", Name: "Run Inversion", StepType: StepTypeAPICall, Endpoint: "/api/inversion/run", Method: "POST", KafkaTopic: "mineralvision.geophysics.inversion.completed", TimeoutSeconds: 7200, RetryCount: 3}, {ID: "step-012-4", Name: "Store Results", StepType: StepTypeAPICall, Endpoint: "/api/inversion/persist", Method: "POST", LedgerEntryType: "inversion_results_stored", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-013", Name: "Advanced Geophysics Modeling", Description: "Run advanced geophysical modeling with uncertainty quantification", Category: "geophysics", UIEntryPoint: "/geophysics/inversion", RequiredPermissions: []string{"geophysics:execute"}, EstimatedDurationMinutes: 90, Tags: []string{"geophysics", "uncertainty", "advanced"}, Steps: []Step{{ID: "step-013-1", Name: "Load Inversion Results", StepType: StepTypeAPICall, Endpoint: "/api/inversion", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-013-2", Name: "Run Advanced Modeling", StepType: StepTypeAPICall, Endpoint: "/api/inversion/advanced", Method: "POST", TimeoutSeconds: 7200, RetryCount: 3}, {ID: "step-013-3", Name: "Quantify Uncertainty", StepType: StepTypeMLInference, Module: "src.api.ml.uncertainty_quantification", KafkaTopic: "mineralvision.geophysics.uncertainty.computed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-013-4", Name: "Generate Report", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-014", Name: "Multi-Sensor Fusion", Description: "Fuse magnetometry, radiometrics, and LiDAR data into unified anomaly layer", Category: "sensor_fusion", UIEntryPoint: "/sensors/sensor-fusion", RequiredPermissions: []string{"sensor_fusion:execute"}, EstimatedDurationMinutes: 30, Tags: []string{"fusion", "magnetometry", "radiometrics", "lidar"}, Steps: []Step{{ID: "step-014-1", Name: "Load Magnetometry Data", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.magnetometry_pipeline", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-014-2", Name: "Load Radiometrics Data", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.radiometrics_pipeline", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-014-3", Name: "Load LiDAR Data", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.lidar_adapter", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-014-4", Name: "Run Kalman Fusion", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.kalman_fusion", KafkaTopic: "mineralvision.sensor.fusion.completed", TimeoutSeconds: 1200, RetryCount: 3}, {ID: "step-014-5", Name: "Generate Anomaly Layer", StepType: StepTypeAPICall, Endpoint: "/api/sensor-fusion/anomaly", Method: "POST", LedgerEntryType: "anomaly_layer_generated", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-015", Name: "Process SEG-Y Seismic Data", Description: "Ingest and visualize SEG-Y seismic survey data", Category: "sensor_fusion", UIEntryPoint: "/sensors/sensor-fusion", RequiredPermissions: []string{"sensor_fusion:write"}, EstimatedDurationMinutes: 25, Tags: []string{"segy", "seismic", "visualization"}, Steps: []Step{{ID: "step-015-1", Name: "Upload SEG-Y File", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-015-2", Name: "Parse SEG-Y Format", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.segy_ingestion", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-015-3", Name: "Store to TileDB", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.tiledb_segy", KafkaTopic: "mineralvision.sensor.segy.stored", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-015-4", Name: "Generate Visualization", StepType: StepTypeVisualization, Module: "src.api.sensor_fusion.segy_visualization", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-016", Name: "Drone GPR Mission Processing", Description: "Process drone-mounted GPR survey data and fuse with other sensors", Category: "sensor_fusion", UIEntryPoint: "/sensors/sensor-fusion", RequiredPermissions: []string{"sensor_fusion:execute"}, EstimatedDurationMinutes: 35, Tags: []string{"drone", "gpr", "fusion"}, Steps: []Step{{ID: "step-016-1", Name: "Upload GPR Data", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-016-2", Name: "Process GPR Pipeline", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.gpr_pipeline", FluvioTopic: "mineralvision.sensor.gpr.processed", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-016-3", Name: "Integrate Drone Telemetry", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.drone_telemetry", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-016-4", Name: "Fuse with Drone GPR", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.drone_gpr", KafkaTopic: "mineralvision.sensor.drone_gpr.fused", TimeoutSeconds: 600, RetryCount: 3}}},

		{ID: "journey-017", Name: "Real-time Streaming Fusion", Description: "Stream and fuse sensor data in real-time from field devices", Category: "sensor_fusion", UIEntryPoint: "/sensors/sensor-fusion", RequiredPermissions: []string{"sensor_fusion:stream"}, EstimatedDurationMinutes: 0, Tags: []string{"streaming", "realtime", "iot"}, Steps: []Step{{ID: "step-017-1", Name: "Connect to Stream", StepType: StepTypeSensorFusion, Module: "src.api.sensor_fusion.streaming_fusion", FluvioTopic: "mineralvision.sensor.stream.raw", TimeoutSeconds: 0, RetryCount: 1}, {ID: "step-017-2", Name: "Apply Deep Learning Fusion", StepType: StepTypeMLInference, Module: "src.api.sensor_fusion.deep_learning_fusion", FluvioTopic: "mineralvision.sensor.stream.fused", TimeoutSeconds: 0, RetryCount: 1}, {ID: "step-017-3", Name: "Publish to Digital Twin", StepType: StepTypeEventPublish, Endpoint: "/api/digital-twin/stream", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-018", Name: "Gold Prospectivity Mapping", Description: "Generate gold prospectivity map using multi-modal ML features", Category: "ml_predictions", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"predictive_modeling:execute"}, EstimatedDurationMinutes: 45, Tags: []string{"gold", "prospectivity", "ml"}, Steps: []Step{{ID: "step-018-1", Name: "Load Feature Data", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/features", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-018-2", Name: "Run Gold Exploration Model", StepType: StepTypeMLInference, Module: "src.api.ml.gold_exploration", KafkaTopic: "mineralvision.ml.gold.inference.completed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-018-3", Name: "Run Prospectivity Workflow", StepType: StepTypeMLInference, Module: "src.api.ml.prospectivity_workflow", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-018-4", Name: "Generate Prospectivity Map", StepType: StepTypeVisualization, Endpoint: "/api/visualization/prospectivity", Method: "POST", LedgerEntryType: "prospectivity_map_generated", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-019", Name: "Lithium Target Assessment", Description: "Assess lithium exploration targets and rank by potential", Category: "ml_predictions", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"predictive_modeling:execute"}, EstimatedDurationMinutes: 40, Tags: []string{"lithium", "targets", "ranking"}, Steps: []Step{{ID: "step-019-1", Name: "Load Lithium Features", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/features", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-019-2", Name: "Run Lithium Model", StepType: StepTypeMLInference, Module: "src.api.ml.lithium_exploration", KafkaTopic: "mineralvision.ml.lithium.inference.completed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-019-3", Name: "Rank Targets", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/rank", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-019-4", Name: "Generate Report", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-020", Name: "Soil Suitability Assessment", Description: "Assess soil suitability for agricultural applications", Category: "ml_predictions", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"predictive_modeling:execute"}, EstimatedDurationMinutes: 30, Tags: []string{"soil", "agriculture", "suitability"}, Steps: []Step{{ID: "step-020-1", Name: "Load Soil Data", StepType: StepTypeAPICall, Endpoint: "/api/samples", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-020-2", Name: "Run Soil Suitability Model", StepType: StepTypeMLInference, Module: "src.api.ml.soil_suitability", KafkaTopic: "mineralvision.ml.soil.inference.completed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-020-3", Name: "Run Advanced Assessment", StepType: StepTypeMLInference, Module: "src.api.ml.advanced_soil_assessment", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-020-4", Name: "Generate Recommendations", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-021", Name: "Uncertainty Quantification", Description: "Quantify prediction uncertainty and generate confidence layers", Category: "ml_predictions", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"predictive_modeling:execute"}, EstimatedDurationMinutes: 25, Tags: []string{"uncertainty", "confidence", "validation"}, Steps: []Step{{ID: "step-021-1", Name: "Load Model Predictions", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/predictions", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-021-2", Name: "Run Uncertainty Quantification", StepType: StepTypeMLInference, Module: "src.api.ml.uncertainty_quantification", KafkaTopic: "mineralvision.ml.uncertainty.computed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-021-3", Name: "Generate Confidence Layers", StepType: StepTypeVisualization, Endpoint: "/api/visualization/confidence", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-022", Name: "Spatial Cross-Validation", Description: "Validate model generalization using spatial cross-validation", Category: "ml_predictions", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"predictive_modeling:execute"}, EstimatedDurationMinutes: 35, Tags: []string{"validation", "spatial_cv", "generalization"}, Steps: []Step{{ID: "step-022-1", Name: "Load Training Data", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/training-data", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-022-2", Name: "Run Spatial CV", StepType: StepTypeMLInference, Module: "src.api.ml.spatial_cv", KafkaTopic: "mineralvision.ml.spatial_cv.completed", TimeoutSeconds: 3600, RetryCount: 3}, {ID: "step-022-3", Name: "Store Metrics", StepType: StepTypeAPICall, Endpoint: "/api/predictive-modeling/metrics", Method: "POST", LedgerEntryType: "cv_metrics_stored", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-023", Name: "Molmo2 Drone Video Analysis", Description: "Analyze drone video footage using Molmo2 ensemble pipeline", Category: "vision_ai", UIEntryPoint: "/molmo2", RequiredPermissions: []string{"molmo:execute"}, EstimatedDurationMinutes: 20, Tags: []string{"molmo2", "drone", "video", "analysis"}, Steps: []Step{{ID: "step-023-1", Name: "Upload Video", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-023-2", Name: "Run Ensemble Pipeline", StepType: StepTypeMLInference, Module: "src.api.molmo.ensemble_pipeline", KafkaTopic: "mineralvision.molmo.ensemble.completed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-023-3", Name: "Run Drone Video Analysis", StepType: StepTypeMLInference, Module: "src.api.molmo.drone_video_analysis", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-023-4", Name: "Store Findings", StepType: StepTypeAPICall, Endpoint: "/api/molmo/findings", Method: "POST", LedgerEntryType: "molmo_analysis_stored", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-024", Name: "SAM3 Image Segmentation", Description: "Segment geological features using SAM3 with domain-specific prompts", Category: "vision_ai", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"sam3:execute"}, EstimatedDurationMinutes: 15, Tags: []string{"sam3", "segmentation", "geological"}, Steps: []Step{{ID: "step-024-1", Name: "Upload Image", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-024-2", Name: "Run SAM3 Segmentation", StepType: StepTypeMLInference, Module: "src.api.vision.sam3.sam3_segmenter", KafkaTopic: "mineralvision.sam3.segmentation.completed", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-024-3", Name: "Store Masks", StepType: StepTypeAPICall, Endpoint: "/api/sam3/masks", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-025", Name: "V-JEPA Feature Extraction", Description: "Extract visual features using V-JEPA for pretraining dataset", Category: "vision_ai", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"jepa:execute"}, EstimatedDurationMinutes: 30, Tags: []string{"vjepa", "features", "pretraining"}, Steps: []Step{{ID: "step-025-1", Name: "Load Image Archive", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-025-2", Name: "Run V-JEPA Integration", StepType: StepTypeMLInference, Module: "src.api.jepa.vjepa_integration", KafkaTopic: "mineralvision.jepa.features.extracted", TimeoutSeconds: 3600, RetryCount: 3}, {ID: "step-025-3", Name: "Store to Lakehouse", StepType: StepTypeMLInference, Module: "src.api.jepa.lakehouse_integration", LedgerEntryType: "jepa_features_stored", TimeoutSeconds: 600, RetryCount: 3}}},

		{ID: "journey-026", Name: "WALDO Object Detection", Description: "Detect mining equipment and features using YOLO11 + RF-DETR ensemble", Category: "vision_ai", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"waldo:execute"}, EstimatedDurationMinutes: 15, Tags: []string{"waldo", "detection", "yolo", "rfdetr"}, Steps: []Step{{ID: "step-026-1", Name: "Upload Image", StepType: StepTypeDataIngestion, Endpoint: "/api/upload", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-026-2", Name: "Run Ensemble Detector", StepType: StepTypeMLInference, Module: "src.api.waldo.ensemble_detector", KafkaTopic: "mineralvision.waldo.detection.completed", TimeoutSeconds: 600, RetryCount: 3}, {ID: "step-026-3", Name: "Store Detections", StepType: StepTypeAPICall, Endpoint: "/api/waldo/detections", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-027", Name: "Digital Twin Session", Description: "Start a digital twin session with real-time streaming and 3D visualization", Category: "digital_twin", UIEntryPoint: "/visualization/3d", RequiredPermissions: []string{"digital_twin:execute"}, EstimatedDurationMinutes: 0, Tags: []string{"digital_twin", "3d", "realtime"}, Steps: []Step{{ID: "step-027-1", Name: "Initialize Digital Twin", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/initialize", Method: "POST", KafkaTopic: "mineralvision.digital_twin.session.started", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-027-2", Name: "Start Real-time Streaming", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/stream/start", Method: "POST", FluvioTopic: "mineralvision.digital_twin.stream", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-027-3", Name: "Load 3D Visualization", StepType: StepTypeVisualization, Module: "src.api.digital_twin.visualization_3d", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-028", Name: "What-If Simulation", Description: "Run what-if simulation scenarios on the digital twin", Category: "digital_twin", UIEntryPoint: "/visualization/3d", RequiredPermissions: []string{"digital_twin:simulate"}, EstimatedDurationMinutes: 20, Tags: []string{"simulation", "what_if", "scenario"}, Steps: []Step{{ID: "step-028-1", Name: "Load Digital Twin State", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/state", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-028-2", Name: "Configure Scenario", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/scenario", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-028-3", Name: "Run Simulation", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/simulate", Method: "POST", KafkaTopic: "mineralvision.digital_twin.simulation.completed", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-028-4", Name: "Generate Scenario Report", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", LedgerEntryType: "simulation_report_generated", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-029", Name: "Blockchain Data Provenance", Description: "Record data provenance on blockchain for regulatory compliance", Category: "compliance", UIEntryPoint: "/settings", RequiredPermissions: []string{"blockchain:write"}, EstimatedDurationMinutes: 5, Tags: []string{"blockchain", "provenance", "compliance"}, Steps: []Step{{ID: "step-029-1", Name: "Select Data Assets", StepType: StepTypeAPICall, Endpoint: "/api/blockchain/assets", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-029-2", Name: "Record Provenance", StepType: StepTypeBlockchainRecord, Endpoint: "/api/blockchain/record", Method: "POST", KafkaTopic: "mineralvision.blockchain.provenance.recorded", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-029-3", Name: "Generate Compliance Certificate", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", LedgerEntryType: "compliance_certificate_generated", TimeoutSeconds: 300, RetryCount: 3}}},

		{ID: "journey-030", Name: "Autonomous Exploration Recommendation", Description: "Generate autonomous exploration recommendations based on fused data", Category: "autonomous", UIEntryPoint: "/ai-insights", RequiredPermissions: []string{"autonomous_exploration:execute"}, EstimatedDurationMinutes: 30, Tags: []string{"autonomous", "exploration", "recommendation"}, Steps: []Step{{ID: "step-030-1", Name: "Load Fused Data", StepType: StepTypeAPICall, Endpoint: "/api/sensor-fusion/fused", Method: "GET", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-030-2", Name: "Run Autonomous Exploration", StepType: StepTypeAPICall, Endpoint: "/api/autonomous-exploration/recommend", Method: "POST", KafkaTopic: "mineralvision.autonomous.recommendation.generated", TimeoutSeconds: 1800, RetryCount: 3}, {ID: "step-030-3", Name: "Human Review", StepType: StepTypeHumanApproval, RequiresApproval: true, PermissionCheck: "autonomous_exploration:approve", TimeoutSeconds: 86400, RetryCount: 1}, {ID: "step-030-4", Name: "Generate Survey Plan", StepType: StepTypeReportGeneration, Endpoint: "/api/reports", Method: "POST", TimeoutSeconds: 300, RetryCount: 3}, {ID: "step-030-5", Name: "Update Digital Twin", StepType: StepTypeAPICall, Endpoint: "/api/digital-twin/plan", Method: "POST", LedgerEntryType: "survey_plan_approved", TimeoutSeconds: 300, RetryCount: 3}}},
	}

	for _, j := range journeys {
		r.Register(j)
	}
}
