"""
Journey Manifest System for MineralVision

Defines the 30 user journeys as composable workflows that map to existing
platform components. Each journey is a sequence of steps that call existing
API endpoints and modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    """Types of journey steps."""
    API_CALL = "api_call"
    ML_INFERENCE = "ml_inference"
    SENSOR_FUSION = "sensor_fusion"
    DATA_INGESTION = "data_ingestion"
    REPORT_GENERATION = "report_generation"
    VISUALIZATION = "visualization"
    BLOCKCHAIN_RECORD = "blockchain_record"
    HUMAN_APPROVAL = "human_approval"
    EVENT_PUBLISH = "event_publish"
    LEDGER_WRITE = "ledger_write"


@dataclass
class JourneyStep:
    """A single step in a user journey."""
    id: str
    name: str
    step_type: StepType
    endpoint: Optional[str] = None
    module: Optional[str] = None
    method: str = "POST"
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    requires_approval: bool = False
    kafka_topic: Optional[str] = None
    fluvio_topic: Optional[str] = None
    permission_check: Optional[str] = None
    ledger_entry_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type.value,
            "endpoint": self.endpoint,
            "module": self.module,
            "method": self.method,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "requires_approval": self.requires_approval,
            "kafka_topic": self.kafka_topic,
            "fluvio_topic": self.fluvio_topic,
            "permission_check": self.permission_check,
            "ledger_entry_type": self.ledger_entry_type,
        }


@dataclass
class JourneyManifest:
    """Definition of a complete user journey."""
    id: str
    name: str
    description: str
    category: str
    steps: List[JourneyStep]
    ui_entry_point: str
    required_permissions: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 5
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "ui_entry_point": self.ui_entry_point,
            "required_permissions": self.required_permissions,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JourneyManifest":
        steps = [
            JourneyStep(
                id=s["id"],
                name=s["name"],
                step_type=StepType(s["step_type"]),
                endpoint=s.get("endpoint"),
                module=s.get("module"),
                method=s.get("method", "POST"),
                input_mapping=s.get("input_mapping", {}),
                output_mapping=s.get("output_mapping", {}),
                timeout_seconds=s.get("timeout_seconds", 300),
                retry_count=s.get("retry_count", 3),
                requires_approval=s.get("requires_approval", False),
                kafka_topic=s.get("kafka_topic"),
                fluvio_topic=s.get("fluvio_topic"),
                permission_check=s.get("permission_check"),
                ledger_entry_type=s.get("ledger_entry_type"),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=data["category"],
            steps=steps,
            ui_entry_point=data["ui_entry_point"],
            required_permissions=data.get("required_permissions", []),
            estimated_duration_minutes=data.get("estimated_duration_minutes", 5),
            tags=data.get("tags", []),
        )


class JourneyRegistry:
    """Registry of all available user journeys."""
    
    def __init__(self):
        self._journeys: Dict[str, JourneyManifest] = {}
        self._load_builtin_journeys()
    
    def _load_builtin_journeys(self):
        """Load the 30 built-in user journeys."""
        journeys = self._define_journeys()
        for journey in journeys:
            self._journeys[journey.id] = journey
        logger.info(f"Loaded {len(self._journeys)} built-in journeys")
    
    def get(self, journey_id: str) -> Optional[JourneyManifest]:
        return self._journeys.get(journey_id)
    
    def list_all(self) -> List[JourneyManifest]:
        return list(self._journeys.values())
    
    def list_by_category(self, category: str) -> List[JourneyManifest]:
        return [j for j in self._journeys.values() if j.category == category]
    
    def register(self, journey: JourneyManifest):
        self._journeys[journey.id] = journey
    
    def export_all(self, path: Path):
        """Export all journeys to JSON file."""
        data = {"journeys": [j.to_dict() for j in self._journeys.values()]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _define_journeys(self) -> List[JourneyManifest]:
        """Define the 30 user journeys based on existing platform components."""
        return [
            # Category: Project Management (1-3)
            JourneyManifest(
                id="journey-001",
                name="Create Exploration Project",
                description="Create a new mineral exploration project and onboard team members",
                category="project_management",
                ui_entry_point="/projects",
                required_permissions=["projects:create", "users:invite"],
                estimated_duration_minutes=10,
                tags=["project", "onboarding", "team"],
                steps=[
                    JourneyStep(
                        id="step-001-1",
                        name="Create Project",
                        step_type=StepType.API_CALL,
                        endpoint="/api/projects",
                        method="POST",
                        kafka_topic="mineralvision.projects.created",
                        permission_check="projects:create",
                        ledger_entry_type="project_created",
                    ),
                    JourneyStep(
                        id="step-001-2",
                        name="Invite Team Members",
                        step_type=StepType.API_CALL,
                        endpoint="/api/users/invite",
                        method="POST",
                        input_mapping={"project_id": "$.step-001-1.id"},
                        kafka_topic="mineralvision.users.invited",
                    ),
                    JourneyStep(
                        id="step-001-3",
                        name="Record Audit Trail",
                        step_type=StepType.BLOCKCHAIN_RECORD,
                        endpoint="/api/blockchain/record",
                        method="POST",
                        input_mapping={"project_id": "$.step-001-1.id", "action": "project_created"},
                    ),
                ],
            ),
            
            # Category: Data Ingestion (2-5)
            JourneyManifest(
                id="journey-002",
                name="Upload Drillhole Data",
                description="Upload and validate drillhole collar, survey, and assay data",
                category="data_ingestion",
                ui_entry_point="/geology/drillholes",
                required_permissions=["drillholes:create", "upload:write"],
                estimated_duration_minutes=15,
                tags=["drillholes", "upload", "validation"],
                steps=[
                    JourneyStep(
                        id="step-002-1",
                        name="Upload Collar File",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                        kafka_topic="mineralvision.upload.started",
                    ),
                    JourneyStep(
                        id="step-002-2",
                        name="Validate Data Format",
                        step_type=StepType.API_CALL,
                        endpoint="/api/drillholes/validate",
                        method="POST",
                        input_mapping={"file_id": "$.step-002-1.file_id"},
                    ),
                    JourneyStep(
                        id="step-002-3",
                        name="Import Drillholes",
                        step_type=StepType.API_CALL,
                        endpoint="/api/drillholes/import",
                        method="POST",
                        input_mapping={"file_id": "$.step-002-1.file_id"},
                        kafka_topic="mineralvision.drillholes.imported",
                        ledger_entry_type="drillholes_imported",
                    ),
                    JourneyStep(
                        id="step-002-4",
                        name="Store to Lakehouse",
                        step_type=StepType.API_CALL,
                        endpoint="/api/drillholes/persist",
                        method="POST",
                        input_mapping={"drillhole_ids": "$.step-002-3.drillhole_ids"},
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-003",
                name="Upload Lab Samples",
                description="Upload laboratory sample results and link to drillholes",
                category="data_ingestion",
                ui_entry_point="/geology/samples",
                required_permissions=["samples:create", "upload:write"],
                estimated_duration_minutes=10,
                tags=["samples", "lab", "assays"],
                steps=[
                    JourneyStep(
                        id="step-003-1",
                        name="Upload Sample File",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                        kafka_topic="mineralvision.upload.started",
                    ),
                    JourneyStep(
                        id="step-003-2",
                        name="Parse LIMS Format",
                        step_type=StepType.DATA_INGESTION,
                        module="src.api.ingestion.lims_ingestion",
                        input_mapping={"file_id": "$.step-003-1.file_id"},
                    ),
                    JourneyStep(
                        id="step-003-3",
                        name="Import Samples",
                        step_type=StepType.API_CALL,
                        endpoint="/api/samples",
                        method="POST",
                        kafka_topic="mineralvision.samples.imported",
                        ledger_entry_type="samples_imported",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-004",
                name="Ingest GNSS Survey Data",
                description="Import GNSS field survey tracks and validate positional accuracy",
                category="data_ingestion",
                ui_entry_point="/gnss",
                required_permissions=["gnss:write"],
                estimated_duration_minutes=8,
                tags=["gnss", "survey", "positioning"],
                steps=[
                    JourneyStep(
                        id="step-004-1",
                        name="Upload GNSS Data",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-004-2",
                        name="Parse GNSS Format",
                        step_type=StepType.DATA_INGESTION,
                        module="src.api.ingestion.gnss_ingestion",
                        input_mapping={"file_id": "$.step-004-1.file_id"},
                    ),
                    JourneyStep(
                        id="step-004-3",
                        name="Validate Accuracy",
                        step_type=StepType.API_CALL,
                        endpoint="/api/gnss/validate",
                        method="POST",
                        fluvio_topic="mineralvision.gnss.validated",
                    ),
                    JourneyStep(
                        id="step-004-4",
                        name="Store Survey",
                        step_type=StepType.API_CALL,
                        endpoint="/api/gnss/surveys",
                        method="POST",
                        ledger_entry_type="gnss_survey_stored",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-005",
                name="Ingest LiDAR Point Cloud",
                description="Import and process LiDAR point cloud data for terrain modeling",
                category="data_ingestion",
                ui_entry_point="/sensors/sensor-fusion",
                required_permissions=["sensor_fusion:write"],
                estimated_duration_minutes=20,
                tags=["lidar", "terrain", "point_cloud"],
                steps=[
                    JourneyStep(
                        id="step-005-1",
                        name="Upload LiDAR Data",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-005-2",
                        name="Parse LiDAR Format",
                        step_type=StepType.DATA_INGESTION,
                        module="src.api.ingestion.lidar_ingestion",
                        input_mapping={"file_id": "$.step-005-1.file_id"},
                    ),
                    JourneyStep(
                        id="step-005-3",
                        name="Process Point Cloud",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.lidar_adapter",
                        fluvio_topic="mineralvision.sensor.lidar.processed",
                    ),
                    JourneyStep(
                        id="step-005-4",
                        name="Generate DEM",
                        step_type=StepType.API_CALL,
                        endpoint="/api/sensor-fusion/lidar/dem",
                        method="POST",
                        kafka_topic="mineralvision.sensor.dem.generated",
                    ),
                ],
            ),
            
            # Category: QA/QC (6-7)
            JourneyManifest(
                id="journey-006",
                name="Run QA/QC Analysis",
                description="Execute quality assurance checks on assay data and flag outliers",
                category="qaqc",
                ui_entry_point="/geology/qaqc",
                required_permissions=["qaqc:execute"],
                estimated_duration_minutes=10,
                tags=["qaqc", "validation", "outliers"],
                steps=[
                    JourneyStep(
                        id="step-006-1",
                        name="Fetch Sample Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/samples",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-006-2",
                        name="Run QA/QC Checks",
                        step_type=StepType.API_CALL,
                        endpoint="/api/qaqc/analyze",
                        method="POST",
                        input_mapping={"sample_ids": "$.step-006-1.sample_ids"},
                        kafka_topic="mineralvision.qaqc.completed",
                    ),
                    JourneyStep(
                        id="step-006-3",
                        name="Flag Outliers",
                        step_type=StepType.API_CALL,
                        endpoint="/api/qaqc/outliers",
                        method="POST",
                        input_mapping={"analysis_id": "$.step-006-2.analysis_id"},
                    ),
                    JourneyStep(
                        id="step-006-4",
                        name="Generate QA/QC Report",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                        input_mapping={"report_type": "qaqc", "analysis_id": "$.step-006-2.analysis_id"},
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-007",
                name="Review QA/QC Exceptions",
                description="Review flagged QA/QC exceptions and approve/reject corrections",
                category="qaqc",
                ui_entry_point="/geology/qaqc",
                required_permissions=["qaqc:approve"],
                estimated_duration_minutes=15,
                tags=["qaqc", "review", "approval"],
                steps=[
                    JourneyStep(
                        id="step-007-1",
                        name="List Pending Exceptions",
                        step_type=StepType.API_CALL,
                        endpoint="/api/qaqc/exceptions",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-007-2",
                        name="Human Review",
                        step_type=StepType.HUMAN_APPROVAL,
                        requires_approval=True,
                        permission_check="qaqc:approve",
                    ),
                    JourneyStep(
                        id="step-007-3",
                        name="Apply Corrections",
                        step_type=StepType.API_CALL,
                        endpoint="/api/qaqc/corrections",
                        method="POST",
                        input_mapping={"approved_corrections": "$.step-007-2.approved"},
                        kafka_topic="mineralvision.qaqc.corrections.applied",
                        ledger_entry_type="qaqc_corrections",
                    ),
                ],
            ),
            
            # Category: Geostatistics (8-11)
            JourneyManifest(
                id="journey-008",
                name="Run Variography Analysis",
                description="Compute experimental variograms and fit theoretical models",
                category="geostatistics",
                ui_entry_point="/geostatistics/variography",
                required_permissions=["geostatistics:execute"],
                estimated_duration_minutes=15,
                tags=["variography", "geostatistics", "spatial"],
                steps=[
                    JourneyStep(
                        id="step-008-1",
                        name="Select Domain",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/domains",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-008-2",
                        name="Compute Experimental Variogram",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/variography/experimental",
                        method="POST",
                        kafka_topic="mineralvision.geostatistics.variogram.computed",
                    ),
                    JourneyStep(
                        id="step-008-3",
                        name="Fit Theoretical Model",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/variography/fit",
                        method="POST",
                        input_mapping={"variogram_id": "$.step-008-2.variogram_id"},
                    ),
                    JourneyStep(
                        id="step-008-4",
                        name="Store Variogram Model",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/variography/save",
                        method="POST",
                        ledger_entry_type="variogram_model_saved",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-009",
                name="Run Kriging Estimation",
                description="Execute kriging interpolation to generate grade estimates",
                category="geostatistics",
                ui_entry_point="/geostatistics/kriging",
                required_permissions=["geostatistics:execute"],
                estimated_duration_minutes=30,
                tags=["kriging", "estimation", "grades"],
                steps=[
                    JourneyStep(
                        id="step-009-1",
                        name="Load Variogram Model",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/variography",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-009-2",
                        name="Configure Kriging Parameters",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/kriging/configure",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-009-3",
                        name="Execute Kriging",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/kriging/run",
                        method="POST",
                        timeout_seconds=1800,
                        kafka_topic="mineralvision.geostatistics.kriging.completed",
                    ),
                    JourneyStep(
                        id="step-009-4",
                        name="Store Results to Lakehouse",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/kriging/persist",
                        method="POST",
                        ledger_entry_type="kriging_results_stored",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-010",
                name="Build Block Model",
                description="Generate a 3D block model from kriged estimates",
                category="geostatistics",
                ui_entry_point="/geostatistics/block-model",
                required_permissions=["geostatistics:execute", "visualization:write"],
                estimated_duration_minutes=45,
                tags=["block_model", "3d", "resource"],
                steps=[
                    JourneyStep(
                        id="step-010-1",
                        name="Define Block Model Grid",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/block-model/grid",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-010-2",
                        name="Populate Blocks",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/block-model/populate",
                        method="POST",
                        timeout_seconds=3600,
                        kafka_topic="mineralvision.geostatistics.blockmodel.populated",
                    ),
                    JourneyStep(
                        id="step-010-3",
                        name="Calculate Resources",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/block-model/resources",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-010-4",
                        name="Export Block Model",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/block-model/export",
                        method="POST",
                        ledger_entry_type="block_model_exported",
                    ),
                    JourneyStep(
                        id="step-010-5",
                        name="Record Provenance",
                        step_type=StepType.BLOCKCHAIN_RECORD,
                        endpoint="/api/blockchain/record",
                        method="POST",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-011",
                name="Generate Grade Shells",
                description="Create grade shell surfaces for resource classification",
                category="geostatistics",
                ui_entry_point="/geostatistics/grade-shells",
                required_permissions=["geostatistics:execute"],
                estimated_duration_minutes=20,
                tags=["grade_shells", "surfaces", "classification"],
                steps=[
                    JourneyStep(
                        id="step-011-1",
                        name="Load Block Model",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/block-model",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-011-2",
                        name="Define Cut-off Grades",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/grade-shells/cutoffs",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-011-3",
                        name="Generate Shells",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/grade-shells/generate",
                        method="POST",
                        kafka_topic="mineralvision.geostatistics.shells.generated",
                    ),
                    JourneyStep(
                        id="step-011-4",
                        name="Export Surfaces",
                        step_type=StepType.API_CALL,
                        endpoint="/api/geostatistics/grade-shells/export",
                        method="POST",
                    ),
                ],
            ),
            
            # Category: Geophysics (12-13)
            JourneyManifest(
                id="journey-012",
                name="Run Geophysics Inversion",
                description="Execute geophysical inversion on survey data",
                category="geophysics",
                ui_entry_point="/geophysics/inversion",
                required_permissions=["geophysics:execute"],
                estimated_duration_minutes=60,
                tags=["inversion", "geophysics", "modeling"],
                steps=[
                    JourneyStep(
                        id="step-012-1",
                        name="Load Survey Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion/surveys",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-012-2",
                        name="Configure Inversion",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion/configure",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-012-3",
                        name="Run Inversion",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion/run",
                        method="POST",
                        timeout_seconds=7200,
                        kafka_topic="mineralvision.geophysics.inversion.completed",
                    ),
                    JourneyStep(
                        id="step-012-4",
                        name="Store Results",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion/persist",
                        method="POST",
                        ledger_entry_type="inversion_results_stored",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-013",
                name="Advanced Geophysics Modeling",
                description="Run advanced geophysical modeling with uncertainty quantification",
                category="geophysics",
                ui_entry_point="/geophysics/inversion",
                required_permissions=["geophysics:execute"],
                estimated_duration_minutes=90,
                tags=["geophysics", "uncertainty", "advanced"],
                steps=[
                    JourneyStep(
                        id="step-013-1",
                        name="Load Inversion Results",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-013-2",
                        name="Run Advanced Modeling",
                        step_type=StepType.API_CALL,
                        endpoint="/api/inversion/advanced",
                        method="POST",
                        timeout_seconds=7200,
                    ),
                    JourneyStep(
                        id="step-013-3",
                        name="Quantify Uncertainty",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.uncertainty_quantification",
                        kafka_topic="mineralvision.geophysics.uncertainty.computed",
                    ),
                    JourneyStep(
                        id="step-013-4",
                        name="Generate Report",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                    ),
                ],
            ),
            
            # Category: Sensor Fusion (14-17)
            JourneyManifest(
                id="journey-014",
                name="Multi-Sensor Fusion",
                description="Fuse magnetometry, radiometrics, and LiDAR data into unified anomaly layer",
                category="sensor_fusion",
                ui_entry_point="/sensors/sensor-fusion",
                required_permissions=["sensor_fusion:execute"],
                estimated_duration_minutes=30,
                tags=["fusion", "magnetometry", "radiometrics", "lidar"],
                steps=[
                    JourneyStep(
                        id="step-014-1",
                        name="Load Magnetometry Data",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.magnetometry_pipeline",
                    ),
                    JourneyStep(
                        id="step-014-2",
                        name="Load Radiometrics Data",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.radiometrics_pipeline",
                    ),
                    JourneyStep(
                        id="step-014-3",
                        name="Load LiDAR Data",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.lidar_adapter",
                    ),
                    JourneyStep(
                        id="step-014-4",
                        name="Run Kalman Fusion",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.kalman_fusion",
                        kafka_topic="mineralvision.sensor.fusion.completed",
                    ),
                    JourneyStep(
                        id="step-014-5",
                        name="Generate Anomaly Layer",
                        step_type=StepType.API_CALL,
                        endpoint="/api/sensor-fusion/anomaly",
                        method="POST",
                        ledger_entry_type="anomaly_layer_generated",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-015",
                name="Process SEG-Y Seismic Data",
                description="Ingest and visualize SEG-Y seismic survey data",
                category="sensor_fusion",
                ui_entry_point="/sensors/sensor-fusion",
                required_permissions=["sensor_fusion:write"],
                estimated_duration_minutes=25,
                tags=["segy", "seismic", "visualization"],
                steps=[
                    JourneyStep(
                        id="step-015-1",
                        name="Upload SEG-Y File",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-015-2",
                        name="Parse SEG-Y Format",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.segy_ingestion",
                    ),
                    JourneyStep(
                        id="step-015-3",
                        name="Store to TileDB",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.tiledb_segy",
                        kafka_topic="mineralvision.sensor.segy.stored",
                    ),
                    JourneyStep(
                        id="step-015-4",
                        name="Generate Visualization",
                        step_type=StepType.VISUALIZATION,
                        module="src.api.sensor_fusion.segy_visualization",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-016",
                name="Drone GPR Mission Processing",
                description="Process drone-mounted GPR survey data and fuse with other sensors",
                category="sensor_fusion",
                ui_entry_point="/sensors/sensor-fusion",
                required_permissions=["sensor_fusion:execute"],
                estimated_duration_minutes=35,
                tags=["drone", "gpr", "fusion"],
                steps=[
                    JourneyStep(
                        id="step-016-1",
                        name="Upload GPR Data",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-016-2",
                        name="Process GPR Pipeline",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.gpr_pipeline",
                        fluvio_topic="mineralvision.sensor.gpr.processed",
                    ),
                    JourneyStep(
                        id="step-016-3",
                        name="Integrate Drone Telemetry",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.drone_telemetry",
                    ),
                    JourneyStep(
                        id="step-016-4",
                        name="Fuse with Drone GPR",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.drone_gpr",
                        kafka_topic="mineralvision.sensor.drone_gpr.fused",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-017",
                name="Real-time Streaming Fusion",
                description="Stream and fuse sensor data in real-time from field devices",
                category="sensor_fusion",
                ui_entry_point="/sensors/sensor-fusion",
                required_permissions=["sensor_fusion:stream"],
                estimated_duration_minutes=0,
                tags=["streaming", "realtime", "iot"],
                steps=[
                    JourneyStep(
                        id="step-017-1",
                        name="Connect to Stream",
                        step_type=StepType.SENSOR_FUSION,
                        module="src.api.sensor_fusion.streaming_fusion",
                        fluvio_topic="mineralvision.sensor.stream.raw",
                    ),
                    JourneyStep(
                        id="step-017-2",
                        name="Apply Deep Learning Fusion",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.sensor_fusion.deep_learning_fusion",
                        fluvio_topic="mineralvision.sensor.stream.fused",
                    ),
                    JourneyStep(
                        id="step-017-3",
                        name="Publish to Digital Twin",
                        step_type=StepType.EVENT_PUBLISH,
                        endpoint="/api/digital-twin/stream",
                        method="POST",
                    ),
                ],
            ),
            
            # Category: AI/ML Predictions (18-22)
            JourneyManifest(
                id="journey-018",
                name="Gold Prospectivity Mapping",
                description="Generate gold prospectivity map using multi-modal ML features",
                category="ml_predictions",
                ui_entry_point="/ai-insights",
                required_permissions=["predictive_modeling:execute"],
                estimated_duration_minutes=45,
                tags=["gold", "prospectivity", "ml"],
                steps=[
                    JourneyStep(
                        id="step-018-1",
                        name="Load Feature Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/features",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-018-2",
                        name="Run Gold Exploration Model",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.gold_exploration",
                        timeout_seconds=1800,
                        kafka_topic="mineralvision.ml.gold.inference.completed",
                    ),
                    JourneyStep(
                        id="step-018-3",
                        name="Run Prospectivity Workflow",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.prospectivity_workflow",
                    ),
                    JourneyStep(
                        id="step-018-4",
                        name="Generate Prospectivity Map",
                        step_type=StepType.VISUALIZATION,
                        endpoint="/api/visualization/prospectivity",
                        method="POST",
                        ledger_entry_type="prospectivity_map_generated",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-019",
                name="Lithium Target Assessment",
                description="Assess lithium exploration targets and rank by potential",
                category="ml_predictions",
                ui_entry_point="/ai-insights",
                required_permissions=["predictive_modeling:execute"],
                estimated_duration_minutes=40,
                tags=["lithium", "targets", "ranking"],
                steps=[
                    JourneyStep(
                        id="step-019-1",
                        name="Load Lithium Features",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/features",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-019-2",
                        name="Run Lithium Model",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.lithium_exploration",
                        timeout_seconds=1800,
                        kafka_topic="mineralvision.ml.lithium.inference.completed",
                    ),
                    JourneyStep(
                        id="step-019-3",
                        name="Rank Targets",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/rank",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-019-4",
                        name="Generate Report",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-020",
                name="Soil Suitability Assessment",
                description="Assess soil suitability for agricultural applications",
                category="ml_predictions",
                ui_entry_point="/ai-insights",
                required_permissions=["predictive_modeling:execute"],
                estimated_duration_minutes=30,
                tags=["soil", "agriculture", "suitability"],
                steps=[
                    JourneyStep(
                        id="step-020-1",
                        name="Load Soil Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/samples",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-020-2",
                        name="Run Soil Suitability Model",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.soil_suitability",
                        kafka_topic="mineralvision.ml.soil.inference.completed",
                    ),
                    JourneyStep(
                        id="step-020-3",
                        name="Run Advanced Assessment",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.advanced_soil_assessment",
                    ),
                    JourneyStep(
                        id="step-020-4",
                        name="Generate Recommendations",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-021",
                name="Uncertainty Quantification",
                description="Quantify prediction uncertainty and generate confidence layers",
                category="ml_predictions",
                ui_entry_point="/ai-insights",
                required_permissions=["predictive_modeling:execute"],
                estimated_duration_minutes=25,
                tags=["uncertainty", "confidence", "validation"],
                steps=[
                    JourneyStep(
                        id="step-021-1",
                        name="Load Model Predictions",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/predictions",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-021-2",
                        name="Run Uncertainty Quantification",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.uncertainty_quantification",
                        kafka_topic="mineralvision.ml.uncertainty.computed",
                    ),
                    JourneyStep(
                        id="step-021-3",
                        name="Generate Confidence Layers",
                        step_type=StepType.VISUALIZATION,
                        endpoint="/api/visualization/confidence",
                        method="POST",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-022",
                name="Spatial Cross-Validation",
                description="Validate model generalization using spatial cross-validation",
                category="ml_predictions",
                ui_entry_point="/ai-insights",
                required_permissions=["predictive_modeling:execute"],
                estimated_duration_minutes=35,
                tags=["validation", "spatial_cv", "generalization"],
                steps=[
                    JourneyStep(
                        id="step-022-1",
                        name="Load Training Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/training-data",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-022-2",
                        name="Run Spatial CV",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.ml.spatial_cv",
                        timeout_seconds=3600,
                        kafka_topic="mineralvision.ml.spatial_cv.completed",
                    ),
                    JourneyStep(
                        id="step-022-3",
                        name="Store Metrics",
                        step_type=StepType.API_CALL,
                        endpoint="/api/predictive-modeling/metrics",
                        method="POST",
                        ledger_entry_type="cv_metrics_stored",
                    ),
                ],
            ),
            
            # Category: Vision/AI (23-26)
            JourneyManifest(
                id="journey-023",
                name="Molmo2 Drone Video Analysis",
                description="Analyze drone video footage using Molmo2 ensemble pipeline",
                category="vision_ai",
                ui_entry_point="/molmo2",
                required_permissions=["molmo:execute"],
                estimated_duration_minutes=20,
                tags=["molmo2", "drone", "video", "analysis"],
                steps=[
                    JourneyStep(
                        id="step-023-1",
                        name="Upload Video",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-023-2",
                        name="Run Ensemble Pipeline",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.molmo.ensemble_pipeline",
                        timeout_seconds=1800,
                        kafka_topic="mineralvision.molmo.ensemble.completed",
                    ),
                    JourneyStep(
                        id="step-023-3",
                        name="Run Drone Video Analysis",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.molmo.drone_video_analysis",
                    ),
                    JourneyStep(
                        id="step-023-4",
                        name="Store Findings",
                        step_type=StepType.API_CALL,
                        endpoint="/api/molmo/findings",
                        method="POST",
                        ledger_entry_type="molmo_analysis_stored",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-024",
                name="SAM3 Image Segmentation",
                description="Segment geological features using SAM3 with domain-specific prompts",
                category="vision_ai",
                ui_entry_point="/ai-insights",
                required_permissions=["sam3:execute"],
                estimated_duration_minutes=15,
                tags=["sam3", "segmentation", "geological"],
                steps=[
                    JourneyStep(
                        id="step-024-1",
                        name="Upload Image",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-024-2",
                        name="Run SAM3 Segmentation",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.vision.sam3.sam3_segmenter",
                        kafka_topic="mineralvision.sam3.segmentation.completed",
                    ),
                    JourneyStep(
                        id="step-024-3",
                        name="Store Masks",
                        step_type=StepType.API_CALL,
                        endpoint="/api/sam3/masks",
                        method="POST",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-025",
                name="V-JEPA Feature Extraction",
                description="Extract visual features using V-JEPA for pretraining dataset",
                category="vision_ai",
                ui_entry_point="/ai-insights",
                required_permissions=["jepa:execute"],
                estimated_duration_minutes=30,
                tags=["vjepa", "features", "pretraining"],
                steps=[
                    JourneyStep(
                        id="step-025-1",
                        name="Load Image Archive",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-025-2",
                        name="Run V-JEPA Integration",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.jepa.vjepa_integration",
                        timeout_seconds=3600,
                        kafka_topic="mineralvision.jepa.features.extracted",
                    ),
                    JourneyStep(
                        id="step-025-3",
                        name="Store to Lakehouse",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.jepa.lakehouse_integration",
                        ledger_entry_type="jepa_features_stored",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-026",
                name="WALDO Object Detection",
                description="Detect mining equipment and features using YOLO11 + RF-DETR ensemble",
                category="vision_ai",
                ui_entry_point="/ai-insights",
                required_permissions=["waldo:execute"],
                estimated_duration_minutes=15,
                tags=["waldo", "detection", "yolo", "rfdetr"],
                steps=[
                    JourneyStep(
                        id="step-026-1",
                        name="Upload Image",
                        step_type=StepType.DATA_INGESTION,
                        endpoint="/api/upload",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-026-2",
                        name="Run Ensemble Detector",
                        step_type=StepType.ML_INFERENCE,
                        module="src.api.waldo.ensemble_detector",
                        kafka_topic="mineralvision.waldo.detection.completed",
                    ),
                    JourneyStep(
                        id="step-026-3",
                        name="Store Detections",
                        step_type=StepType.API_CALL,
                        endpoint="/api/waldo/detections",
                        method="POST",
                    ),
                ],
            ),
            
            # Category: Digital Twin & Visualization (27-28)
            JourneyManifest(
                id="journey-027",
                name="Digital Twin Session",
                description="Start a digital twin session with real-time streaming and 3D visualization",
                category="digital_twin",
                ui_entry_point="/visualization/3d",
                required_permissions=["digital_twin:execute"],
                estimated_duration_minutes=0,
                tags=["digital_twin", "3d", "realtime"],
                steps=[
                    JourneyStep(
                        id="step-027-1",
                        name="Initialize Digital Twin",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/initialize",
                        method="POST",
                        kafka_topic="mineralvision.digital_twin.session.started",
                    ),
                    JourneyStep(
                        id="step-027-2",
                        name="Start Real-time Streaming",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/stream/start",
                        method="POST",
                        fluvio_topic="mineralvision.digital_twin.stream",
                    ),
                    JourneyStep(
                        id="step-027-3",
                        name="Load 3D Visualization",
                        step_type=StepType.VISUALIZATION,
                        module="src.api.digital_twin.visualization_3d",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-028",
                name="What-If Simulation",
                description="Run what-if simulation scenarios on the digital twin",
                category="digital_twin",
                ui_entry_point="/visualization/3d",
                required_permissions=["digital_twin:simulate"],
                estimated_duration_minutes=20,
                tags=["simulation", "what_if", "scenario"],
                steps=[
                    JourneyStep(
                        id="step-028-1",
                        name="Load Digital Twin State",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/state",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-028-2",
                        name="Configure Scenario",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/scenario",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-028-3",
                        name="Run Simulation",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/simulate",
                        method="POST",
                        timeout_seconds=1800,
                        kafka_topic="mineralvision.digital_twin.simulation.completed",
                    ),
                    JourneyStep(
                        id="step-028-4",
                        name="Generate Scenario Report",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                        ledger_entry_type="simulation_report_generated",
                    ),
                ],
            ),
            
            # Category: Compliance & Governance (29-30)
            JourneyManifest(
                id="journey-029",
                name="Blockchain Data Provenance",
                description="Record data provenance on blockchain for regulatory compliance",
                category="compliance",
                ui_entry_point="/settings",
                required_permissions=["blockchain:write"],
                estimated_duration_minutes=5,
                tags=["blockchain", "provenance", "compliance"],
                steps=[
                    JourneyStep(
                        id="step-029-1",
                        name="Select Data Assets",
                        step_type=StepType.API_CALL,
                        endpoint="/api/blockchain/assets",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-029-2",
                        name="Record Provenance",
                        step_type=StepType.BLOCKCHAIN_RECORD,
                        endpoint="/api/blockchain/record",
                        method="POST",
                        kafka_topic="mineralvision.blockchain.provenance.recorded",
                    ),
                    JourneyStep(
                        id="step-029-3",
                        name="Generate Compliance Certificate",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                        ledger_entry_type="compliance_certificate_generated",
                    ),
                ],
            ),
            
            JourneyManifest(
                id="journey-030",
                name="Autonomous Exploration Recommendation",
                description="Generate autonomous exploration recommendations based on fused data",
                category="autonomous",
                ui_entry_point="/ai-insights",
                required_permissions=["autonomous_exploration:execute"],
                estimated_duration_minutes=30,
                tags=["autonomous", "exploration", "recommendation"],
                steps=[
                    JourneyStep(
                        id="step-030-1",
                        name="Load Fused Data",
                        step_type=StepType.API_CALL,
                        endpoint="/api/sensor-fusion/fused",
                        method="GET",
                    ),
                    JourneyStep(
                        id="step-030-2",
                        name="Run Autonomous Exploration",
                        step_type=StepType.API_CALL,
                        endpoint="/api/autonomous-exploration/recommend",
                        method="POST",
                        kafka_topic="mineralvision.autonomous.recommendation.generated",
                    ),
                    JourneyStep(
                        id="step-030-3",
                        name="Human Review",
                        step_type=StepType.HUMAN_APPROVAL,
                        requires_approval=True,
                        permission_check="autonomous_exploration:approve",
                    ),
                    JourneyStep(
                        id="step-030-4",
                        name="Generate Survey Plan",
                        step_type=StepType.REPORT_GENERATION,
                        endpoint="/api/reports",
                        method="POST",
                    ),
                    JourneyStep(
                        id="step-030-5",
                        name="Update Digital Twin",
                        step_type=StepType.API_CALL,
                        endpoint="/api/digital-twin/plan",
                        method="POST",
                        ledger_entry_type="survey_plan_approved",
                    ),
                ],
            ),
        ]


# Global registry instance
_registry: Optional[JourneyRegistry] = None


def get_journey_registry() -> JourneyRegistry:
    """Get the global journey registry instance."""
    global _registry
    if _registry is None:
        _registry = JourneyRegistry()
    return _registry
