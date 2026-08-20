"""Validated request and response contracts for oil-spill intelligence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ObservationSource(str, Enum):
    """Evidence source for an oil-spill assessment."""

    DRONE_RGB = "drone_rgb"
    SATELLITE_OPTICAL = "satellite_optical"
    SATELLITE_SAR = "satellite_sar"
    FLUOROSENSOR = "fluorosensor"
    MANUAL_ANNOTATION = "manual_annotation"


class ReviewStatus(str, Enum):
    """Human-review state for an incident assessment."""

    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    NEEDS_RESURVEY = "needs_resurvey"


class Severity(str, Enum):
    """Screening severity based on detected surface area."""

    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class GeographicBounds(BaseModel):
    """Geographic footprint of a nadir image in WGS84 coordinates."""

    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    @model_validator(mode="after")
    def validate_extent(self) -> "GeographicBounds":
        if self.west >= self.east:
            raise ValueError("west must be smaller than east")
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        return self


class MaskAnalysisRequest(BaseModel):
    """Request to assess a precomputed oil-probability or binary segmentation mask."""

    mask_base64: str = Field(..., min_length=1, description="Base64 PNG/JPEG mask; non-zero values indicate oil.")
    image_width_px: int = Field(..., gt=0, le=20000)
    image_height_px: int = Field(..., gt=0, le=20000)
    source: ObservationSource
    model_id: str = Field(..., min_length=1, max_length=128)
    model_version: str = Field(..., min_length=1, max_length=128)
    image_id: Optional[str] = Field(None, max_length=255)
    project_id: Optional[str] = Field(None, max_length=36)
    observed_at: Optional[datetime] = None
    ground_sampling_distance_m: Optional[float] = Field(None, gt=0, le=1000)
    geographic_bounds: Optional[GeographicBounds] = None
    probability_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_component_area_px: int = Field(25, ge=1, le=10_000_000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("mask_base64")
    @classmethod
    def strip_data_url_prefix(cls, value: str) -> str:
        """Accept common data-URL mask inputs while persisting only base64 content."""
        if "," in value and value.lstrip().startswith("data:image/"):
            return value.split(",", maxsplit=1)[1]
        return value


class ReviewRequest(BaseModel):
    """Operator review of an algorithmic or externally supplied assessment."""

    status: ReviewStatus
    reviewer: str = Field(..., min_length=1, max_length=255)
    note: Optional[str] = Field(None, max_length=5000)

    @model_validator(mode="after")
    def prevent_pending_review(self) -> "ReviewRequest":
        if self.status == ReviewStatus.PENDING_REVIEW:
            raise ValueError("review status must be confirmed, rejected, or needs_resurvey")
        return self


class SearchPlanRequest(BaseModel):
    """Parameters for a non-executable coverage-priority recommendation."""

    cell_size_m: float = Field(25.0, gt=0, le=5000)
    drone_count: int = Field(1, ge=1, le=100)
    buffer_m: float = Field(100.0, ge=0, le=100_000)


class OilSpillAssessmentResponse(BaseModel):
    """Structured, reviewable result of a mask or model analysis."""

    incident_id: str
    source: ObservationSource
    model_id: str
    model_version: str
    review_status: ReviewStatus
    severity: Severity
    oil_pixel_count: int
    oil_fraction: float
    oil_area_m2: Optional[float]
    oil_area_hectares: Optional[float]
    confidence: Optional[float]
    quality_flags: List[str]
    geometry_geojson: Optional[Dict[str, Any]]
    mask_dimensions: List[int]
    observed_at: Optional[datetime]
    created_at: datetime


class SearchPlanResponse(BaseModel):
    """Coverage priority recommendation for a reviewed assessment."""

    incident_id: str
    advisory_only: bool = True
    recommended_search_area_m2: Optional[float]
    priority_cells: List[Dict[str, Any]]
    notes: List[str]


class ModelLifecycleStatus(str, Enum):
    """Governance state for a registered segmentation model."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    RETIRED = "retired"


class EvaluationSplit(str, Enum):
    """Dataset partition category recorded with an evaluation run."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    SEALED_HOLDOUT = "sealed_holdout"


class IncidentEventType(str, Enum):
    """Reviewable operational events associated with an incident."""

    EVIDENCE_ATTACHED = "evidence_attached"
    RESURVEY_REQUESTED = "resurvey_requested"
    COVERAGE_PLAN_CREATED = "coverage_plan_created"
    REVIEW_COMPLETED = "review_completed"
    EXPORTED = "exported"
    DOMAIN_SHIFT_FLAGGED = "domain_shift_flagged"


class SegmentationMetrics(BaseModel):
    """Oil-class metrics calculated on a documented evaluation dataset."""

    oil_f1: float = Field(..., ge=0.0, le=1.0)
    oil_iou: float = Field(..., ge=0.0, le=1.0)
    oil_precision: float = Field(..., ge=0.0, le=1.0)
    oil_recall: float = Field(..., ge=0.0, le=1.0)
    expected_calibration_error: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModelRegistrationRequest(BaseModel):
    """A versioned local segmentation artifact eligible for controlled evaluation."""

    model_id: str = Field(..., min_length=1, max_length=128)
    model_version: str = Field(..., min_length=1, max_length=128)
    engine: str = Field(..., pattern="^(torchscript|onnx)$")
    artifact_sha256: str = Field(..., pattern="^[a-fA-F0-9]{64}$")
    intended_domains: List[str] = Field(..., min_length=1, max_length=20)
    model_card_url: Optional[str] = Field(None, max_length=2048)
    notes: Optional[str] = Field(None, max_length=5000)


class ModelRegistrationResponse(BaseModel):
    id: str
    model_id: str
    model_version: str
    engine: str
    artifact_sha256: str
    intended_domains: List[str]
    lifecycle_status: ModelLifecycleStatus
    created_at: datetime


class EvaluationRunRequest(BaseModel):
    """Evaluation evidence required before model promotion."""

    dataset_fingerprint: str = Field(..., min_length=8, max_length=128)
    split: EvaluationSplit
    domain: str = Field(..., min_length=1, max_length=255)
    sample_count: int = Field(..., ge=1, le=10_000_000)
    metrics: SegmentationMetrics
    jepa_backbone: Optional[str] = Field(None, max_length=128)
    reviewer: str = Field(..., min_length=1, max_length=255)
    notes: Optional[str] = Field(None, max_length=5000)


class EvaluationRunResponse(BaseModel):
    id: str
    model_id: str
    model_version: str
    dataset_fingerprint: str
    split: EvaluationSplit
    domain: str
    sample_count: int
    metrics: SegmentationMetrics
    jepa_backbone: Optional[str]
    created_at: datetime


class ModelApprovalRequest(BaseModel):
    """Human approval of a model after its sealed-holdout evidence clears the gate."""

    reviewer: str = Field(..., min_length=1, max_length=255)
    note: Optional[str] = Field(None, max_length=5000)


class ModelPromotionResponse(BaseModel):
    model_id: str
    model_version: str
    eligible: bool
    lifecycle_status: ModelLifecycleStatus
    reasons: List[str]


class IncidentEventRequest(BaseModel):
    event_type: IncidentEventType
    actor: str = Field(..., min_length=1, max_length=255)
    details: Dict[str, Any] = Field(default_factory=dict)


class IncidentEventResponse(BaseModel):
    id: str
    incident_id: str
    event_type: IncidentEventType
    actor: str
    details: Dict[str, Any]
    created_at: datetime


class OperationsSummaryResponse(BaseModel):
    total_incidents: int
    pending_review: int
    confirmed: int
    needs_resurvey: int
    high_or_critical: int
    approved_models: int
    candidate_models: int


class TemporalConsensusRequest(BaseModel):
    """Aligned model probability masks, optionally paired with real JEPA embeddings."""

    masks_base64: List[str] = Field(..., min_length=2, max_length=64)
    image_width_px: int = Field(..., ge=1, le=20_000)
    image_height_px: int = Field(..., ge=1, le=20_000)
    jepa_embeddings: Optional[List[List[float]]] = None


class TemporalConsensusResponse(BaseModel):
    """Fused probability evidence and explicit temporal-quality diagnostics."""

    fused_mask_base64: str
    frame_weights: List[float]
    temporal_stability: float
    used_jepa_embeddings: bool
    quality_flags: List[str]
