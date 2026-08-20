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
