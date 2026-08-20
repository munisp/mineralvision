"""Oil-spill intelligence extension for MineralVision."""

from .analysis import MaskAssessment, MaskValidationError, assess_mask, decode_probability_mask
from .models import ModelInferenceError, ModelNotConfiguredError, model_from_environment
from .schemas import (
    GeographicBounds,
    MaskAnalysisRequest,
    ObservationSource,
    OilSpillAssessmentResponse,
    ReviewRequest,
    ReviewStatus,
    SearchPlanRequest,
    SearchPlanResponse,
    Severity,
)

__all__ = [
    "GeographicBounds",
    "MaskAnalysisRequest",
    "MaskAssessment",
    "MaskValidationError",
    "ModelInferenceError",
    "ModelNotConfiguredError",
    "ObservationSource",
    "OilSpillAssessmentResponse",
    "ReviewRequest",
    "ReviewStatus",
    "SearchPlanRequest",
    "SearchPlanResponse",
    "Severity",
    "assess_mask",
    "decode_probability_mask",
    "model_from_environment",
]
