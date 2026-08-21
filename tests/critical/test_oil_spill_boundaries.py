"""OIL-01 through OIL-06: Oil-spill raw imagery, model governance, and incident review boundary tests."""
import hashlib
import os
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MineralVision_Final_Package", "src"))

from api.oil_spill.schemas import ReviewRequest, ReviewStatus
from api.oil_spill.analysis import assess_mask, classify_severity, MaskAssessment
from api.oil_spill.governance import evaluate_promotion_eligibility, fuse_temporal_probabilities


@pytest.fixture
def sample_mask():
    """Normalized [0,1] probability mask with a simulated oil region."""
    mask = np.zeros((64, 64), dtype=np.float64)
    mask[10:20, 10:20] = 0.8
    return mask


# OIL-01: Unregistered/mismatched ONNX model artifact
class TestOIL01UnregisteredModel:
    def test_abstract_model_cannot_be_instantiated(self):
        from api.oil_spill.models import OilSpillSegmentationModel
        with pytest.raises(TypeError):
            OilSpillSegmentationModel(descriptor=None)


# OIL-02: Valid mask assessment
class TestOIL02ValidAssessment:
    def test_mask_assessment_produces_area(self, sample_mask):
        result = assess_mask(
            probability_map=sample_mask,
            threshold=0.5,
            min_component_area_px=1,
            ground_sampling_distance_m=0.5,
            geographic_bounds=None,
        )
        assert isinstance(result, MaskAssessment)
        assert result.oil_area_m2 is None or result.oil_area_m2 >= 0

    def test_severity_classification(self, sample_mask):
        result = assess_mask(probability_map=sample_mask, threshold=0.5, min_component_area_px=1, ground_sampling_distance_m=0.5, geographic_bounds=None)
        assert result.severity.value in ("low", "moderate", "high", "critical")


# OIL-03: Invalid inputs
class TestOIL03InvalidInput:
    def test_empty_mask_rejected(self):
        with pytest.raises((ValueError, IndexError, TypeError)):
            assess_mask(probability_map=np.array([], dtype=np.float64), threshold=0.5, min_component_area_px=1, ground_sampling_distance_m=0.5, geographic_bounds=None)

    def test_non_2d_mask_rejected(self):
        with pytest.raises((ValueError, IndexError, TypeError)):
            assess_mask(probability_map=np.zeros((10, 10, 3), dtype=np.float64), threshold=0.5, min_component_area_px=1, ground_sampling_distance_m=0.5, geographic_bounds=None)


# OIL-04: Low-confidence assessment
class TestOIL04LowConfidence:
    def test_below_threshold_produces_zero_area(self):
        low_mask = np.full((64, 64), 0.04, dtype=np.float64)  # ~4% probability
        result = assess_mask(probability_map=low_mask, threshold=0.5, min_component_area_px=1, ground_sampling_distance_m=0.5, geographic_bounds=None)
        assert result.oil_area_m2 is None or result.oil_area_m2 == 0

    def test_all_zero_mask_produces_no_oil(self):
        zero_mask = np.zeros((64, 64), dtype=np.float64)
        result = assess_mask(probability_map=zero_mask, threshold=0.5, min_component_area_px=1, ground_sampling_distance_m=0.5, geographic_bounds=None)
        assert result.oil_area_m2 is None or result.oil_area_m2 == 0


# OIL-05: Review schema validation
class TestOIL05ReviewSchema:
    def test_valid_review_request(self):
        req = ReviewRequest(status=ReviewStatus.CONFIRMED, reviewer="analyst@example.com", note="Verified")
        assert req.status == ReviewStatus.CONFIRMED

    def test_pending_review_rejected(self):
        with pytest.raises(Exception):
            ReviewRequest(status=ReviewStatus.PENDING_REVIEW, reviewer="analyst@example.com")

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            ReviewRequest(status="invalid_state", reviewer="analyst@example.com")


# OIL-06: Model governance promotion
class TestOIL06ModelGovernance:
    def test_promotion_requires_all_thresholds(self):
        evaluations = [("offshore_rgb", "sealed_holdout", 150, {"oil_f1": 0.96, "oil_iou": 0.96, "oil_precision": 0.98, "oil_recall": 0.98})]
        result = evaluate_promotion_eligibility(evaluations, intended_domains=["offshore_rgb"])
        assert result.eligible is False

    def test_promotion_passes_when_all_thresholds_met(self):
        evaluations = [("offshore_rgb", "sealed_holdout", 150, {"oil_f1": 0.98, "oil_iou": 0.96, "oil_precision": 0.98, "oil_recall": 0.98})]
        result = evaluate_promotion_eligibility(evaluations, intended_domains=["offshore_rgb"])
        assert result.eligible is True

    def test_temporal_consensus_without_embeddings(self):
        # Masks must be float in [0, 1] per the governance contract
        masks = [np.full((32, 32), 0.8, dtype=np.float64) for _ in range(5)]
        result = fuse_temporal_probabilities(masks, jepa_embeddings=None)
        assert result.probability_map.shape == (32, 32)
        assert result.used_jepa_embeddings is False
