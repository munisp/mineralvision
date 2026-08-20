"""Unit tests for the MineralVision oil-spill intelligence extension."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from api.oil_spill.analysis import (
    MaskValidationError,
    assess_mask,
    build_coverage_priority_cells,
    decode_probability_mask,
)
from api.oil_spill.models import ModelNotConfiguredError, model_from_environment
from api.oil_spill.schemas import GeographicBounds, MaskAnalysisRequest, ObservationSource, Severity


def encode_mask(mask: np.ndarray) -> str:
    """Encode a normalized test mask as the same base64 evidence accepted by the API."""
    buffer = io.BytesIO()
    Image.fromarray(np.round(mask * 255).astype(np.uint8), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_mask_analysis_produces_area_geometry_and_severity():
    probability = np.zeros((10, 10), dtype=np.float32)
    probability[2:6, 3:8] = 0.9  # 20-pixel connected slick
    probability[0, 0] = 1.0  # Isolated speckle must be removed
    bounds = GeographicBounds(west=4.0, south=51.0, east=4.001, north=51.001)

    result = assess_mask(
        probability,
        threshold=0.5,
        min_component_area_px=4,
        ground_sampling_distance_m=2.0,
        geographic_bounds=bounds,
    )

    assert result.candidate_pixels == 21
    assert result.retained_pixels == 20
    assert result.component_count == 1
    assert result.oil_area_m2 == 80.0
    assert result.severity == Severity.LOW
    assert result.geometry_geojson is not None
    assert result.geometry_geojson["type"] == "MultiPolygon"
    assert result.confidence == pytest.approx(0.8571, abs=0.0001)
    assert "not_georeferenced" not in result.quality_flags


def test_mask_analysis_flags_unactionable_components_and_missing_georeferencing():
    probability = np.zeros((8, 8), dtype=np.float32)
    probability[1, 1] = 1.0
    probability[6, 6] = 0.8

    result = assess_mask(
        probability,
        threshold=0.5,
        min_component_area_px=4,
        ground_sampling_distance_m=None,
        geographic_bounds=None,
    )

    assert result.retained_pixels == 0
    assert result.oil_area_m2 is None
    assert result.geometry_geojson is None
    assert result.severity == Severity.UNKNOWN
    assert result.confidence is None
    assert "all_candidate_components_below_minimum_area" in result.quality_flags
    assert "not_georeferenced" in result.quality_flags


def test_decode_mask_requires_matching_dimensions():
    mask = np.zeros((4, 5), dtype=np.float32)
    encoded = encode_mask(mask)

    with pytest.raises(MaskValidationError, match="mask dimensions"):
        decode_probability_mask(encoded, expected_width=6, expected_height=4)


def test_request_accepts_data_url_evidence_and_validates_footprint():
    mask = np.zeros((4, 5), dtype=np.float32)
    request = MaskAnalysisRequest(
        mask_base64="data:image/png;base64," + encode_mask(mask),
        image_width_px=5,
        image_height_px=4,
        source=ObservationSource.DRONE_RGB,
        model_id="unet-efficientnet-b4",
        model_version="v1.0.0",
        ground_sampling_distance_m=0.25,
    )
    assert not request.mask_base64.startswith("data:")

    with pytest.raises(ValueError, match="west must be smaller"):
        GeographicBounds(west=4.1, south=51.0, east=4.0, north=51.1)


def test_coverage_plan_is_advisory_and_prioritized_from_incident_geometry():
    probability = np.zeros((10, 10), dtype=np.float32)
    probability[3:7, 3:7] = 1.0
    assessment = assess_mask(
        probability,
        threshold=0.5,
        min_component_area_px=4,
        ground_sampling_distance_m=1.0,
        geographic_bounds=GeographicBounds(west=4.0, south=51.0, east=4.002, north=51.002),
    )

    recommended_area_m2, cells, notes = build_coverage_priority_cells(
        assessment.geometry_geojson,
        cell_size_m=25,
        drone_count=2,
        buffer_m=50,
    )

    assert recommended_area_m2 is not None and recommended_area_m2 > 0
    assert cells
    assert cells[0]["priority_score"] >= cells[-1]["priority_score"]
    assert any("Advisory only" in note for note in notes)


def test_image_inference_fails_closed_without_a_configured_model(monkeypatch):
    monkeypatch.delenv("OIL_SPILL_MODEL_PATH", raising=False)
    with pytest.raises(ModelNotConfiguredError, match="Image inference is disabled"):
        model_from_environment()


from api.oil_spill.governance import (
    PROMOTION_THRESHOLDS,
    evaluate_promotion_eligibility,
    fuse_temporal_probabilities,
)
from api.oil_spill.schemas import EvaluationSplit


def _passing_metrics() -> dict[str, float]:
    return dict(PROMOTION_THRESHOLDS)


def test_promotion_requires_sealed_evidence_for_every_intended_domain():
    decision = evaluate_promotion_eligibility(
        [("drone_rgb_daylight", EvaluationSplit.SEALED_HOLDOUT, 150, _passing_metrics())],
        ["drone_rgb_daylight", "port_glare"],
    )

    assert not decision.eligible
    assert "port_glare" in decision.reasons[0]


def test_promotion_gate_requires_all_oil_class_metrics_and_minimum_sample_size():
    weak_metrics = _passing_metrics()
    weak_metrics["oil_iou"] = 0.94
    decision = evaluate_promotion_eligibility(
        [("drone_rgb_daylight", EvaluationSplit.SEALED_HOLDOUT, 150, weak_metrics)],
        ["drone_rgb_daylight"],
    )
    assert not decision.eligible

    passing = evaluate_promotion_eligibility(
        [("drone_rgb_daylight", EvaluationSplit.SEALED_HOLDOUT, 150, _passing_metrics())],
        ["drone_rgb_daylight"],
    )
    assert passing.eligible
    assert not passing.reasons


def test_temporal_consensus_uses_real_embeddings_without_fabricated_jepa_outputs():
    maps = [
        np.full((3, 3), 0.8, dtype=np.float32),
        np.full((3, 3), 0.7, dtype=np.float32),
        np.full((3, 3), 0.1, dtype=np.float32),
    ]
    result = fuse_temporal_probabilities(
        maps,
        jepa_embeddings=[[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0]],
    )

    assert result.used_jepa_embeddings
    assert sum(result.frame_weights) == pytest.approx(1.0, abs=1e-6)
    assert result.probability_map.mean() > 0.5


def test_temporal_consensus_marks_median_fallback_when_no_embeddings_are_supplied():
    result = fuse_temporal_probabilities(
        [np.full((2, 2), 0.2, dtype=np.float32), np.full((2, 2), 0.8, dtype=np.float32)],
    )

    assert not result.used_jepa_embeddings
    assert result.probability_map.mean() == pytest.approx(0.5)
    assert "jepa_embeddings_not_supplied_median_fusion_used" in result.quality_flags
