"""
Numeric fusion tests for the WALDO YOLO11 + RF-DETR ensemble, plus
anti-silent-mock tests for the RF-DETR backbone.

All assertions are on real math: detector outputs are constructed directly
(no models needed); the RF-DETR backbone is exercised with rfdetr blocked.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_SRC = os.path.join(REPO_ROOT, "MineralVision_Final_Package", "src")
for p in (REPO_ROOT, FINAL_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.waldo.ensemble_detector import BoxFusion

RFDETR_BACKBONE_PATH = os.path.join(
    REPO_ROOT, "MineralVision_WALDO_Production_Package", "src",
    "waldo_integration", "rfdetr_backbone.py")


def load_backbone_module():
    """Load rfdetr_backbone.py directly (the waldo_integration package
    __init__ imports ultralytics, which is an optional ML dep)."""
    spec = importlib.util.spec_from_file_location(
        "mv_rfdetr_backbone", RFDETR_BACKBONE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def test_iou_exact_known_boxes():
    # overlap region [2,2]-[4,4]: area 4; areas 16+16 -> union 28 -> IoU 1/7
    iou = BoxFusion.compute_iou([0, 0, 4, 4], [2, 2, 6, 6])
    assert iou == pytest.approx(4.0 / 28.0, abs=1e-12)
    # identical boxes -> 1
    assert BoxFusion.compute_iou([1, 1, 5, 5], [1, 1, 5, 5]) == pytest.approx(1.0)
    # disjoint -> 0
    assert BoxFusion.compute_iou([0, 0, 2, 2], [3, 3, 5, 5]) == 0.0
    # containment: inner area 4, union 16 -> 0.25
    assert BoxFusion.compute_iou([0, 0, 4, 4], [1, 1, 3, 3]) == pytest.approx(0.25)
    # edge-touching boxes -> zero intersection area -> 0
    assert BoxFusion.compute_iou([0, 0, 2, 2], [2, 0, 4, 2]) == 0.0


# ---------------------------------------------------------------------------
# Weighted Box Fusion
# ---------------------------------------------------------------------------

def test_wbf_fuses_overlapping_boxes_to_weighted_average():
    # Two models, equal weights; one overlapping pair + one singleton.
    boxes_list = [
        [[0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 120.0, 120.0]],   # model 0
        [[2.0, 1.0, 12.0, 11.0]],                                  # model 1
    ]
    scores_list = [[0.9, 0.5], [0.7]]
    labels_list = [[1, 1], [1]]

    fused_boxes, fused_scores, fused_labels, contributing = \
        BoxFusion.weighted_box_fusion(boxes_list, scores_list, labels_list,
                                      iou_threshold=0.5)

    assert len(fused_boxes) == 2  # one fused cluster + singleton
    # Identify cluster (2 contributing models) vs singleton
    idx_pair = contributing.index([0, 1])
    idx_single = 1 - idx_pair

    # Weighted average with weights w_i = (0.5 * score_i) / sum:
    # w0 = 0.45/0.8 = 0.5625, w1 = 0.35/0.8 = 0.4375
    w0, w1 = 0.5625, 0.4375
    expected = [0.0 * w0 + 2.0 * w1, 0.0 * w0 + 1.0 * w1,
                10.0 * w0 + 12.0 * w1, 10.0 * w0 + 11.0 * w1]
    assert fused_boxes[idx_pair] == pytest.approx(expected, abs=1e-9)

    # conf_type="avg" -> mean(0.9, 0.7) = 0.8, boosted by 2-model agreement
    # factor 1.1 -> min(1, 0.88)
    assert fused_scores[idx_pair] == pytest.approx(0.88, abs=1e-9)
    assert fused_labels[idx_pair] == 1

    # Singleton keeps its box; score = 0.5, no consensus boost
    assert fused_boxes[idx_single] == pytest.approx([100, 100, 120, 120])
    assert fused_scores[idx_single] == pytest.approx(0.5, abs=1e-9)
    assert contributing[idx_single] == [0]


def test_wbf_respects_unequal_model_weights():
    boxes_list = [[[0.0, 0.0, 10.0, 10.0]], [[4.0, 0.0, 14.0, 10.0]]]
    scores_list = [[1.0], [1.0]]
    labels_list = [[0], [0]]
    # weights normalized: model0 = 0.75, model1 = 0.25
    fused_boxes, _, _, _ = BoxFusion.weighted_box_fusion(
        boxes_list, scores_list, labels_list,
        weights=[3.0, 1.0], iou_threshold=0.2)
    assert len(fused_boxes) == 1
    # equal scores -> coordinate weights follow model weights
    assert fused_boxes[0][0] == pytest.approx(0.75 * 0.0 + 0.25 * 4.0, abs=1e-9)
    assert fused_boxes[0][2] == pytest.approx(0.75 * 10.0 + 0.25 * 14.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Soft-NMS (Gaussian decay)
# ---------------------------------------------------------------------------

def test_soft_nms_gaussian_decay():
    # top box score 0.9; neighbor IoU with top = 0.5 (verified below), score 0.8
    top = [0.0, 0.0, 4.0, 4.0]
    nbr = [2.0, 0.0, 6.0, 4.0]  # intersection 2*4=8, union 32-8=24 -> IoU 1/3
    boxes = [top, nbr]
    scores = [0.9, 0.8]
    labels = [0, 0]
    models = [0, 1]
    sigma = 0.5
    out_boxes, out_scores, out_labels, out_models = BoxFusion.soft_nms(
        boxes, scores, labels, models, sigma=sigma, score_threshold=0.0)

    assert out_boxes[0] == pytest.approx(top)          # top box kept first
    assert out_scores[0] == pytest.approx(0.9)         # top score untouched
    iou = BoxFusion.compute_iou(top, nbr)
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-12)
    expected_decay = float(np.exp(-(iou ** 2) / sigma))
    # neighbor suppressed by exactly the Gaussian factor
    assert out_scores[1] == pytest.approx(0.8 * expected_decay, abs=1e-9)
    assert out_scores[1] < 0.8                          # actually suppressed
    assert len(out_boxes) == 2                          # but not removed


def test_soft_nms_threshold_drops_fully_suppressed_box():
    # identical boxes: IoU=1 -> decay exp(-1/0.5)=exp(-2) ~ 0.135
    boxes = [[0, 0, 4, 4], [0, 0, 4, 4]]
    scores = [0.9, 0.1]
    out_boxes, out_scores, _, _ = BoxFusion.soft_nms(
        boxes, scores, [0, 0], [0, 1], sigma=0.5, score_threshold=0.05)
    # 0.1 * exp(-2) = 0.0135 < 0.05 -> second box dropped
    assert len(out_boxes) == 1
    assert out_scores[0] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Consensus fusion
# ---------------------------------------------------------------------------

def test_consensus_only_when_models_agree():
    # model0: two boxes; model1 agrees with only the first (IoU >= 0.5).
    boxes_list = [
        [[0.0, 0.0, 4.0, 4.0], [100.0, 100.0, 110.0, 110.0]],
        [[0.5, 0.5, 4.5, 4.5], [200.0, 200.0, 210.0, 210.0]],
    ]
    scores_list = [[0.9, 0.6], [0.7, 0.95]]
    labels_list = [[3, 3], [3, 3]]

    # IoU of agreeing pair: inter 3.5*3.5=12.25, union 32-12.25=19.75 -> 0.620
    iou = BoxFusion.compute_iou(boxes_list[0][0], boxes_list[1][0])
    assert iou >= 0.5

    fused_boxes, fused_scores, fused_labels, contributing = \
        BoxFusion.consensus_fusion(boxes_list, scores_list, labels_list,
                                   iou_threshold=0.5, min_models=2)
    # Only the agreeing pair survives
    assert len(fused_boxes) == 1
    assert fused_labels == [3]
    assert contributing == [[0, 1]]
    # averaged box and score
    assert fused_boxes[0] == pytest.approx([0.25, 0.25, 4.25, 4.25])
    assert fused_scores[0] == pytest.approx((0.9 + 0.7) / 2, abs=1e-9)


def test_consensus_drops_below_iou_threshold():
    # same class, but IoU < 0.5 -> no consensus
    boxes_list = [
        [[0.0, 0.0, 4.0, 4.0]],
        [[3.0, 0.0, 7.0, 4.0]],  # inter 1*4=4, union 32-4=28 -> 0.143
    ]
    scores_list = [[0.9], [0.9]]
    labels_list = [[0], [0]]
    fused = BoxFusion.consensus_fusion(boxes_list, scores_list, labels_list,
                                       iou_threshold=0.5, min_models=2)
    assert fused == ([], [], [], [])


# ---------------------------------------------------------------------------
# No silent mock
# ---------------------------------------------------------------------------

def test_no_silent_mock_raises_by_default(monkeypatch):
    bb = load_backbone_module()
    assert not bb.RFDETR_AVAILABLE or True  # env may or may not have rfdetr
    monkeypatch.delitem(sys.modules, "rfdetr", raising=False)
    monkeypatch.setattr(bb, "RFDETR_AVAILABLE", False)  # simulate blocked import
    monkeypatch.delenv("MV_ALLOW_MOCK_FALLBACK", raising=False)

    cfg = bb.RFDETRConfig()
    with pytest.raises(bb.RFDETRUnavailableError):
        bb.RFDETRDetector(cfg)


def test_mock_fallback_marks_detections(monkeypatch):
    bb = load_backbone_module()
    monkeypatch.setattr(bb, "RFDETR_AVAILABLE", False)
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")

    cfg = bb.RFDETRConfig(confidence_threshold=0.0)
    detector = bb.RFDETRDetector(cfg)
    assert isinstance(detector.model, bb.MockRFDETRModel)

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    detections = detector.detect(image)
    for det in detections:
        assert det.metadata.get("mock") is True
        assert det.to_dict()["metadata"]["mock"] is True


def test_finetuner_evaluate_raises_without_backend(monkeypatch):
    bb = load_backbone_module()
    monkeypatch.setattr(bb, "RFDETR_AVAILABLE", False)
    tuner = bb.RFDETRFineTuner(bb.RFDETRTrainingConfig())
    with pytest.raises(bb.RFDETRUnavailableError):
        tuner.evaluate("/nonexistent/ckpt.pt", "/nonexistent/data")


def test_journey_026_names_real_callable():
    try:
        from src.api.orchestration.journeys import get_journey_registry
    except ImportError:
        from api.orchestration.journeys import get_journey_registry
    j26 = get_journey_registry().get("journey-026")
    assert j26 is not None
    step = next(s for s in j26.steps if s.id == "step-026-2")
    assert step.module == "src.api.waldo.ensemble_detector"
    assert step.function == "run_ensemble_detection"
    try:
        from src.api.waldo import ensemble_detector as ed
    except ImportError:
        from api.waldo import ensemble_detector as ed
    assert callable(getattr(ed, step.function))
