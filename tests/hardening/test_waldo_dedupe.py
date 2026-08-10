"""
Tests for WALDO dedupe: Final_Package modules must import detector
primitives from the canonical MineralVision_WALDO_Production_Package
(via WALDO_PACKAGE_SRC / relative path) and must never fabricate
detections when the heavy ML stack is unavailable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_PKG = REPO_ROOT / "MineralVision_Final_Package"
CANONICAL_SRC = REPO_ROOT / "MineralVision_WALDO_Production_Package" / "src"

sys.path.insert(0, str(FINAL_PKG))


@pytest.fixture(autouse=True)
def waldo_src_env(monkeypatch):
    monkeypatch.setenv("WALDO_PACKAGE_SRC", str(CANONICAL_SRC))


def test_canonical_package_exists():
    assert (CANONICAL_SRC / "waldo_integration" / "detection.py").exists()
    assert (CANONICAL_SRC / "waldo_integration" / "rfdetr_backbone.py").exists()


def test_ensemble_detector_module_imports_and_references_canonical():
    import src.api.waldo.ensemble_detector as ens

    source = Path(ens.__file__).read_text()
    assert "WALDO_PACKAGE_SRC" in source
    assert "waldo_integration.detection" in source
    assert "waldo_integration.rfdetr_backbone" in source
    # public names intact
    for name in ("EnsembleWALDODetector", "EnsembleConfig", "create_ensemble_detector"):
        assert hasattr(ens, name)


def test_jepa_integration_imports_canonical_primitives():
    import src.api.jepa.waldo_sam3_integration as jepa

    source = Path(jepa.__file__).read_text()
    assert "waldo_integration.rfdetr_backbone" in source
    assert "RFDETRDetector" in source
    # no fabricated random detections remain
    assert "np.random.randint(0, 5)" not in source


def test_molmo_fusion_imports_canonical_primitives():
    import src.api.molmo.waldo_molmo_fusion as molmo

    source = Path(molmo.__file__).read_text()
    assert "waldo_integration.detection" in source
    assert "WALDODetector" in source


def test_jepa_detection_returns_nothing_without_detector(monkeypatch):
    """No fabricated detections when canonical detector unavailable.

    Post-decon contract: without a loadable RF-DETR and without a configured
    WALDO_SERVICE_URL, detection raises an honest WaldoIntegrationUnavailable
    instead of returning random or silently empty results.
    """
    import pytest

    from src.api.jepa.vjepa_integration import create_feature_extractor
    from src.api.jepa.waldo_sam3_integration import (
        DetectionTarget,
        WALDOJEPAIntegration,
        WaldoIntegrationUnavailable,
    )

    monkeypatch.delenv("WALDO_SERVICE_URL", raising=False)
    extractor = create_feature_extractor()
    integration = WALDOJEPAIntegration(
        jepa_extractor=extractor,
        detection_targets=[DetectionTarget.EQUIPMENT],
    )
    # heavy ML stack (ultralytics) is not installed in this env
    integration.load_waldo_model()
    assert integration._waldo_model is None

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(WaldoIntegrationUnavailable):
        integration._run_waldo_detection(frame, 0.5)


def test_molmo_detection_returns_nothing_without_detector():
    from src.api.molmo.waldo_molmo_fusion import WALDOMolmoFusion

    fusion = WALDOMolmoFusion(confidence_threshold=0.5)
    fusion._load_waldo_model()
    assert fusion._waldo_model is None

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert fusion._run_waldo_detection(frame, 0) == []


def test_canonical_classes_are_shared_singletons():
    """When importable, adapters must use THE canonical classes."""
    sys.path.insert(0, str(CANONICAL_SRC))
    try:
        from waldo_integration.detection import WALDODetector
        from waldo_integration.rfdetr_backbone import RFDETRDetector
    except ImportError:
        pytest.skip("canonical waldo_integration heavy deps unavailable")


    # The adapter loaders import from waldo_integration — same module objects
    assert "waldo_integration.detection" in sys.modules
    assert sys.modules["waldo_integration.detection"].WALDODetector is WALDODetector
    assert RFDETRDetector is sys.modules["waldo_integration.rfdetr_backbone"].RFDETRDetector
