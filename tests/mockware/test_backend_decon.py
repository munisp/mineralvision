"""F3 backend decontamination tests — honest failure by default, flagged
synthetic outputs only under MV_ALLOW_MOCK_FALLBACK=true. No skips."""

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL = os.path.join(REPO, "MineralVision_Final_Package", "src")


def load_module(name, relpath):
    """Load a module directly from a file path, isolated from package init."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(FINAL, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def clear_flag(monkeypatch):
    monkeypatch.delenv("MV_ALLOW_MOCK_FALLBACK", raising=False)
    yield


# ---------------------------------------------------------------- fix 1
def _stub_heavy_ml_deps(monkeypatch):
    """Stub torch/lightning/mlflow/tfp/geopandas/sklearn so the module imports."""
    def mk(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    class _Base:
        def __init_subclass__(cls, **kw):
            pass

        def __init__(self, *a, **kw):
            pass

    torch = mk("torch")
    torch.nn = mk("torch.nn")
    torch.optim = mk("torch.optim")
    torch_utils = mk("torch.utils")
    torch_data = mk("torch.utils.data", Dataset=_Base, DataLoader=object,
                    random_split=lambda *a, **k: None)
    torch_utils.data = torch_data
    torch.utils = torch_utils
    pl = mk("pytorch_lightning", LightningModule=_Base)
    mlflow = mk("mlflow", pytorch=mk("mlflow.pytorch"))
    tfp = mk("tensorflow_probability")
    gpd = mk("geopandas")
    sk_pre = mk("sklearn.preprocessing", StandardScaler=object)
    sk_ms = mk("sklearn.model_selection", train_test_split=lambda *a, **k: None)
    sklearn = mk("sklearn", preprocessing=sk_pre, model_selection=sk_ms)
    for name, mod in [("torch", torch), ("torch.nn", torch.nn),
                      ("torch.optim", torch.optim), ("torch.utils", torch_utils),
                      ("torch.utils.data", torch_data),
                      ("pytorch_lightning", pl), ("mlflow", mlflow),
                      ("mlflow.pytorch", mlflow.pytorch),
                      ("tensorflow_probability", tfp), ("geopandas", gpd),
                      ("sklearn", sklearn), ("sklearn.preprocessing", sk_pre),
                      ("sklearn.model_selection", sk_ms)]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_deposit_training_raises_without_real_data(monkeypatch, tmp_path):
    _stub_heavy_ml_deps(monkeypatch)
    m = load_module("mdp_test1", "api/ml/predictive_modeling/mineral_deposit_prediction.py")
    with pytest.raises(m.DataUnavailableError) as exc:
        m.MineralDepositDataset(str(tmp_path))
    assert "location_id" in str(exc.value)


def test_deposit_training_synthetic_tagged_with_flag(monkeypatch, tmp_path):
    _stub_heavy_ml_deps(monkeypatch)
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    m = load_module("mdp_test2", "api/ml/predictive_modeling/mineral_deposit_prediction.py")
    ds = m.MineralDepositDataset(str(tmp_path))
    assert ds.synthetic is True
    assert ds.features.shape == (1000, 50)


# ---------------------------------------------------------------- fix 2
def test_reference_workflow_refuses_without_flag():
    m = load_module("rw_test1", "api/workflows/reference_workflows.py")
    wf = m.create_gold_orogenic_workflow()
    with pytest.raises(m.DemonstrationWorkflowError) as exc:
        wf.execute([])
    assert "SIMULATED" in str(exc.value)


def test_reference_workflow_simulated_tag_with_flag(monkeypatch):
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    m = load_module("rw_test2", "api/workflows/reference_workflows.py")
    wf = m.create_gold_orogenic_workflow()
    result = wf.execute([])
    assert result.simulated is True
    d = result.to_dict()
    assert d["simulated"] is True  # top-level marker


# ---------------------------------------------------------------- fix 3
def test_rfdetr_detector_honest_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rfdetr", None)  # force ImportError
    m = load_module("ep_test1", "api/molmo/ensemble_pipeline.py")
    det = m.RFDETRDetector()
    with pytest.raises(m.BackendUnavailableError) as exc:
        det.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert "rfdetr" in str(exc.value)


def test_sam3_segmenter_honest_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sam3", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    m = load_module("ep_test2", "api/molmo/ensemble_pipeline.py")
    seg = m.SAM3Segmenter()
    with pytest.raises(m.BackendUnavailableError):
        seg.auto_segment(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(m.BackendUnavailableError):
        seg.segment_from_boxes(np.zeros((8, 8, 3), dtype=np.uint8),
                               [m.BoundingBox(x1=0, y1=0, x2=4, y2=4)])


def test_vjepa_embedder_raises_when_jepa_unavailable(monkeypatch):
    # force the dual-context imports to yield a torch-less backend
    fake = types.ModuleType("fake_torch_core")
    fake.TORCH_AVAILABLE = False
    monkeypatch.setitem(sys.modules, "src.api.jepa.torch_core", fake)
    monkeypatch.setitem(sys.modules, "api.jepa.torch_core", fake)
    m = load_module("ep_test3", "api/molmo/ensemble_pipeline.py")
    emb = m.VJEPAEmbedder()
    with pytest.raises(m.BackendUnavailableError) as exc:
        emb.embed_frames([np.zeros((16, 16, 3), dtype=np.uint8)])
    assert "torch" in str(exc.value).lower()


def test_vjepa_embedder_uses_real_jepa_when_available(monkeypatch):
    pytest.importorskip("torch")
    sys.path.insert(0, FINAL)
    monkeypatch.setitem(sys.modules, "src.api.jepa.torch_core",
                        importlib.import_module("api.jepa.torch_core"))
    m = load_module("ep_test4", "api/molmo/ensemble_pipeline.py")
    emb = m.VJEPAEmbedder(model_path=None)
    out = emb.embed_frames([np.random.rand(3, 224, 224).astype(np.float32)])
    out = np.asarray(out)
    assert out.ndim == 1 and out.size > 0
    assert np.isfinite(out).all()
    assert not np.allclose(out, 0)


# ---------------------------------------------------------------- fix 4
def test_segy_loader_honest_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "segyio", None)
    m = load_module("segy_test1", "api/sensor_fusion/segy_visualization.py")
    viewer = m.SEGYViewerIntegration()
    with pytest.raises(NotImplementedError) as exc:
        viewer.load_segy("/nonexistent.sgy")
    assert "segyio" in str(exc.value)


def test_segy_synthetic_opt_in_tagged(monkeypatch):
    m = load_module("segy_test2", "api/sensor_fusion/segy_visualization.py")
    # direct call without flag refused
    with pytest.raises(RuntimeError):
        m.make_synthetic_volume()
    # even with flag, load_segy defaults to the honest path
    monkeypatch.setenv("MV_ALLOW_MOCK_FALLBACK", "true")
    monkeypatch.setitem(sys.modules, "segyio", None)
    viewer = m.SEGYViewerIntegration()
    with pytest.raises(NotImplementedError):
        viewer.load_segy("/nonexistent.sgy")
    # explicit opt-in works and is tagged
    vol = m.make_synthetic_volume(seed=7)
    assert vol.metadata["synthetic"] is True
    vol2 = viewer.load_segy("/nonexistent.sgy", synthetic=True)
    assert vol2.metadata["synthetic"] is True


# ---------------------------------------------------------------- fix 5
def test_climate_era5_strict_raises_and_fallback_tagged():
    pytest.importorskip("xarray")
    m = load_module("clim_test1", "api/climate_resilience/advanced_climate.py")
    region = {"min_lon": 120.0, "max_lon": 121.0, "min_lat": -30.0, "max_lat": -29.0}
    cfg = m.ClimateAPIConfig(api_key=None, cache_enabled=False)
    provider = m.ERA5Provider(cfg)
    tr = ("2024-01-01", "2024-01-03")
    # strict -> honest error (no CDS credentials/library in sandbox)
    with pytest.raises(m.ClimateDataUnavailableError):
        provider.fetch_data("temperature", region, tr, strict=True)
    # default -> synthetic but explicitly tagged
    ds = provider.fetch_data("temperature", region, tr)
    assert ds.attrs["synthetic"] is True
    assert "synthetic_reason" in ds.attrs


def test_climate_openweather_strict_raises_and_fallback_tagged():
    pytest.importorskip("xarray")
    pytest.importorskip("requests")
    m = load_module("clim_test2", "api/climate_resilience/advanced_climate.py")
    region = {"min_lon": 120.0, "max_lon": 121.0, "min_lat": -30.0, "max_lat": -29.0}
    cfg = m.ClimateAPIConfig(api_key="invalid", base_url="http://127.0.0.1:9/none",
                             timeout_seconds=2)
    provider = m.OpenWeatherMapProvider(cfg)
    with pytest.raises(m.ClimateDataUnavailableError):
        provider.fetch_data("current", region, ("2024-01-01", "2024-01-01"), strict=True)
    ds = provider.fetch_data("current", region, ("2024-01-01", "2024-01-01"))
    assert ds.attrs["synthetic"] is True
    assert "synthetic_reason" in ds.attrs


# ---------------------------------------------------------------- fix 6
def test_foundation_models_random_projection_disclosed():
    m = load_module("fm_test1", "api/ml/foundation_models.py")
    a = m.MultispectralAdapter(n_bands=4, patch_size=16, embedding_dim=32)
    assert isinstance(a, m.RandomProjectionAdapter)
    meta = a.encoder_metadata()
    assert meta["encoder_type"] == "random_projection"
    assert meta["not_learned"] is True
    cfg = a.get_config().to_dict()
    assert cfg["encoder_type"] == "random_projection"
    assert cfg["not_learned"] is True
    g = m.GeophysicsAdapter(list(m.DataModality)[0], patch_size=32, embedding_dim=16)
    assert isinstance(g, m.RandomProjectionAdapter)
    assert g.encoder_metadata()["not_learned"] is True
    # encode/decode round-trip shapes intact after refactor
    e = a.encode(np.random.rand(2, 4, 32, 32))
    assert e.shape == (2, 4, 32)
    assert a.decode(e).shape == (2, 4, 4 * 16 * 16)
