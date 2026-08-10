"""
JEPA decontamination tests.

Asserts that no code path in ``api.jepa`` silently returns random or
fabricated data:

- VJEPAEncoder.encode / VJEPAPredictor.predict / VJEPAPretrainer.train_epoch
  raise JEPAUnavailableError when the torch_core backend is absent
  (simulated by monkeypatching sys.modules).
- MultiScaleMasking still generates valid, disjoint context/target masks.
- AnomalyDetector with hand-crafted embeddings ranks a planted outlier
  with the highest anomaly score.
- LocalParquetBackend writes real parquet readable by pyarrow, or
  honestly labeled JSON with backend == "json".
- WALDO/SAM3 integration raises WaldoIntegrationUnavailable when no real
  detector/segmenter/service is configured.

No mocks, no skips. Torch is NOT required for these tests.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pytest

# Dual-context import support (repo-root or src layout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "MineralVision_Final_Package", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from api.jepa.vjepa_integration import (
    AnomalyDetector,
    Embedding,
    FaissIndex,
    ImageryType,
    JEPAUnavailableError,
    MultiScaleMasking,
    VJEPAConfig,
    VJEPAEncoder,
    VJEPAFeatureExtractor,
    VJEPAPredictor,
    VJEPAPretrainer,
)
from api.jepa.waldo_sam3_integration import (
    SAM3JEPAIntegration,
    JEPAPrompt,
    WaldoIntegrationUnavailable,
    WALDOJEPAIntegration,
)
from api.jepa.lakehouse_integration import LocalParquetBackend


@pytest.fixture
def no_torch_core(monkeypatch):
    """Simulate the absence of api.jepa.torch_core (both import contexts)."""
    for name in ("src.api.jepa.torch_core", "api.jepa.torch_core"):
        monkeypatch.setitem(sys.modules, name, None)
    return None


# ---------------------------------------------------------------------------
# 1. No fabricated embeddings / predictions / training losses
# ---------------------------------------------------------------------------

def test_encode_raises_without_torch_core(no_torch_core):
    encoder = VJEPAEncoder(VJEPAConfig())
    encoder.load_pretrained()
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(JEPAUnavailableError):
        encoder.encode(frames)


def test_encode_all_tokens_raises_without_torch_core(no_torch_core):
    encoder = VJEPAEncoder(VJEPAConfig())
    frames = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    with pytest.raises(JEPAUnavailableError):
        encoder.encode(frames, return_all_tokens=True)


def test_predictor_predict_raises_without_torch_core(no_torch_core):
    predictor = VJEPAPredictor(VJEPAConfig())
    ctx = np.zeros((1, 10, 384), dtype=np.float32)
    ctx_mask = np.ones((1, 10), dtype=bool)
    tgt_mask = np.zeros((1, 10), dtype=bool)
    with pytest.raises((JEPAUnavailableError, NotImplementedError)):
        predictor.predict(ctx, ctx_mask, tgt_mask)


def test_train_epoch_raises_without_torch_core(no_torch_core):
    pretrainer = VJEPAPretrainer(config=VJEPAConfig(), data_loaders=[])
    with pytest.raises(JEPAUnavailableError):
        pretrainer.train_epoch(epoch=0)


def test_feature_extractor_raises_without_torch_core(no_torch_core):
    extractor = VJEPAFeatureExtractor(VJEPAConfig())
    with pytest.raises(JEPAUnavailableError):
        extractor.extract_features([np.zeros((32, 32, 3), dtype=np.uint8)])


def test_error_message_is_honest(no_torch_core):
    encoder = VJEPAEncoder(VJEPAConfig())
    with pytest.raises(JEPAUnavailableError) as exc_info:
        encoder.encode(np.zeros((1, 32, 32, 3), dtype=np.uint8))
    message = str(exc_info.value).lower()
    assert "torch_core" in message
    assert "fake" in message or "random" in message or "fabricat" in message


# ---------------------------------------------------------------------------
# 2. MultiScaleMasking still generates valid masks
# ---------------------------------------------------------------------------

def test_multiscale_masking_generates_valid_masks():
    rng_state = np.random.get_state()
    np.random.seed(42)
    try:
        config = VJEPAConfig()
        masking = MultiScaleMasking(config, ImageryType.SATELLITE_RGB)
        batch_size, T, H, W = 4, 8, 14, 14
        context_masks, target_masks = masking.generate_masks(batch_size, (T, H, W))
    finally:
        np.random.set_state(rng_state)

    assert context_masks.shape == (batch_size, T, H, W)
    assert target_masks.shape == (batch_size, T, H, W)
    assert context_masks.dtype == bool
    assert target_masks.dtype == bool

    # Masks are disjoint, targets are non-empty, context retains patches.
    for b in range(batch_size):
        assert not np.any(context_masks[b] & target_masks[b])
        assert target_masks[b].sum() > 0
        assert context_masks[b].sum() > 0


# ---------------------------------------------------------------------------
# 3. AnomalyDetector ranks a planted outlier highest (hand-crafted embeddings)
# ---------------------------------------------------------------------------

def _hand_crafted_embedding(embedding_id: str, vector: np.ndarray, imagery_id: str) -> Embedding:
    return Embedding(
        embedding_id=embedding_id,
        vector=vector.tolist(),
        imagery_id=imagery_id,
    )


def test_anomaly_detector_ranks_planted_outlier_highest():
    rng = np.random.default_rng(0)
    dim = 64

    # Tight cluster of "normal" embeddings around the unit vector e0.
    base = np.zeros(dim)
    base[0] = 1.0
    normal_vectors = [base + rng.normal(0, 0.01, size=dim) for _ in range(20)]
    # Planted outlier: orthogonal direction, far from the cluster.
    outlier = np.zeros(dim)
    outlier[1] = 1.0

    config = VJEPAConfig()
    extractor = VJEPAFeatureExtractor(config)

    # Feed hand-crafted embeddings directly into the real detector math by
    # replacing the extraction callables with plain functions returning our
    # crafted vectors (no MagicMock, no random model output).
    crafted = {"normals": normal_vectors, "outlier": outlier}

    def fake_extract_batch(samples, pooling="cls"):
        return [
            _hand_crafted_embedding(f"normal_{i}", crafted["normals"][i], f"img_{i}")
            for i, _ in enumerate(samples)
        ]

    state = {"query": None}

    def fake_extract_features(frames, pooling="cls", layer="last"):
        return _hand_crafted_embedding("query", state["query"], "query_img")

    extractor.extract_batch = fake_extract_batch
    extractor.extract_features = fake_extract_features

    index = FaissIndex(dimension=dim, index_type="Flat")
    detector = AnomalyDetector(extractor, index, threshold=2.0)
    detector.build_baseline([{"frames": [], "metadata": {}} for _ in normal_vectors])

    scores = {}
    for name, vec in [("outlier", outlier)] + [
        (f"normal_{i}", v) for i, v in enumerate(normal_vectors[:5])
    ]:
        state["query"] = vec
        result = detector.detect({"frames": [], "metadata": {"imagery_id": name}})
        scores[name] = result.anomaly_score

    assert max(scores, key=scores.get) == "outlier"
    assert scores["outlier"] > max(
        score for name, score in scores.items() if name != "outlier"
    )


# ---------------------------------------------------------------------------
# 4. Lakehouse backend honesty
# ---------------------------------------------------------------------------

def test_lakehouse_backend_writes_honest_format(tmp_path):
    backend = LocalParquetBackend(str(tmp_path))
    assert backend.backend in ("parquet", "json")

    records = [
        {"embedding_id": "e1", "value": 1.5, "vector": [0.1, 0.2, 0.3], "site": "alpha"},
        {"embedding_id": "e2", "value": 2.5, "vector": [0.4, 0.5, 0.6], "site": "beta"},
    ]
    assert backend.write_records("jepa_embeddings", records) == 2

    if backend.backend == "parquet":
        parquet_file = tmp_path / "jepa_embeddings.parquet"
        assert parquet_file.exists()
        assert not (tmp_path / "jepa_embeddings.json").exists()

        import pyarrow.parquet as pq
        table = pq.read_table(parquet_file)
        assert table.num_rows == 2
        assert set(table.column_names) >= {"embedding_id", "value", "site"}
    else:
        json_file = tmp_path / "jepa_embeddings.json"
        assert json_file.exists()
        with open(json_file) as f:
            on_disk = json.load(f)
        assert len(on_disk) == 2

    # Round-trip through the backend's own reader.
    read_back = backend.read_records("jepa_embeddings")
    assert len(read_back) == 2
    ids = {r["embedding_id"] for r in read_back}
    assert ids == {"e1", "e2"}

    # Filters still work on the real stored data.
    filtered = backend.read_records("jepa_embeddings", filters={"site": "alpha"})
    assert len(filtered) == 1
    assert filtered[0]["embedding_id"] == "e1"


def test_lakehouse_json_fallback_is_honestly_labeled(tmp_path, monkeypatch):
    """When pyarrow is absent the backend must say backend == 'json' and
    write .json files (never JSON-in-.parquet-clothing)."""
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    backend = LocalParquetBackend(str(tmp_path))
    assert backend.backend == "json"

    backend.write_records("t", [{"a": 1}])
    assert (tmp_path / "t.json").exists()
    assert not (tmp_path / "t.parquet").exists()
    with open(tmp_path / "t.json") as f:
        assert json.load(f) == [{"a": 1}]


def test_local_json_backend_alias_deprecation(tmp_path):
    import warnings
    from api.jepa.lakehouse_integration import LocalJSONBackend, LocalParquetBackend as LPB

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend = LocalJSONBackend(str(tmp_path))
    assert any(w.category is DeprecationWarning for w in caught)
    assert isinstance(backend, LPB)
    assert backend.backend in ("parquet", "json")


# ---------------------------------------------------------------------------
# 5. WALDO / SAM3 raise honest errors when no real backend is configured
# ---------------------------------------------------------------------------

def test_waldo_detect_raises_without_detector_or_service(monkeypatch, no_torch_core):
    monkeypatch.delenv("WALDO_SERVICE_URL", raising=False)
    # Force canonical WALDO import to fail so no detector can load.
    monkeypatch.setitem(sys.modules, "waldo_integration.rfdetr_backbone", None)
    monkeypatch.setitem(sys.modules, "waldo_integration", None)

    extractor = VJEPAFeatureExtractor(VJEPAConfig())
    integration = WALDOJEPAIntegration(jepa_extractor=extractor)
    integration._waldo_model = None

    with pytest.raises(WaldoIntegrationUnavailable):
        integration.detect(np.zeros((64, 64, 3), dtype=np.uint8), use_jepa_refinement=False)


def test_sam3_segmentation_raises_without_model_or_service(monkeypatch):
    monkeypatch.delenv("SAM3_SERVICE_URL", raising=False)
    extractor = VJEPAFeatureExtractor(VJEPAConfig())
    integration = SAM3JEPAIntegration(jepa_extractor=extractor)
    integration._sam3_model = None

    prompt = JEPAPrompt(prompt_type="point", coordinates=[(16, 16)])
    with pytest.raises(WaldoIntegrationUnavailable):
        integration._run_sam3_segmentation(np.zeros((64, 64, 3), dtype=np.uint8), prompt)


def test_sam3_attention_generation_is_loud_not_random():
    extractor = VJEPAFeatureExtractor(VJEPAConfig())
    integration = SAM3JEPAIntegration(jepa_extractor=extractor)

    with pytest.raises(WaldoIntegrationUnavailable):
        integration._generate_attention_points([0.0] * 8, [0.0] * 8, (64, 64))
    with pytest.raises(WaldoIntegrationUnavailable):
        integration._generate_attention_bbox([0.0] * 8, [0.0] * 8, (64, 64))


def test_boundary_features_are_real_and_deterministic():
    extractor = VJEPAFeatureExtractor(VJEPAConfig())
    integration = SAM3JEPAIntegration(jepa_extractor=extractor)

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1  # filled 16x16 square

    feats1 = integration._extract_boundary_features(mask)
    feats2 = integration._extract_boundary_features(mask)

    assert feats1 is not None
    assert len(feats1) == 8
    assert feats1 == feats2  # deterministic, not random
    # area fraction of a 16x16 square in 32x32 = 0.25
    assert abs(feats1[0] - 0.25) < 1e-6
    # centroid normalized = center of the square ~ (15.5/32, 15.5/32)
    assert abs(feats1[2] - 15.5 / 32) < 1e-6
    assert abs(feats1[3] - 15.5 / 32) < 1e-6


def test_distillation_requires_real_student(monkeypatch, no_torch_core):
    from api.jepa.waldo_sam3_integration import FeatureDistillation

    teacher = VJEPAFeatureExtractor(VJEPAConfig())
    distiller = FeatureDistillation(teacher_extractor=teacher, student_model=None)

    with pytest.raises(JEPAUnavailableError):
        distiller.distill_to_detector(
            detector_backbone=None,
            training_images=[np.zeros((32, 32, 3), dtype=np.uint8)],
            num_epochs=1,
        )


def test_distillation_student_features_come_from_student(no_torch_core):
    """With a real (hand-crafted deterministic) student callable, the loss
    is computed from actual student outputs — no random student features."""
    from api.jepa.waldo_sam3_integration import FeatureDistillation

    teacher = VJEPAFeatureExtractor(VJEPAConfig())
    dim = teacher.config.embedding_dim

    student_calls = []

    class HandCraftedStudent:
        def extract_features(self, images):
            student_calls.append(len(images))
            # Deterministic features derived from the input images.
            return np.stack([
                np.full(dim, float(np.asarray(img).mean()), dtype=np.float32)
                for img in images
            ])

    distiller = FeatureDistillation(
        teacher_extractor=teacher,
        student_model=HandCraftedStudent(),
    )

    # Teacher extraction is replaced with deterministic hand-crafted vectors
    # so the whole pipeline is exercised without torch (no mocks: plain
    # deterministic function, real loss math).
    def teacher_features(frames, pooling="cls", layer="last"):
        vec = np.linspace(0.0, 1.0, dim, dtype=np.float32)
        return _hand_crafted_embedding("t", vec, "t_img")

    distiller.teacher.extract_features = teacher_features

    images = [np.full((8, 8, 3), 10, dtype=np.uint8), np.full((8, 8, 3), 200, dtype=np.uint8)]
    losses = distiller.compute_distillation_loss(
        images,
        distiller._extract_student_features(images),
    )
    assert student_calls == [2]
    assert np.isfinite(losses["total_loss"])
    assert losses["total_loss"] > 0.0  # student != teacher -> positive real loss
