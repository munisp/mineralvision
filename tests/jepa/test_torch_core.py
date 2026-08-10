"""Seeded real tests for the PyTorch I-JEPA core (JEPA_REBUILD_SPEC contract).

No mocks, no skips: these tests exercise the real torch backend end-to-end,
plus an honest TORCH_AVAILABLE=False path via import monkeypatching.
"""

import importlib
import sys

import numpy as np
import pytest
import torch

from api.jepa import torch_core
from api.jepa.torch_core import (
    TORCH_AVAILABLE,
    JEPAConfig,
    JEPAModel,
    jepa_backend,
    make_masks,
)

assert TORCH_AVAILABLE, "torch must be installed to run the real JEPA tests"

IMG = 96
SEED = 1234


def _structured_images(n, rng, size=IMG):
    """Synthetic structured images: stripes and blobs (not noise)."""
    imgs = np.zeros((n, size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        kind = i % 2
        if kind == 0:  # stripes
            freq = rng.integers(4, 10)
            phase = rng.uniform(0, np.pi)
            band = 0.5 + 0.5 * np.sin(2 * np.pi * freq * xx / size + phase)
            imgs[i, ..., 0] = band
            imgs[i, ..., 1] = 1.0 - band
            imgs[i, ..., 2] = 0.3 * np.sin(2 * np.pi * freq * yy / size)
        else:  # blobs
            for _ in range(4):
                cy, cx = rng.uniform(10, size - 10, 2)
                r = rng.uniform(6, 16)
                mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r**2
                color = rng.uniform(0.2, 1.0, 3)
                imgs[i][mask] = color
    return np.clip(imgs, 0.0, 1.0).astype(np.float32)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = JEPAModel(JEPAConfig(), device="cpu")
    m.rng = np.random.default_rng(SEED)
    return m


@pytest.fixture(scope="module")
def batch():
    rng = np.random.default_rng(7)
    return _structured_images(8, rng)


def test_backend_reports_torch():
    assert jepa_backend() == "torch"


def test_make_masks_shapes_and_disjoint():
    rng = np.random.default_rng(0)
    ctx, tgt = make_masks(8, 36, rng=rng)
    assert ctx.shape[0] == tgt.shape[0] == 8
    assert ctx.ndim == 2 and tgt.ndim == 2
    assert 5 <= ctx.shape[1] <= 8  # ~0.15-0.20 of 36
    assert tgt.shape[1] >= 28  # ~0.85-1.0 of 36, minus ctx reserve
    for c, t in zip(ctx, tgt):
        assert len(np.intersect1d(c, t)) == 0
        assert c.min() >= 0 and t.max() < 36


def test_shapes_and_norm(model, batch):
    emb = model.embed_image(batch[0])
    assert emb.shape == (384,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
    out = model.encode_target(batch)
    assert out.shape == (8, 36, 384)


def test_train_step_learns(model, batch):
    torch.manual_seed(SEED + 1)
    model.rng = np.random.default_rng(SEED + 1)
    losses = [model.train_step(batch) for _ in range(15)]
    assert all(isinstance(l, float) for l in losses)
    first = float(np.mean(losses[:3]))
    last = float(np.mean(losses[-3:]))
    assert last < first, f"no learning signal: first3={first:.4f} last3={last:.4f}"


def test_ema_target_differs_from_context(model, batch):
    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    model.train_step(batch)
    t_after = list(model.target_encoder.parameters())
    c_after = list(model.context_encoder.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, t_after))
    assert changed, "target encoder did not EMA-update"
    differs = any(
        not torch.allclose(t, c) for t, c in zip(t_after, c_after)
    )
    assert differs, "target encoder identical to context encoder (no EMA lag)"


def test_embedding_semantics(model):
    torch.manual_seed(SEED + 2)
    model.rng = np.random.default_rng(SEED + 2)
    rng = np.random.default_rng(99)
    imgs = _structured_images(8, rng)
    for _ in range(10):
        model.train_step(imgs)

    def crop(img, top, left, size=64):
        c = img[top : top + size, left : left + size]
        # resize-free crop embedding: pad back to 96 via repeat to keep ViT input size
        out = np.zeros_like(img)
        out[:size, :size] = c
        return out

    a1 = model.embed_image(crop(imgs[0], 0, 0))
    a2 = model.embed_image(crop(imgs[0], 16, 16))
    b1 = model.embed_image(crop(imgs[1], 0, 0))

    def cos(x, y):
        return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    assert cos(a1, a2) > cos(a1, b1)


def test_save_load_roundtrip(model, batch, tmp_path):
    emb_before = model.embed_image(batch[0])
    p = tmp_path / "jepa.pt"
    model.save(p)
    loaded = JEPAModel.load(p, device="cpu")
    emb_after = loaded.embed_image(batch[0])
    np.testing.assert_allclose(emb_before, emb_after, atol=1e-6)


def test_deterministic_init():
    torch.manual_seed(42)
    m1 = JEPAModel(JEPAConfig())
    torch.manual_seed(42)
    m2 = JEPAModel(JEPAConfig())
    for p1, p2 in zip(m1.context_encoder.parameters(), m2.context_encoder.parameters()):
        assert torch.equal(p1, p2)


def test_unavailable_path_honest(monkeypatch):
    """Simulate a missing torch install: module imports fine, backend is honest."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", None)
    reloaded = importlib.reload(torch_core)
    try:
        assert reloaded.TORCH_AVAILABLE is False
        assert reloaded.jepa_backend() == "unavailable"
        with pytest.raises(RuntimeError, match="torch"):
            reloaded.JEPAModel(reloaded.JEPAConfig())
    finally:
        monkeypatch.undo()
        importlib.reload(torch_core)
    assert torch_core.TORCH_AVAILABLE is True
