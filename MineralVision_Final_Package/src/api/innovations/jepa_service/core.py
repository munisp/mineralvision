"""Lazy bridge to the JEPA torch core (src/api/jepa/torch_core.py).

The torch core is built by a parallel workstream against the binding
interface contract in ``JEPA_REBUILD_SPEC.md``.  This module imports it
lazily inside functions and reports honestly when it (or torch itself) is
unavailable — never faking a model.

Model lifecycle: module-level lazy singleton ``get_model()``.  If the env
var ``MV_JEPA_CHECKPOINT`` points at an existing checkpoint file the model
is loaded from it; otherwise a fresh model is built from a default
``JEPAConfig()`` with a fixed torch manual seed (deterministic per process).
The instance persists for the life of the process; ``/train/step`` mutates
it in place and persists a checkpoint to ``MV_JEPA_CHECKPOINT`` when set.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional

CHECKPOINT_ENV = "MV_JEPA_CHECKPOINT"
DEFAULT_SEED = 0

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

def torch_available() -> bool:
    """True when torch can be imported in this environment."""
    return importlib.util.find_spec("torch") is not None


def faiss_available() -> bool:
    """True when faiss can be imported (informational; scoring here is
    exact numpy kNN regardless)."""
    return (importlib.util.find_spec("faiss") is not None
            or importlib.util.find_spec("faiss_cpu") is not None)


def load_torch_core() -> Optional[Any]:
    """Import the JEPA torch core module, or return None when unavailable.

    Dual-context import: repo-root (``src.api...``) then package-root
    (``api...``).  Any ImportError (including the core's own torch import
    guard failing) yields None.
    """
    if not torch_available():
        return None
    for name in ("src.api.jepa.torch_core", "api.jepa.torch_core"):
        try:
            module = __import__(name, fromlist=["*"])
        except ImportError:
            continue
        if getattr(module, "TORCH_AVAILABLE", False):
            return module
        return None
    return None


def capabilities() -> Dict[str, Any]:
    """Real capability probe; never raises."""
    torch_ok = torch_available()
    core = load_torch_core()
    core_ok = core is not None
    config: Dict[str, Any] = {}
    if core_ok:
        try:
            cfg = core.JEPAConfig()
            config = {k: getattr(cfg, k)
                      for k in ("img_size", "patch", "embed_dim", "depth",
                                "heads", "pred_depth", "ema_momentum")
                      if hasattr(cfg, k)}
        except Exception:  # pragma: no cover - defensive, probe must not fail
            config = {}
    checkpoint = os.environ.get(CHECKPOINT_ENV) or None
    return {
        "backend": core.jepa_backend() if core_ok else "unavailable",
        "torch_available": torch_ok,
        "torch_core_available": core_ok,
        "faiss_available": faiss_available(),
        "anomaly_backend": "numpy-exact-knn",
        "config": config,
        "checkpoint_env": CHECKPOINT_ENV,
        "checkpoint_path": checkpoint,
        "checkpoint_loaded": _MODEL is not None and _MODEL_FROM_CHECKPOINT,
    }


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

class CoreUnavailableError(RuntimeError):
    """Raised when the JEPA torch core cannot be imported."""


_MODEL: Optional[Any] = None
_MODEL_FROM_CHECKPOINT: bool = False


def get_model() -> Any:
    """Return the process-wide JEPAModel, building it on first call.

    Raises CoreUnavailableError (mapped to 503 by the routes) when the torch
    core is not importable.
    """
    global _MODEL, _MODEL_FROM_CHECKPOINT
    if _MODEL is not None:
        return _MODEL
    core = load_torch_core()
    if core is None:
        raise CoreUnavailableError(
            "jepa torch core unavailable: install torch and provide "
            "src/api/jepa/torch_core.py")
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    if checkpoint and os.path.isfile(checkpoint):
        _MODEL = core.JEPAModel.load(checkpoint, device="cpu")
        _MODEL_FROM_CHECKPOINT = True
    else:
        try:
            import torch
            torch.manual_seed(DEFAULT_SEED)
        except ImportError:  # pragma: no cover - torch_available pre-checked
            pass
        _MODEL = core.JEPAModel(core.JEPAConfig(), device="cpu")
        _MODEL_FROM_CHECKPOINT = False
    return _MODEL


def reset_model() -> None:
    """Drop the singleton (used by tests to re-probe environments)."""
    global _MODEL, _MODEL_FROM_CHECKPOINT
    _MODEL = None
    _MODEL_FROM_CHECKPOINT = False


def persist_checkpoint(model: Any) -> Optional[str]:
    """Save the model to ``MV_JEPA_CHECKPOINT`` when set; return the path."""
    path = os.environ.get(CHECKPOINT_ENV)
    if not path:
        return None
    model.save(path)
    return path
