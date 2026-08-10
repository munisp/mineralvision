"""HTTP layer for the JEPA innovation service (thin; see logic.py/core.py).

Every endpoint that needs the model goes through ``core.get_model()``; when
the torch core is unavailable the service answers an honest 503
"jepa torch core unavailable" instead of faking embeddings.  Pure-numpy
payloads (raw embedding vectors) keep working without torch.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import core, logic

router = APIRouter(
    prefix="/innovations/jepa",
    tags=["jepa"])

# In-memory anomaly baseline: list of {"embedding": np.ndarray, "label": str}
BASELINE_STORE: List[Dict[str, Any]] = []

PATCH_GRID = 6  # img_size 96 / patch 16 -> 6x6 patch tokens


# ---------------------------------------------------------------------------
# Payload models
# ---------------------------------------------------------------------------

class ImageEmbeddingRequest(BaseModel):
    """One image: nested list rows x cols [x 3] or base64 PNG."""
    image: Any


class BatchEmbeddingRequest(BaseModel):
    images: List[Any] = Field(min_length=1)


class TrainStepRequest(BaseModel):
    images: List[Any] = Field(min_length=1)
    steps: int = Field(default=1, ge=1, le=1000)


class BaselineRequest(BaseModel):
    """Register baseline embeddings from images (needs torch core) or from
    raw embedding vectors (works without it)."""
    images: Optional[List[Any]] = None
    embeddings: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    replace: bool = Field(default=False,
                          description="clear the baseline store first")


class AnomalyScoreRequest(BaseModel):
    images: Optional[List[Any]] = None
    embeddings: Optional[List[List[float]]] = None
    k: int = Field(default=5, ge=1)


class ChangeScoreRequest(BaseModel):
    image_before: Any
    image_after: Any
    top_k: int = Field(default=5, ge=1, le=36)


class CoreScanEmbedRequest(BaseModel):
    """Same image payload shape as core-scan ingest; sliced into N vertical
    segments and each segment embedded."""
    image: Any
    n_segments: int = Field(default=4, ge=1)
    hole_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode(image: Any) -> np.ndarray:
    try:
        return logic.decode_image_unit(image)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid image payload: {exc}") from exc


def _require_model() -> Any:
    try:
        return core.get_model()
    except core.CoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _embed(model: Any, image: Any) -> np.ndarray:
    img = _decode(image)
    emb = np.asarray(model.embed_image(img), dtype=np.float64).ravel()
    if not np.all(np.isfinite(emb)):
        raise HTTPException(status_code=500,
                            detail="model returned non-finite embedding")
    return emb


def _embeddings_from_payload(images: Optional[List[Any]],
                             embeddings: Optional[List[List[float]]],
                             model: Optional[Any]) -> List[np.ndarray]:
    """Resolve a request's images/embeddings into embedding vectors."""
    out: List[np.ndarray] = []
    if embeddings:
        for i, e in enumerate(embeddings):
            vec = np.asarray(e, dtype=np.float64).ravel()
            if vec.size == 0 or not np.all(np.isfinite(vec)):
                raise HTTPException(
                    status_code=422,
                    detail=f"embedding {i} must be a non-empty finite vector")
            out.append(vec)
    if images:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="jepa torch core unavailable: cannot embed images")
        out.extend(_embed(model, im) for im in images)
    if not out:
        raise HTTPException(
            status_code=422,
            detail="provide at least one of 'images' or 'embeddings'")
    dims = {v.shape[0] for v in out}
    if len(dims) > 1:
        raise HTTPException(status_code=422,
                            detail="inconsistent embedding dimensions")
    return out


def _embed_or_none() -> Optional[Any]:
    try:
        return core.get_model()
    except core.CoreUnavailableError:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/capabilities")
def get_capabilities() -> Dict[str, Any]:
    """Real probe of torch / torch_core / faiss availability; never fails."""
    return core.capabilities()


@router.post("/embeddings/image")
def embed_image(req: ImageEmbeddingRequest) -> Dict[str, Any]:
    model = _require_model()
    emb = _embed(model, req.image)
    return {"embedding": emb.tolist(),
            "dim": int(emb.shape[0]),
            "norm": float(np.linalg.norm(emb))}


@router.post("/embeddings/batch")
def embed_batch(req: BatchEmbeddingRequest) -> Dict[str, Any]:
    model = _require_model()
    embs = [_embed(model, im) for im in req.images]
    return {"embeddings": [e.tolist() for e in embs],
            "count": len(embs),
            "dim": int(embs[0].shape[0]),
            "norms": [float(np.linalg.norm(e)) for e in embs]}


@router.post("/train/step")
def train_step(req: TrainStepRequest) -> Dict[str, Any]:
    model = _require_model()
    imgs = np.stack([_decode(im) for im in req.images])
    losses: List[float] = []
    for _ in range(req.steps):
        loss = float(model.train_step(imgs))
        if not math.isfinite(loss):
            raise HTTPException(status_code=500,
                                detail="train_step returned non-finite loss")
        losses.append(loss)
    checkpoint = core.persist_checkpoint(model)
    return {"losses": losses,
            "steps": req.steps,
            "batch_size": int(imgs.shape[0]),
            "final_loss": losses[-1],
            "checkpoint_saved_to": checkpoint}


@router.post("/anomaly/baseline")
def register_baseline(req: BaselineRequest) -> Dict[str, Any]:
    if req.replace:
        BASELINE_STORE.clear()
    model = _embed_or_none() if req.images else None
    if req.images and model is None:
        raise HTTPException(
            status_code=503,
            detail="jepa torch core unavailable: cannot embed images")
    vecs = _embeddings_from_payload(req.images, req.embeddings, model)
    labels = req.labels or [None] * len(vecs)
    if len(labels) != len(vecs):
        raise HTTPException(
            status_code=422,
            detail="labels length must match number of embeddings")
    for vec, label in zip(vecs, labels):
        BASELINE_STORE.append({"embedding": vec, "label": label})
    return {"registered": len(vecs),
            "baseline_size": len(BASELINE_STORE),
            "dim": int(vecs[0].shape[0])}


@router.post("/anomaly/score")
def anomaly_score(req: AnomalyScoreRequest) -> Dict[str, Any]:
    if not BASELINE_STORE:
        raise HTTPException(
            status_code=409,
            detail="no baseline registered; POST /anomaly/baseline first")
    model = _embed_or_none() if req.images else None
    if req.images and model is None:
        raise HTTPException(
            status_code=503,
            detail="jepa torch core unavailable: cannot embed images")
    vecs = _embeddings_from_payload(req.images, req.embeddings, model)
    baseline = np.stack([b["embedding"] for b in BASELINE_STORE])
    if baseline.shape[1] != vecs[0].shape[0]:
        raise HTTPException(
            status_code=422,
            detail=f"query dim {vecs[0].shape[0]} != baseline dim "
                   f"{baseline.shape[1]}")
    scores, neighbor_idx = logic.knn_anomaly_scores(
        np.stack(vecs), baseline, k=req.k)
    ranks = logic.rank_descending(scores)
    results = [{"score": float(s), "rank": int(r),
                "neighbors": [int(i) for i in idx]}
               for s, r, idx in zip(scores, ranks, neighbor_idx)]
    return {"k": min(req.k, len(BASELINE_STORE)),
            "baseline_size": len(BASELINE_STORE),
            "results": results}


@router.post("/change/score")
def change_score(req: ChangeScoreRequest) -> Dict[str, Any]:
    model = _require_model()
    img_a = _decode(req.image_before)
    img_b = _decode(req.image_after)
    emb_a = np.asarray(model.embed_image(img_a), dtype=np.float64).ravel()
    emb_b = np.asarray(model.embed_image(img_b), dtype=np.float64).ravel()
    change = logic.cosine_distance(emb_a, emb_b)

    patches = np.asarray(
        model.encode_target(np.stack([img_a, img_b])), dtype=np.float64)
    if patches.ndim != 3 or patches.shape[0] != 2:
        raise HTTPException(
            status_code=500,
            detail="encode_target must return array of shape (2, N, D)")
    n = patches.shape[1]
    grid = int(round(math.sqrt(n)))
    if grid * grid != n:
        raise HTTPException(
            status_code=500,
            detail=f"patch count {n} is not a square grid")
    change_map, mean_patch = logic.change_map_from_patches(
        patches[0], patches[1], grid=grid)
    return {"change_score": change,
            "mean_patch_distance": mean_patch,
            "grid": grid,
            "change_map": change_map,
            "top_regions": logic.top_changed_regions(change_map, req.top_k)}


@router.post("/corescan/embed")
def corescan_embed(req: CoreScanEmbedRequest) -> Dict[str, Any]:
    model = _require_model()
    img = _decode(req.image)
    try:
        segments = logic.slice_vertical_segments(img, req.n_segments)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out = []
    for seg in segments:
        emb = np.asarray(model.embed_image(seg["image"]),
                         dtype=np.float64).ravel()
        out.append({"index": seg["index"],
                    "row_start": seg["row_start"],
                    "row_end": seg["row_end"],
                    "embedding": emb.tolist(),
                    "norm": float(np.linalg.norm(emb))})
    return {"hole_id": req.hole_id,
            "n_segments": len(out),
            "dim": int(out[0]["embedding"].__len__()),
            "segments": out}
