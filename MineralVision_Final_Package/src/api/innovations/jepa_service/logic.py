"""Pure numpy logic for the JEPA innovation service.

No torch here: image decoding reuses the platform core-scan decoder via a
dual-context import; anomaly scoring (exact kNN cosine distance), change-map
assembly from target-encoder patch embeddings, and vertical photo slicing are
plain numpy so they can be unit-tested without the torch core.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # dual-context import (repo-root vs package-root execution)
    from src.api.innovations.core_scan import logic as core_logic
except ImportError:  # pragma: no cover
    from api.innovations.core_scan import logic as core_logic

EPS = 1e-9

# Re-export the platform decoder: nested list rows x cols [x 3] or base64 PNG
# -> float array (rows, cols, 3) in [0, 255].
decode_image = core_logic.decode_image


def decode_image_unit(image: Any) -> np.ndarray:
    """Decode an image payload to float32 (rows, cols, 3) in [0, 1]."""
    arr = decode_image(image)
    arr = np.asarray(arr, dtype=np.float32)
    if float(arr.max()) > 1.0 + EPS:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Embeddings helpers
# ---------------------------------------------------------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector (zero vector stays zero)."""
    vec = np.asarray(vec, dtype=np.float64).ravel()
    n = float(np.linalg.norm(vec))
    if n < EPS:
        return vec
    return vec / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance 1 - cos(a, b) in [0, 2]."""
    an = l2_normalize(a)
    bn = l2_normalize(b)
    return float(1.0 - float(np.dot(an, bn)))


# ---------------------------------------------------------------------------
# Anomaly scoring — exact kNN cosine distance against a baseline set
# ---------------------------------------------------------------------------

def knn_anomaly_scores(queries: np.ndarray,
                       baseline: np.ndarray,
                       k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Score each query by mean cosine distance to its k nearest baseline
    embeddings (exact, numpy).

    Returns ``(scores, neighbor_indices)`` where ``scores[i]`` is the anomaly
    score of query ``i`` and ``neighbor_indices[i]`` are the indices (into
    ``baseline``) of the k nearest baseline embeddings used.
    """
    q = np.asarray(queries, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    if b.ndim != 2 or b.shape[0] == 0:
        raise ValueError("baseline must be a non-empty (n, d) array")
    if b.shape[1] != q.shape[1]:
        raise ValueError(
            f"embedding dim mismatch: query d={q.shape[1]} "
            f"vs baseline d={b.shape[1]}")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, b.shape[0])

    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), EPS)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
    dist = 1.0 - qn @ bn.T  # (n_queries, n_baseline) cosine distance

    idx = np.argsort(dist, axis=1)[:, :k]
    scores = np.take_along_axis(dist, idx, axis=1).mean(axis=1)
    return scores.astype(float), idx


def rank_descending(scores: Sequence[float]) -> List[int]:
    """Rank each score within its batch; rank 1 = highest (most anomalous).

    Ties share the better rank (standard competition ranking).
    """
    s = np.asarray(scores, dtype=float)
    ranks = np.empty(s.shape[0], dtype=int)
    for i in range(s.shape[0]):
        ranks[i] = 1 + int(np.sum(s > s[i]))
    return ranks.tolist()


# ---------------------------------------------------------------------------
# Change detection — per-patch target-encoder embedding distances
# ---------------------------------------------------------------------------

def change_map_from_patches(patches_a: np.ndarray,
                            patches_b: np.ndarray,
                            grid: Optional[int] = None
                            ) -> Tuple[List[List[float]], float]:
    """Per-patch cosine distances between two patch-embedding sets.

    ``patches_a``/``patches_b`` are (N, D) target-encoder patch embeddings of
    the two images (same N, same D).  Returns ``(change_map, mean_distance)``
    where ``change_map`` is a ``grid x grid`` nested list (grid defaults to
    ``int(sqrt(N))``; N must be a perfect square when grid is omitted).
    """
    a = np.asarray(patches_a, dtype=np.float64)
    b = np.asarray(patches_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            "patch embeddings must be two (N, D) arrays of equal shape")
    n = a.shape[0]
    if n == 0:
        raise ValueError("no patches")
    if grid is None:
        g = int(round(np.sqrt(n)))
        if g * g != n:
            raise ValueError(
                f"patch count {n} is not a perfect square; pass grid")
        grid = g
    if grid * grid != n:
        raise ValueError(f"grid {grid}x{grid} != patch count {n}")

    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), EPS)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), EPS)
    dists = 1.0 - np.sum(an * bn, axis=1)  # (N,)
    change_map = dists.reshape(grid, grid).astype(float).tolist()
    return change_map, float(dists.mean())


def top_changed_regions(change_map: Sequence[Sequence[float]],
                        top_k: int = 5) -> List[Dict[str, Any]]:
    """Top-k highest-distance cells of a change map, with row/col coords."""
    m = np.asarray(change_map, dtype=float)
    if m.ndim != 2:
        raise ValueError("change_map must be 2-D")
    flat = [(float(m[r, c]), int(r), int(c))
            for r in range(m.shape[0]) for c in range(m.shape[1])]
    flat.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [{"row": r, "col": c, "distance": d}
            for d, r, c in flat[:max(0, int(top_k))]]


# ---------------------------------------------------------------------------
# Core-scan bridge — vertical segmentation of a core photo
# ---------------------------------------------------------------------------

def slice_vertical_segments(img: np.ndarray,
                            n_segments: int) -> List[Dict[str, Any]]:
    """Slice an (rows, cols, 3) photo into ``n_segments`` bands stacked
    vertically (contiguous row spans, top to bottom).

    Returns a list of ``{"index", "row_start", "row_end", "image"}`` dicts;
    row spans are as equal as possible and cover the full height.
    """
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError("image must be (rows, cols, channels)")
    rows = arr.shape[0]
    n = int(n_segments)
    if n < 1:
        raise ValueError("n_segments must be >= 1")
    if n > rows:
        raise ValueError(f"n_segments {n} exceeds image height {rows}")
    edges = np.linspace(0, rows, n + 1).astype(int)
    out = []
    for i in range(n):
        r0, r1 = int(edges[i]), int(edges[i + 1])
        out.append({"index": i, "row_start": r0, "row_end": r1,
                    "image": arr[r0:r1]})
    return out
