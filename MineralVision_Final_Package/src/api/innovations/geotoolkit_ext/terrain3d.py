"""Innovation 8 — terrain-3d mesh service.

DTM nested-list grid + bounds -> decimated triangle mesh JSON
(vertices / normals / indices / optional height-based colors) and an
optional hillshade PNG (real hillshade math, azimuth 315, altitude 45).
"""

import base64
import io
import math
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

router = APIRouter()


class Bounds(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float


class MeshRequest(BaseModel):
    dtm: List[List[float]]            # rows = y (north->south), cols = x (west->east)
    bounds: Bounds
    decimation: int = 1               # stride; 1 = full resolution
    colors: bool = False              # height-based vertex colors
    hillshade: bool = False           # include hillshade PNG (base64)
    hillshade_azimuth: float = 315.0
    hillshade_altitude: float = 45.0


def build_mesh(dtm: np.ndarray, bounds: Bounds, decimation: int):
    """Return vertices (n,3), normals (n,3), indices (m,3) for a grid mesh."""
    z = dtm[::decimation, ::decimation]
    rows, cols = z.shape
    if rows < 2 or cols < 2:
        raise HTTPException(status_code=422, detail="decimated grid must be at least 2x2")
    xs = np.linspace(bounds.minx, bounds.maxx, cols)
    ys = np.linspace(bounds.maxy, bounds.miny, rows)  # row 0 = north edge
    X, Y = np.meshgrid(xs, ys)
    vertices = np.stack([X, Y, z], axis=-1).reshape(-1, 3)

    # two triangles per quad; indices CCW
    idx = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            i00 = r * cols + c
            i01 = r * cols + c + 1
            i10 = (r + 1) * cols + c
            i11 = (r + 1) * cols + c + 1
            idx.append((i00, i10, i01))
            idx.append((i01, i10, i11))
    indices = np.array(idx, dtype=np.int64)

    # face normals via real cross products
    v0 = vertices[indices[:, 0]]
    v1 = vertices[indices[:, 1]]
    v2 = vertices[indices[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    face_n = face_n / norms

    # accumulate to vertex normals
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, indices[:, k], face_n)
    nl = np.linalg.norm(normals, axis=1, keepdims=True)
    nl[nl == 0] = 1.0
    normals = normals / nl
    return vertices, normals, indices


def height_colors(zvals: np.ndarray) -> np.ndarray:
    """Map normalized height to a simple terrain ramp (green->brown->white)."""
    zmin, zmax = float(zvals.min()), float(zvals.max())
    t = (zvals - zmin) / (zmax - zmin) if zmax > zmin else np.zeros_like(zvals)
    r = np.clip(0.2 + 1.6 * t, 0, 1)
    g = np.clip(0.6 - 0.2 * t + 0.4 * np.maximum(t - 0.8, 0) * 5, 0, 1)
    b = np.clip(0.2 - 0.1 * t + 0.8 * np.maximum(t - 0.7, 0) / 0.3, 0, 1)
    return np.stack([r, g, b], axis=-1)


def hillshade_png(dtm: np.ndarray, bounds: Bounds, azimuth: float, altitude: float) -> str:
    """Real hillshade: hs = cos(alt)*cos(slope) + sin(alt)*sin(slope)*cos(az - aspect)."""
    z = dtm.astype(float)
    rows, cols = z.shape
    cell_x = (bounds.maxx - bounds.minx) / (cols - 1)
    cell_y = (bounds.maxy - bounds.miny) / (rows - 1)
    dz_dy, dz_dx = np.gradient(z, cell_y, cell_x)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az = math.radians(azimuth)
    alt = math.radians(altitude)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    img = Image.fromarray((hs * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@router.post("/terrain/mesh")
def terrain_mesh(req: MeshRequest) -> Dict[str, Any]:
    dtm = np.asarray(req.dtm, dtype=float)
    if dtm.ndim != 2 or min(dtm.shape) < 2:
        raise HTTPException(status_code=422, detail="dtm must be a 2-D grid of at least 2x2")
    if req.decimation < 1:
        raise HTTPException(status_code=422, detail="decimation must be >= 1")

    vertices, normals, indices = build_mesh(dtm, req.bounds, req.decimation)

    out: Dict[str, Any] = {
        "vertices": [[round(float(v), 6) for v in row] for row in vertices],
        "normals": [[round(float(v), 6) for v in row] for row in normals],
        "indices": indices.astype(int).tolist(),
        "vertex_count": int(vertices.shape[0]),
        "triangle_count": int(indices.shape[0]),
        "decimation": req.decimation,
        "grid_shape": list(dtm[::req.decimation, ::req.decimation].shape),
        "bounds": req.bounds.model_dump(),
    }
    if req.colors:
        cols = height_colors(vertices[:, 2])
        out["colors"] = [[round(float(c), 4) for c in row] for row in cols]
    if req.hillshade:
        out["hillshade_png_base64"] = hillshade_png(
            dtm, req.bounds, req.hillshade_azimuth, req.hillshade_altitude)
        out["hillshade_params"] = {"azimuth": req.hillshade_azimuth, "altitude": req.hillshade_altitude}
    return out
