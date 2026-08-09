"""
MineralVision geomodel innovation — implicit geological modeling
(GEOSPATIAL_SPEC competitive gap #2; Leapfrog-style RBF implicit surfaces).

Endpoints (prefix /innovations/geomodel):
- POST /surfaces/fit        — RBF implicit surface from contacts + orientations
- POST /surfaces/evaluate   — scalar field on a 3D grid + crossing count
- POST /surfaces/isosurface — marching-cubes/tetrahedra mesh (three.js-ready)
- POST /models/build        — multi-surface stratigraphic voxel model (+faults)
- POST /models/section      — 2D section of unit labels through the model
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import engine

router = APIRouter(prefix="/innovations/geomodel", tags=["geomodel"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Point3(BaseModel):
    x: float
    y: float
    z: float


class Orientation(Point3):
    dip: float = Field(..., ge=0, le=90, description="degrees below horizontal")
    azimuth: float = Field(..., description="dip direction, deg clockwise from north")


class PolarityPoint(Point3):
    side: int = Field(..., description="+1 interior/above, -1 exterior/below")


class FitRequest(BaseModel):
    contacts: List[Point3] = Field(default_factory=list)
    orientations: List[Orientation] = Field(default_factory=list)
    polarity: List[PolarityPoint] = Field(default_factory=list)
    kernel: str = Field("thin_plate_spline", description="thin_plate_spline|cubic")
    name: Optional[str] = None


class GridSpec(BaseModel):
    bounds: List[List[float]] = Field(..., description="[[xmin,ymin,zmin],[xmax,ymax,zmax]]")
    shape: List[int] = Field(..., description="[nx, ny, nz]")


class FaultSpec(BaseModel):
    point: List[float]
    dip: float
    azimuth: float
    throw: float = Field(..., description="vertical throw (m) applied to upthrown side")


class EvaluateRequest(BaseModel):
    surface_id: str
    grid: GridSpec
    faults: List[FaultSpec] = Field(default_factory=list)


class IsosurfaceRequest(BaseModel):
    surface_id: str
    grid: GridSpec
    backend: str = Field("auto", description="auto|skimage|tetrahedra")
    faults: List[FaultSpec] = Field(default_factory=list)


class BuildRequest(BaseModel):
    surface_ids: List[str] = Field(..., min_length=1,
                                   description="ordered top -> bottom")
    grid: GridSpec
    faults: List[FaultSpec] = Field(default_factory=list)
    unit_names: Optional[List[str]] = None


class SectionRequest(BaseModel):
    model_id: Optional[str] = None
    surface_ids: Optional[List[str]] = None  # re-evaluate fields exactly
    origin: List[float] = Field(..., description="section origin corner [x,y,z]")
    u: List[float] = Field(..., description="in-plane unit vector, column direction")
    v: List[float] = Field(..., description="in-plane unit vector, row direction")
    nu: int = Field(..., ge=2, le=2000)
    nv: int = Field(..., ge=2, le=2000)
    du: float = Field(..., gt=0)
    dv: float = Field(..., gt=0)
    faults: List[FaultSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/surfaces/fit")
def fit_surface(req: FitRequest):
    if req.kernel not in ("thin_plate_spline", "cubic"):
        raise HTTPException(422, "kernel must be thin_plate_spline or cubic")
    try:
        surf = engine.fit_surface(
            contacts=[[p.x, p.y, p.z] for p in req.contacts],
            orientations=[o.model_dump() for o in req.orientations],
            polarity=[p.model_dump() for p in req.polarity],
            kernel=req.kernel, name=req.name or "")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    res = surf.contact_residuals()
    return {
        "surface_id": surf.id,
        "name": surf.name,
        "kernel": req.kernel,
        "n_contacts": surf.n_contacts,
        "n_orientations": surf.n_orientations,
        "n_polarity": surf.n_polarity,
        "field_stats": {
            "contact_residual_max": float(res.max()) if len(res) else 0.0,
            "contact_residual_mean": float(res.mean()) if len(res) else 0.0,
            "center": surf.center.tolist(),
            "scale": surf.scale,
        },
    }


def _get_surface(sid: str) -> engine.FittedSurface:
    s = engine.SURFACE_REGISTRY.get(sid)
    if s is None:
        raise HTTPException(404, f"surface '{sid}' not found")
    return s


def _grid_spacing(grid: GridSpec):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = grid.bounds
    nx, ny, nz = (int(s) for s in grid.shape)
    return ([xmin, ymin, zmin],
            [(xmax - xmin) / max(nx - 1, 1),
             (ymax - ymin) / max(ny - 1, 1),
             (zmax - zmin) / max(nz - 1, 1)])


@router.post("/surfaces/evaluate")
def evaluate_surface(req: EvaluateRequest):
    surf = _get_surface(req.surface_id)
    try:
        vol = engine.evaluate_grid(surf, req.grid.bounds, req.grid.shape,
                                   [f.model_dump() for f in req.faults])
    except Exception as exc:
        raise HTTPException(422, str(exc))
    return {
        "surface_id": surf.id,
        "shape": [int(s) for s in vol.shape],  # (nz, ny, nx)
        "axis_order": ["z", "y", "x"],
        "values": vol.tolist(),
        "min": float(vol.min()),
        "max": float(vol.max()),
        "isosurface_crossing_cells": engine.count_crossings(vol),
    }


@router.post("/surfaces/isosurface")
def isosurface_endpoint(req: IsosurfaceRequest):
    surf = _get_surface(req.surface_id)
    origin, spacing = _grid_spacing(req.grid)
    vol = engine.evaluate_grid(surf, req.grid.bounds, req.grid.shape,
                               [f.model_dump() for f in req.faults])
    try:
        verts, faces, backend = engine.isosurface(vol, origin, spacing, req.backend)
    except ImportError:
        raise HTTPException(503, "skimage not available; use backend='tetrahedra'")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    normals = engine.mesh_normals(verts, faces)
    return {
        "surface_id": surf.id,
        "backend": backend,
        "n_vertices": int(len(verts)),
        "n_faces": int(len(faces)),
        "vertices": [[round(float(c), 6) for c in row] for row in verts],
        "faces": faces.tolist(),
        "indices": faces.tolist(),  # alias, consistent with geotoolkit-ext terrain mesh
        "normals": [[round(float(c), 6) for c in row] for row in normals],
        "coordinate_order": ["x", "y", "z"],
    }


@router.post("/models/build")
def build_model(req: BuildRequest):
    try:
        model = engine.build_model(
            req.surface_ids, req.grid.bounds, req.grid.shape,
            [f.model_dump() for f in req.faults], req.unit_names)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    cv = model.cell_volume
    counts = {name: int((model.labels == i).sum())
              for i, name in enumerate(model.unit_names)}
    return {
        "model_id": model.id,
        "shape": list(model.shape),
        "axis_order": ["z", "y", "x"],
        "n_units": len(model.unit_names),
        "unit_counts": counts,
        "unit_volumes": {name: cnt * cv for name, cnt in counts.items()},
        "cell_volume": cv,
        "total_cells": int(model.labels.size),
        "labels": model.labels.tolist(),
    }


@router.post("/models/section")
def model_section(req: SectionRequest):
    surfaces = None
    if req.surface_ids:
        surfaces = [_get_surface(s) for s in req.surface_ids]
    model = None
    if req.model_id:
        model = engine.MODEL_REGISTRY.get(req.model_id)
        if model is None:
            raise HTTPException(404, f"model '{req.model_id}' not found")
    if model is None and surfaces is None:
        raise HTTPException(422, "provide model_id and/or surface_ids")
    labels = engine.extract_section(
        model, req.origin, req.u, req.v, req.nu, req.nv, req.du, req.dv,
        surfaces=surfaces, faults=[f.model_dump() for f in req.faults])
    counts = {}
    for lab in np.unique(labels):
        counts[int(lab)] = int((labels == lab).sum())
    return {
        "shape": [int(req.nv), int(req.nu)],
        "axis_order": ["v", "u"],
        "labels": labels.tolist(),
        "unit_pixel_counts": counts,
        "basis": {"origin": req.origin, "u": req.u, "v": req.v,
                  "du": req.du, "dv": req.dv},
    }


__all__ = ["router"]
