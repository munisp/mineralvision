"""
Implicit geological modeling engine (competitive gap #2).

Core approach: fit a scalar field f(x, y, z) to structural data with scipy's
RBFInterpolator (thin-plate spline or cubic polyharmonic kernel) using

- contact points           -> f = 0
- orientation constraints  -> paired off-contact points along the plane normal
                              (dip/azimuth -> normal vector) with values +/- d
- polarity points          -> interior (+) / exterior (-) control values

The geological boundary is the f = 0 isosurface, extracted either with
skimage.measure.marching_cubes (when importable) or a self-contained
marching-tetrahedra implementation (numpy only) which is tested directly.
"""

from __future__ import annotations

import math
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Orientation math
# ---------------------------------------------------------------------------

def plane_normal(dip_deg: float, azimuth_deg: float) -> np.ndarray:
    """Upward unit normal of a plane given dip (deg below horizontal) and dip
    direction azimuth (deg clockwise from north=+y)."""
    d = math.radians(dip_deg)
    a = math.radians(azimuth_deg)
    n = np.array([-math.sin(d) * math.sin(a),
                  -math.sin(d) * math.cos(a),
                  math.cos(d)])
    return n / np.linalg.norm(n)


# ---------------------------------------------------------------------------
# Scalar-field fit
# ---------------------------------------------------------------------------

class FittedSurface:
    def __init__(self, rbf: RBFInterpolator, name: str,
                 n_contacts: int, n_orientations: int, n_polarity: int,
                 contact_pts: np.ndarray, center: np.ndarray, scale: float):
        self.id = uuid.uuid4().hex[:12]
        self.rbf = rbf
        self.name = name
        self.n_contacts = n_contacts
        self.n_orientations = n_orientations
        self.n_polarity = n_polarity
        self.contact_pts = contact_pts
        self.center = center
        self.scale = scale

    def evaluate(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.asarray(self.rbf(pts), dtype=float)

    def contact_residuals(self) -> np.ndarray:
        if len(self.contact_pts) == 0:
            return np.array([])
        return np.abs(self.evaluate(self.contact_pts))


SURFACE_REGISTRY: Dict[str, FittedSurface] = {}
MODEL_REGISTRY: Dict[str, "VoxelModel"] = {}


def fit_surface(contacts: Sequence[Sequence[float]],
                orientations: Sequence[dict],
                polarity: Sequence[dict],
                kernel: str = "thin_plate_spline",
                epsilon_frac: float = 0.02,
                polarity_value: Optional[float] = None,
                name: str = "") -> FittedSurface:
    """Fit implicit scalar field. Orientations: dicts with x,y,z,dip,azimuth.
    Polarity: dicts with x,y,z,side (+1/-1)."""
    cpts = np.asarray(contacts, dtype=float).reshape(-1, 3) if contacts else np.zeros((0, 3))
    if len(cpts) < 3 and not polarity:
        raise ValueError("need at least 3 contact points (or polarity points)")

    # characteristic length scale of the data
    allc = [cpts]
    for o in orientations:
        allc.append(np.array([[o["x"], o["y"], o["z"]]]))
    for p in polarity:
        allc.append(np.array([[p["x"], p["y"], p["z"]]]))
    allpts = np.vstack(allc)
    span = float(np.max(allpts.max(axis=0) - allpts.min(axis=0))) or 1.0
    center = allpts.mean(axis=0)
    eps = epsilon_frac * span
    pval = float(polarity_value) if polarity_value else 0.1 * span

    pts, vals = [cpts], [np.zeros(len(cpts))]
    for o in orientations:
        p = np.array([o["x"], o["y"], o["z"]], dtype=float)
        n = plane_normal(o["dip"], o["azimuth"])
        pts.append(np.array([p + eps * n, p - eps * n]))
        vals.append(np.array([eps, -eps]))
    for p in polarity:
        side = 1.0 if p["side"] >= 0 else -1.0
        pts.append(np.array([[p["x"], p["y"], p["z"]]], dtype=float))
        vals.append(np.array([side * pval]))

    X = np.vstack(pts)
    y = np.concatenate(vals)
    # Normalize coordinates for conditioning; wrap rbf in an affine transform.
    Xn = (X - center) / span
    deg = 1 if kernel == "thin_plate_spline" else 2
    if len(X) < 4:
        deg = 0
    rbf = RBFInterpolator(Xn, y, kernel=kernel, degree=deg, smoothing=0.0)

    class _ScaledRBF:
        def __call__(self, P):
            return rbf((np.atleast_2d(P) - center) / span)

    surf = FittedSurface(_ScaledRBF(), name, len(cpts), len(orientations),
                         len(polarity), cpts, center, span)
    SURFACE_REGISTRY[surf.id] = surf
    return surf


def evaluate_grid(surf: FittedSurface, bounds: Sequence[Sequence[float]],
                  shape: Sequence[int],
                  faults: Sequence[dict] = ()) -> np.ndarray:
    """Evaluate field on a regular grid. Returns array shaped (nz, ny, nx).
    Faults: dicts with point [x,y,z], dip, azimuth, throw (m, vertical).
    Points on the upthrown side of the fault plane are evaluated at their
    pre-offset position (p + throw in z)."""
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    nx, ny, nz = (int(s) for s in shape)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    for f in faults or ():
        n = plane_normal(f["dip"], f["azimuth"])
        fp = np.asarray(f["point"], dtype=float)
        side = (P - fp) @ n
        P = np.where((side > 0)[:, None], P + np.array([0, 0, float(f["throw"])]), P)
    vals = surf.evaluate(P)
    return vals.reshape(nz, ny, nx)


def count_crossings(vol: np.ndarray) -> int:
    """Number of grid cells whose 8 corner values straddle zero."""
    cnt = 0
    for k in range(vol.shape[0] - 1):
        for j in range(vol.shape[1] - 1):
            for i in range(vol.shape[2] - 1):
                c = vol[k:k + 2, j:j + 2, i:i + 2]
                if c.min() <= 0 <= c.max() and c.min() != c.max():
                    cnt += 1
    return int(cnt)


# ---------------------------------------------------------------------------
# Isosurface extraction
# ---------------------------------------------------------------------------

# 6-tetrahedra decomposition of a unit cube (corner indices 0..7, binary i,j,k
# bit order: idx = i + 2j + 4k)
_TETS = [
    (0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
    (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
]


def marching_tetrahedra(vol: np.ndarray, origin: Sequence[float],
                        spacing: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the f=0 isosurface from a (nz, ny, nx) grid via marching
    tetrahedra. Returns (vertices (m,3), faces (t,3)) with welded vertices."""
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)  # (dx, dy, dz)
    nz, ny, nx = vol.shape
    raw_tris: List[List[np.ndarray]] = []

    def corner(i, j, k):
        return origin + spacing * np.array([i, j, k], dtype=float)

    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                cv = np.empty(8)
                cp = np.empty((8, 3))
                for b, (di, dj, dk) in enumerate(
                        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]):
                    cv[b] = vol[k + dk, j + dj, i + di]
                    cp[b] = corner(i + di, j + dj, k + dk)
                if cv.min() > 0 or cv.max() < 0:
                    continue
                for tet in _TETS:
                    tv = cv[list(tet)]
                    tp = cp[list(tet)]
                    tri = []
                    # edges of tetrahedron
                    for a in range(4):
                        for b in range(a + 1, 4):
                            va, vb = tv[a], tv[b]
                            if (va < 0) == (vb < 0):
                                continue
                            t = va / (va - vb) if va != vb else 0.5
                            tri.append(tp[a] + t * (tp[b] - tp[a]))
                    if len(tri) == 3:
                        raw_tris.append(tri)
                    elif len(tri) == 4:
                        raw_tris.append([tri[0], tri[1], tri[2]])
                        raw_tris.append([tri[0], tri[2], tri[3]])

    if not raw_tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    # weld vertices by quantized coordinates
    vmap: Dict[Tuple[int, int, int], int] = {}
    verts: List[np.ndarray] = []
    faces = np.empty((len(raw_tris), 3), dtype=int)
    for fi, tri in enumerate(raw_tris):
        for ci, p in enumerate(tri):
            key = tuple(np.round(p * 1e9).astype(np.int64))
            idx = vmap.get(key)
            if idx is None:
                idx = len(verts)
                vmap[key] = idx
                verts.append(p)
            faces[fi, ci] = idx
    return np.array(verts), faces


def isosurface(vol: np.ndarray, origin: Sequence[float],
               spacing: Sequence[float], backend: str = "auto"
               ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Extract isosurface; backend auto|skimage|tetrahedra. Returns
    (vertices, faces, backend_used)."""
    if backend in ("auto", "skimage"):
        try:
            from skimage.measure import marching_cubes  # type: ignore
            dx, dy, dz = spacing
            verts, faces, _, _ = marching_cubes(
                vol, level=0.0, spacing=(dz, dy, dx), allow_degenerate=False)
            verts = verts[:, ::-1] + np.asarray(origin, dtype=float)  # z,y,x -> x,y,z
            return verts, faces.astype(int), "skimage"
        except ImportError:
            if backend == "skimage":
                raise
    verts, faces = marching_tetrahedra(vol, origin, spacing)
    return verts, faces, "tetrahedra"


def mesh_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals."""
    normals = np.zeros_like(verts)
    if len(faces) == 0:
        return normals
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    fn = np.cross(a, b)
    for c in range(3):
        np.add.at(normals, faces[:, c], fn)
    ln = np.linalg.norm(normals, axis=1)
    ln[ln == 0] = 1.0
    return normals / ln[:, None]


# ---------------------------------------------------------------------------
# Stratigraphic voxel model
# ---------------------------------------------------------------------------

class VoxelModel:
    def __init__(self, labels: np.ndarray, bounds, shape, unit_names):
        self.id = uuid.uuid4().hex[:12]
        self.labels = labels          # (nz, ny, nx) int
        self.bounds = bounds
        self.shape = tuple(int(s) for s in shape)
        self.unit_names = unit_names

    @property
    def cell_volume(self) -> float:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.bounds
        nx, ny, nz = self.shape
        return ((xmax - xmin) / max(nx - 1, 1) *
                (ymax - ymin) / max(ny - 1, 1) *
                (zmax - zmin) / max(nz - 1, 1))


def build_model(surface_ids: Sequence[str], bounds, shape,
                faults: Sequence[dict] = (),
                unit_names: Optional[Sequence[str]] = None) -> VoxelModel:
    """Stack ordered surfaces (top->bottom). Unit 0 is above the first
    surface (f1 > 0), unit i between surface i and i+1, unit n below all."""
    surfs = []
    for sid in surface_ids:
        s = SURFACE_REGISTRY.get(sid)
        if s is None:
            raise KeyError(f"surface '{sid}' not found")
        surfs.append(s)
    if not surfs:
        raise ValueError("at least one surface required")
    fields = [evaluate_grid(s, bounds, shape, faults) for s in surfs]
    labels = np.zeros(fields[0].shape, dtype=int)
    for f in fields:
        labels += (f < 0).astype(int)
    n_units = len(surfs) + 1
    names = list(unit_names) if unit_names else [f"unit_{i}" for i in range(n_units)]
    if len(names) != n_units:
        raise ValueError(f"need {n_units} unit names")
    model = VoxelModel(labels, [list(bounds[0]), list(bounds[1])], shape, names)
    MODEL_REGISTRY[model.id] = model
    return model


def extract_section(model: VoxelModel, origin: Sequence[float],
                    u: Sequence[float], v: Sequence[float],
                    nu: int, nv: int, du: float, dv: float,
                    surfaces: Optional[Sequence[FittedSurface]] = None,
                    faults: Sequence[dict] = ()) -> np.ndarray:
    """Sample unit labels on a 2D section plane:
    P(a, b) = origin + a*u*du + b*v*dv, a in [0,nu), b in [0,nv).
    Returns (nv, nu) label array. Re-evaluates the model's fields when
    surfaces are given, otherwise nearest-neighbor samples the voxel model."""
    origin = np.asarray(origin, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    A, B = np.meshgrid(np.arange(nu) * du, np.arange(nv) * dv)
    P = origin[None, None, :] + A[..., None] * u + B[..., None] * v
    Pf = P.reshape(-1, 3)
    if surfaces:
        fields = []
        for s in surfaces:
            Q = Pf.copy()
            for f in faults or ():
                n = plane_normal(f["dip"], f["azimuth"])
                side = (Q - np.asarray(f["point"], dtype=float)) @ n
                Q = np.where((side > 0)[:, None], Q + np.array([0, 0, float(f["throw"])]), Q)
            fields.append(s.evaluate(Q))
        labels = np.zeros(len(Pf), dtype=int)
        for f in fields:
            labels += (f < 0).astype(int)
        return labels.reshape(nv, nu)
    # nearest-neighbor voxel lookup
    (xmin, ymin, zmin), (xmax, ymax, zmax) = model.bounds
    nx, ny, nz = model.shape
    ix = np.clip(np.round((Pf[:, 0] - xmin) / max(xmax - xmin, 1e-9) * (nx - 1)).astype(int), 0, nx - 1)
    iy = np.clip(np.round((Pf[:, 1] - ymin) / max(ymax - ymin, 1e-9) * (ny - 1)).astype(int), 0, ny - 1)
    iz = np.clip(np.round((Pf[:, 2] - zmin) / max(zmax - zmin, 1e-9) * (nz - 1)).astype(int), 0, nz - 1)
    return model.labels[iz, iy, ix].reshape(nv, nu)
