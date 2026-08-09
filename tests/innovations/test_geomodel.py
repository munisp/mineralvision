"""
Tests for the geomodel implicit modeling engine.

Analytic verification (seeded, no mocks/skips):
- planar contacts + orientations on z = 2x + 1 -> field recovers the plane
  (residuals ~0, isosurface vertices lie on the plane, recovered normal ok)
- spherical contacts -> isosurface radius within tolerance (both backends)
- stacked 2-surface model -> exact unit assignment counts
- section of a planar model -> straight-line boundary between units
- marching-tetrahedra fallback tested directly (backend="tetrahedra")
"""

import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_PKG = os.path.join(REPO_ROOT, "MineralVision_Final_Package")
SRC = os.path.join(FINAL_PKG, "src")
for p in (REPO_ROOT, FINAL_PKG, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from src.api.innovations.geomodel import engine, router  # noqa: E402

rng = np.random.default_rng(42)

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def plane_z(x):
    return 2.0 * x + 1.0


def plane_dip_az():
    # f = z - 2x - 1; gradient (-2, 0, 1): plane dips toward -x (az 270),
    # dip = atan(|dh/dx|) = atan(2)
    return math.degrees(math.atan(2.0)), 270.0


def fit_plane_surface():
    dip, az = plane_dip_az()
    contacts = []
    for _ in range(30):
        x, y = rng.uniform(0.5, 9.5, 2)
        contacts.append({"x": float(x), "y": float(y), "z": plane_z(x)})
    orientations = [
        {"x": 2.0, "y": 3.0, "z": plane_z(2.0), "dip": dip, "azimuth": az},
        {"x": 7.0, "y": 8.0, "z": plane_z(7.0), "dip": dip, "azimuth": az},
    ]
    r = client.post("/innovations/geomodel/surfaces/fit", json={
        "contacts": contacts, "orientations": orientations,
        "kernel": "thin_plate_spline", "name": "plane"})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# 1. fit
# ---------------------------------------------------------------------------

def test_fit_planar_contacts_recovers_plane():
    body = fit_plane_surface()
    assert body["n_contacts"] == 30
    assert body["n_orientations"] == 2
    # RBF with exact TPS interpolation -> contacts satisfied
    assert body["field_stats"]["contact_residual_max"] < 1e-8

    sid = body["surface_id"]
    # Evaluate along a vertical line: field must switch sign at the plane.
    r = client.post("/innovations/geomodel/surfaces/evaluate", json={
        "surface_id": sid,
        "grid": {"bounds": [[0, 0, -6], [10, 10, 18]], "shape": [6, 6, 12]}})
    assert r.status_code == 200, r.text
    ev = r.json()
    assert ev["isosurface_crossing_cells"] > 0
    assert ev["min"] < 0 < ev["max"]

    # Field at probe points matches the analytic plane function sign/magnitude
    # proportionally: sign(f) == sign(z - 2x - 1).
    surf = engine.SURFACE_REGISTRY[sid]
    probes = np.array([[3.0, 4.0, plane_z(3.0) + 2.0],
                       [3.0, 4.0, plane_z(3.0) - 2.0],
                       [8.0, 1.0, plane_z(8.0) + 5.0]])
    vals = surf.evaluate(probes)
    assert np.sign(vals[0]) == 1 and np.sign(vals[1]) == -1
    assert np.sign(vals[2]) == 1

    # Recovered plane orientation: gradient of f at center is parallel to the
    # analytic normal (-2, 0, 1)/sqrt(5).
    c = np.array([5.0, 5.0, plane_z(5.0)])
    h = 0.25
    grad = np.array([
        surf.evaluate(c + [h, 0, 0])[0] - surf.evaluate(c - [h, 0, 0])[0],
        surf.evaluate(c + [0, h, 0])[0] - surf.evaluate(c - [0, h, 0])[0],
        surf.evaluate(c + [0, 0, h])[0] - surf.evaluate(c - [0, 0, h])[0],
    ])
    grad /= np.linalg.norm(grad)
    analytic = np.array([-2.0, 0.0, 1.0]) / math.sqrt(5.0)
    assert np.dot(grad, analytic) == pytest.approx(1.0, abs=0.05)


def test_fit_orientation_math():
    n = engine.plane_normal(0.0, 0.0)
    assert n == pytest.approx(np.array([0, 0, 1]))
    n = engine.plane_normal(90.0, 90.0)  # vertical, dips east -> normal +x (up-dip side)
    assert n == pytest.approx(np.array([1, 0, 0]), abs=1e-12)
    n = engine.plane_normal(45.0, 180.0)  # dips south -> normal tilts north
    assert n[1] == pytest.approx(-math.sin(math.radians(45)), abs=1e-12)
    assert n[2] == pytest.approx(math.cos(math.radians(45)), abs=1e-12)
    # analytic check: plane z = 2x + 1 dips toward -x (az 270), dip atan(2);
    # up-normal must equal gradient of z-2x-1 normalized.
    dip, az = plane_dip_az()
    n = engine.plane_normal(dip, az)
    analytic = np.array([-2.0, 0.0, 1.0]) / math.sqrt(5.0)
    assert np.dot(n, analytic) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 2/3. evaluate + isosurface (both backends, sphere radius check)
# ---------------------------------------------------------------------------

def fit_sphere_surface(radius=4.0, center=(5.0, 5.0, 5.0), n=60):
    # contacts on sphere + polarity points inside/outside
    contacts = []
    for _ in range(n):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = np.array(center) + radius * v
        contacts.append({"x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    polarity = [
        {"x": center[0], "y": center[1], "z": center[2], "side": 1},
        {"x": center[0] + 3 * radius, "y": center[1], "z": center[2], "side": -1},
    ]
    r = client.post("/innovations/geomodel/surfaces/fit", json={
        "contacts": contacts, "polarity": polarity, "kernel": "cubic",
        "name": "sphere"})
    assert r.status_code == 200, r.text
    return r.json()["surface_id"], center


@pytest.mark.parametrize("backend", ["tetrahedra", "skimage"])
def test_isosurface_sphere_radius(backend):
    if backend == "skimage":
        pytest.importorskip("skimage")
    sid, center = fit_sphere_surface()
    bounds = [[0, 0, 0], [10, 10, 10]]
    r = client.post("/innovations/geomodel/surfaces/isosurface", json={
        "surface_id": sid,
        "grid": {"bounds": bounds, "shape": [24, 24, 24]},
        "backend": backend})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == backend
    assert body["n_vertices"] > 50 and body["n_faces"] > 50
    verts = np.array(body["vertices"])
    radii = np.linalg.norm(verts - np.array(center), axis=1)
    assert radii.mean() == pytest.approx(4.0, abs=0.35)
    assert radii.std() < 0.35
    # normals finite and unit-length
    normals = np.array(body["normals"])
    nl = np.linalg.norm(normals, axis=1)
    assert np.all(nl > 0.99)


def test_isosurface_tetrahedra_matches_skimage_on_plane():
    body = fit_plane_surface()
    sid = body["surface_id"]
    origin, spacing = [0, 0, -6], [10 / 15, 10 / 15, 24 / 15]
    grid = {"bounds": [[0, 0, -6], [10, 10, 18]], "shape": [16, 16, 16]}
    surf = engine.SURFACE_REGISTRY[sid]
    vol = engine.evaluate_grid(surf, grid["bounds"], grid["shape"])
    vt, ft, bt = engine.isosurface(vol, origin, spacing, "tetrahedra")
    assert bt == "tetrahedra" and len(vt) > 0
    # every tetrahedra-mesh vertex lies on the analytic plane
    dev = np.abs(vt[:, 2] - 2.0 * vt[:, 0] - 1.0)
    assert dev.max() < 1e-6
    pytest.importorskip("skimage")
    vs, fs, bs = engine.isosurface(vol, origin, spacing, "skimage")
    dev2 = np.abs(vs[:, 2] - 2.0 * vs[:, 0] - 1.0)
    assert dev2.max() < 0.05  # skimage interpolates on the discrete grid


# ---------------------------------------------------------------------------
# 4. models/build — stacked 2-surface stratigraphy
# ---------------------------------------------------------------------------

def test_two_surface_model_unit_counts():
    # Two horizontal planes: z = 6 (top) and z = 3 (bottom), over a
    # 0..10 box with 21^3 grid -> exact band counts.
    def fit_horizontal(z0, name):
        contacts = [{"x": float(x), "y": float(y), "z": z0}
                    for x in (1, 5, 9) for y in (1, 5, 9)]
        orientations = [{"x": 5, "y": 5, "z": z0, "dip": 0.0, "azimuth": 0.0}]
        r = client.post("/innovations/geomodel/surfaces/fit", json={
            "contacts": contacts, "orientations": orientations,
            "kernel": "thin_plate_spline", "name": name})
        assert r.status_code == 200, r.text
        return r.json()["surface_id"]

    s_top = fit_horizontal(6.0, "top")
    s_bot = fit_horizontal(3.0, "bottom")
    # 20 z-levels 0, 10/19, ..., 10: no node exactly on z=3 or z=6 planes.
    r = client.post("/innovations/geomodel/models/build", json={
        "surface_ids": [s_top, s_bot],
        "grid": {"bounds": [[0, 0, 0], [10, 10, 10]], "shape": [20, 20, 20]},
        "unit_names": ["cap", "middle", "basement"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_units"] == 3
    counts = body["unit_counts"]
    # step = 10/19: z>6 -> k=12..19 (8 levels); 3<z<6 -> k=6..11 (6);
    # z<3 -> k=0..5 (6). Sign convention: above surface = f>0.
    per_xy = 20 * 20
    assert counts["cap"] == 8 * per_xy
    assert counts["middle"] == 6 * per_xy
    assert counts["basement"] == 6 * per_xy
    vols = body["unit_volumes"]
    assert vols["cap"] == pytest.approx(8 * per_xy * body["cell_volume"])
    assert body["cell_volume"] == pytest.approx((10 / 19) ** 3)


def test_fault_offset_shifts_units():
    # One horizontal plane z=5; fault vertical plane at x=5 (dip 90, az 90)
    # with throw +2: east side evaluated at z+2 -> boundary appears at z=3.
    contacts = [{"x": float(x), "y": float(y), "z": 5.0}
                for x in (1, 5, 9) for y in (1, 5, 9)]
    r = client.post("/innovations/geomodel/surfaces/fit", json={
        "contacts": contacts,
        "orientations": [{"x": 5, "y": 5, "z": 5.0, "dip": 0.0, "azimuth": 0.0}],
        "kernel": "thin_plate_spline"})
    sid = r.json()["surface_id"]
    # 20 levels: no node exactly on z=5 (or the fault-shifted z=3 boundary:
    # step 10/19 -> levels ..., 2.63, 3.16, ...).
    grid = {"bounds": [[0, 0, 0], [10, 10, 10]], "shape": [20, 20, 20]}
    plain = client.post("/innovations/geomodel/models/build", json={
        "surface_ids": [sid], "grid": grid}).json()
    faulted = client.post("/innovations/geomodel/models/build", json={
        "surface_ids": [sid], "grid": grid,
        "faults": [{"point": [5, 0, 5], "dip": 90.0, "azimuth": 90.0,
                    "throw": 2.0}]}).json()
    # Without fault: unit_0 (above, f>0) occupies z>5 -> 10 z-levels.
    assert plain["unit_counts"]["unit_0"] == 10 * 20 * 20
    # Fault throw +2 shifts one side's evaluation to z+2 -> boundary at z=3
    # there. With step 10/19 that is 4 extra z-levels of unit_0 on the shifted
    # side (10 of 20 x-columns).
    expected = 10 * 20 * 20 + 4 * 20 * 10
    assert faulted["unit_counts"]["unit_0"] == expected


# ---------------------------------------------------------------------------
# 5. models/section — planar model section is a straight line
# ---------------------------------------------------------------------------

def test_section_of_planar_model_is_straight_line():
    body = fit_plane_surface()
    sid = body["surface_id"]
    # vertical x-z section at y=5: u=+x, v=+z
    r = client.post("/innovations/geomodel/models/section", json={
        "surface_ids": [sid],
        "origin": [0.0, 5.0, -4.0],
        "u": [1, 0, 0], "v": [0, 0, 1],
        "nu": 60, "nv": 60, "du": 0.15, "dv": 0.4})
    assert r.status_code == 200, r.text
    body = r.json()
    labels = np.array(body["labels"])  # (nv, nu), rows = z, cols = x
    # Boundary row for each column: straight line z = 2x + 1.
    # row index of first unit-1 cell from top -> z_cross = -4 + row*0.4 ~ 2x+1
    for col in range(0, 60, 10):
        x = col * 0.15
        col_lab = labels[:, col]
        below = np.where(col_lab == 1)[0]   # below plane (f<0), bottom rows
        above = np.where(col_lab == 0)[0]   # above plane, top rows
        assert len(below) > 0 and len(above) > 0
        z_first_above = -4.0 + above.min() * 0.4
        # allow one row of quantization (dv) around the analytic crossing
        assert z_first_above == pytest.approx(2.0 * x + 1.0, abs=0.45)
    # straightness: boundary row is affine in column index (slope = 2*du/dv)
    bnd = np.array([np.where(labels[:, c] == 1)[0].max() for c in range(60)])
    cols = np.arange(60)
    slope, intercept = np.polyfit(cols, bnd, 1)
    assert slope == pytest.approx(2.0 * 0.15 / 0.4, abs=0.15)


def test_section_requires_target_and_validates():
    r = client.post("/innovations/geomodel/models/section", json={
        "origin": [0, 0, 0], "u": [1, 0, 0], "v": [0, 0, 1],
        "nu": 5, "nv": 5, "du": 1.0, "dv": 1.0})
    assert r.status_code == 422
    r = client.post("/innovations/geomodel/surfaces/evaluate", json={
        "surface_id": "missing",
        "grid": {"bounds": [[0, 0, 0], [1, 1, 1]], "shape": [2, 2, 2]}})
    assert r.status_code == 404
