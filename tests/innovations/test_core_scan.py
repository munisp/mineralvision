"""Tests for the drill-core scan pipeline (planted geometry + signatures).

Synthetic core photo: dark tray background with bright core strips, planted
missing-core gaps and planted dark transverse fracture lines.  Synthetic
spectral rows with known band-ratio indices.  Seeded noise only; no mocks,
no skips.
"""

import base64
import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from api.innovations.core_scan import logic, router

# --------------------------------------------------------------------------
# Synthetic scene geometry
# --------------------------------------------------------------------------

TRAY_BGR = (50, 50, 50)        # dark tray background
CORE_RGB = (190, 165, 130)     # bright tan core
FRACTURE_RGB = (90, 90, 90)    # dark transverse fracture line (dimmer than
                               # core, brighter than empty tray)

N_ROWS, N_COLS = 400, 120
# core strip occupies columns 20..100
C0, C1 = 20, 100

# Box registration: two boxes with a planted depth gap 10.0 -> 12.0 m.
# ppm = 20 everywhere (200 px / 10 m, 200 px / 10 m).
BOXES = [
    {"row_start": 0, "row_end": 200, "start_depth_m": 0.0, "end_depth_m": 10.0},
    {"row_start": 200, "row_end": 400, "start_depth_m": 12.0, "end_depth_m": 22.0},
]
PPM = 20.0

# planted missing-core gaps (pixel rows re-painted with tray colour)
MISSING = [(50, 70), (90, 100), (250, 300)]   # [start, end) px rows
N_MISSING = sum(e - s for s, e in MISSING)
PLANTED_RECOVERY = (N_ROWS - N_MISSING) / N_ROWS  # 320/400 = 0.8

# planted fracture lines (row -> width in px)
FRACTURES = [(30, 2), (150, 3), (330, 2)]


def make_photo(seed=7):
    """Synthetic tray photo with planted gaps and fractures."""
    rng = np.random.default_rng(seed)
    img = np.zeros((N_ROWS, N_COLS, 3))
    img[:, :] = np.array(TRAY_BGR) + rng.normal(0, 2, img.shape)
    img[:, C0:C1] = np.array(CORE_RGB) + rng.normal(0, 2, (N_ROWS, C1 - C0, 3))
    for s, e in MISSING:  # missing core -> tray shows through
        img[s:e, C0:C1] = np.array(TRAY_BGR) + rng.normal(0, 2, (e - s, C1 - C0, 3))
    for row, w in FRACTURES:
        img[row:row + w, C0:C1] = np.array(FRACTURE_RGB)
    return np.clip(img, 0, 255).astype(np.uint8)


def spectral_row(clay_ratio=1.0, iron_ratio=1.0, carb_ratio=1.0,
                 sil_ratio=1.0):
    """9-band (ASTER-style) profile with planted ratios.

    clay = b7/b5, iron = b4/b2, carbonate = b8/b7, silica = b9/b8.
    """
    b = np.ones(9)
    b[1] = 1.0; b[3] = iron_ratio            # b2, b4
    b[4] = 1.0; b[6] = clay_ratio            # b5, b7
    b[7] = carb_ratio * clay_ratio           # b8 (carb = b8/b7)
    b[8] = sil_ratio * b[7]                  # b9 (silica = b9/b8)
    return b.tolist()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def ingested(client):
    photo = make_photo()
    resp = client.post("/innovations/core-scan/scan/ingest", json={
        "image": photo.tolist(), "boxes": BOXES, "hole_id": "DDH-001"})
    assert resp.status_code == 200, resp.text
    return resp.json()


# --------------------------------------------------------------------------
# Logic: registration
# --------------------------------------------------------------------------

def test_registration_exact_geometry():
    boxes = logic.validate_boxes(BOXES)
    qc = logic.registration_qc(boxes, N_ROWS)
    assert qc["pixels_per_meter"] == pytest.approx(PPM)
    assert qc["depth_gaps"] == [{"from_m": 10.0, "to_m": 12.0, "length_m": 2.0}]
    assert qc["depth_span_m"] == pytest.approx(22.0)
    assert qc["registered_length_m"] == pytest.approx(20.0)
    assert qc["coverage_fraction"] == pytest.approx(20.0 / 22.0)


def test_depth_row_roundtrip():
    boxes = logic.validate_boxes(BOXES)
    assert logic.depth_to_row(boxes, 0.0) == 0
    assert logic.depth_to_row(boxes, 5.0) == 100
    assert logic.depth_to_row(boxes, 11.0) is None       # inside the gap
    assert logic.depth_to_row(boxes, 12.0) == 200
    assert logic.row_to_depth(boxes, 250) == pytest.approx(14.5)
    assert logic.row_to_depth(boxes, 500) is None


def test_depth_intervals_respect_gap_and_box_boundary():
    boxes = logic.validate_boxes(BOXES)
    segs = logic.depth_intervals(boxes, 3.0)
    # box 1: 0-3,3-6,6-9,9-10 (truncated); box 2: 12-15,15-18,18-21,21-22
    assert [(s["from_m"], s["to_m"]) for s in segs] == [
        (0, 3), (3, 6), (6, 9), (9, 10), (12, 15), (15, 18), (18, 21), (21, 22)]


# --------------------------------------------------------------------------
# Logic: spectral indices reuse platform cores
# --------------------------------------------------------------------------

def test_spectral_indices_exact_ratios():
    idx = logic.spectral_indices(spectral_row(clay_ratio=2.0, iron_ratio=1.8,
                                              carb_ratio=1.3, sil_ratio=1.2))
    assert idx["clay"] == pytest.approx(2.0)
    assert idx["iron_oxide"] == pytest.approx(1.8)
    assert idx["carbonate"] == pytest.approx(1.3)
    assert idx["silica"] == pytest.approx(1.2)


def test_planted_high_clay_classifies_clay():
    idx = logic.spectral_indices(spectral_row(clay_ratio=2.5))
    cls, conf = logic.classify_indices(idx)
    assert cls == "clay"
    assert conf == pytest.approx(2.5 / 1.4)  # clay threshold 1.4


def test_barren_when_all_below_threshold():
    cls, _ = logic.classify_indices(logic.spectral_indices(spectral_row()))
    assert cls == "barren"


# --------------------------------------------------------------------------
# Logic: photo quality on planted scene
# --------------------------------------------------------------------------

def test_recovery_matches_planted_fraction():
    q = logic.core_quality(make_photo().astype(float),
                           logic.validate_boxes(BOXES))
    assert q["recovery_fraction"] == pytest.approx(PLANTED_RECOVERY, abs=0.02)


def test_planted_fractures_counted():
    q = logic.core_quality(make_photo().astype(float),
                           logic.validate_boxes(BOXES))
    assert q["n_fractures"] == len(FRACTURES)
    depths = sorted(f["depth_m"] for f in q["fractures"])
    # fracture rows 31, 151, 331 -> depths 31/20, 151/20, 12 + 131/20
    assert depths[0] == pytest.approx(31 / PPM, abs=0.1)
    assert depths[1] == pytest.approx(151 / PPM, abs=0.1)
    assert depths[2] == pytest.approx(12 + 131 / PPM, abs=0.1)
    assert q["fractures_per_meter"] == pytest.approx(
        len(FRACTURES) / q["recovered_length_m"])


def test_rqd_fraction_for_unfractured_core_is_one():
    img = make_photo()
    for row, w in FRACTURES:  # erase fractures
        img[row:row + w, C0:C1] = CORE_RGB
    q = logic.core_quality(img.astype(float), logic.validate_boxes(BOXES))
    assert q["n_fractures"] == 0
    assert q["rqd_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Logic: alteration log merging
# --------------------------------------------------------------------------

def test_alteration_run_lengths_exact_metres():
    map_log = [
        {"depth_from_m": 0.0, "depth_to_m": 1.0, "length_m": 1.0,
         "mineral_class": "clay", "proxy": False, "indices": {"clay": 2.0}},
        {"depth_from_m": 1.0, "depth_to_m": 2.0, "length_m": 1.0,
         "mineral_class": "clay", "proxy": False, "indices": {"clay": 3.0}},
        {"depth_from_m": 2.0, "depth_to_m": 3.5, "length_m": 1.5,
         "mineral_class": "carbonate", "proxy": False,
         "indices": {"carbonate": 1.4}},
        {"depth_from_m": 3.5, "depth_to_m": 4.0, "length_m": 0.5,
         "mineral_class": "no_core", "proxy": False, "indices": {}},
        {"depth_from_m": 4.0, "depth_to_m": 5.0, "length_m": 1.0,
         "mineral_class": "clay", "proxy": False, "indices": {"clay": 1.8}},
    ]
    res = logic.alteration_log(map_log)
    assert res["n_runs"] == 4  # no_core gap breaks the clay run
    runs = res["runs"]
    assert runs[0]["length_m"] == pytest.approx(2.0)
    assert runs[0]["alteration_zone"] == "phyllic"
    assert runs[0]["mean_indices"]["clay"] == pytest.approx(2.5)
    assert runs[1]["mineral_class"] == "carbonate"
    assert runs[1]["length_m"] == pytest.approx(1.5)
    assert runs[1]["alteration_zone"] == "propylitic"
    assert runs[3]["depth_from_m"] == 4.0
    assert res["metres_by_class"]["clay"] == pytest.approx(3.0)
    assert res["metres_by_zone"]["phyllic"] == pytest.approx(3.0)
    lines = res["csv"].strip().split("\n")
    assert lines[0].startswith("depth_from_m")
    assert len(lines) == 1 + 4
    assert lines[1].startswith("0.000,2.000,2.000,clay,phyllic")


def test_custom_zonation_config():
    map_log = [{"depth_from_m": 0.0, "depth_to_m": 2.0, "length_m": 2.0,
                "mineral_class": "clay", "proxy": False, "indices": {}}]
    res = logic.alteration_log(map_log, zonation={"argillic": ["clay"]})
    assert res["runs"][0]["alteration_zone"] == "argillic"
    assert res["metres_by_zone"] == {"argillic": pytest.approx(2.0)}


# --------------------------------------------------------------------------
# API: ingest
# --------------------------------------------------------------------------

def test_ingest_returns_registration_qc(ingested):
    assert ingested["hole_id"] == "DDH-001"
    qc = ingested["registration_qc"]
    assert qc["pixels_per_meter"] == pytest.approx(PPM)
    assert qc["depth_gaps"] == [{"from_m": 10.0, "to_m": 12.0,
                                 "length_m": 2.0}]
    assert ingested["image_shape"] == [N_ROWS, N_COLS, 3]


def test_ingest_base64_png(client):
    photo = make_photo()
    buf = io.BytesIO()
    Image.fromarray(photo).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = client.post("/innovations/core-scan/scan/ingest", json={
        "image": b64, "boxes": BOXES})
    assert resp.status_code == 200, resp.text
    assert resp.json()["image_shape"] == [N_ROWS, N_COLS, 3]


def test_ingest_validation_error(client):
    bad = [dict(BOXES[0], row_end=5000)]  # exceeds image rows
    resp = client.post("/innovations/core-scan/scan/ingest", json={
        "image": make_photo().tolist(), "boxes": bad})
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# API: mineral map (spectral)
# --------------------------------------------------------------------------

def _ingest_with_spectral(client):
    photo = make_photo()
    rows = []
    # clay zone for depths 0..5 (rows 0..100), carbonate 5..10, silica 12..22
    for r in range(0, 100):
        rows.append({"row": r, "bands": spectral_row(clay_ratio=2.5)})
    for r in range(100, 200):
        rows.append({"row": r, "bands": spectral_row(carb_ratio=1.5)})
    for r in range(200, 400):
        rows.append({"row": r, "bands": spectral_row(sil_ratio=1.4)})
    resp = client.post("/innovations/core-scan/scan/ingest", json={
        "image": photo.tolist(), "boxes": BOXES, "spectral_rows": rows})
    assert resp.status_code == 200, resp.text
    return resp.json()["scan_id"]


def test_mineral_map_spectral_classes(client):
    scan_id = _ingest_with_spectral(client)
    resp = client.post("/innovations/core-scan/scan/mineral-map", json={
        "scan_id": scan_id, "segment_m": 1.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "spectral"
    log = body["log"]
    assert len(log) == 20  # 20 registered metres at 1 m segments
    for entry in log[:5]:
        assert entry["mineral_class"] == "clay", entry
        assert entry["proxy"] is False
        assert entry["indices"]["clay"] == pytest.approx(2.5)
    for entry in log[5:10]:
        assert entry["mineral_class"] == "carbonate", entry
    for entry in log[10:]:
        assert entry["mineral_class"] == "silica", entry
    assert log[0]["depth_from_m"] == 0.0
    assert log[10]["depth_from_m"] == 12.0  # gap honoured


def test_mineral_map_rgb_proxy_flag(client, ingested):
    resp = client.post("/innovations/core-scan/scan/mineral-map", json={
        "scan_id": ingested["scan_id"], "segment_m": 2.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["source"] == "rgb_proxy"
    classified = [e for e in body["log"] if e["mineral_class"] != "no_core"]
    assert classified
    assert all(e["proxy"] is True for e in classified)
    # every classified segment exposes honest proxy indices
    assert all("iron_oxide" in e["indices"] for e in classified)


def test_mineral_map_unknown_scan(client):
    resp = client.post("/innovations/core-scan/scan/mineral-map", json={
        "scan_id": "nope"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# API: alteration log end-to-end
# --------------------------------------------------------------------------

def test_alteration_log_end_to_end_exact_lengths(client):
    scan_id = _ingest_with_spectral(client)
    resp = client.post("/innovations/core-scan/scan/alteration-log", json={
        "scan_id": scan_id, "segment_m": 1.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    runs = body["runs"]
    assert [(r["mineral_class"], r["length_m"]) for r in runs] == [
        ("clay", 5.0), ("carbonate", 5.0), ("silica", 10.0)]
    assert runs[0]["alteration_zone"] == "phyllic"
    assert runs[1]["alteration_zone"] == "propylitic"
    assert runs[2]["alteration_zone"] == "potassic"
    assert body["metres_by_zone"]["phyllic"] == pytest.approx(5.0)
    assert body["metres_by_zone"]["potassic"] == pytest.approx(10.0)
    csv_lines = body["csv"].strip().split("\n")
    assert len(csv_lines) == 4
    assert csv_lines[1] == "0.000,5.000,5.000,clay,phyllic,False"
    assert csv_lines[3].startswith("12.000,22.000,10.000,silica")


# --------------------------------------------------------------------------
# API: quality end-to-end
# --------------------------------------------------------------------------

def test_quality_endpoint_matches_planted_scene(client, ingested):
    resp = client.post("/innovations/core-scan/scan/quality", json={
        "scan_id": ingested["scan_id"]})
    assert resp.status_code == 200, resp.text
    q = resp.json()
    assert q["recovery_fraction"] == pytest.approx(PLANTED_RECOVERY, abs=0.02)
    assert q["recovery_pct"] == pytest.approx(80.0, abs=2.0)
    assert q["n_fractures"] == len(FRACTURES)
    assert q["registered_length_m"] == pytest.approx(20.0)
    assert q["method"].startswith("photo-derived")
