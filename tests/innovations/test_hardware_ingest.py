"""Tests for /innovations/hardware-ingest — real tiny fixtures, real parsing.

Fixtures are crafted in-test: a minimal valid LAS 1.2 binary (struct-built),
a Bruker-style pXRF CSV, NMEA-0183 GGA sentences, a LAS 2.0 well-log text,
and deterministic numpy coherence arrays. No mocks of computation.
"""

import importlib.util
import struct

import numpy as np
import pytest
from api.innovations.hardware_ingest import router as hw_router
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(hw_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# real fixture builders
# ---------------------------------------------------------------------------

def _build_las_12() -> bytes:
    """Minimal valid LAS 1.2 file: 3 points, point format 0."""
    pts = [  # (x_raw, y_raw, z_raw, intensity, flags, classification)
        (10000, 20000, 3000, 100, 0b001_001, 2),
        (10010, 20010, 3010, 150, 0b001_001, 2),
        (10020, 20020, 3020, 200, 0b001_001, 5),
    ]
    scale = (0.01, 0.01, 0.01)
    offset = (500000.0, 4100000.0, 0.0)
    xs = [p[0] * scale[0] + offset[0] for p in pts]
    ys = [p[1] * scale[1] + offset[1] for p in pts]
    zs = [p[2] * scale[2] + offset[2] for p in pts]

    header = b"LASF"
    header += struct.pack("<H", 0)            # file source id
    header += struct.pack("<H", 0)            # global encoding
    header += b"\x00" * 16                    # GUID
    header += struct.pack("<B", 1)            # version major
    header += struct.pack("<B", 2)            # version minor
    header += b"MVTEST".ljust(32, b"\x00")    # system identifier
    header += b"pytest".ljust(32, b"\x00")    # generating software
    header += struct.pack("<H", 1)            # creation day
    header += struct.pack("<H", 2024)         # creation year
    header += struct.pack("<H", 227)          # header size
    header += struct.pack("<I", 227)          # offset to point data
    header += struct.pack("<I", 0)            # number of VLRs
    header += struct.pack("<B", 0)            # point data format
    header += struct.pack("<H", 20)           # point record length
    header += struct.pack("<I", len(pts))     # legacy point count
    header += struct.pack("<5I", len(pts), 0, 0, 0, 0)  # points by return
    header += struct.pack("<3d", *scale)
    header += struct.pack("<3d", *offset)
    header += struct.pack("<6d", max(xs), min(xs), max(ys), min(ys),
                          max(zs), min(zs))
    assert len(header) == 227

    records = b""
    for x, y, z, inten, flags, cls in pts:
        records += struct.pack("<3iHBBbBH", x, y, z, inten, flags, cls,
                               0, 0, 0)
        assert len(records) % 20 == 0
    return header + records


LAS_BYTES = _build_las_12()
# expected real coordinates
EXP_X = [500000.0 + 10000 * 0.01, 500000.0 + 10010 * 0.01,
         500000.0 + 10020 * 0.01]
EXP_Z = [30.0, 30.10, 30.20]

BRUKER_CSV = (
    "Spectrum,Sample,Fe,Cu,Zn\n"
    "1,SOIL-01,45200.5,320.4,89.1\n"
    "2,SOIL-02,38900.0,410.2,<0.5\n"
    "3,SOIL-03,51000.7,285.9,120.3\n"
)

# two real GGA fixes ~111 m apart in longitude
NMEA_TEXT = (
    "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\n"
    "$GPGGA,123520,4807.038,N,01131.006,E,1,08,0.9,545.5,M,46.9,M,,*48\n"
)

WELLLOG_LAS = """~VERSION INFORMATION
 VERS.                          2.0 :   CWLS LOG ASCII STANDARD - VERSION 2.0
 WRAP.                           NO :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
 STRT.M                        100.0 :
 STOP.M                        104.0 :
 STEP.M                          1.0 :
 NULL.                        -999.25 :
 WELL.                    MV-TEST-001 :
~CURVE INFORMATION
 DEPT.M                              :  1  DEPTH
 GR  .API                            :  2  GAMMA RAY
 RHOB.G/C3                           :  3  BULK DENSITY
~PARAMETER INFORMATION
 RUN .                              1 :  RUN NUMBER
~A  DEPTH     GR       RHOB
100.0    45.5     2.35
101.0    50.1     2.40
102.0    -999.25  2.42
103.0    62.3     -999.25
104.0    58.0     2.55
"""

H2, W2 = 10, 10
COH_T1 = np.full((H2, W2), 0.8)
COH_T2 = np.full((H2, W2), 0.8)
COH_T2[2:6, 3:7] = 0.3  # 16-cell real coherence drop (e.g. ground deformation)

_DELTA = COH_T2 - COH_T1  # -0.5 on the 16-cell block, 0 elsewhere
EXP_DELTA_MEAN = float(_DELTA.mean())          # -0.08
EXP_DELTA_STD = float(_DELTA.std())
EXP_N_CHANGED = 16
EXP_CHANGED_FRAC = 16 / 100


def _has(modname: str) -> bool:
    return importlib.util.find_spec(modname) is not None


LAZ_BACKEND_AVAILABLE = _has("laspy") and _has("lazrs")


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_truthful(client):
    r = client.get("/innovations/hardware-ingest/capabilities")
    assert r.status_code == 200
    body = r.json()
    for name in ("laspy", "lazrs", "rasterio", "numpy", "scipy"):
        assert name in body["backends"]
        assert body["backends"][name]["available"] in (True, False)
    # numpy/scipy are installed in this environment — must be reported real
    assert body["backends"]["numpy"]["available"] is True
    assert body["backends"]["scipy"]["available"] is True
    # LAZ support must exactly match laspy+lazrs reality
    laz = body["hardware"]["lidar"]["formats"]["laz"]
    assert laz["supported"] == LAZ_BACKEND_AVAILABLE
    if not LAZ_BACKEND_AVAILABLE:
        assert "lazrs" in laz["remediation"]
    # uncompressed LAS is always supported
    assert body["hardware"]["lidar"]["formats"]["las"]["supported"] is True
    # truthful GNSS matrix: no dedicated rinex/kml parsers in the codebase
    assert body["hardware"]["gnss"]["formats"]["nmea"]["supported"] is True
    assert body["hardware"]["gnss"]["formats"]["rinex"]["supported"] is False
    assert body["hardware"]["gnss"]["formats"]["kml"]["supported"] is False
    # GeoTIFF InSAR input support must match rasterio reality
    gtiff = body["hardware"]["insar"]["inputs"]["geotiff"]
    assert gtiff["supported"] == _has("rasterio")


# ---------------------------------------------------------------------------
# LiDAR LAS
# ---------------------------------------------------------------------------


def test_lidar_las_real_header_and_points(client):
    r = client.post("/innovations/hardware-ingest/lidar/las",
                    files={"file": ("tiny.las", LAS_BYTES, "application/octet-stream")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "las_1.2"
    assert body["backend"] == "built-in LASReader"
    assert body["point_count"] == 3
    assert body["point_format"] == 0
    assert body["bounds"]["min_x"] == pytest.approx(min(EXP_X), abs=1e-9)
    assert body["bounds"]["max_x"] == pytest.approx(max(EXP_X), abs=1e-9)
    assert body["bounds"]["min_z"] == pytest.approx(30.0, abs=1e-9)
    assert body["bounds"]["max_z"] == pytest.approx(30.20, abs=1e-9)
    assert body["sample_point_count"] == 3
    p0 = body["sample_points"][0]
    assert p0["x"] == pytest.approx(EXP_X[0], abs=1e-9)
    assert p0["z"] == pytest.approx(30.0, abs=1e-9)
    assert p0["intensity"] == 100
    assert p0["classification"] == 2
    p2 = body["sample_points"][2]
    assert p2["classification"] == 5
    assert p2["intensity"] == 200


def test_lidar_las_invalid_signature_422(client):
    r = client.post("/innovations/hardware-ingest/lidar/las",
                    files={"file": ("bad.las", b"NOTALASFILE" * 40,
                                    "application/octet-stream")})
    assert r.status_code == 422
    assert "signature" in r.json()["detail"].lower()


def test_lidar_laz_honest_failure_without_backend(client):
    if LAZ_BACKEND_AVAILABLE:
        pytest.skip("laspy+lazrs installed; honest-failure path not reachable")
    r = client.post("/innovations/hardware-ingest/lidar/las",
                    files={"file": ("tiny.laz", LAS_BYTES,
                                    "application/octet-stream")})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "pip install" in detail
    assert "lazrs" in detail or "laspy" in detail
    # never fabricated
    assert "points" not in r.text.lower() or "sample_points" not in r.text


# ---------------------------------------------------------------------------
# pXRF
# ---------------------------------------------------------------------------


def test_xrf_bruker_csv_real_values(client):
    r = client.post("/innovations/hardware-ingest/xrf/csv",
                    files={"file": ("export.csv", BRUKER_CSV.encode(), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["vendor_detected"] == "bruker_s1_titan"
    assert body["reading_count"] == 3
    assert set(body["elements_detected"]) == {"Fe", "Cu", "Zn"}
    fe = body["element_statistics"]["Fe"]
    assert fe["count"] == 3
    assert fe["min"] == pytest.approx(38900.0, abs=1e-6)
    assert fe["max"] == pytest.approx(51000.7, abs=1e-6)
    assert fe["mean"] == pytest.approx(
        (45200.5 + 38900.0 + 51000.7) / 3, abs=1e-6)
    # Zn row 2 is '<0.5' -> below detection, excluded from stats
    zn = body["element_statistics"]["Zn"]
    assert zn["count"] == 2
    assert zn["below_detection_count"] == 1
    assert zn["max"] == pytest.approx(120.3, abs=1e-6)
    assert body["qc_flags_summary"].get("BDL", 0) >= 1


def test_xrf_unknown_vendor_param_422(client):
    r = client.post("/innovations/hardware-ingest/xrf/csv?vendor=acme",
                    files={"file": ("export.csv", BRUKER_CSV.encode(), "text/csv")})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GNSS
# ---------------------------------------------------------------------------


def test_gnss_nmea_real_fixes(client):
    r = client.post("/innovations/hardware-ingest/gnss/nmea",
                    files={"file": ("track.nmea", NMEA_TEXT.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dedicated_parser"] is True
    assert body["parser"] == "NMEAParser"
    assert body["observation_count"] == 2
    # 4807.038 N = 48 + 7.038/60
    exp_lat = 48 + 7.038 / 60
    assert body["bounds"]["min_lat"] == pytest.approx(exp_lat, abs=1e-9)
    # 01131.006 E = 11 + 31.006/60
    assert body["bounds"]["max_lon"] == pytest.approx(11 + 31.006 / 60, abs=1e-9)


def test_gnss_unsupported_fmt_422(client):
    r = client.post("/innovations/hardware-ingest/gnss/geojson",
                    files={"file": ("x.geojson", b"{}", "application/json")})
    assert r.status_code == 422


def test_gnss_rinex_truthful_fallback_warning(client):
    # RINEX obs content has no dedicated parser; endpoint must say so.
    rinex = "     3.03           OBSERVATION DATA    M (MIXED)           RINEX VERSION / TYPE\n"
    r = client.post("/innovations/hardware-ingest/gnss/rinex",
                    files={"file": ("site.obs", rinex.encode(), "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["dedicated_parser"] is False
    assert "warning" in body


# ---------------------------------------------------------------------------
# well-log LAS
# ---------------------------------------------------------------------------


def test_welllog_las_real_stats(client):
    r = client.post("/innovations/hardware-ingest/welllog/las",
                    files={"file": ("mv-test-001.las", WELLLOG_LAS.encode(),
                                    "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 5
    assert body["n_curves"] == 3
    assert body["depth_mnemonic"] == "DEPT"
    assert body["null_value"] == pytest.approx(-999.25)
    assert body["depth_range"] == {"start": 100.0, "stop": 104.0}
    curves = {c["mnemonic"]: c for c in body["curves"]}
    assert curves["GR"]["unit"] == "API"
    # GR has one null (-999.25 at 102.0) -> 4 valid values
    assert curves["GR"]["count"] == 4
    assert curves["GR"]["min"] == pytest.approx(45.5, abs=1e-9)
    assert curves["GR"]["max"] == pytest.approx(62.3, abs=1e-9)
    assert curves["GR"]["mean"] == pytest.approx(
        (45.5 + 50.1 + 62.3 + 58.0) / 4, abs=1e-9)
    # RHOB has one null (at 103.0)
    assert curves["RHOB"]["count"] == 4
    assert curves["RHOB"]["mean"] == pytest.approx(
        (2.35 + 2.40 + 2.42 + 2.55) / 4, abs=1e-9)
    assert body["well"]["WELL"] == "MV-TEST-001"


def test_welllog_missing_data_block_422(client):
    bad = "~VERSION INFORMATION\n VERS. 2.0 :\n~CURVE INFORMATION\n DEPT.M : DEPTH\n"
    r = client.post("/innovations/hardware-ingest/welllog/las",
                    files={"file": ("bad.las", bad.encode(), "text/plain")})
    assert r.status_code == 422
    assert "~A" in r.json()["detail"]


# ---------------------------------------------------------------------------
# InSAR coherence change
# ---------------------------------------------------------------------------


def test_insar_coherence_change_real_metrics(client):
    r = client.post(
        "/innovations/hardware-ingest/insar/coherence-change",
        json={"coherence_t1": COH_T1.tolist(), "coherence_t2": COH_T2.tolist(),
              "abs_threshold": 0.2, "z_threshold": 2.0, "pixel_size": 20.0})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["synthetic"] is False
    assert body["n_cells"] == 100
    assert body["delta_mean"] == pytest.approx(EXP_DELTA_MEAN, abs=1e-12)
    assert body["delta_std"] == pytest.approx(EXP_DELTA_STD, abs=1e-12)
    assert body["delta_min"] == pytest.approx(-0.5, abs=1e-12)
    assert body["delta_max"] == pytest.approx(0.0, abs=1e-12)
    assert body["n_changed_cells"] == EXP_N_CHANGED
    assert body["changed_fraction"] == pytest.approx(EXP_CHANGED_FRAC, abs=1e-12)
    assert body["n_regions"] == 1
    region = body["regions"][0]
    assert region["n_pixels"] == 16
    assert region["area"] == pytest.approx(16 * 20.0 ** 2, abs=1e-9)
    assert region["mean_delta"] == pytest.approx(-0.5, abs=1e-12)


def test_insar_no_change_zero_fraction(client):
    same = COH_T1.tolist()
    r = client.post(
        "/innovations/hardware-ingest/insar/coherence-change",
        json={"coherence_t1": same, "coherence_t2": same})
    assert r.status_code == 200
    body = r.json()
    assert body["delta_std"] == 0.0
    assert body["n_changed_cells"] == 0
    assert body["changed_fraction"] == 0.0


def test_insar_shape_mismatch_422(client):
    r = client.post(
        "/innovations/hardware-ingest/insar/coherence-change",
        json={"coherence_t1": COH_T1.tolist(),
              "coherence_t2": COH_T2[:5].tolist()})
    assert r.status_code == 422
    assert "shape mismatch" in r.json()["detail"]


def test_insar_no_payload_422(client):
    r = client.post("/innovations/hardware-ingest/insar/coherence-change")
    assert r.status_code == 422
