"""Deterministic tests for the JORC resource classification engine."""

import numpy as np
import pytest
from api.innovations.jorc_reporter import logic, router
from fastapi import FastAPI
from fastapi.testclient import TestClient

VARIOGRAM_RANGE = 100.0
BLOCK_VOLUME = 10.0 * 10.0 * 5.0  # 500 m3


def _grid_blocks(nx=9, ny=9, spacing=10.0, grade=2.0, density=2.7):
    blocks = []
    for i in range(nx):
        for j in range(ny):
            blocks.append({
                "x": i * spacing, "y": j * spacing, "z": 0.0,
                "grade": grade, "density": density,
            })
    return blocks


def _samples(spacing):
    pts = np.arange(0, 81, spacing)
    return [{"x": float(x), "y": float(y), "z": 0.0} for x in pts for y in pts]


# ---------------------------------------------------------------- logic ----

def test_classification_by_drill_spacing():
    """Blocks near dense samples classify Measured; distant blocks drop class."""
    blocks = _grid_blocks()
    # Dense 10 m drilling on a 100 m variogram range.
    classified = logic.classify_blocks(
        blocks, _samples(10.0), variogram_range=VARIOGRAM_RANGE,
        search=logic.SearchEllipsoid(25.0, 25.0, 25.0))
    assert all(c.resource_class == logic.ResourceClass.MEASURED
               for c in classified)

    # Sparse 40 m drilling: nearest distance 0-28 m, fewer than 4 samples
    # within a 25 m ellipse -> nothing can be Measured.
    classified = logic.classify_blocks(
        blocks, _samples(40.0), variogram_range=VARIOGRAM_RANGE,
        search=logic.SearchEllipsoid(25.0, 25.0, 25.0))
    classes = {c.resource_class for c in classified}
    assert logic.ResourceClass.MEASURED not in classes
    assert classes <= {logic.ResourceClass.INDICATED,
                       logic.ResourceClass.INFERRED,
                       logic.ResourceClass.UNCLASSIFIED}


def test_exact_block_classification_rules():
    """Hand-built geometry with known distances -> exact class assignment."""
    samples = [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 10.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 10.0, "z": 0.0},
        {"x": 10.0, "y": 10.0, "z": 0.0},
    ]
    search = logic.SearchEllipsoid(30.0, 30.0, 30.0)
    blocks = [
        # 5,5 -> 4 samples in ellipse, nearest ~7.07 m <= 0.25*100 -> Measured
        {"x": 5.0, "y": 5.0, "z": 0.0, "grade": 1.0, "density": 2.7},
        # 35,35 -> nearest sample 35.36 m > 30 radius -> 0 in ellipse
        # -> Unclassified
        {"x": 35.0, "y": 35.0, "z": 0.0, "grade": 1.0, "density": 2.7},
        # 15,0 -> all 4 samples within 30 m (dists 15, 5, 18.0, 11.2),
        # nearest 5 <= 25 -> Measured
        {"x": 15.0, "y": 0.0, "z": 0.0, "grade": 1.0, "density": 2.7},
        # 25,25 -> nearest 21.2 m <= 25 but only 3 samples within 30 m
        # ((10,0) and (0,10) at 29.15 m, (10,10) at 21.2 m) -> count 3 < 4
        # so not Measured; indicated rule (<=0.5*range, >=3 samples) matches
        {"x": 25.0, "y": 25.0, "z": 0.0, "grade": 1.0, "density": 2.7},
    ]
    classified = logic.classify_blocks(
        blocks, samples, variogram_range=VARIOGRAM_RANGE, search=search)
    got = [c.resource_class for c in classified]
    assert got[0] == logic.ResourceClass.MEASURED
    assert got[1] == logic.ResourceClass.UNCLASSIFIED
    assert got[2] == logic.ResourceClass.MEASURED
    assert got[3] == logic.ResourceClass.INDICATED
    # Check the measured distances explicitly.
    assert classified[0].nearest_sample_distance == pytest.approx(np.sqrt(50))
    assert classified[0].n_samples_in_ellipse == 4
    assert classified[3].nearest_sample_distance == pytest.approx(np.sqrt(450))
    assert classified[3].n_samples_in_ellipse == 3


def test_summary_mass_balance_and_grades():
    """Per-class tonnage sums to total; weighted grades exact."""
    blocks = _grid_blocks(grade=3.0, density=2.5)
    classified = logic.classify_blocks(
        blocks, _samples(10.0), variogram_range=VARIOGRAM_RANGE,
        search=logic.SearchEllipsoid(25.0, 25.0, 25.0))
    summary = logic.summarize_by_class(classified, BLOCK_VOLUME)
    total_tonnage = sum(s["tonnage"] for s in summary.values())
    assert total_tonnage == pytest.approx(len(blocks) * BLOCK_VOLUME * 2.5)
    m = summary["measured"]
    assert m["n_blocks"] == len(blocks)
    assert m["avg_grade"] == pytest.approx(3.0)
    assert m["metal_content"] == pytest.approx(m["tonnage"] * 3.0)
    assert summary["indicated"]["n_blocks"] == 0


def test_report_sections_and_qaqc_fields():
    blocks = _grid_blocks()
    samples = _samples(20.0)
    search = logic.SearchEllipsoid(40.0, 40.0, 40.0)
    rules = logic.default_rules()
    classified = logic.classify_blocks(
        blocks, samples, VARIOGRAM_RANGE, search, rules)
    qaqc = {"standards_within_control_pct": 95.0, "blank_contamination_events": 0}
    report = logic.build_report(
        classified, samples, VARIOGRAM_RANGE, search, rules,
        block_volume=BLOCK_VOLUME, qaqc_summary=qaqc,
        project_name="test", element="Au")

    for section in ("data_summary", "qaqc_statement", "estimation_params",
                    "classification_table", "grade_tonnage_by_class"):
        assert section in report
    assert report["data_summary"]["n_samples"] == len(samples)
    assert report["qaqc_statement"]["standards_within_control_pct"] == 95.0
    assert report["qaqc_statement"]["blank_contamination_events"] == 0
    # Placeholder fields remain when not supplied.
    assert report["qaqc_statement"]["duplicate_mean_hard_pct"] is None
    # Classification table percentages add to 100.
    pct = sum(v["pct_of_blocks"]
              for v in report["classification_table"].values())
    assert pct == pytest.approx(100.0)


def test_deterministic_repeat():
    blocks = _grid_blocks()
    samples = _samples(15.0)
    search = logic.SearchEllipsoid(30.0, 30.0, 30.0)
    a = logic.classify_blocks(blocks, samples, VARIOGRAM_RANGE, search)
    b = logic.classify_blocks(blocks, samples, VARIOGRAM_RANGE, search)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_anisotropic_search():
    """Anisotropic ellipsoid: sample inside along major axis, outside along
    minor at the same Euclidean distance."""
    from api.geostatistics.kriging import SearchEllipsoid
    search = SearchEllipsoid(radius_major=40.0, radius_minor=10.0,
                             radius_vertical=10.0, azimuth=0.0, dip=0.0)
    # azimuth 0 -> major axis along X (core rotation convention).
    assert search.contains(30.0, 0.0, 0.0)      # 30 m along major
    assert not search.contains(0.0, 30.0, 0.0)  # 30 m along minor


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        logic.classify_blocks(_grid_blocks(), [], VARIOGRAM_RANGE)
    with pytest.raises(ValueError):
        logic.classify_blocks(_grid_blocks(), _samples(20.0), 0.0)


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_classify_and_report(client):
    payload = {
        "blocks": _grid_blocks(),
        "samples": _samples(10.0),
        "variogram_range": VARIOGRAM_RANGE,
        "block_volume": BLOCK_VOLUME,
        "search_ellipsoid": {"radius_major": 25.0, "radius_minor": 25.0,
                             "radius_vertical": 25.0},
    }
    r = client.post("/innovations/jorc_reporter/classify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n_blocks"] == 81
    assert body["summary_by_class"]["measured"]["n_blocks"] == 81

    r = client.post("/innovations/jorc_reporter/report",
                    json={**payload, "project_name": "api-test"})
    assert r.status_code == 200
    report = r.json()
    assert report["project"] == "api-test"
    assert report["classification_table"]["measured"]["tonnage"] == pytest.approx(
        81 * BLOCK_VOLUME * 2.7)


def test_api_default_rules(client):
    r = client.get("/innovations/jorc_reporter/rules/defaults")
    assert r.status_code == 200
    rules = r.json()["rules"]
    assert rules["measured"] == {"max_range_fraction": 0.25, "min_samples": 4}
    assert rules["indicated"] == {"max_range_fraction": 0.5, "min_samples": 3}
    assert rules["inferred"] == {"max_range_fraction": 1.0, "min_samples": 2}


def test_api_validation_error(client):
    r = client.post("/innovations/jorc_reporter/classify", json={
        "blocks": _grid_blocks(), "samples": [],
        "variogram_range": 100.0, "block_volume": 500.0,
    })
    assert r.status_code == 422
