"""Tests for commodity_discovery — real gold/lithium engines behind the API."""

import os
import sys

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'MineralVision_Final_Package', 'src'))

from api.innovations.commodity_discovery import router
from api.innovations.commodity_discovery.logic import classify_zonation

PREFIX = "/innovations/commodity-discovery"


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def orogenic_rich():
    return {"sample_id": "rich", "x": 100.0, "y": 200.0,
            "elements": {"Au": 0.5, "As": 500, "Sb": 50, "W": 50, "Bi": 10}}


def orogenic_barren():
    return {"sample_id": "barren", "x": 101.0, "y": 201.0,
            "elements": {"Au": 0.001, "As": 1.0, "Sb": 0.1, "W": 0.1, "Bi": 0.01}}


# ---------------------------------------------------------------------------
# Router import
# ---------------------------------------------------------------------------

def test_router_imports_standalone():
    assert router.prefix == PREFIX
    assert "commodity-discovery" in router.tags
    paths = {r.path for r in router.routes}
    assert f"{PREFIX}/gold/score-samples" in paths
    assert f"{PREFIX}/gold/alteration" in paths
    assert f"{PREFIX}/gold/regolith" in paths
    assert f"{PREFIX}/lithium/score-pegmatite" in paths
    assert f"{PREFIX}/lithium/brine" in paths
    assert f"{PREFIX}/discovery/workflow" in paths
    assert f"{PREFIX}/deposit-types" in paths


# ---------------------------------------------------------------------------
# Gold scoring
# ---------------------------------------------------------------------------

def test_gold_pathfinder_rich_outscores_barren(client):
    resp = client.post(f"{PREFIX}/gold/score-samples", json={
        "deposit_type": "orogenic",
        "samples": [orogenic_barren(), orogenic_rich()],
    })
    assert resp.status_code == 200
    results = {r["sample_id"]: r for r in resp.json()["results"]}
    assert results["rich"]["pathfinder_score"] > 0.3
    assert results["barren"]["pathfinder_score"] == 0.0
    assert results["rich"]["rank"] == 1
    assert results["barren"]["rank"] == 2


def test_orogenic_vs_porphyry_scoring_differs(client):
    """Orogenic weights As/Sb/W; porphyry weights Cu/Mo — same sample, different score."""
    as_rich = {"sample_id": "as", "x": 0, "y": 0,
               "elements": {"Au": 0.2, "As": 500, "Sb": 50, "W": 50, "Cu": 5, "Mo": 0.1}}
    cu_rich = {"sample_id": "cu", "x": 0, "y": 0,
               "elements": {"Au": 0.2, "As": 5, "Sb": 0.5, "W": 0.1, "Cu": 2000, "Mo": 50}}
    oro = client.post(f"{PREFIX}/gold/score-samples", json={
        "deposit_type": "orogenic", "samples": [as_rich, cu_rich]}).json()["results"]
    por = client.post(f"{PREFIX}/gold/score-samples", json={
        "deposit_type": "porphyry_gold", "samples": [as_rich, cu_rich]}).json()["results"]
    oro_by_id = {r["sample_id"]: r["pathfinder_score"] for r in oro}
    por_by_id = {r["sample_id"]: r["pathfinder_score"] for r in por}
    assert oro_by_id["as"] > oro_by_id["cu"]
    assert por_by_id["cu"] > por_by_id["as"]
    # Ratios are deposit-type specific
    oro_ratios = {r["sample_id"]: r["element_ratios"] for r in oro}
    assert "Au/As" in oro_ratios["as"]
    por_ratios = {r["sample_id"]: r["element_ratios"] for r in por}
    assert "Cu/Mo" in por_ratios["cu"]


# ---------------------------------------------------------------------------
# Gold alteration
# ---------------------------------------------------------------------------

def test_gold_alteration_geochem_classification(client):
    # High K2O vs Na2O -> sericite/phyllic; K2O/(K2O+Na2O)*100 > 70
    sample = {"sample_id": "s1", "x": 0, "y": 0,
              "elements": {"K2O": 8.0, "Na2O": 0.5, "MgO": 2.0, "CaO": 0.2,
                           "SiO2": 70.0, "Al2O3": 5.0, "Fe2O3": 2.0, "FeO": 1.0}}
    resp = client.post(f"{PREFIX}/gold/alteration", json={
        "geochem_samples": [sample]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["geochemical"]["sericite_index"]["mean"] > 70
    assert "phyllic" in body["classification"]


def test_gold_alteration_spectral_index(client):
    rng = np.random.default_rng(0)
    data = (0.3 + 0.05 * rng.normal(size=(4, 4, 6))).tolist()
    wavelengths = [500, 660, 875, 950, 2215, 2330]
    resp = client.post(f"{PREFIX}/gold/alteration", json={
        "hyperspectral_data": data, "wavelengths": wavelengths,
        "spectral_indices": ["ferric_iron", "clay_content"]})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["spectral"].keys()) == {"ferric_iron", "clay_content"}
    assert body["spectral"]["ferric_iron"]["mean"] > 0


# ---------------------------------------------------------------------------
# Gold regolith
# ---------------------------------------------------------------------------

def test_gold_regolith_from_dem(client):
    # Flat plain -> thick regolith; planted steep ridge -> thin/outcrop
    dem = np.full((8, 8), 100.0)
    dem[:, 6:] += np.linspace(0, 120, 8)[:, None] * 0 + np.linspace(0, 120, 2)[None, :]
    dem[0, 7] = 400.0  # sharp peak -> steep slope
    resp = client.post(f"{PREFIX}/gold/regolith", json={
        "dem": dem.tolist(), "rainfall": 500.0, "cell_size": 25.0})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["thickness_min_m"] <= body["thickness_mean_m"] <= body["thickness_max_m"]
    # flat area thickness near base (500/50 = 10m), steep cells much thinner
    flat_thickness = body["thickness_grid"][3][1]
    steep_thickness = body["thickness_grid"][0][7]
    assert flat_thickness > steep_thickness
    assert body["regolith_class_counts"]


# ---------------------------------------------------------------------------
# Lithium pegmatite
# ---------------------------------------------------------------------------

def test_lithium_fractionation_series_monotonic(client):
    """Planted zonation series: rising Rb at constant K -> rising fractionation index."""
    samples = []
    # K=3% constant; Rb rises -> K/Rb falls -> more evolved
    for i, rb in enumerate([150.0, 500.0, 1000.0, 1300.0]):
        samples.append({"sample_id": f"p{i}", "x": float(i), "y": 0.0,
                        "li": 200.0, "cs": 60.0, "rb": rb, "ta": 30.0,
                        "sn": 60.0, "k": 3.0})
    resp = client.post(f"{PREFIX}/lithium/score-pegmatite", json={
        "deposit_type": "pegmatite_lct", "samples": samples})
    assert resp.status_code == 200
    results = resp.json()["results"]
    fis = [r["fractionation_index"] for r in results]
    assert fis[0] < fis[1] < fis[2] < fis[3]
    # zonation classification consistent with fractionation index
    zonations = [r["zonation"] for r in results]
    assert zonations[-1] in ("intermediate_zone", "core_zone")
    assert zonations[0] == "border_zone"
    for r in results:
        assert r["zonation"] == classify_zonation(r["fractionation_index"])


def test_lithium_pegmatite_rich_outscores_barren(client):
    rich = {"sample_id": "li_rich", "x": 0, "y": 0,
            "li": 2000.0, "cs": 300.0, "rb": 1500.0, "ta": 100.0, "sn": 100.0, "k": 2.0}
    barren = {"sample_id": "li_barren", "x": 1, "y": 1,
              "li": 5.0, "cs": 1.0, "rb": 20.0, "ta": 0.5, "sn": 1.0, "k": 3.0}
    resp = client.post(f"{PREFIX}/lithium/score-pegmatite", json={
        "samples": [rich, barren]})
    results = {r["sample_id"]: r for r in resp.json()["results"]}
    assert results["li_rich"]["pathfinder_score"] > results["li_barren"]["pathfinder_score"]
    assert results["li_rich"]["rank"] == 1


# ---------------------------------------------------------------------------
# Lithium brine
# ---------------------------------------------------------------------------

def test_brine_mg_li_classification_boundaries(client):
    cases = [
        (290.0, "excellent"),      # Mg/Li = 2.9 < 3
        (500.0, "good"),           # 5.0 < 6
        (900.0, "moderate"),       # 9.0 < 10
        (1500.0, "challenging"),   # 15.0 < 20
        (2500.0, "difficult"),     # 25.0 >= 20
    ]
    samples = [{"sample_id": f"b{i}", "x": 0, "y": 0, "lithium": 100.0,
                "magnesium": mg, "chloride": 120000.0, "sodium": 80000.0}
               for i, (mg, _) in enumerate(cases)]
    resp = client.post(f"{PREFIX}/lithium/brine", json={"samples": samples})
    assert resp.status_code == 200
    results = resp.json()["results"]
    for r, (_, expected) in zip(results, cases):
        assert r["mg_li_interpretation"] == expected
    # no-lithium sample
    resp2 = client.post(f"{PREFIX}/lithium/brine", json={"samples": [
        {"sample_id": "noli", "x": 0, "y": 0, "lithium": 0.0, "magnesium": 100.0}]})
    r2 = resp2.json()["results"][0]
    assert r2["mg_li_interpretation"] == "no_lithium"
    assert r2["mg_li_ratio"] is None


def test_brine_type_classification_and_evaporation(client):
    salar = {"sample_id": "salar", "x": 0, "y": 0, "lithium": 600.0,
             "sodium": 90000.0, "chloride": 150000.0, "magnesium": 2000.0}
    oilfield = {"sample_id": "oil", "x": 1, "y": 1, "lithium": 150.0,
                "calcium": 20000.0, "chloride": 150000.0, "sodium": 10000.0}
    resp = client.post(f"{PREFIX}/lithium/brine", json={"samples": [salar, oilfield]})
    results = {r["sample_id"]: r for r in resp.json()["results"]}
    assert results["salar"]["brine_classification"] == "Na-Cl"
    assert results["oil"]["brine_classification"] == "Ca-Cl"
    # evaporation index vs seawater Cl (19000 mg/L)
    assert results["salar"]["evaporation_index"] == pytest.approx(150000 / 19000, rel=1e-4)
    assert results["salar"]["rank"] == 1  # higher Li


# ---------------------------------------------------------------------------
# Discovery workflow
# ---------------------------------------------------------------------------

def test_workflow_ranked_zones_monotonic(client):
    samples = []
    # Two clusters: rich around (0,0), barren around (2000,2000)
    for i, base in enumerate([0.5, 0.8, 1.0]):
        samples.append({"sample_id": f"r{i}", "x": 10.0 * i, "y": 10.0 * i,
                        "elements": {"Au": base, "As": 500 * base, "Sb": 50 * base,
                                     "W": 50 * base}})
    for i in range(3):
        samples.append({"sample_id": f"b{i}", "x": 2000.0 + 10 * i, "y": 2000.0,
                        "elements": {"Au": 0.001, "As": 1.0, "Sb": 0.1, "W": 0.1}})
    resp = client.post(f"{PREFIX}/discovery/workflow", json={
        "commodity": "gold", "deposit_type": "orogenic",
        "samples": samples, "cell_size": 500.0})
    assert resp.status_code == 200
    body = resp.json()
    zones = body["ranked_zones"]
    assert len(zones) == 2
    scores = [z["zone_score"] for z in zones]
    assert scores == sorted(scores, reverse=True)
    assert [z["rank"] for z in zones] == [1, 2]
    assert zones[0]["best_sample_id"].startswith("r")
    assert zones[0]["zone_score"] > 0.8  # rich cluster mean normalized score
    assert zones[1]["zone_score"] == 0.0  # barren cluster scores all zero
    assert "score" in zones[0]["explanation"]


def test_workflow_lithium(client):
    samples = [
        {"sample_id": "p0", "x": 0.0, "y": 0.0, "li": 1500.0, "cs": 200.0,
         "rb": 1000.0, "ta": 80.0, "k": 2.0},
        {"sample_id": "p1", "x": 3000.0, "y": 0.0, "li": 10.0, "cs": 1.0,
         "rb": 30.0, "ta": 0.5, "k": 3.0},
    ]
    resp = client.post(f"{PREFIX}/discovery/workflow", json={
        "commodity": "lithium", "samples": samples, "cell_size": 500.0})
    assert resp.status_code == 200
    zones = resp.json()["ranked_zones"]
    assert zones[0]["best_sample_id"] == "p0"
    assert zones[0]["zone_score"] >= zones[1]["zone_score"]


# ---------------------------------------------------------------------------
# Deposit types catalogue
# ---------------------------------------------------------------------------

def test_deposit_types_catalogue(client):
    resp = client.get(f"{PREFIX}/deposit-types")
    assert resp.status_code == 200
    body = resp.json()
    gold_types = {g["deposit_type"] for g in body["gold"]}
    assert {"orogenic", "epithermal_high_sulfidation", "epithermal_low_sulfidation",
            "porphyry_gold", "placer"} <= gold_types
    oro = next(g for g in body["gold"] if g["deposit_type"] == "orogenic")
    assert set(oro["diagnostic_elements"]["primary"]) == {"Au", "As", "Sb", "W", "Bi", "Te"}
    li_types = {l["deposit_type"] for l in body["lithium"]}
    assert {"pegmatite_lct", "brine_salar", "clay_hectorite"} <= li_types
    lct = next(l for l in body["lithium"] if l["deposit_type"] == "pegmatite_lct")
    assert "Li" in lct["diagnostic_elements"]["primary"]


# ---------------------------------------------------------------------------
# 422 validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,payload", [
    (f"{PREFIX}/gold/score-samples", {
        "deposit_type": "not_a_deposit", "samples": [orogenic_rich()]}),
    (f"{PREFIX}/gold/score-samples", {
        "deposit_type": "orogenic", "samples": []}),
    (f"{PREFIX}/gold/score-samples", {
        "deposit_type": "orogenic", "samples": [{"sample_id": "x"}]}),
    (f"{PREFIX}/gold/alteration", {}),
    (f"{PREFIX}/gold/alteration", {
        "hyperspectral_data": [[[1.0, 2.0]]], "wavelengths": [500, 600, 700]}),
    (f"{PREFIX}/gold/regolith", {"dem": "not_an_array"}),
    (f"{PREFIX}/gold/regolith", {"dem": [[1.0]], "rainfall": -5}),
    (f"{PREFIX}/lithium/score-pegmatite", {
        "deposit_type": "bogus", "samples": [{"sample_id": "s", "x": 0, "y": 0}]}),
    (f"{PREFIX}/lithium/brine", {"samples": []}),
    (f"{PREFIX}/lithium/brine", {"samples": [
        {"sample_id": "s", "x": 0, "y": 0, "brine_type": "bogus_brine"}]}),
    (f"{PREFIX}/discovery/workflow", {"commodity": "uranium", "samples": [
        {"sample_id": "s", "x": 0, "y": 0, "elements": {"Au": 1}}]}),
    (f"{PREFIX}/discovery/workflow", {"commodity": "gold", "samples": []}),
    (f"{PREFIX}/discovery/workflow", {"commodity": "gold", "samples": [
        {"sample_id": "s", "x": 0, "y": 0, "elements": {"Au": 1}}],
        "cell_size": -10}),
])
def test_malformed_input_rejected_422(client, url, payload):
    resp = client.post(url, json=payload)
    assert resp.status_code == 422, f"{url} returned {resp.status_code}: {resp.text}"
