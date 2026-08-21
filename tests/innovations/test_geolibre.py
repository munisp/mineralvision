"""Tests for /innovations/geolibre — real builder math, real sqlite DB seeding.

No mocks of computation. The export endpoint is tested against a real sqlite
database with seeded ProjectModel/DrillholeModel/SampleModel rows via a
FastAPI dependency override of ``get_db``.
"""

import os
import sys
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.database import Base, DrillholeModel, ProjectModel, SampleModel, get_db
from api.innovations.geolibre import project_builder as pb
from api.innovations.geolibre.routes import router


# ---------------------------------------------------------------------------
# fixtures: isolated PostgreSQL DB with seeded project + drillholes + assays
# ---------------------------------------------------------------------------

PROJECT_ID = str(uuid.uuid4())
HOLES = [  # (hole_id, x, y, z, depth)
    ("DH001", 119.50, -23.50, 540.0, 250.0),
    ("DH002", 119.52, -23.48, 535.0, 300.0),
    ("DH003", 119.54, -23.52, 545.0, 180.0),
]
AU_GRADES = {"DH001": [1.5, 2.5], "DH002": [0.8], "DH003": [3.1, 0.4, 1.2]}
EXP_MEAN = {"DH001": 2.0, "DH002": 0.8, "DH003": 1.5666666666666667}


@pytest.fixture()
def db_session():
    database_url = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        raise RuntimeError("MV_TEST_DATABASE_URL/DATABASE_URL must be an isolated PostgreSQL URL for GeoLibre tests")
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(ProjectModel(id=PROJECT_ID, name="Yilgarn Gold", status="active"))
    hole_rows = {}
    for hid, x, y, z, td in HOLES:
        row = DrillholeModel(id=str(uuid.uuid4()), hole_id=hid,
                             project_id=PROJECT_ID, collar_x=x, collar_y=y,
                             collar_z=z, total_depth=td, status="completed")
        session.add(row)
        hole_rows[hid] = row
    session.flush()
    for hid, grades in AU_GRADES.items():
        for i, g in enumerate(grades):
            session.add(SampleModel(id=str(uuid.uuid4()),
                                    sample_id=f"{hid}-S{i}",
                                    drillhole_id=hole_rows[hid].id,
                                    from_depth=10.0 * i, to_depth=10.0 * (i + 1),
                                    assay_data={"au": g, "cu": 0.1 * (i + 1)}))
    session.commit()
    yield session
    session.close()


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(router)

    def _override():
        yield db_session

    app.dependency_overrides[get_db] = _override
    return TestClient(app)


# ---------------------------------------------------------------------------
# builder schema validity (pure python)
# ---------------------------------------------------------------------------


def test_new_project_schema():
    doc = pb.new_project("T", center=[119.5, -23.5], zoom=10.0,
                         bounds=[119.0, -24.0, 120.0, -23.0])
    assert doc["format_version"] == 1
    assert doc["generator"] == "mineralvision"
    assert doc["center"] == [119.5, -23.5]
    assert doc["bounds"] == [119.0, -24.0, 120.0, -23.0]
    assert doc["layers"] == [] and doc["legends"] == []
    assert pb.validate_project(doc) == []


def test_builder_rejects_bad_geometry_args():
    with pytest.raises(pb.ProjectBuilderError):
        pb.new_project("T", center=[200.0, 0.0], zoom=10.0)
    with pytest.raises(pb.ProjectBuilderError):
        pb.new_project("T", center=[0.0, 0.0], zoom=30.0)
    with pytest.raises(pb.ProjectBuilderError):
        pb.new_project("T", center=[0.0, 0.0], zoom=10.0, basemap="nope")


def test_layer_builders_and_describe():
    doc = pb.new_project("T", center=[119.5, -23.5], zoom=11.0)
    pb.add_tile_layer(doc, "tiles", "/innovations/geotoolkit/tiles/features/{z}/{x}/{y}",
                      opacity=0.9, attribution="MineralVision")
    pb.add_geojson_layer(doc, "collars", {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "geometry": {"type": "Point", "coordinates": [119.5, -23.5]},
         "properties": {"grade": 2.0}}]})
    pb.add_cog_layer(doc, "dem", "https://example.org/dem.tif", rescale=[0, 1000])
    pb.add_geoparquet_layer(doc, "tenements", "https://example.org/t.parquet")
    pb.add_heatmap_layer(doc, "heat",
                         {"type": "FeatureCollection", "features": [
                             {"type": "Feature",
                              "geometry": {"type": "Point", "coordinates": [119.5, -23.5]},
                              "properties": {"grade": 2.0}}]},
                         weight_property="grade")
    pb.add_legend(doc, "heat", title="Au", color_ramp="viridis", vmin=0, vmax=3, unit="ppm")
    assert pb.validate_project(doc) == []
    summary = pb.describe_project(doc)
    assert summary["valid"] is True
    assert summary["n_layers"] == 5
    assert summary["layers_by_type"] == {
        "xyz-tile": 1, "geojson": 1, "cog": 1, "geoparquet": 1, "heatmap": 1}
    assert summary["inline_feature_count"] == 2
    assert summary["n_legends"] == 1
    assert summary["size_bytes"] > 0


def test_builder_rejects_bad_layers():
    doc = pb.new_project("T", center=[0.0, 0.0], zoom=5.0)
    with pytest.raises(pb.ProjectBuilderError):
        pb.add_tile_layer(doc, "bad", "https://example.org/no-tokens")
    with pytest.raises(pb.ProjectBuilderError):
        pb.add_geojson_layer(doc, "badgj", {"type": "Nonsense"})
    with pytest.raises(pb.ProjectBuilderError):
        pb.add_cog_layer(doc, "badcog", "")
    with pytest.raises(pb.ProjectBuilderError):
        pb.add_heatmap_layer(doc, "badheat",
                             {"type": "FeatureCollection", "features": [
                                 {"type": "Feature",
                                  "geometry": {"type": "Point", "coordinates": [0, 0]},
                                  "properties": {}}]},
                             weight_property="grade")
    with pytest.raises(pb.ProjectBuilderError):
        pb.add_legend(doc, "ghost", title="x")
    assert doc["layers"] == []  # nothing partially added


def test_describe_project_flags_invalid():
    bad = {"generator": "mineralvision", "layers": [{"id": "a", "type": "weird"}]}
    summary = pb.describe_project(bad)
    assert summary["valid"] is False
    assert any("missing required key" in p for p in summary["problems"])
    assert any("unknown type" in p for p in summary["problems"])


# ---------------------------------------------------------------------------
# export endpoint (real DB)
# ---------------------------------------------------------------------------


def test_export_contains_exact_drillhole_coordinates(client):
    r = client.get(f"/innovations/geolibre/projects/{PROJECT_ID}/export")
    assert r.status_code == 200
    doc = r.json()
    assert doc["generator"] == "mineralvision"
    assert doc["name"] == "Yilgarn Gold"
    assert doc["source_project_id"] == PROJECT_ID

    collars = next(l for l in doc["layers"] if l["id"] == "drillhole-collars")
    feats = collars["source"]["data"]["features"]
    assert len(feats) == 3
    by_hole = {f["properties"]["hole_id"]: f for f in feats}
    for hid, x, y, z, td in HOLES:
        coords = by_hole[hid]["geometry"]["coordinates"]
        assert coords == [pytest.approx(x), pytest.approx(y)]
        assert by_hole[hid]["properties"]["total_depth"] == td

    # center/bounds derived from real collar extents
    assert doc["center"] == [pytest.approx(119.52), pytest.approx(-23.50)]
    assert doc["bounds"][0] < 119.50 and doc["bounds"][2] > 119.54


def test_export_includes_real_tile_route_and_heatmap(client):
    doc = client.get(f"/innovations/geolibre/projects/{PROJECT_ID}/export").json()
    tiles = next(l for l in doc["layers"] if l["type"] == "xyz-tile")
    assert tiles["source"]["tiles"] == [
        "/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes"]

    heat = next(l for l in doc["layers"] if l["type"] == "heatmap")
    assert heat["source"]["weightProperty"] == "grade"
    feats = heat["source"]["data"]["features"]
    assert len(feats) == 3
    for f in feats:
        hid = f["properties"]["hole_id"]
        assert f["properties"]["grade"] == pytest.approx(EXP_MEAN[hid])
    legend = doc["legends"][0]
    assert legend["vmin"] == pytest.approx(0.8)
    assert legend["vmax"] == pytest.approx(2.0)


def test_export_404_for_project_without_drillholes(client):
    r = client.get("/innovations/geolibre/projects/no-such-project/export")
    assert r.status_code == 404
    assert "no drillholes" in r.json()["detail"]


# ---------------------------------------------------------------------------
# validate endpoint round-trip
# ---------------------------------------------------------------------------


def test_validate_roundtrip(client):
    doc = client.get(f"/innovations/geolibre/projects/{PROJECT_ID}/export").json()
    r = client.post("/innovations/geolibre/projects/validate", json={"document": doc})
    assert r.status_code == 200
    summary = r.json()
    assert summary["valid"] is True
    assert summary["problems"] == []
    assert summary["n_layers"] == len(doc["layers"])
    assert summary["inline_feature_count"] == 6  # 3 collars + 3 heat points
    assert summary["generator"] == "mineralvision"


def test_validate_rejects_garbage(client):
    r = client.post("/innovations/geolibre/projects/validate",
                    json={"document": {"foo": 1}})
    assert r.status_code == 200  # honest summary, not a crash
    body = r.json()
    assert body["valid"] is False
    assert body["problems"]


# ---------------------------------------------------------------------------
# map-html
# ---------------------------------------------------------------------------


def _export_doc(client):
    return client.get(f"/innovations/geolibre/projects/{PROJECT_ID}/export").json()


def test_map_html_fallback_contains_real_urls_and_marker(client):
    doc = _export_doc(client)
    r = client.post("/innovations/geolibre/map-html",
                    json={"document": doc, "prefer_backend": "fallback"})
    assert r.status_code == 200
    html = r.text
    assert "generator: mineralvision-fallback" in html
    assert "maplibre" in html.lower()
    # real platform tile route embedded verbatim
    assert "/innovations/geotoolkit/tiles/features/{z}/{x}/{y}?layer=drillholes" in html
    # real drillhole coordinates embedded in the page
    assert "119.5" in html and "-23.5" in html
    assert "DH001" in html


def test_map_html_503_when_geolibre_forced_but_absent(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "geolibre", None)  # force ImportError
    doc = _export_doc(client)
    r = client.post("/innovations/geolibre/map-html",
                    json={"document": doc, "prefer_backend": "geolibre"})
    assert r.status_code == 503
    assert "pip install geolibre" in r.json()["detail"]


def test_map_html_rejects_invalid_document(client):
    r = client.post("/innovations/geolibre/map-html",
                    json={"document": {"layers": [{"id": "x", "type": "bogus"}]},
                          "prefer_backend": "fallback"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------


def test_capabilities_truthful(client):
    r = client.get("/innovations/geolibre/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["geolibre_package"]["available"] in (True, False)
    if body["geolibre_package"]["available"] is False:
        assert body["geolibre_package"]["remediation"] == "pip install geolibre"
    assert set(body["supported_layer_types"]) == {
        "xyz-tile", "geojson", "cog", "geoparquet", "heatmap"}
    assert "osm" in body["basemap_catalog"]
    assert body["generator"] == "mineralvision"
    assert "drillhole_features" in body["platform_tile_routes"]
