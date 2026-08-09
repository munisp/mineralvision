"""Deterministic tests for the submission_packager innovation (B4-15)."""

import hashlib
import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.submission_packager import router
from api.innovations.submission_packager.logic import (
    MANIFEST_NAME,
    build_manifest,
    package_submission,
    render_files,
    validate_submission,
)
from api.innovations.submission_packager.templates import TEMPLATES

FULL_DATA = {
    "tenement_details": {"tenement_id": "E70/5432", "holder": "MV Exploration Pty Ltd",
                         "grant_date": "2020-06-01", "expiry_date": "2030-05-31"},
    "activities_summary": {"year": 2025, "rc_metres": 4200, "soil_samples": 1150,
                           "summary": "RC drilling and infill geochemistry completed."},
    "drillholes": [
        {"hole_id": "RC002", "east": 501000, "north": 7501000, "elevation": 512, "depth": 180, "dip": -60, "azimuth": 90},
        {"hole_id": "RC001", "east": 500500, "north": 7500500, "elevation": 509, "depth": 150, "dip": -60, "azimuth": 90},
    ],
    "assay_qaqc": [
        {"sample_id": "S0002", "hole_id": "RC001", "from_m": 10, "to_m": 11, "commodity": "Au", "value": 1.2, "qaqc_flag": "pass"},
        {"sample_id": "S0001", "hole_id": "RC001", "from_m": 0, "to_m": 1, "commodity": "Au", "value": 0.4, "qaqc_flag": "pass"},
    ],
    "expenditure": [
        {"category": "drilling", "description": "RC program", "amount": 610000, "currency": "AUD"},
        {"category": "geochemistry", "description": "Assays", "amount": 145000, "currency": "AUD"},
    ],
    "environmental_statement": {"clearing_permits_current": True, "rehab_progress_pct": 40},
}


class TestTemplates:
    def test_two_templates_registered(self):
        assert set(TEMPLATES) == {"wa_dmirs_annual", "generic"}

    def test_wa_requires_environmental_statement_generic_does_not(self):
        wa_required = {s["key"] for s in TEMPLATES["wa_dmirs_annual"]["sections"] if s["required"]}
        gen_required = {s["key"] for s in TEMPLATES["generic"]["sections"] if s["required"]}
        assert "environmental_statement" in wa_required
        assert "environmental_statement" not in gen_required
        # data valid for generic but invalid for WA (missing env statement)
        data = {k: v for k, v in FULL_DATA.items() if k != "environmental_statement"}
        assert validate_submission("generic", data).valid is True
        assert validate_submission("wa_dmirs_annual", data).valid is False


class TestValidator:
    def test_complete_data_passes(self):
        result = validate_submission("wa_dmirs_annual", FULL_DATA)
        assert result.valid is True
        assert result.issues == []

    def test_missing_required_sections_fail_with_names(self):
        data = {k: v for k, v in FULL_DATA.items() if k not in ("drillholes", "expenditure")}
        result = validate_submission("wa_dmirs_annual", data)
        assert result.valid is False
        assert {i.section for i in result.issues} == {"drillholes", "expenditure"}

    def test_empty_collection_counts_as_missing(self):
        data = dict(FULL_DATA)
        data["drillholes"] = []
        assert validate_submission("wa_dmirs_annual", data).valid is False

    def test_packaging_refused_when_incomplete(self):
        with pytest.raises(ValueError, match="drillholes"):
            package_submission("wa_dmirs_annual", {"tenement_details": {"x": 1}})
        with pytest.raises(ValueError, match="unknown template"):
            package_submission("nt_unknown", FULL_DATA)


class TestRendering:
    def test_csv_sorted_by_first_column(self):
        files = dict(render_files("wa_dmirs_annual", FULL_DATA))
        drill_csv = files["drillholes.csv"].decode()
        lines = drill_csv.strip().split("\n")
        assert lines[0] == "hole_id,east,north,elevation,depth,dip,azimuth"
        assert lines[1].startswith("RC001")  # sorted despite RC002 first in input
        assay_csv = files["assay_qaqc.csv"].decode()
        assert assay_csv.strip().split("\n")[1].startswith("S0001")

    def test_render_only_present_sections(self):
        data = {k: v for k, v in FULL_DATA.items() if k in ("tenement_details", "drillholes")}
        names = {name for name, _ in render_files("wa_dmirs_annual", data)}
        assert names == {"tenement_details.json", "drillholes.csv"}


class TestPackaging:
    def test_zip_contents_match_manifest_hashes(self):
        result = package_submission("wa_dmirs_annual", FULL_DATA)
        with zipfile.ZipFile(io.BytesIO(result.zip_bytes)) as zf:
            names = zf.namelist()
            assert names == sorted(names)  # sorted entries
            for entry in result.manifest["files"]:
                content = zf.read(entry["name"])
                assert hashlib.sha256(content).hexdigest() == entry["sha256"]
                assert len(content) == entry["size"]
            manifest_in_zip = json.loads(zf.read(MANIFEST_NAME))
            assert manifest_in_zip["template"] == "wa_dmirs_annual"
            assert manifest_in_zip["files"] == result.manifest["files"]

    def test_zip_is_byte_deterministic(self):
        first = package_submission("wa_dmirs_annual", FULL_DATA).zip_bytes
        second = package_submission("wa_dmirs_annual", FULL_DATA).zip_bytes
        assert first == second
        with zipfile.ZipFile(io.BytesIO(first)) as zf:
            assert {i.date_time for i in zf.infolist()} == {(1980, 1, 1, 0, 0, 0)}

    def test_expected_file_set(self):
        result = package_submission("wa_dmirs_annual", FULL_DATA)
        names = {e["name"] for e in result.manifest["files"]}
        assert names == {
            "tenement_details.json", "activities_summary.json", "drillholes.csv",
            "assay_qaqc.csv", "expenditure.csv", "environmental_statement.json",
        }


class TestAPI:
    @pytest.fixture()
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_validate_endpoint_catches_omission(self, client):
        resp = client.post("/innovations/submission_packager/validate", json={
            "template": "wa_dmirs_annual",
            "data": {k: v for k, v in FULL_DATA.items() if k != "environmental_statement"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert body["issues"][0]["section"] == "environmental_statement"

    def test_package_endpoint_returns_zip(self, client):
        resp = client.post("/innovations/submission_packager/package", json={
            "template": "wa_dmirs_annual", "data": FULL_DATA,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            assert MANIFEST_NAME in zf.namelist()

    def test_package_endpoint_422_on_incomplete(self, client):
        resp = client.post("/innovations/submission_packager/package", json={
            "template": "wa_dmirs_annual", "data": {"tenement_details": {}},
        })
        assert resp.status_code == 422

    def test_preview_returns_manifest(self, client):
        resp = client.post("/innovations/submission_packager/package/preview", json={
            "template": "generic", "data": FULL_DATA,
        })
        assert resp.status_code == 200
        assert resp.json()["template"] == "generic"

    def test_templates_endpoint(self, client):
        resp = client.get("/innovations/submission_packager/templates")
        assert resp.status_code == 200
        assert set(resp.json()) == {"wa_dmirs_annual", "generic"}
