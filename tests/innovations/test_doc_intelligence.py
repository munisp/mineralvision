"""Deterministic tests for the doc_intelligence innovation (B5-17)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.doc_intelligence import router
from api.innovations.doc_intelligence.logic import (
    extract_all,
    extract_commodities,
    extract_coordinates,
    extract_dates,
    extract_hole_ids,
    extract_intervals,
)

FIXTURE = """
RIDGE A PROSPECT - RC DRILLING SUMMARY REPORT
Prepared 15 March 2025. Field program completed 2025-03-10.

Collar RC001 was drilled at zone 50J 500500E 7500500N (lat -23.4567, lon 119.5678)
to 180m depth. Assay results received 10/03/2025.

Significant intercepts:
RC001: from 45.0 to 46.0 m @ 2.34 g/t Au
RC001: 78-79m @ 0.87 g/t Au
DH012: from 120.5 to 121.5 m, 15.6 ppm Ag
RC023: 10-11m @ 1.2% Cu

Gold mineralisation is associated with quartz veining; silver and copper credits noted.
Coordinates verified at -23.4567, 119.5678.
"""


class TestHoleIds:
    def test_exact_hole_ids(self):
        holes = extract_hole_ids(FIXTURE)
        assert [h["hole_id"] for h in holes] == ["RC001", "DH012", "RC023"]
        assert all(h["confidence"] == 0.99 for h in holes)

    def test_suffix_and_uniqueness(self):
        holes = extract_hole_ids("DD12A DD12A RC7 DH999 and DHDD RC")
        assert [h["hole_id"] for h in holes] == ["DD12A", "RC7", "DH999"]


class TestIntervals:
    def test_exact_interval_extraction(self):
        intervals = extract_intervals(FIXTURE)
        assert len(intervals) == 4

        first = intervals[0]
        assert first["hole_id"] == "RC001"
        assert first["from_m"] == 45.0 and first["to_m"] == 46.0
        assert first["value"] == 2.34 and first["unit"] == "g/t"
        assert first["commodity"] == {"symbol": "Au", "name": "gold"}
        assert first["confidence"] == 1.0  # base .6 + fromto .1 + hole .15 + commodity .15

        dash = intervals[1]
        assert dash["from_m"] == 78.0 and dash["to_m"] == 79.0
        assert dash["confidence"] == 0.9  # no from/to phrasing bonus

        silver = intervals[2]
        assert silver["hole_id"] == "DH012" and silver["unit"] == "ppm"
        assert silver["commodity"]["symbol"] == "Ag"

        copper = intervals[3]
        assert copper["unit"] == "%" and copper["commodity"]["symbol"] == "Cu"

    def test_invalid_interval_rejected(self):
        assert extract_intervals("RC001: 50-40m @ 1.0 g/t Au") == []  # to <= from

    def test_missing_components_lower_confidence(self):
        [iv] = extract_intervals("intercept of 30-31m @ 1.5 g/t")
        assert iv["hole_id"] is None and iv["commodity"] is None
        assert iv["confidence"] == 0.6


class TestCommodities:
    def test_commodity_counts(self):
        comms = {c["symbol"]: c for c in extract_commodities(FIXTURE)}
        assert comms["Au"]["name"] == "gold"
        # Au: 2 interval mentions + 1 "Gold" name mention; Ag + 1 "silver"; Cu + 1 "copper"
        assert comms["Au"]["mentions"] == 3
        assert comms["Ag"]["mentions"] == 2
        assert comms["Cu"]["mentions"] == 2


class TestDates:
    def test_dates_normalized(self):
        dates = [d["date"] for d in extract_dates(FIXTURE)]
        assert "2025-03-15" in dates  # 15 March 2025
        assert "2025-03-10" in dates  # ISO + numeric 10/03/2025 deduped by span
        iso = next(d for d in extract_dates(FIXTURE) if d["date"] == "2025-03-15")
        assert iso["confidence"] == 0.9
        numeric = [d for d in extract_dates("on 05/06/2024 drilling")]
        assert numeric[0] == {"date": "2024-06-05", "confidence": 0.7, "span": [3, 13]}


class TestCoordinates:
    def test_utm_and_decdeg(self):
        coords = extract_coordinates(FIXTURE)
        utm = [c for c in coords if c["system"] == "UTM"]
        assert utm[0]["zone"] == "50J"
        assert utm[0]["east"] == 500500.0 and utm[0]["north"] == 7500500.0
        assert utm[0]["confidence"] == 0.95

        labeled = [c for c in coords if c["system"] == "decdeg" and c["confidence"] == 0.95]
        assert labeled[0]["lat"] == -23.4567 and labeled[0]["lon"] == 119.5678

        bare = [c for c in coords if c["system"] == "decdeg" and c["confidence"] == 0.75]
        assert len(bare) == 1  # trailing bare pair only; labeled pair not double-counted

    def test_out_of_range_rejected(self):
        assert extract_coordinates("lat -95.0, lon 200.0") == []


class TestExtractAll:
    def test_counts(self):
        result = extract_all(FIXTURE)
        assert len(result.hole_ids) == 3
        assert len(result.intervals) == 4
        assert len(result.dates) == 3
        assert len(result.coordinates) == 3


class TestAPI:
    @pytest.fixture()
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_extract_text_endpoint(self, client):
        resp = client.post("/innovations/doc_intelligence/extract/text",
                           json={"text": FIXTURE, "filename": "ridge_a.md"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["counts"] == {"hole_ids": 3, "intervals": 4, "commodities": 3,
                                  "dates": 3, "coordinates": 3}
        assert body["source"] == "ridge_a.md"
        assert body["intervals"][0]["value"] == 2.34

    def test_markdown_upload(self, client):
        resp = client.post(
            "/innovations/doc_intelligence/extract/document",
            files={"file": ("notes.md", FIXTURE.encode(), "text/markdown")},
        )
        assert resp.status_code == 200
        assert resp.json()["counts"]["hole_ids"] == 3

    def test_pdf_without_pypdf_returns_501(self, client, monkeypatch):
        import api.innovations.doc_intelligence.routes as routes_mod

        def boom(_bytes):
            raise RuntimeError("pypdf_unavailable")

        monkeypatch.setattr(routes_mod, "extract_text_from_pdf", boom)
        resp = client.post(
            "/innovations/doc_intelligence/extract/document",
            files={"file": ("old_report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert resp.status_code == 501
        assert "pypdf" in resp.json()["detail"]

    def test_non_utf8_rejected(self, client):
        resp = client.post(
            "/innovations/doc_intelligence/extract/document",
            files={"file": ("data.txt", b"\xff\xfe\x00bad", "text/plain")},
        )
        assert resp.status_code == 422
