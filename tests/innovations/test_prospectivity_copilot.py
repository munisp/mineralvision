"""Tests for prospectivity_copilot — deterministic NL parser + DB execution."""

import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'MineralVision_Final_Package', 'src'))

from api.database import Base, ProjectModel, DrillholeModel, SampleModel
from api.innovations.prospectivity_copilot.logic import (
    parse_query, explain_query, execute_query, UNIT_TO_GPT,
)


# ---------------------------------------------------------------------------
# AST tests (>= 15 phrases)
# ---------------------------------------------------------------------------

def test_ast_list_gold_wa():
    pq = parse_query("list gold projects in Western Australia")
    assert pq.intent == "list"
    assert pq.entity == "projects"
    assert pq.commodities == ["gold"]
    assert pq.symbols == ["Au"]
    assert pq.regions == ["western australia"]
    assert pq.grade_constraints == []
    assert pq.distance_constraint is None


def test_ast_rank_drillholes_grade():
    pq = parse_query("rank drillholes where gold > 2 g/t in the Pilbara")
    assert pq.intent == "rank"
    assert pq.entity == "drillholes"
    assert pq.commodities == ["gold"]
    assert pq.regions == ["pilbara"]
    assert len(pq.grade_constraints) == 1
    c = pq.grade_constraints[0]
    assert (c.commodity, c.op, c.value, c.unit) == ("gold", ">", 2.0, "g/t")
    assert c.value_gpt == pytest.approx(2.0)


def test_ast_count_projects():
    pq = parse_query("how many projects have copper")
    assert pq.intent == "count"
    assert pq.entity == "projects"
    assert pq.commodities == ["copper"]


def test_ast_porphyry_chile():
    pq = parse_query("show porphyry copper projects in Chile")
    assert pq.intent == "list"
    assert pq.deposit_types == ["porphyry"]
    assert pq.commodities == ["copper"]
    assert pq.regions == ["chile"]


def test_ast_lithium_pegmatite():
    pq = parse_query("find lithium pegmatite projects")
    assert pq.commodities == ["lithium"]
    assert pq.deposit_types == ["pegmatite"]


def test_ast_top_within_km():
    pq = parse_query("top 5 gold drillholes within 10 km of Kalgoorlie")
    assert pq.intent == "rank"
    assert pq.entity == "drillholes"
    assert pq.commodities == ["gold"]
    d = pq.distance_constraint
    assert d is not None and d.distance_km == pytest.approx(10.0)
    assert d.reference == "kalgoorlie"


def test_ast_explain_intent():
    pq = parse_query("explain why epithermal gold targets rank highest")
    assert pq.intent == "explain"
    assert pq.commodities == ["gold"]
    assert pq.deposit_types == ["epithermal"]


def test_ast_count_samples_silver_ppm():
    pq = parse_query("count samples with silver >= 100 ppm")
    assert pq.intent == "count"
    assert pq.entity == "samples"
    c = pq.grade_constraints[0]
    assert (c.commodity, c.op, c.value, c.unit) == ("silver", ">=", 100.0, "ppm")
    assert c.value_gpt == pytest.approx(100.0)  # 1 ppm == 1 g/t


def test_ast_vms_two_commodities():
    pq = parse_query("list vms copper zinc projects in Quebec")
    assert pq.deposit_types == ["vms"]
    assert pq.commodities == ["copper", "zinc"]
    assert pq.regions == ["quebec"]


def test_ast_within_miles_conversion():
    pq = parse_query("find VMS deposits within 2 miles of Noranda")
    d = pq.distance_constraint
    assert d is not None
    assert d.distance_km == pytest.approx(3.218688)
    assert d.reference == "noranda"


def test_ast_symbol_au():
    pq = parse_query("rank drillholes with Au > 2 g/t")
    assert pq.intent == "rank"
    assert pq.entity == "drillholes"
    assert pq.commodities == ["gold"]
    assert len(pq.grade_constraints) == 1
    assert pq.grade_constraints[0].symbol == "Au"


def test_ast_uranium_south_australia():
    pq = parse_query("list uranium projects in South Australia")
    assert pq.commodities == ["uranium"]
    assert pq.regions == ["south australia"]


def test_ast_iocg_gawler_rank():
    pq = parse_query("best IOCG projects in the Gawler")
    assert pq.intent == "rank"
    assert pq.deposit_types == ["iocg"]
    assert pq.regions == ["gawler"]


def test_ast_mineral_spodumene_maps_lithium():
    pq = parse_query("list projects with spodumene")
    assert pq.commodities == ["lithium"]


def test_ast_battery_metals_group():
    pq = parse_query("count battery metals projects")
    assert pq.intent == "count"
    assert set(pq.commodities) == {"lithium", "cobalt", "nickel", "carbon",
                                   "manganese"}


def test_ast_generic_grade_phrase():
    pq = parse_query("show orogenic gold projects with grade at least 1 g/t")
    assert pq.intent == "list"
    assert pq.commodities == ["gold"]
    assert pq.deposit_types == ["orogenic"]
    assert len(pq.grade_constraints) == 1
    c = pq.grade_constraints[0]
    assert (c.commodity, c.op, c.value) == ("gold", ">=", 1.0)


def test_ast_percent_unit_normalization():
    pq = parse_query("list projects with iron above 40 %")
    c = pq.grade_constraints[0]
    assert (c.commodity, c.op) == ("iron", ">")
    assert c.value_gpt == pytest.approx(40.0 * UNIT_TO_GPT["%"])


def test_ast_ppb_unit_normalization():
    pq = parse_query("samples with gold > 500 ppb")
    c = pq.grade_constraints[0]
    assert c.value_gpt == pytest.approx(0.5)


def test_ast_within_metres():
    pq = parse_query("how many drillholes are within 500 m of Kalgoorlie")
    assert pq.intent == "count"
    assert pq.entity == "drillholes"
    assert pq.distance_constraint.distance_km == pytest.approx(0.5)


def test_ast_preposition_not_indium():
    # "in" must not be read as indium
    pq = parse_query("list vms copper zinc projects in Quebec")
    assert "indium" not in pq.commodities


def test_explanation_is_plain_language():
    pq = parse_query("rank drillholes where gold > 2 g/t in the Pilbara")
    text = explain_query(pq)
    assert "Rank drillholes" in text
    assert "gold" in text and "Pilbara" in text.lower() or "pilbara" in text
    assert "greater than 2.0 g/t" in text


# ---------------------------------------------------------------------------
# DB execution tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()

    gold = ProjectModel(id=str(uuid.uuid4()), name="Golden Mile",
                        location="Kalgoorlie, Western Australia",
                        commodities=["gold"], status="active")
    copper = ProjectModel(id=str(uuid.uuid4()), name="Andina Norte",
                          location="Antofagasta, Chile",
                          commodities=["copper"], status="active")
    lithium = ProjectModel(id=str(uuid.uuid4()), name="Greenbushes East",
                           location="Pilbara, Western Australia",
                           commodities=["lithium"], status="active")
    s.add_all([gold, copper, lithium])
    s.flush()

    # Kalgoorlie collars near (151, -31); Chile far away; Pilbara mid
    h1 = DrillholeModel(id=str(uuid.uuid4()), hole_id="DH001",
                        project_id=gold.id, collar_x=151.0, collar_y=-31.0,
                        collar_z=400.0, total_depth=200.0, status="completed")
    h2 = DrillholeModel(id=str(uuid.uuid4()), hole_id="DH002",
                        project_id=gold.id, collar_x=151.01, collar_y=-31.0,
                        collar_z=400.0, total_depth=250.0, status="completed")
    h3 = DrillholeModel(id=str(uuid.uuid4()), hole_id="DH100",
                        project_id=copper.id, collar_x=-70.0, collar_y=-23.0,
                        collar_z=3000.0, total_depth=400.0, status="completed")
    s.add_all([h1, h2, h3])
    s.flush()

    s.add_all([
        SampleModel(id=str(uuid.uuid4()), sample_id="S1", drillhole_id=h1.id,
                    from_depth=50.0, to_depth=52.0,
                    assay_data={"Au": 3.5, "Ag": 12.0}),
        SampleModel(id=str(uuid.uuid4()), sample_id="S2", drillhole_id=h1.id,
                    from_depth=80.0, to_depth=82.0,
                    assay_data={"Au": 1.2}),
        SampleModel(id=str(uuid.uuid4()), sample_id="S3", drillhole_id=h2.id,
                    from_depth=100.0, to_depth=101.0,
                    assay_data={"Au": 5.8}),
        SampleModel(id=str(uuid.uuid4()), sample_id="S4", drillhole_id=h3.id,
                    from_depth=200.0, to_depth=202.0,
                    assay_data={"Cu": 8500.0}),
    ])
    s.commit()
    yield s
    s.close()


def test_execute_list_gold_projects(session):
    pq = parse_query("list gold projects in Western Australia")
    out = execute_query(session, pq)
    assert out["count"] == 1
    assert out["results"][0]["name"] == "Golden Mile"
    assert "gold" in out["explanation"]


def test_execute_rank_drillholes_grade(session):
    pq = parse_query("rank drillholes where gold > 2 g/t")
    out = execute_query(session, pq)
    assert out["count"] == 2
    # DH002 (5.8 g/t) must outrank DH001 (3.5 g/t)
    assert [r["hole_id"] for r in out["results"]] == ["DH002", "DH001"]
    assert out["results"][0]["max_grade_gpt"] == pytest.approx(5.8)


def test_execute_count_samples_silver(session):
    pq = parse_query("count samples with silver >= 10 ppm")
    out = execute_query(session, pq)
    assert out["count"] == 1
    assert out["results"] == []  # count intent returns no rows


def test_execute_within_distance(session):
    pq = parse_query("list drillholes within 5 km of Kalgoorlie")
    out = execute_query(session, pq)
    holes = {r["hole_id"] for r in out["results"]}
    assert holes == {"DH001", "DH002"}  # Chile hole excluded


def test_execute_copper_project_chile(session):
    pq = parse_query("list copper projects in Chile")
    out = execute_query(session, pq)
    assert out["count"] == 1
    assert out["results"][0]["name"] == "Andina Norte"


def test_execute_no_match(session):
    pq = parse_query("list diamond projects in Antarctica")
    out = execute_query(session, pq)
    assert out["count"] == 0


# ---------------------------------------------------------------------------
# Router smoke test
# ---------------------------------------------------------------------------

def test_router_parse_endpoint():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.innovations.prospectivity_copilot import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    r = client.post("/innovations/prospectivity_copilot/parse",
                    json={"question": "rank drillholes where gold > 2 g/t"})
    assert r.status_code == 200
    body = r.json()
    assert body["parsed"]["intent"] == "rank"
    assert body["parsed"]["grade_constraints"][0]["value_gpt"] == pytest.approx(2.0)
    assert "gold" in body["explanation"]
