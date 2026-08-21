"""
Tests for the geodb bridge innovation module.

Seeds the isolated PostgreSQL/PostGIS CI database with real drillholes and
samples via the platform SQLAlchemy models, builds the spatial index, and
runs real bbox/near queries asserting actual returned IDs and distances. It
also tests the lakehouse parquet sync and Sedona availability status. No
mocks or SQLite fallback are used in CI.
"""

import atexit
import os
import sys
import tempfile
import uuid

import pytest
from sqlalchemy import or_

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "MineralVision_Final_Package", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "MineralVision_Enhanced", "lakehouse_architecture"))

_TMPDIR = tempfile.mkdtemp(prefix="geodb_test_")
_LAKEHOUSE = os.path.join(_TMPDIR, "lakehouse")

_PREV_ENV = {k: os.environ.get(k) for k in ("DATABASE_URL", "GEODB_LAKEHOUSE_PATH")}
_TEST_DATABASE_URL = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
if not _TEST_DATABASE_URL.startswith(("postgres://", "postgresql://", "postgresql+")):
    raise RuntimeError("MV_TEST_DATABASE_URL/DATABASE_URL must be an isolated PostgreSQL URL for GeoDB tests")
os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
os.environ["GEODB_LAKEHOUSE_PATH"] = _LAKEHOUSE


def _restore_env():
    for _k, _v in _PREV_ENV.items():
        if _v is None:
            os.environ.pop(_k, None)
        else:
            os.environ[_k] = _v


atexit.register(_restore_env)


@pytest.fixture(scope="module", autouse=True)
def _restore_env_after_module():
    """Restore DATABASE_URL/GEODB_LAKEHOUSE_PATH so later test modules
    (e.g. observability alembic tests) are not polluted."""
    yield
    _restore_env()

from api.database import (  # noqa: E402
    Base,
    DrillholeModel,
    ProjectModel,
    SampleModel,
    UserModel,
)
from api.innovations.geodb import (  # noqa: E402
    router,  # noqa: E402
    service,  # noqa: E402
)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

app = FastAPI()
app.include_router(router)
client = TestClient(app)

# Deterministic collar coordinates
COLLARS = [
    ("DH-001", 100.0, 200.0),
    ("DH-002", 150.0, 250.0),
    ("DH-003", 500.0, 600.0),
    ("DH-004", 105.0, 205.0),
]

HOLE_IDS = {}
SAMPLE_IDS = {}
TEST_RUN_ID = uuid.uuid4().hex
PROJECT_ID = {"value": None}


def _remove_prior_geodb_fixture_data(session):
    """Remove only records created by earlier interrupted runs of this fixture."""
    prior_users = session.query(UserModel).filter(
        or_(
            UserModel.username == "geodb_tester",
            UserModel.username.like("geodb_tester_%"),
        )
    ).all()
    for prior_user in prior_users:
        projects = session.query(ProjectModel).filter(
            ProjectModel.owner_id == prior_user.id,
            ProjectModel.name == "GeoDB Test Project",
        ).all()
        for project in projects:
            drillholes = session.query(DrillholeModel).filter(
                DrillholeModel.project_id == project.id
            ).all()
            for drillhole in drillholes:
                session.query(SampleModel).filter(
                    SampleModel.drillhole_id == drillhole.id
                ).delete(synchronize_session=False)
                session.delete(drillhole)
            session.delete(project)
        session.delete(prior_user)
    session.commit()


@pytest.fixture(scope="module", autouse=True)
def seed_database():
    # Other test modules may mutate DATABASE_URL during collection/execution.
    # GeoDB is PostgreSQL-only, so reassert the isolated CI URL immediately
    # before constructing its local auxiliary-table engine.
    os.environ["DATABASE_URL"] = _TEST_DATABASE_URL
    service._engine_cache.pop(_TEST_DATABASE_URL, None)
    engine = service.get_engine()
    assert engine.dialect.name == "postgresql"
    Base.metadata.create_all(engine)
    session = service.get_session()
    try:
        _remove_prior_geodb_fixture_data(session)
        user = UserModel(
            id=str(uuid.uuid4()),
            username=f"geodb_tester_{TEST_RUN_ID}",
            email=f"geodb_{TEST_RUN_ID}@test.example",
            password_hash="x",
        )
        session.add(user)
        project = ProjectModel(
            id=str(uuid.uuid4()), name="GeoDB Test Project", owner_id=user.id
        )
        session.add(project)
        session.flush()
        PROJECT_ID["value"] = project.id
        for hole_id, x, y in COLLARS:
            dh = DrillholeModel(
                id=str(uuid.uuid4()),
                hole_id=hole_id,
                project_id=project.id,
                collar_x=x,
                collar_y=y,
                collar_z=350.0,
                total_depth=250.0,
                azimuth=0.0,
                dip=-90.0,
                status="completed",
            )
            session.add(dh)
            session.flush()
            HOLE_IDS[hole_id] = dh.id
            for i in range(2):
                s = SampleModel(
                    id=str(uuid.uuid4()),
                    sample_id=f"{hole_id}-S{i+1}",
                    drillhole_id=dh.id,
                    from_depth=10.0 * i,
                    to_depth=10.0 * (i + 1),
                    lithology="granite",
                    assay_data={"Au_ppm": 0.5 + i},
                )
                session.add(s)
                session.flush()
                SAMPLE_IDS[s.sample_id] = s.id
        session.commit()
    finally:
        session.close()
    yield


def test_spatial_enable_postgis():
    resp = client.post("/innovations/geodb/spatial/enable")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dialect"] == "postgres"
    assert body["enabled"] is True
    assert body["mode"] == "postgis"


def test_spatial_status():
    resp = client.get("/innovations/geodb/spatial/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["database_dialect"] == "postgres"
    assert body["spatial_enabled"] is True
    assert body["mode"] == "postgis"


def test_index_drillholes():
    resp = client.post(f"/innovations/geodb/spatial/index/drillholes?project_id={PROJECT_ID['value']}")
    assert resp.status_code == 200
    body = resp.json()
    # 4 drillholes + 8 samples = 12 indexed entities
    assert body["indexed"] == 12
    assert body["by_type"]["drillhole"] == 4
    assert body["by_type"]["sample"] == 8
    bbox = body["bbox"]
    assert bbox["min_x"] == 100.0
    assert bbox["min_y"] == 200.0
    assert bbox["max_x"] == 500.0
    assert bbox["max_y"] == 600.0
    # geometry metadata rows persisted
    status = client.get("/innovations/geodb/spatial/status").json()
    assert status["spatial_feature_rows"] == 12
    assert status["indexed_entities"] == 12


def test_bbox_query_returns_real_ids():
    client.post(f"/innovations/geodb/spatial/index/drillholes?project_id={PROJECT_ID['value']}")
    resp = client.post(
        "/innovations/geodb/spatial/query/bbox",
        json={"min_x": 90.0, "min_y": 190.0, "max_x": 200.0, "max_y": 300.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    drillhole_ids = {
        r["id"] for r in body["results"] if r["entity_type"] == "drillhole"
    }
    # DH-001, DH-002, DH-004 fall inside; DH-003 (500,600) does not
    assert drillhole_ids == {
        HOLE_IDS["DH-001"],
        HOLE_IDS["DH-002"],
        HOLE_IDS["DH-004"],
    }
    # bbox fully excluding everything returns zero
    resp2 = client.post(
        "/innovations/geodb/spatial/query/bbox",
        json={"min_x": 900.0, "min_y": 900.0, "max_x": 1000.0, "max_y": 1000.0},
    )
    assert resp2.json()["count"] == 0


def test_bbox_query_entity_type_filter():
    client.post(f"/innovations/geodb/spatial/index/drillholes?project_id={PROJECT_ID['value']}")
    resp = client.post(
        "/innovations/geodb/spatial/query/bbox",
        json={
            "min_x": 90.0,
            "min_y": 190.0,
            "max_x": 200.0,
            "max_y": 300.0,
            "entity_types": ["drillhole"],
        },
    )
    body = resp.json()
    assert body["count"] == 3
    assert all(r["entity_type"] == "drillhole" for r in body["results"])


def test_near_query_real_distances():
    client.post(f"/innovations/geodb/spatial/index/drillholes?project_id={PROJECT_ID['value']}")
    # query at exact collar of DH-001: nearest drillhole distance must be 0
    resp = client.post(
        "/innovations/geodb/spatial/query/near",
        json={"x": 100.0, "y": 200.0, "k": 4, "entity_types": ["drillhole"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 4
    results = body["results"]
    # sorted ascending by distance
    dists = [r["distance"] for r in results]
    assert dists == sorted(dists)
    assert results[0]["id"] == HOLE_IDS["DH-001"]
    assert results[0]["distance"] == pytest.approx(0.0)
    # DH-004 at (105,205) -> sqrt(50); DH-002 at (150,250) -> sqrt(5000)
    assert results[1]["id"] == HOLE_IDS["DH-004"]
    assert results[1]["distance"] == pytest.approx((50.0) ** 0.5)
    assert results[2]["id"] == HOLE_IDS["DH-002"]
    assert results[2]["distance"] == pytest.approx((5000.0) ** 0.5)


def test_near_query_max_distance():
    client.post(f"/innovations/geodb/spatial/index/drillholes?project_id={PROJECT_ID['value']}")
    resp = client.post(
        "/innovations/geodb/spatial/query/near",
        json={
            "x": 100.0,
            "y": 200.0,
            "k": 100,
            "max_distance": 10.0,
            "entity_types": ["drillhole"],
        },
    )
    body = resp.json()
    ids = {r["id"] for r in body["results"]}
    assert ids == {HOLE_IDS["DH-001"], HOLE_IDS["DH-004"]}


def test_lakehouse_sync_writes_real_parquet():
    import pyarrow.parquet as pq

    resp = client.post(
        "/innovations/geodb/lakehouse/sync",
        json={"project_id": PROJECT_ID["value"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["drillholes"]["row_count"] == 4
    assert body["samples"]["row_count"] == 8

    dh_path = body["drillholes"]["path"]
    s_path = body["samples"]["path"]
    assert os.path.exists(dh_path)
    assert os.path.exists(s_path)

    dh_table = pq.read_table(dh_path)
    assert dh_table.num_rows == 4
    hole_ids = set(dh_table.column("hole_id").to_pylist())
    assert hole_ids == {"DH-001", "DH-002", "DH-003", "DH-004"}

    s_table = pq.read_table(s_path)
    assert s_table.num_rows == 8

    status = client.get("/innovations/geodb/lakehouse/status").json()
    assert status["files"]["drillholes/drillholes.parquet"]["exists"] is True
    assert status["files"]["samples/samples.parquet"]["exists"] is True
    assert status["last_sync"]["drillholes"]["row_count"] == 4


def test_sedona_status_structured():
    resp = client.get("/innovations/geodb/sedona/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "sedona_available" in body
    assert "platform_module_available" in body
    assert isinstance(body["sedona_available"], bool)
    if not body["sedona_available"]:
        assert body["detail"]  # honest reason


def test_sedona_knn_availability_contract():
    status = client.get("/innovations/geodb/sedona/status").json()
    resp = client.post(
        "/innovations/geodb/sedona/knn", json={"x": 100.0, "y": 200.0, "k": 3}
    )
    if status["sedona_available"]:
        assert resp.status_code == 200
    else:
        assert resp.status_code == 503
        assert "Sedona" in resp.json()["detail"]
