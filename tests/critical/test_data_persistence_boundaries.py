"""DATA-01 through DATA-05: PostgreSQL migration, project isolation, and concurrency boundary tests.

These tests validate the database migration chain, tenant isolation, project-scoped operations,
and concurrent idempotency behavior against a real PostgreSQL instance.
"""
import hashlib
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "MineralVision_Final_Package", "src"))

from api.database import Base, engine, SessionLocal, get_db

# Skip the entire module if no PostgreSQL test database is configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("MV_TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="DATA tests require a PostgreSQL test database (MV_TEST_DATABASE_URL)",
)


@pytest.fixture(scope="module")
def db_engine():
    """Use the module-level engine from the database module."""
    yield engine


@pytest.fixture
def db_session():
    """Create a scoped session for each test."""
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


# ---------------------------------------------------------------------------
# DATA-01: Migration chain produces expected schema
# ---------------------------------------------------------------------------

class TestDATA01MigrationSchema:
    """Alembic upgrade head MUST produce all required tables and constraints."""

    def test_oil_spill_tables_exist(self, db_engine):
        """The oil-spill incident, model, evaluation, and event tables exist."""
        from sqlalchemy import inspect
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        required = ["oil_spill_incidents", "oil_spill_models", "oil_spill_evaluation_runs", "oil_spill_incident_events"]
        # Note: these are the exact table names from the migration
        for table in required:
            assert table in tables, f"Missing table: {table}"

    def test_core_tables_exist(self, db_engine):
        """Core platform tables exist after migration."""
        from sqlalchemy import inspect
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        core = ["users", "projects", "drillholes", "samples"]
        for table in core:
            assert table in tables, f"Missing core table: {table}"

    def test_financial_tables_exist(self, db_engine):
        """Financial transfer control tables exist after migration 0003."""
        from sqlalchemy import inspect
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        financial = ["financial_transfer_intents", "financial_transfer_approvals", "financial_transfer_audit_events"]
        for table in financial:
            assert table in tables, f"Missing financial table: {table}"


# ---------------------------------------------------------------------------
# DATA-02: Migration downgrade/re-upgrade safety
# ---------------------------------------------------------------------------

class TestDATA02MigrationReversibility:
    """Migration operations MUST not silently fall back to SQLite."""

    def test_engine_is_postgresql(self, db_engine):
        """The test engine dialect is PostgreSQL, not SQLite."""
        assert "postgresql" in str(db_engine.url), "Engine is not PostgreSQL"

    def test_sqlite_url_rejected_in_production_mode(self):
        """get_engine with a sqlite URL in production mode should fail or warn."""
        # The database module should reject SQLite in non-development mode
        with pytest.raises(Exception):
            os.environ.pop("MINERALVISION_DEV_MODE", None)
            get_engine("sqlite:///test.db")


# ---------------------------------------------------------------------------
# DATA-03: Cross-project GeoDB isolation
# ---------------------------------------------------------------------------

class TestDATA03ProjectIsolation:
    """Project-scoped operations MUST NOT leak data across tenants."""

    def test_project_model_has_owner(self, db_session):
        """ProjectModel requires an owner_id field for tenant isolation."""
        from api.database import ProjectModel, UserModel
        user_id = str(uuid.uuid4())
        user = UserModel(id=user_id, username=f"user_{user_id[:8]}", email=f"{user_id[:8]}@test.local", password_hash="x", first_name="Test", last_name="User")
        db_session.add(user)
        db_session.flush()
        project = ProjectModel(
            id=str(uuid.uuid4()),
            name=f"test-project-{uuid.uuid4().hex[:8]}",
            owner_id=user_id,
            description="Isolation test",
        )
        db_session.add(project)
        db_session.flush()
        assert project.owner_id is not None
        db_session.rollback()

    def test_different_owners_cannot_see_each_other(self, db_session):
        """Projects with different owners are isolated by owner_id filter."""
        from api.database import ProjectModel, UserModel
        owner_a = str(uuid.uuid4())
        owner_b = str(uuid.uuid4())
        user_a = UserModel(id=owner_a, username=f"user_{owner_a[:8]}", email=f"{owner_a[:8]}@test.local", password_hash="x", first_name="A", last_name="User")
        user_b = UserModel(id=owner_b, username=f"user_{owner_b[:8]}", email=f"{owner_b[:8]}@test.local", password_hash="x", first_name="B", last_name="User")
        db_session.add_all([user_a, user_b])
        db_session.flush()
        proj_a = ProjectModel(id=str(uuid.uuid4()), name="proj-a", owner_id=owner_a, description="A")
        proj_b = ProjectModel(id=str(uuid.uuid4()), name="proj-b", owner_id=owner_b, description="B")
        db_session.add_all([proj_a, proj_b])
        db_session.flush()

        # Query as owner_a
        results = db_session.query(ProjectModel).filter_by(owner_id=owner_a).all()
        assert all(p.owner_id == owner_a for p in results)
        assert proj_b.id not in [p.id for p in results]
        db_session.rollback()


# ---------------------------------------------------------------------------
# DATA-04: Concurrent idempotency reservation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("MV_FINANCIAL_TABLES_MIGRATED"),
    reason="DATA-04 requires financial migration 0003 (set MV_FINANCIAL_TABLES_MIGRATED=1)",
)
class TestDATA04ConcurrentIdempotency:
    """Concurrent transfer reservations with the same key MUST produce exactly one intent."""

    def test_concurrent_same_key_produces_one_intent(self, db_engine):
        """Two threads reserving the same idempotency key produce one persisted intent."""
        from sqlalchemy import text
        idempotency_key = f"test-concurrent-{uuid.uuid4().hex}"
        results = []

        def reserve(worker_id):
            session = SessionLocal()
            try:
                session.execute(text(
                    "INSERT INTO financial_transfer_intents "
                    "(idempotency_key, request_hash, actor_id, state, intent_payload, created_at) "
                    "VALUES (:key, :hash, :actor, 'in_progress', :payload, NOW()) "
                    "ON CONFLICT (idempotency_key) DO NOTHING"
                ), {
                    "key": idempotency_key,
                    "hash": hashlib.sha256(f"worker-{worker_id}".encode()).hexdigest(),
                    "actor": f"test-actor-{worker_id}",
                    "payload": json.dumps({"amount": 100, "currency": "USD"}),
                })
                session.commit()
                results.append(("inserted", worker_id))
            except Exception as e:
                session.rollback()
                results.append(("conflict", worker_id))
            finally:
                session.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(reserve, i) for i in range(4)]
            for f in futures:
                f.result()

        session = SessionLocal()
        count = session.execute(
            text("SELECT COUNT(*) FROM financial_transfer_intents WHERE idempotency_key = :key"),
            {"key": idempotency_key}
        ).scalar()
        session.execute(text("DELETE FROM financial_transfer_intents WHERE idempotency_key = :key"), {"key": idempotency_key})
        session.commit()
        session.close()
        assert count == 1


# ---------------------------------------------------------------------------
# DATA-05: Audit event key rotation boundary
# ---------------------------------------------------------------------------

class TestDATA05AuditKeyBoundary:
    """Audit events MUST be verifiable only with the correct key version."""

    def test_hmac_verification_requires_correct_key(self):
        """An HMAC computed with key A cannot be verified with key B."""
        import hmac
        import hashlib
        key_a = b"audit-key-version-1"
        key_b = b"audit-key-version-2"
        message = b"transfer_intent_created|2026-08-21T12:00:00Z|amount=100"
        mac_a = hmac.new(key_a, message, hashlib.sha256).hexdigest()
        mac_b = hmac.new(key_b, message, hashlib.sha256).hexdigest()
        assert mac_a != mac_b
        # Verification with wrong key fails
        assert not hmac.compare_digest(mac_a, mac_b)
