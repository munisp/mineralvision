"""
Persistence for geophysical inversion jobs (SQLAlchemy).

Database URL resolution order:
1. ``MV_INVERSION_JOBS_DB`` environment variable (any SQLAlchemy URL);
2. default: SQLite file ``mv_inversion_jobs.sqlite3`` in the CWD.

Tests override the ``get_session`` FastAPI dependency with their own
session factory (e.g. in-memory SQLite with ``StaticPool``), so no file or
external service is required.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import Float, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class InversionJob(Base):
    __tablename__ = "inversion_job"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: uuid.uuid4().hex)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending",
        doc="pending | running | completed | failed")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_result": self.result_json is not None,
        }


def make_session_factory(url: str | None = None):
    """Create (engine, session_factory) and ensure the schema exists."""
    if url is None:
        url = os.environ.get("MV_INVERSION_JOBS_DB",
                             "sqlite:///mv_inversion_jobs.sqlite3")
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


_default_engine, SessionLocal = make_session_factory()


def get_session():
    """FastAPI dependency: one session per request (overridable in tests)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
