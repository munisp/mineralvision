"""
Persistence for rig telemetry (SQLAlchemy).

Database URL resolution order:
1. ``MV_DRILL_TELEMETRY_DB`` environment variable;
2. default: SQLite file ``mv_drill_telemetry.sqlite3`` in the CWD.

Tests override the ``get_session`` FastAPI dependency (e.g. in-memory SQLite
with ``StaticPool``), so no file or external service is required.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class TelemetryPoint(Base):
    __tablename__ = "telemetry_point"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rig_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[str] = mapped_column(String(40), nullable=False)
    depth: Mapped[float] = mapped_column(Float, nullable=False)
    rop: Mapped[float] = mapped_column(Float, nullable=False,
                                       doc="rate of penetration")
    torque: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rpm: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    vibration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    ingested_at: Mapped[str] = mapped_column(
        String(40), nullable=False,
        default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "id": self.id, "rig_id": self.rig_id, "timestamp": self.timestamp,
            "depth": self.depth, "rop": self.rop, "torque": self.torque,
            "rpm": self.rpm, "vibration": self.vibration,
        }


def make_session_factory(url: str | None = None):
    """Create (engine, session_factory) and ensure the schema exists."""
    if url is None:
        url = os.environ.get("MV_DRILL_TELEMETRY_DB",
                             "sqlite:///mv_drill_telemetry.sqlite3")
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
