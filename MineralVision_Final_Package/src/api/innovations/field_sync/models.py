"""SQLAlchemy models for offline field data synchronization."""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Index

from .db import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EntityStateModel(Base):
    """Current server-side state of a synced entity (e.g. a field sample)."""

    __tablename__ = "sync_entity_state"

    entity_id = Column(String(128), primary_key=True)
    entity_type = Column(String(32), nullable=True)  # field_log | sample | photo | ...
    version = Column(Integer, nullable=False, default=0)  # monotonically increasing
    data = Column(JSON, nullable=False, default=dict)
    deleted = Column(Boolean, nullable=False, default=False)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class SyncOpModel(Base):
    """Applied (or attempted) client operation — the change log."""

    __tablename__ = "sync_ops"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_op_id = Column(String(64), nullable=False, unique=True)  # idempotency key
    entity_id = Column(String(128), nullable=False, index=True)
    entity_type = Column(String(32), nullable=True)
    device_id = Column(String(64), nullable=True, index=True)  # originating device
    op = Column(String(16), nullable=False)  # create | update | delete
    base_version = Column(Integer, nullable=False)
    applied_version = Column(Integer, nullable=True)  # version after apply; NULL when conflict
    status = Column(String(16), nullable=False)  # applied | conflict | duplicate
    payload = Column(JSON, nullable=False, default=dict)
    client_ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_sync_ops_entity_version", "entity_id", "applied_version"),
    )


class ConflictModel(Base):
    """Conflict record: server wins by default; both versions retained."""

    __tablename__ = "sync_conflicts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(128), nullable=False, index=True)
    client_op_id = Column(String(64), nullable=False)
    op = Column(String(16), nullable=False)
    base_version = Column(Integer, nullable=False)
    server_version = Column(Integer, nullable=False)
    client_payload = Column(JSON, nullable=False, default=dict)
    server_payload = Column(JSON, nullable=False, default=dict)
    resolution = Column(String(32), nullable=False, default="server_wins")
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class DeviceCursorModel(Base):
    """Per-device sync cursor for delta downloads.

    ``cursor`` is the highest applied entity version this device has already
    seen; the next pull returns only ops with applied_version > cursor.
    """

    __tablename__ = "sync_device_cursors"

    device_id = Column(String(64), primary_key=True)
    cursor = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
