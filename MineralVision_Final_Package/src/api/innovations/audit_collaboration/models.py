"""SQLAlchemy models for audit trail, comments and settings versioning."""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, Index

from .db import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class AuditEventModel(Base):
    """Append-only audit event with before/after JSON diff. Never updated."""

    __tablename__ = "collab_audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    actor = Column(String(128), nullable=False, index=True)
    action = Column(String(64), nullable=False)
    entity_type = Column(String(64), nullable=False)  # project | drillhole | target | ...
    entity_id = Column(String(128), nullable=False)
    before = Column(JSON, nullable=False, default=dict)
    after = Column(JSON, nullable=False, default=dict)
    diff = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_collab_audit_entity", "entity_type", "entity_id", "id"),
    )


class CommentModel(Base):
    """Threaded comment on an entity; parent_id forms the thread."""

    __tablename__ = "collab_comments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(64), nullable=False)
    entity_id = Column(String(128), nullable=False)
    parent_id = Column(Integer, nullable=True)  # NULL → top-level
    author = Column(String(128), nullable=False)
    body = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_collab_comments_entity", "entity_type", "entity_id", "id"),
    )


class SettingsSnapshotModel(Base):
    """Immutable snapshot of project settings; monotonic version per project."""

    __tablename__ = "collab_settings_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(128), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    settings = Column(JSON, nullable=False, default=dict)
    actor = Column(String(128), nullable=False)
    note = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_collab_settings_project_version", "project_id", "version", unique=True),
    )
