"""SQLAlchemy models for indigenous knowledge governance."""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, Text, Index

from .db import Base


class KnowledgeRecordModel(Base):
    """An indigenous knowledge record with access tier, consent and attribution."""

    __tablename__ = "indigenous_knowledge_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    community = Column(String(255), nullable=False)
    tier = Column(String(16), nullable=False, index=True)  # public | restricted | sacred
    content = Column(Text, nullable=False)
    consent_reference = Column(String(255), nullable=False)  # consent instrument / agreement id
    attribution = Column(String(255), nullable=False)  # how knowledge holders must be credited
    created_by = Column(String(128), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


class AccessAuditModel(Base):
    """Per-access audit row: who read what, when, at which tier."""

    __tablename__ = "indigenous_access_audit"

    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(Integer, nullable=False, index=True)
    actor = Column(String(128), nullable=False)
    actor_role = Column(String(64), nullable=False)
    action = Column(String(32), nullable=False)  # read | list | export
    tier = Column(String(16), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    __table_args__ = (
        Index("ix_indigenous_audit_ts", "timestamp"),
    )
