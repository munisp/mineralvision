"""SQLAlchemy models for webhooks, deliveries and API keys."""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean

from .db import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class WebhookModel(Base):
    """A webhook subscription: URL + secret bound to event topics."""

    __tablename__ = "hub_webhooks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(512), nullable=False)
    secret = Column(String(128), nullable=False)  # HMAC signing secret
    topics = Column(JSON, nullable=False, default=list)
    name = Column(String(128), nullable=False, default="")
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class DeliveryModel(Base):
    """Delivery log entry: one row per webhook delivery attempt series."""

    __tablename__ = "hub_deliveries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    webhook_id = Column(Integer, nullable=False, index=True)
    topic = Column(String(128), nullable=False)
    payload = Column(JSON, nullable=False, default=dict)
    status = Column(String(16), nullable=False)  # success | failed
    attempts = Column(Integer, nullable=False, default=0)
    last_status_code = Column(Integer, nullable=True)
    signature = Column(String(80), nullable=False)
    error = Column(String(512), nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class ApiKeyModel(Base):
    """Scoped API key; the secret is stored ONLY as a bcrypt hash."""

    __tablename__ = "hub_api_keys"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(32), nullable=False, unique=True, index=True)  # public lookup id
    key_hash = Column(String(128), nullable=False)  # bcrypt of full key
    name = Column(String(128), nullable=False)
    tenant_id = Column(String(128), nullable=False, default="", index=True)
    scopes = Column(JSON, nullable=False, default=list)  # e.g. ["read", "write"]
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class EvidenceRecordModel(Base):
    """Tenant-bound source evidence with canonical lineage and model provenance."""

    __tablename__ = "hub_evidence_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(40), nullable=False, unique=True, index=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    source_system = Column(String(64), nullable=False)
    source_ref = Column(String(1024), nullable=False)
    source_version = Column(String(256), nullable=False)
    observed_at = Column(DateTime, nullable=False)
    ingested_at = Column(DateTime, nullable=False, default=_utcnow)
    geometry = Column(JSON, nullable=False, default=dict)
    payload = Column(JSON, nullable=False, default=dict)
    model_run = Column(JSON, nullable=False, default=dict)
    lineage_hash = Column(String(64), nullable=False, index=True)


class WritebackProposalModel(Base):
    """A reviewed candidate update; no external write happens in this table."""

    __tablename__ = "hub_writeback_proposals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    proposal_id = Column(String(40), nullable=False, unique=True, index=True)
    tenant_id = Column(String(128), nullable=False, index=True)
    evidence_id = Column(String(40), nullable=False, index=True)
    target_system = Column(String(64), nullable=False)
    target_ref = Column(String(1024), nullable=False)
    state = Column(String(32), nullable=False, default="staged", index=True)
    request_hash = Column(String(64), nullable=False, index=True)
    candidate_payload = Column(JSON, nullable=False, default=dict)
    dry_run = Column(JSON, nullable=False, default=dict)
    submitted_by = Column(String(128), nullable=False)
    reviewer_id = Column(String(128), nullable=True)
    mfa_verified = Column(Boolean, nullable=False, default=False)
    review_reason = Column(String(2048), nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    approved_at = Column(DateTime, nullable=True)
