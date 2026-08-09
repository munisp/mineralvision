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
    scopes = Column(JSON, nullable=False, default=list)  # e.g. ["read", "write"]
    active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
