"""SQLAlchemy model for the assay chain-of-custody ledger."""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Index

from .db import Base


class CustodyLedgerEntry(Base):
    """One append-only link in the chain-of-custody hash chain.

    entry_hash  = SHA256(prev_hash || canonical_json(payload) || iso_timestamp || actor)
    signature   = HMAC-SHA256(server_key, entry_hash)
    """

    __tablename__ = "custody_ledger_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)  # monotonic sequence
    entity_id = Column(String(128), nullable=False, index=True)
    entity_type = Column(String(32), nullable=False)  # sample_batch | dispatch | lab_receipt | results
    event_type = Column(String(32), nullable=False)
    actor = Column(String(128), nullable=False)
    payload = Column(JSON, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    prev_hash = Column(String(64), nullable=False)
    entry_hash = Column(String(64), nullable=False, unique=True)
    signature = Column(String(64), nullable=False)

    __table_args__ = (
        Index("ix_custody_entity_seq", "entity_id", "id"),
    )
