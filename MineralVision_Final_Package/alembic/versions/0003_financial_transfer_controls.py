"""Add durable financial transfer idempotency and audit controls.

Revision ID: 0003_financial_transfer_controls
Revises: 0002_oil_spill_postgres
Create Date: 2026-08-21
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_financial_transfer_controls"
down_revision = "0002_oil_spill_postgres"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "financial_transfer_intents",
        sa.Column("idempotency_key", sa.String(length=128), primary_key=True),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("actor_id", sa.String(length=255), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("intent_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("receipt", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("state IN ('in_progress', 'posted', 'reconciliation_required', 'rejected')", name="ck_financial_transfer_intents_state"),
    )
    op.create_index("ix_financial_transfer_intents_actor_id", "financial_transfer_intents", ["actor_id"])
    op.create_index("ix_financial_transfer_intents_state", "financial_transfer_intents", ["state"])

    op.create_table(
        "financial_transfer_approvals",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String(length=128), sa.ForeignKey("financial_transfer_intents.idempotency_key", ondelete="RESTRICT"), nullable=False),
        sa.Column("approver_id", sa.String(length=255), nullable=False),
        sa.Column("assurance", sa.String(length=32), nullable=False),
        sa.Column("challenge_id", sa.String(length=255), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("idempotency_key", "approver_id", name="uq_financial_transfer_approval_actor"),
        sa.CheckConstraint("decision IN ('approved', 'rejected')", name="ck_financial_transfer_approval_decision"),
    )
    op.create_index("ix_financial_transfer_approvals_intent", "financial_transfer_approvals", ["idempotency_key"])

    op.create_table(
        "financial_transfer_audit_events",
        sa.Column("sequence", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("idempotency_key", sa.String(length=128), sa.ForeignKey("financial_transfer_intents.idempotency_key", ondelete="RESTRICT"), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_id", sa.String(length=255), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("event_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("key_version", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_financial_transfer_audit_events_intent_sequence", "financial_transfer_audit_events", ["idempotency_key", "sequence"])
    op.create_index("ix_financial_transfer_audit_events_actor", "financial_transfer_audit_events", ["actor_id"])


def downgrade() -> None:
    op.drop_index("ix_financial_transfer_audit_events_actor", table_name="financial_transfer_audit_events")
    op.drop_index("ix_financial_transfer_audit_events_intent_sequence", table_name="financial_transfer_audit_events")
    op.drop_table("financial_transfer_audit_events")
    op.drop_index("ix_financial_transfer_approvals_intent", table_name="financial_transfer_approvals")
    op.drop_table("financial_transfer_approvals")
    op.drop_index("ix_financial_transfer_intents_state", table_name="financial_transfer_intents")
    op.drop_index("ix_financial_transfer_intents_actor_id", table_name="financial_transfer_intents")
    op.drop_table("financial_transfer_intents")
