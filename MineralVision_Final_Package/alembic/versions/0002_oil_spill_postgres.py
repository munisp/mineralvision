"""Add PostgreSQL oil-spill intelligence and model-governance tables.

Revision ID: 0002_oil_spill_postgres
Revises: 0001_initial
Create Date: 2026-08-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_oil_spill_postgres"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "oil_spill_incidents",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("image_id", sa.String(length=255), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("review_status", sa.String(length=50), nullable=False, server_default="pending_review"),
        sa.Column("severity", sa.String(length=50), nullable=False, server_default="unknown"),
        sa.Column("oil_pixel_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("oil_fraction", sa.Float(), nullable=False, server_default="0"),
        sa.Column("oil_area_m2", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("quality_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("geometry_geojson", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("image_width_px", sa.Integer(), nullable=False),
        sa.Column("image_height_px", sa.Integer(), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=True),
        sa.Column("source_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("reviewer", sa.String(length=255), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_oil_spill_incidents_project_id", "oil_spill_incidents", ["project_id"])
    op.create_index("ix_oil_spill_incidents_image_id", "oil_spill_incidents", ["image_id"])
    op.create_index("ix_oil_spill_incidents_source", "oil_spill_incidents", ["source"])
    op.create_index("ix_oil_spill_incidents_review_status", "oil_spill_incidents", ["review_status"])
    op.create_index("ix_oil_spill_incidents_severity", "oil_spill_incidents", ["severity"])

    op.create_table(
        "oil_spill_models",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_id", sa.String(length=128), nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False, unique=True),
        sa.Column("intended_domains", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="[]"),
        sa.Column("model_card_url", sa.String(length=2048), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("lifecycle_status", sa.String(length=32), nullable=False, server_default="candidate"),
        sa.Column("approved_by", sa.String(length=255), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("model_id", "model_version", name="uq_oil_spill_model_version"),
    )
    op.create_index("ix_oil_spill_models_model_id", "oil_spill_models", ["model_id"])
    op.create_index("ix_oil_spill_models_lifecycle_status", "oil_spill_models", ["lifecycle_status"])

    op.create_table(
        "oil_spill_evaluation_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("model_registration_id", sa.String(length=36), sa.ForeignKey("oil_spill_models.id"), nullable=False),
        sa.Column("dataset_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("split", sa.String(length=32), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("jepa_backbone", sa.String(length=128), nullable=True),
        sa.Column("reviewer", sa.String(length=255), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_oil_spill_evaluation_runs_model_registration_id", "oil_spill_evaluation_runs", ["model_registration_id"])
    op.create_index("ix_oil_spill_evaluation_runs_dataset_fingerprint", "oil_spill_evaluation_runs", ["dataset_fingerprint"])
    op.create_index("ix_oil_spill_evaluation_runs_split", "oil_spill_evaluation_runs", ["split"])
    op.create_index("ix_oil_spill_evaluation_runs_domain", "oil_spill_evaluation_runs", ["domain"])

    op.create_table(
        "oil_spill_incident_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("incident_id", sa.String(length=36), sa.ForeignKey("oil_spill_incidents.id"), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=255), nullable=False),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_oil_spill_incident_events_incident_id", "oil_spill_incident_events", ["incident_id"])
    op.create_index("ix_oil_spill_incident_events_event_type", "oil_spill_incident_events", ["event_type"])


def downgrade() -> None:
    op.drop_index("ix_oil_spill_incident_events_event_type", table_name="oil_spill_incident_events")
    op.drop_index("ix_oil_spill_incident_events_incident_id", table_name="oil_spill_incident_events")
    op.drop_table("oil_spill_incident_events")
    op.drop_index("ix_oil_spill_evaluation_runs_domain", table_name="oil_spill_evaluation_runs")
    op.drop_index("ix_oil_spill_evaluation_runs_split", table_name="oil_spill_evaluation_runs")
    op.drop_index("ix_oil_spill_evaluation_runs_dataset_fingerprint", table_name="oil_spill_evaluation_runs")
    op.drop_index("ix_oil_spill_evaluation_runs_model_registration_id", table_name="oil_spill_evaluation_runs")
    op.drop_table("oil_spill_evaluation_runs")
    op.drop_index("ix_oil_spill_models_lifecycle_status", table_name="oil_spill_models")
    op.drop_index("ix_oil_spill_models_model_id", table_name="oil_spill_models")
    op.drop_table("oil_spill_models")
    op.drop_index("ix_oil_spill_incidents_severity", table_name="oil_spill_incidents")
    op.drop_index("ix_oil_spill_incidents_review_status", table_name="oil_spill_incidents")
    op.drop_index("ix_oil_spill_incidents_source", table_name="oil_spill_incidents")
    op.drop_index("ix_oil_spill_incidents_image_id", table_name="oil_spill_incidents")
    op.drop_index("ix_oil_spill_incidents_project_id", table_name="oil_spill_incidents")
    op.drop_table("oil_spill_incidents")
