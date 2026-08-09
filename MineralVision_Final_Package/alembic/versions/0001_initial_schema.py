"""Initial schema — all tables from src.api.database.Base.metadata.

Autogenerate-style revision written explicitly against the existing models:
users, projects, drillholes, samples, qaqc_records, reports, audit_logs.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-09
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("first_name", sa.String(length=100), nullable=True),
        sa.Column("last_name", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "projects",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("commodities", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("owner_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_projects_name", "projects", ["name"], unique=False)

    op.create_table(
        "drillholes",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("hole_id", sa.String(length=100), nullable=False),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("collar_x", sa.Float(), nullable=False),
        sa.Column("collar_y", sa.Float(), nullable=False),
        sa.Column("collar_z", sa.Float(), nullable=False),
        sa.Column("total_depth", sa.Float(), nullable=False),
        sa.Column("azimuth", sa.Float(), nullable=True),
        sa.Column("dip", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("assay_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_drillholes_hole_id", "drillholes", ["hole_id"], unique=False)

    op.create_table(
        "samples",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("sample_id", sa.String(length=100), nullable=False),
        sa.Column("drillhole_id", sa.String(length=36), sa.ForeignKey("drillholes.id"), nullable=False),
        sa.Column("from_depth", sa.Float(), nullable=False),
        sa.Column("to_depth", sa.Float(), nullable=False),
        sa.Column("sample_type", sa.String(length=50), nullable=True),
        sa.Column("lithology", sa.String(length=100), nullable=True),
        sa.Column("assay_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_samples_sample_id", "samples", ["sample_id"], unique=False)

    op.create_table(
        "qaqc_records",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("sample_id", sa.String(length=36), sa.ForeignKey("samples.id"), nullable=True),
        sa.Column("qc_type", sa.String(length=50), nullable=False),
        sa.Column("expected_value", sa.Float(), nullable=True),
        sa.Column("actual_value", sa.Float(), nullable=True),
        sa.Column("tolerance", sa.Float(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("report_type", sa.String(length=50), nullable=False),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.Column("created_by", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=True),
        sa.Column("entity_id", sa.String(length=36), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("reports")
    op.drop_table("qaqc_records")
    op.drop_index("ix_samples_sample_id", table_name="samples")
    op.drop_table("samples")
    op.drop_index("ix_drillholes_hole_id", table_name="drillholes")
    op.drop_table("drillholes")
    op.drop_index("ix_projects_name", table_name="projects")
    op.drop_table("projects")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
