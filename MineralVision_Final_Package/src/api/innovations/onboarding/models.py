"""SQLAlchemy models for stakeholder onboarding (orgs, memberships, invitations).

Plain String/Integer/DateTime columns only — compatible with both SQLite and
PostgreSQL. Registered on the module-private Base in db.py, NOT the platform
Base (alembic-drift guard).
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from .db import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class OrganizationModel(Base):
    __tablename__ = "onboarding_organizations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(128), nullable=False, unique=True, index=True)
    status = Column(String(32), nullable=False, default="active")  # active | suspended
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class MembershipModel(Base):
    __tablename__ = "onboarding_memberships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=False, index=True)
    org_id = Column(Integer, nullable=False, index=True)
    role = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False, default="active")  # active | revoked
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class InvitationModel(Base):
    __tablename__ = "onboarding_invitations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    org_id = Column(Integer, nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    role = Column(String(64), nullable=False)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)  # sha256 hex
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    invited_by = Column(String(36), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class PasswordResetTokenModel(Base):
    __tablename__ = "onboarding_password_reset_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)  # sha256 hex
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
