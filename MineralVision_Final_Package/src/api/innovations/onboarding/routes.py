"""HTTP layer for stakeholder onboarding: orgs, memberships, invitations,
and a real password-reset token flow.

Auth model: every endpoint requires a valid platform JWT except the
token-bearing public flows (invitation validate/accept, password-reset
request/confirm) — those are authorized by possession of the secret token.
"""

import hashlib
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .db import get_db
from .email import EmailService, get_email_service
from .models import (
    InvitationModel,
    MembershipModel,
    OrganizationModel,
    PasswordResetTokenModel,
)

# Dual-context import of the platform database + auth helpers.
# CRITICAL: auth_middleware holds the JWT signing secret at module level, so we
# must bind to WHICHEVER module instance is already loaded (``api.*`` vs
# ``src.api.*``) — importing the other context would create a second instance
# with a different ephemeral secret and every token would fail validation.
# Dual-context import of the platform database + auth helpers.
# CRITICAL: auth_middleware holds the JWT signing secret at module level, so we
# must bind to the SAME module instance (``api.*`` vs ``src.api.*``) as the
# consuming application — importing the other context would create a second
# instance with a different ephemeral secret and tokens would fail validation.
# We select the context from our own package name (i.e. how the app imported
# this innovation), NOT from import order.
if __package__ and __package__.startswith("src."):  # src.api.innovations.onboarding
    from src.api.auth_middleware import TokenPayload, hash_password, require_auth
    from src.api.database import UserModel
    from src.api.database import get_db as platform_get_db
else:  # api.innovations.onboarding (MineralVision_Final_Package/src on sys.path)
    from api.auth_middleware import TokenPayload, hash_password, require_auth
    from api.database import UserModel
    from api.database import get_db as platform_get_db

router = APIRouter()

# Roles that may be granted through invitations / memberships.
ALLOWED_INVITE_ROLES = {
    "viewer",
    "geologist",
    "resource_geologist",
    "field_technician",
    "investor",
    "regulator",
    "custodian",
    "org_admin",
}

INVITATION_TTL_HOURS = 72
PASSWORD_RESET_TTL_HOURS = 1

# Membership roles allowed to invite others into an org.
_ORG_ADMIN_ROLES = {"org_admin", "admin"}


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _public_base_url() -> str:
    return os.getenv("MV_PUBLIC_BASE_URL", "http://localhost:5173").rstrip("/")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or secrets.token_hex(4)


def _maybe_return_token(token: str, delivery_mode: str) -> Optional[str]:
    """Dev convenience: only ever expose a raw token for console delivery AND
    when explicitly opted in via env. Default off."""
    if delivery_mode == "console" and os.getenv("MV_ONBOARDING_RETURN_TOKEN", "").lower() == "true":
        return token
    return None


def _get_membership(db: Session, user_id: str, org_id: int) -> Optional[MembershipModel]:
    return (
        db.query(MembershipModel)
        .filter(
            MembershipModel.user_id == user_id,
            MembershipModel.org_id == org_id,
            MembershipModel.status == "active",
        )
        .first()
    )


def _org_to_dict(org: OrganizationModel) -> dict:
    return {
        "id": org.id,
        "name": org.name,
        "slug": org.slug,
        "status": org.status,
        "created_at": org.created_at.isoformat() if org.created_at else None,
    }


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class OrgCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: Optional[str] = Field(None, max_length=128)


class InvitationCreateRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    role: str


class InvitationAcceptRequest(BaseModel):
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class PasswordResetRequestBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)


class PasswordResetConfirmBody(BaseModel):
    token: str = Field(..., min_length=10)
    new_password: str = Field(..., min_length=8)


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------

@router.post("/orgs", status_code=201)
def create_org(
    req: OrgCreateRequest,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """Create an organization; the caller becomes its org_admin."""
    slug = req.slug or _slugify(req.name)
    if db.query(OrganizationModel).filter(OrganizationModel.slug == slug).first():
        raise HTTPException(status_code=409, detail=f"Organization slug '{slug}' already exists")

    org = OrganizationModel(name=req.name, slug=slug, status="active")
    db.add(org)
    db.flush()

    membership = MembershipModel(
        user_id=user.user_id, org_id=org.id, role="org_admin", status="active"
    )
    db.add(membership)
    db.commit()
    db.refresh(org)
    return _org_to_dict(org)


@router.get("/orgs-mine")
def list_my_orgs(
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """List organizations the current user belongs to."""
    memberships = (
        db.query(MembershipModel)
        .filter(MembershipModel.user_id == user.user_id, MembershipModel.status == "active")
        .all()
    )
    orgs = []
    for m in memberships:
        org = db.query(OrganizationModel).filter(OrganizationModel.id == m.org_id).first()
        if org:
            orgs.append({**_org_to_dict(org), "my_role": m.role})
    return {"organizations": orgs}


@router.get("/orgs/{org_id}")
def get_org(
    org_id: int,
    db: Session = Depends(get_db),
    user: TokenPayload = Depends(require_auth),
):
    """Org detail incl. members — org members only."""
    org = db.query(OrganizationModel).filter(OrganizationModel.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if user.role != "admin" and not _get_membership(db, user.user_id, org_id):
        raise HTTPException(status_code=403, detail="Not a member of this organization")

    members = (
        db.query(MembershipModel)
        .filter(MembershipModel.org_id == org_id, MembershipModel.status == "active")
        .all()
    )
    return {
        **_org_to_dict(org),
        "members": [
            {"user_id": m.user_id, "role": m.role, "status": m.status,
             "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in members
        ],
    }


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------

@router.post("/orgs/{org_id}/invitations", status_code=201)
def create_invitation(
    org_id: int,
    req: InvitationCreateRequest,
    db: Session = Depends(get_db),
    email_service: EmailService = Depends(get_email_service),
    user: TokenPayload = Depends(require_auth),
):
    """Invite a stakeholder into an org (org admins only)."""
    if req.role not in ALLOWED_INVITE_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown role '{req.role}'. Allowed: {sorted(ALLOWED_INVITE_ROLES)}",
        )

    org = db.query(OrganizationModel).filter(OrganizationModel.id == org_id).first()
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    inviter = _get_membership(db, user.user_id, org_id)
    is_privileged = user.role == "admin" or (inviter and inviter.role in _ORG_ADMIN_ROLES)
    if not is_privileged:
        raise HTTPException(status_code=403, detail="Only org admins can invite members")

    token = secrets.token_urlsafe(32)
    invitation = InvitationModel(
        org_id=org_id,
        email=req.email,
        role=req.role,
        token_hash=_hash_token(token),
        expires_at=_utcnow() + timedelta(hours=INVITATION_TTL_HOURS),
        invited_by=user.user_id,
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)

    link = f"{_public_base_url()}/accept-invite/{token}"
    delivery = email_service.send(
        to=req.email,
        subject=f"You're invited to join {org.name} on MineralVision",
        body=(
            f"You have been invited to join '{org.name}' on MineralVision "
            f"with the role '{req.role}'.\n\n"
            f"Accept the invitation (valid {INVITATION_TTL_HOURS}h):\n{link}\n"
        ),
    )

    response = {
        "invitation_id": invitation.id,
        "org_id": org_id,
        "email": req.email,
        "role": req.role,
        "expires_at": invitation.expires_at.isoformat(),
        "email_delivery": delivery,
    }
    exposed = _maybe_return_token(token, delivery)
    if exposed is not None:
        response["token"] = exposed
    return response


def _load_valid_invitation(db: Session, token: str) -> InvitationModel:
    invitation = (
        db.query(InvitationModel)
        .filter(InvitationModel.token_hash == _hash_token(token))
        .first()
    )
    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")
    if invitation.accepted_at is not None:
        raise HTTPException(status_code=410, detail="Invitation has already been used")
    if invitation.expires_at < _utcnow():
        raise HTTPException(status_code=410, detail="Invitation has expired")
    return invitation


@router.get("/invitations/{token}")
def validate_invitation(token: str, db: Session = Depends(get_db)):
    """Public: validate an invitation token (410 if expired/used)."""
    invitation = _load_valid_invitation(db, token)
    org = db.query(OrganizationModel).filter(OrganizationModel.id == invitation.org_id).first()
    return {
        "email": invitation.email,
        "role": invitation.role,
        "org": _org_to_dict(org) if org else None,
        "expires_at": invitation.expires_at.isoformat(),
    }


@router.post("/invitations/{token}/accept", status_code=201)
def accept_invitation(
    token: str,
    req: InvitationAcceptRequest,
    db: Session = Depends(get_db),
    platform_db: Session = Depends(platform_get_db),
):
    """Public: accept an invitation — creates the real platform user (bcrypt)
    and an org membership. The client then logs in via /auth/login."""
    invitation = _load_valid_invitation(db, token)

    existing = platform_db.query(UserModel).filter(UserModel.email == invitation.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="A user with this email already exists")

    base_username = invitation.email.split("@")[0] or "user"
    username = base_username
    while platform_db.query(UserModel).filter(UserModel.username == username).first():
        username = f"{base_username}-{secrets.token_hex(2)}"

    first_name, last_name = "", ""
    if req.full_name:
        parts = req.full_name.split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""

    new_user = UserModel(
        id=str(uuid.uuid4()),
        username=username,
        email=invitation.email,
        password_hash=hash_password(req.password),
        first_name=first_name,
        last_name=last_name,
        role=invitation.role,
        is_active=True,
    )
    platform_db.add(new_user)
    platform_db.commit()
    platform_db.refresh(new_user)

    invitation.accepted_at = _utcnow()
    membership = _get_membership(db, new_user.id, invitation.org_id)
    if not membership:
        membership = MembershipModel(
            user_id=new_user.id, org_id=invitation.org_id,
            role=invitation.role, status="active",
        )
        db.add(membership)
    db.commit()

    org = db.query(OrganizationModel).filter(OrganizationModel.id == invitation.org_id).first()
    return {
        "success": True,
        "user_id": new_user.id,
        "email": new_user.email,
        "role": invitation.role,
        "org": _org_to_dict(org) if org else None,
        "message": "Account created. Log in via /auth/login.",
    }


# ---------------------------------------------------------------------------
# Password reset (real token flow; replaces the auth.py stub semantics)
# ---------------------------------------------------------------------------

@router.post("/password-reset/request")
def request_password_reset(
    req: PasswordResetRequestBody,
    db: Session = Depends(get_db),
    platform_db: Session = Depends(platform_get_db),
    email_service: EmailService = Depends(get_email_service),
):
    """Request a password reset. Always 200 — never leaks account existence."""
    user = platform_db.query(UserModel).filter(UserModel.email == req.email).first()
    delivery = "none"
    if user:
        token = secrets.token_urlsafe(32)
        reset = PasswordResetTokenModel(
            user_id=user.id,
            email=user.email,
            token_hash=_hash_token(token),
            expires_at=_utcnow() + timedelta(hours=PASSWORD_RESET_TTL_HOURS),
        )
        db.add(reset)
        db.commit()
        link = f"{_public_base_url()}/reset-password/{token}"
        delivery = email_service.send(
            to=user.email,
            subject="MineralVision password reset",
            body=(
                f"A password reset was requested for your MineralVision account.\n\n"
                f"Reset your password (valid {PASSWORD_RESET_TTL_HOURS}h):\n{link}\n\n"
                "If you did not request this, ignore this message."
            ),
        )
        exposed = _maybe_return_token(token, delivery)
        response = {"status": "ok", "email_delivery": delivery}
        if exposed is not None:
            response["token"] = exposed
        return response
    return {"status": "ok", "email_delivery": delivery}


@router.post("/password-reset/confirm")
def confirm_password_reset(
    req: PasswordResetConfirmBody,
    db: Session = Depends(get_db),
    platform_db: Session = Depends(platform_get_db),
):
    """Confirm a password reset with the emailed token."""
    reset = (
        db.query(PasswordResetTokenModel)
        .filter(PasswordResetTokenModel.token_hash == _hash_token(req.token))
        .first()
    )
    if not reset:
        raise HTTPException(status_code=404, detail="Reset token not found")
    if reset.used_at is not None:
        raise HTTPException(status_code=410, detail="Reset token has already been used")
    if reset.expires_at < _utcnow():
        raise HTTPException(status_code=410, detail="Reset token has expired")

    user = platform_db.query(UserModel).filter(UserModel.id == reset.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = hash_password(req.new_password)
    platform_db.commit()
    reset.used_at = _utcnow()
    db.commit()
    return {"success": True, "message": "Password updated. Log in via /auth/login."}
