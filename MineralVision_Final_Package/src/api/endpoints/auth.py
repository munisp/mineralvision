"""
API endpoints for Authentication.

Database-backed authentication using bcrypt password hashing and PyJWT
access tokens (see src.api.auth_middleware for the security contracts).
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid

from sqlalchemy.orm import Session

from ..database import get_db, UserModel
from ..auth_middleware import (
    AUTH_MODE, create_access_token, verify_password, hash_password,
    require_auth, blacklist_token, TokenPayload, JWT_EXPIRATION_HOURS
)

router = APIRouter()


class LoginRequest(BaseModel):
    """Schema for login request (username or email + password)."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Schema for registration request."""
    username: str = Field(..., min_length=3, max_length=100)
    email: str
    password: str = Field(..., min_length=8)
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class RefreshRequest(BaseModel):
    """Schema for token refresh request."""
    refreshToken: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    """Schema for password change request."""
    currentPassword: str
    newPassword: str = Field(..., min_length=8)


class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    email: str


def _require_local_auth() -> None:
    if AUTH_MODE == "oidc":
        raise HTTPException(
            status_code=410,
            detail="Local credentials are disabled. Authenticate and manage passwords through the configured identity provider.",
        )


def _user_response(user: UserModel) -> Dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "firstName": user.first_name or "",
        "lastName": user.last_name or "",
        "name": f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username,
        "roles": [user.role],
        "role": user.role,
        "isActive": user.is_active
    }


def _token_response(user: UserModel) -> Dict[str, Any]:
    token = create_access_token({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role
    })
    expires_in = JWT_EXPIRATION_HOURS * 3600
    return {
        "access_token": token,
        "accessToken": token,
        "refreshToken": token,
        "token_type": "bearer",
        "tokenType": "Bearer",
        "expires_in": expires_in,
        "expiresIn": expires_in,
        "session": {"token": token, "expires_in": expires_in},
        "user": _user_response(user)
    }


@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Development-only local login; production OIDC delegates to Keycloak."""
    _require_local_auth()
    user = db.query(UserModel).filter(
        (UserModel.username == request.username) | (UserModel.email == request.username)
    ).first()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is disabled")

    return _token_response(user)


@router.post("/register", status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Development-only local registration; production OIDC delegates to Keycloak."""
    _require_local_auth()
    if db.query(UserModel).filter(UserModel.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")

    if db.query(UserModel).filter(UserModel.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = UserModel(
        id=str(uuid.uuid4()),
        username=request.username,
        email=request.email,
        password_hash=hash_password(request.password),
        first_name=request.first_name,
        last_name=request.last_name,
        role="user"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "User registered successfully",
        "user_id": user.id,
        "user": _user_response(user)
    }


@router.post("/logout")
async def logout(user: TokenPayload = Depends(require_auth)):
    """Development logout; OIDC logout must use the provider end-session endpoint."""
    if AUTH_MODE == "oidc":
        raise HTTPException(status_code=410, detail="Use the identity provider end-session endpoint for OIDC logout")
    blacklist_token(user.jti)
    return {"status": "logged_out"}


@router.post("/refresh")
async def refresh_token(user: TokenPayload = Depends(require_auth),
                        db: Session = Depends(get_db)):
    """Development-only token refresh; OIDC clients use provider refresh rotation."""
    _require_local_auth()
    db_user = db.query(UserModel).filter(UserModel.id == user.user_id).first()
    if not db_user or not db_user.is_active:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    # Rotate: blacklist the presented token, issue a fresh one
    blacklist_token(user.jti)
    return _token_response(db_user)


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    user: TokenPayload = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Development-only password change; OIDC delegates this to Keycloak."""
    _require_local_auth()
    db_user = db.query(UserModel).filter(UserModel.id == user.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(request.currentPassword, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    db_user.password_hash = hash_password(request.newPassword)
    db_user.updated_at = datetime.utcnow()
    db.commit()

    # Invalidate the token used for this request
    blacklist_token(user.jti)

    return {"status": "password_changed"}


@router.post("/reset-password")
async def reset_password(request: PasswordResetRequest):
    """Development-only reset placeholder; OIDC delegates this to Keycloak."""
    _require_local_auth()
    return {
        "status": "reset_email_sent",
        "message": "If the email exists, a password reset link has been sent"
    }


@router.get("/me")
async def get_current_user_info(
    user: TokenPayload = Depends(require_auth),
    db: Session = Depends(get_db)
):
    """Return local or OIDC token identity without requiring a duplicate local user."""
    if AUTH_MODE == "oidc":
        return {
            "id": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "role": user.role,
            "mfa_verified": user.mfa_verified,
        }
    db_user = db.query(UserModel).filter(UserModel.id == user.user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    return _user_response(db_user)
