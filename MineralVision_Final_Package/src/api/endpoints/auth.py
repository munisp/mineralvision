"""
API endpoints for Authentication.

This module provides endpoints for user authentication including
login, logout, token refresh, and password management.
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import uuid
import hashlib
import secrets

router = APIRouter()

# Runtime storage for API operations
tokens_db: Dict[str, dict] = {}
sessions_db: Dict[str, dict] = {}

# Import users_db from users module (shared storage)
from .users import users_db, hash_password


class LoginRequest(BaseModel):
    """Schema for login request."""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Schema for login response."""
    accessToken: str
    refreshToken: str
    tokenType: str = "Bearer"
    expiresIn: int
    user: Dict[str, Any]


class RefreshRequest(BaseModel):
    """Schema for token refresh request."""
    refreshToken: str


class PasswordChangeRequest(BaseModel):
    """Schema for password change request."""
    currentPassword: str
    newPassword: str = Field(..., min_length=8)


class PasswordResetRequest(BaseModel):
    """Schema for password reset request."""
    email: str


def generate_token() -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(32)


def create_tokens(user_id: str) -> tuple:
    """Create access and refresh tokens for a user."""
    access_token = generate_token()
    refresh_token = generate_token()
    
    # Store tokens
    tokens_db[access_token] = {
        "user_id": user_id,
        "type": "access",
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
    }
    
    tokens_db[refresh_token] = {
        "user_id": user_id,
        "type": "refresh",
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(days=7)).isoformat()
    }
    
    return access_token, refresh_token


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return tokens."""
    # Find user by email
    user = None
    for u in users_db.values():
        if u.get("email") == request.email:
            user = u
            break
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify password
    if user.get("password") != hash_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Check if user is active
    if not user.get("isActive", True):
        raise HTTPException(status_code=401, detail="Account is disabled")
    
    # Create tokens
    access_token, refresh_token = create_tokens(user["id"])
    
    # Update last login
    user["lastLogin"] = datetime.utcnow().isoformat()
    users_db[user["id"]] = user
    
    # Create session
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "user_id": user["id"],
        "access_token": access_token,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return LoginResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        tokenType="Bearer",
        expiresIn=3600,
        user={
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "roles": user["roles"]
        }
    )


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """Logout user and invalidate token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract token from header
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = parts[1]
    
    # Remove token
    if token in tokens_db:
        del tokens_db[token]
    
    # Remove associated sessions
    sessions_to_remove = [
        sid for sid, session in sessions_db.items()
        if session.get("access_token") == token
    ]
    for sid in sessions_to_remove:
        del sessions_db[sid]
    
    return {"status": "logged_out"}


@router.post("/refresh", response_model=LoginResponse)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token."""
    token_data = tokens_db.get(request.refreshToken)
    
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    if token_data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    # Check expiration
    expires_at = datetime.fromisoformat(token_data["expires_at"])
    if datetime.utcnow() > expires_at:
        del tokens_db[request.refreshToken]
        raise HTTPException(status_code=401, detail="Refresh token expired")
    
    user_id = token_data["user_id"]
    user = users_db.get(user_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Create new tokens
    access_token, refresh_token = create_tokens(user_id)
    
    # Invalidate old refresh token
    del tokens_db[request.refreshToken]
    
    return LoginResponse(
        accessToken=access_token,
        refreshToken=refresh_token,
        tokenType="Bearer",
        expiresIn=3600,
        user={
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "roles": user["roles"]
        }
    )


@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    authorization: Optional[str] = Header(None)
):
    """Change user password."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract and validate token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = parts[1]
    token_data = tokens_db.get(token)
    
    if not token_data or token_data.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")
    
    user_id = token_data["user_id"]
    user = users_db.get(user_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Verify current password
    if user.get("password") != hash_password(request.currentPassword):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Update password
    user["password"] = hash_password(request.newPassword)
    user["updatedAt"] = datetime.utcnow().isoformat()
    users_db[user_id] = user
    
    return {"status": "password_changed"}


@router.post("/reset-password")
async def reset_password(request: PasswordResetRequest):
    """Request password reset."""
    # Find user by email
    user = None
    for u in users_db.values():
        if u.get("email") == request.email:
            user = u
            break
    
    # Always return success to prevent email enumeration
    return {
        "status": "reset_email_sent",
        "message": "If the email exists, a password reset link has been sent"
    }


@router.get("/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current authenticated user."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract and validate token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = parts[1]
    token_data = tokens_db.get(token)
    
    if not token_data or token_data.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid access token")
    
    # Check expiration
    expires_at = datetime.fromisoformat(token_data["expires_at"])
    if datetime.utcnow() > expires_at:
        del tokens_db[token]
        raise HTTPException(status_code=401, detail="Access token expired")
    
    user_id = token_data["user_id"]
    user = users_db.get(user_id)
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "roles": user["roles"],
        "isActive": user["isActive"],
        "lastLogin": user.get("lastLogin")
    }
