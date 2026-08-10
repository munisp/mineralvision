"""
API endpoints for User management.

Database-backed CRUD operations for users and role management,
integrating with the RBAC module. Passwords are hashed with bcrypt
(see src.api.auth_middleware).
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from ..database import get_db, UserModel
from ..auth_middleware import hash_password, require_role, TokenPayload

# Import RBAC module
from ..auth.rbac import (
    RBACManager,
    User,
    Role,
    Permission,
    create_rbac_manager
)

router = APIRouter()

# Initialize RBAC manager
rbac_manager = create_rbac_manager()

# Roles valid for assignment: the RoleManager catalogue plus the legacy
# self-registered "user" role. Unknown roles are rejected with 422.
VALID_ROLES = set(rbac_manager.system.role_manager.roles.keys()) | {"user"}


def _validate_roles(roles: list) -> None:
    unknown = [r for r in roles if r not in VALID_ROLES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown role(s): {unknown}. Valid roles: {sorted(VALID_ROLES)}"
        )


class UserCreate(BaseModel):
    """Schema for creating a user."""
    email: str
    name: str
    password: str = Field(..., min_length=8)
    roles: List[str] = Field(default_factory=lambda: ["viewer"])
    metadata: Optional[Dict[str, Any]] = None


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[str] = None
    name: Optional[str] = None
    password: Optional[str] = None
    roles: Optional[List[str]] = None
    isActive: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """Schema for user response."""
    id: str
    email: str
    name: str
    roles: List[str]
    isActive: bool
    createdAt: str
    updatedAt: str
    lastLogin: Optional[str] = None


class RoleUpdateRequest(BaseModel):
    """Schema for role update request."""
    roles: List[str]


def _to_response(user: UserModel) -> UserResponse:
    name = f"{user.first_name or ''} {user.last_name or ''}".strip() or user.username
    return UserResponse(
        id=user.id,
        email=user.email,
        name=name,
        roles=[user.role],
        isActive=user.is_active,
        createdAt=user.created_at.isoformat(),
        updatedAt=user.updated_at.isoformat(),
        lastLogin=None
    )


@router.get("", response_model=List[UserResponse])
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    isActive: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """List all users with optional filtering (admin only)."""
    query = db.query(UserModel)
    if role:
        query = query.filter(UserModel.role == role)
    if isActive is not None:
        query = query.filter(UserModel.is_active == isActive)
    users = query.offset(offset).limit(limit).all()
    return [_to_response(u) for u in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: Session = Depends(get_db)):
    """Get a specific user by ID."""
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return _to_response(user)


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(
    user: UserCreate,
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """Create a new user (admin only)."""
    if db.query(UserModel).filter(UserModel.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    _validate_roles(user.roles)
    role = user.roles[0] if user.roles else "viewer"
    name_parts = user.name.split(" ", 1)

    db_user = UserModel(
        id=str(uuid.uuid4()),
        username=user.email,
        email=user.email,
        password_hash=hash_password(user.password),
        first_name=name_parts[0] if name_parts else "",
        last_name=name_parts[1] if len(name_parts) > 1 else "",
        role=role,
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Register with RBAC manager
    try:
        rbac_manager.create_user(
            user_id=db_user.id,
            email=db_user.email,
            name=user.name,
            roles=user.roles
        )
    except Exception:
        pass  # RBAC registration is best-effort

    return _to_response(db_user)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user: UserUpdate,
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """Update an existing user (admin only)."""
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    update_data = user.model_dump(exclude_unset=True)

    if update_data.get("email"):
        db_user.email = update_data["email"]
        db_user.username = update_data["email"]
    if update_data.get("name"):
        parts = update_data["name"].split(" ", 1)
        db_user.first_name = parts[0]
        db_user.last_name = parts[1] if len(parts) > 1 else ""
    if update_data.get("password"):
        db_user.password_hash = hash_password(update_data["password"])
    if update_data.get("roles"):
        _validate_roles(update_data["roles"])
        db_user.role = update_data["roles"][0]
    if update_data.get("isActive") is not None:
        db_user.is_active = update_data["isActive"]

    db_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_user)

    return _to_response(db_user)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: str,
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """Delete a user (admin only)."""
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    db.delete(db_user)
    db.commit()
    return None


@router.put("/{user_id}/roles", response_model=UserResponse)
async def update_user_roles(
    user_id: str,
    request: RoleUpdateRequest,
    db: Session = Depends(get_db),
    admin: TokenPayload = Depends(require_role(["admin"]))
):
    """Update user roles (admin only)."""
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    _validate_roles(request.roles)
    if request.roles:
        db_user.role = request.roles[0]
    db_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_user)

    # Update RBAC manager
    try:
        rbac_manager.update_user_roles(user_id, request.roles)
    except Exception:
        pass

    return _to_response(db_user)


@router.get("/{user_id}/permissions")
async def get_user_permissions(user_id: str, db: Session = Depends(get_db)):
    """Get all permissions for a user."""
    db_user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    try:
        permissions = rbac_manager.get_user_permissions(user_id)
        if permissions:
            return {
                "userId": user_id,
                "roles": [db_user.role],
                "permissions": permissions
            }
    except Exception:
        pass

    # Derive permissions from the user's role
    default_permissions = {
        "admin": ["read", "write", "delete", "admin"],
        "editor": ["read", "write"],
        "viewer": ["read"],
        "user": ["read"]
    }
    return {
        "userId": user_id,
        "roles": [db_user.role],
        "permissions": default_permissions.get(db_user.role, ["read"])
    }
