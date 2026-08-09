"""
API endpoints for User management.

This module provides CRUD operations for users and role management,
integrating with the RBAC module.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
import uuid
import hashlib

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

# Runtime storage for API operations
users_db: Dict[str, dict] = {}


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


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


@router.get("", response_model=List[UserResponse])
async def list_users(
    role: Optional[str] = Query(None, description="Filter by role"),
    isActive: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all users with optional filtering."""
    users = list(users_db.values())
    
    # Apply filters
    if role:
        users = [u for u in users if role in u.get("roles", [])]
    if isActive is not None:
        users = [u for u in users if u.get("isActive") == isActive]
    
    # Apply pagination
    users = users[offset:offset + limit]
    
    # Remove password from response
    return [UserResponse(**{k: v for k, v in u.items() if k != "password"}) for u in users]


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get a specific user by ID."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    user = users_db[user_id]
    return UserResponse(**{k: v for k, v in user.items() if k != "password"})


@router.post("", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    """Create a new user."""
    # Check for duplicate email
    for existing in users_db.values():
        if existing.get("email") == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    
    user_data = {
        "id": user_id,
        "email": user.email,
        "name": user.name,
        "password": hash_password(user.password),
        "roles": user.roles,
        "isActive": True,
        "createdAt": now,
        "updatedAt": now,
        "lastLogin": None,
        "metadata": user.metadata or {}
    }
    
    # Register with RBAC manager
    try:
        rbac_manager.create_user(
            user_id=user_id,
            email=user.email,
            name=user.name,
            roles=user.roles
        )
    except Exception:
        pass  # RBAC registration is optional
    
    users_db[user_id] = user_data
    return UserResponse(**{k: v for k, v in user_data.items() if k != "password"})


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user: UserUpdate):
    """Update an existing user."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    existing = users_db[user_id]
    update_data = user.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            if key == "password":
                existing["password"] = hash_password(value)
            else:
                existing[key] = value
    
    existing["updatedAt"] = datetime.utcnow().isoformat()
    users_db[user_id] = existing
    
    return UserResponse(**{k: v for k, v in existing.items() if k != "password"})


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: str):
    """Delete a user."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    del users_db[user_id]
    return None


@router.put("/{user_id}/roles", response_model=UserResponse)
async def update_user_roles(user_id: str, request: RoleUpdateRequest):
    """Update user roles."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    existing = users_db[user_id]
    existing["roles"] = request.roles
    existing["updatedAt"] = datetime.utcnow().isoformat()
    
    # Update RBAC manager
    try:
        rbac_manager.update_user_roles(user_id, request.roles)
    except Exception:
        pass
    
    users_db[user_id] = existing
    return UserResponse(**{k: v for k, v in existing.items() if k != "password"})


@router.get("/{user_id}/permissions")
async def get_user_permissions(user_id: str):
    """Get all permissions for a user."""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    user = users_db[user_id]
    
    try:
        permissions = rbac_manager.get_user_permissions(user_id)
        return {
            "userId": user_id,
            "roles": user["roles"],
            "permissions": permissions
        }
    except Exception:
        # Return default permissions based on roles
        default_permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "editor": ["read", "write"],
            "viewer": ["read"]
        }
        
        all_permissions = set()
        for role in user["roles"]:
            all_permissions.update(default_permissions.get(role, ["read"]))
        
        return {
            "userId": user_id,
            "roles": user["roles"],
            "permissions": list(all_permissions)
        }


@router.get("/roles/available")
async def list_available_roles():
    """List all available roles."""
    return {
        "roles": [
            {"name": "admin", "description": "Full system access"},
            {"name": "project_manager", "description": "Manage projects and team members"},
            {"name": "geologist", "description": "Access to geology and drillhole data"},
            {"name": "geostatistician", "description": "Access to geostatistics and modeling"},
            {"name": "geophysicist", "description": "Access to geophysics and inversion"},
            {"name": "editor", "description": "Read and write access"},
            {"name": "viewer", "description": "Read-only access"}
        ]
    }
