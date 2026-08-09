"""
Role-Based Access Control (RBAC) Module for MineralVision Platform.

Provides comprehensive RBAC capabilities including:
- User management
- Role definitions and permissions
- Resource-level access control
- Audit trails and logging
- Session management
- API key management
- Multi-tenant support
"""

from .rbac import (
    Permission,
    ResourceType,
    AuditAction,
    UserStatus,
    Permission_,
    Role,
    User,
    Session,
    APIKey,
    AuditLogEntry,
    Tenant,
    PasswordHasher,
    TokenGenerator,
    RoleManager,
    UserManager,
    SessionManager,
    APIKeyManager,
    AuditLogger,
    TenantManager,
    RBACSystem,
    create_rbac_system,
)

__all__ = [
    "Permission",
    "ResourceType",
    "AuditAction",
    "UserStatus",
    "Permission_",
    "Role",
    "User",
    "Session",
    "APIKey",
    "AuditLogEntry",
    "Tenant",
    "PasswordHasher",
    "TokenGenerator",
    "RoleManager",
    "UserManager",
    "SessionManager",
    "APIKeyManager",
    "AuditLogger",
    "TenantManager",
    "RBACSystem",
    "create_rbac_system",
]
