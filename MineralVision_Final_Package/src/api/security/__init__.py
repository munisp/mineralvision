"""
MineralVision Security Module.

This module provides security and multi-tenancy features:
- Strong tenant isolation
- Secrets management with encryption
- SSO/SAML integration
- Least-privilege RBAC
- Tamper-evident audit logging
"""

from .tenancy import (
    TenantTier,
    Permission,
    AuthProvider,
    Tenant,
    Role,
    User,
    Session,
    Secret,
    SecurityEvent,
    EncryptionService,
    SecretsManager,
    RBACManager,
    TenantManager,
    SecurityAuditLog,
    SAMLProvider,
    SecurityManager,
    create_security_manager,
    create_tenant,
    create_rbac_manager,
)

__all__ = [
    # Enums
    'TenantTier',
    'Permission',
    'AuthProvider',
    
    # Data classes
    'Tenant',
    'Role',
    'User',
    'Session',
    'Secret',
    'SecurityEvent',
    
    # Services
    'EncryptionService',
    'SecretsManager',
    'RBACManager',
    'TenantManager',
    'SecurityAuditLog',
    'SAMLProvider',
    
    # Main class
    'SecurityManager',
    
    # Factory functions
    'create_security_manager',
    'create_tenant',
    'create_rbac_manager',
]
