"""
Role-Based Access Control (RBAC) Module for MineralVision Platform.

Comprehensive RBAC including:
1. User management
2. Role definitions and permissions
3. Resource-level access control
4. Audit trails and logging
5. Session management
6. API key management
7. Multi-tenant support
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
import hashlib
import secrets
import uuid
import json


class Permission(Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    APPROVE = "approve"
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"
    AUDIT = "audit"


class ResourceType(Enum):
    """Resource types in the system."""
    PROJECT = "project"
    DRILLHOLE = "drillhole"
    SAMPLE = "sample"
    ASSAY = "assay"
    BLOCK_MODEL = "block_model"
    RESOURCE_ESTIMATE = "resource_estimate"
    REPORT = "report"
    SURFACE = "surface"
    CROSS_SECTION = "cross_section"
    VARIOGRAM = "variogram"
    QAQC = "qaqc"
    USER = "user"
    ROLE = "role"
    AUDIT_LOG = "audit_log"
    API_KEY = "api_key"
    TENANT = "tenant"


class AuditAction(Enum):
    """Audit action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    FAILED_LOGIN = "failed_login"
    PASSWORD_CHANGE = "password_change"
    PERMISSION_CHANGE = "permission_change"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REJECT = "reject"
    SHARE = "share"
    API_CALL = "api_call"


class UserStatus(Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"
    PENDING = "pending"
    SUSPENDED = "suspended"


@dataclass
class Permission_:
    """Permission definition."""
    name: str
    resource_type: ResourceType
    actions: Set[Permission]
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def allows(self, action: Permission, resource: Optional[Dict] = None) -> bool:
        """Check if permission allows action."""
        if action not in self.actions:
            return False
        
        if not self.conditions or not resource:
            return True
        
        for key, value in self.conditions.items():
            if key in resource and resource[key] != value:
                return False
        
        return True


@dataclass
class Role:
    """Role definition."""
    id: str
    name: str
    description: str
    permissions: List[Permission_]
    is_system_role: bool = False
    tenant_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    
    def has_permission(self, resource_type: ResourceType, action: Permission,
                      resource: Optional[Dict] = None) -> bool:
        """Check if role has permission."""
        for perm in self.permissions:
            if perm.resource_type == resource_type and perm.allows(action, resource):
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "permissions": [
                {
                    "name": p.name,
                    "resource_type": p.resource_type.value,
                    "actions": [a.value for a in p.actions]
                }
                for p in self.permissions
            ],
            "is_system_role": self.is_system_role,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class User:
    """User account."""
    id: str
    username: str
    email: str
    password_hash: str
    first_name: str = ""
    last_name: str = ""
    roles: List[str] = field(default_factory=list)
    status: UserStatus = UserStatus.ACTIVE
    tenant_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_changed_at: Optional[datetime] = None
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def is_locked(self) -> bool:
        if self.status == UserStatus.LOCKED:
            return True
        if self.locked_until and datetime.now() < self.locked_until:
            return True
        return False
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "roles": self.roles,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "mfa_enabled": self.mfa_enabled
        }
        
        if include_sensitive:
            data["failed_login_attempts"] = self.failed_login_attempts
            data["locked_until"] = self.locked_until.isoformat() if self.locked_until else None
        
        return data


@dataclass
class Session:
    """User session."""
    id: str
    user_id: str
    token: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    ip_address: str = ""
    user_agent: str = ""
    is_active: bool = True
    last_activity: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        return self.is_active and not self.is_expired


@dataclass
class APIKey:
    """API key for programmatic access."""
    id: str
    user_id: str
    name: str
    key_hash: str
    prefix: str
    permissions: List[Permission_]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 1000
    allowed_ips: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        return self.is_active and not self.is_expired


@dataclass
class AuditLogEntry:
    """Audit log entry."""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    action: AuditAction
    resource_type: ResourceType
    resource_id: Optional[str]
    tenant_id: Optional[str]
    ip_address: str = ""
    user_agent: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "username": self.username,
            "action": self.action.value,
            "resource_type": self.resource_type.value,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "ip_address": self.ip_address,
            "details": self.details,
            "status": self.status,
            "error_message": self.error_message
        }


@dataclass
class Tenant:
    """Multi-tenant organization."""
    id: str
    name: str
    slug: str
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    settings: Dict[str, Any] = field(default_factory=dict)
    quota: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "settings": self.settings,
            "quota": self.quota
        }


class PasswordHasher:
    """Password hashing utility."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Hash a password."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_input = f"{salt}:{password}".encode('utf-8')
        password_hash = hashlib.sha256(hash_input).hexdigest()
        
        return f"{salt}:{password_hash}"
    
    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify a password against stored hash."""
        try:
            salt, _ = stored_hash.split(':')
            computed_hash = PasswordHasher.hash_password(password, salt)
            return secrets.compare_digest(computed_hash, stored_hash)
        except ValueError:
            return False


class TokenGenerator:
    """Token generation utility."""
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate a session token."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_api_key() -> Tuple[str, str, str]:
        """Generate an API key. Returns (full_key, prefix, hash)."""
        prefix = secrets.token_hex(4)
        secret = secrets.token_hex(24)
        full_key = f"mv_{prefix}_{secret}"
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()
        
        return full_key, prefix, key_hash


class RoleManager:
    """Manage roles and permissions."""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self._create_default_roles()
    
    def _create_default_roles(self):
        """Create default system roles."""
        admin_permissions = [
            Permission_(
                name="full_access",
                resource_type=rt,
                actions={Permission.READ, Permission.WRITE, Permission.DELETE, 
                        Permission.ADMIN, Permission.EXECUTE, Permission.APPROVE,
                        Permission.EXPORT, Permission.IMPORT, Permission.SHARE, Permission.AUDIT}
            )
            for rt in ResourceType
        ]
        
        self.roles["admin"] = Role(
            id="admin",
            name="Administrator",
            description="Full system access",
            permissions=admin_permissions,
            is_system_role=True
        )
        
        geologist_permissions = [
            Permission_(
                name="project_access",
                resource_type=ResourceType.PROJECT,
                actions={Permission.READ, Permission.WRITE}
            ),
            Permission_(
                name="drillhole_access",
                resource_type=ResourceType.DRILLHOLE,
                actions={Permission.READ, Permission.WRITE, Permission.DELETE}
            ),
            Permission_(
                name="sample_access",
                resource_type=ResourceType.SAMPLE,
                actions={Permission.READ, Permission.WRITE, Permission.DELETE}
            ),
            Permission_(
                name="assay_access",
                resource_type=ResourceType.ASSAY,
                actions={Permission.READ, Permission.WRITE}
            ),
            Permission_(
                name="qaqc_access",
                resource_type=ResourceType.QAQC,
                actions={Permission.READ, Permission.WRITE, Permission.EXECUTE}
            ),
            Permission_(
                name="cross_section_access",
                resource_type=ResourceType.CROSS_SECTION,
                actions={Permission.READ, Permission.WRITE, Permission.EXPORT}
            ),
            Permission_(
                name="surface_access",
                resource_type=ResourceType.SURFACE,
                actions={Permission.READ, Permission.WRITE}
            )
        ]
        
        self.roles["geologist"] = Role(
            id="geologist",
            name="Geologist",
            description="Geological data management",
            permissions=geologist_permissions,
            is_system_role=True
        )
        
        resource_geologist_permissions = geologist_permissions + [
            Permission_(
                name="block_model_access",
                resource_type=ResourceType.BLOCK_MODEL,
                actions={Permission.READ, Permission.WRITE, Permission.EXECUTE}
            ),
            Permission_(
                name="variogram_access",
                resource_type=ResourceType.VARIOGRAM,
                actions={Permission.READ, Permission.WRITE, Permission.EXECUTE}
            ),
            Permission_(
                name="resource_estimate_access",
                resource_type=ResourceType.RESOURCE_ESTIMATE,
                actions={Permission.READ, Permission.WRITE, Permission.EXECUTE}
            ),
            Permission_(
                name="report_access",
                resource_type=ResourceType.REPORT,
                actions={Permission.READ, Permission.WRITE, Permission.EXPORT}
            )
        ]
        
        self.roles["resource_geologist"] = Role(
            id="resource_geologist",
            name="Resource Geologist",
            description="Resource estimation and modeling",
            permissions=resource_geologist_permissions,
            is_system_role=True
        )
        
        viewer_permissions = [
            Permission_(
                name="view_access",
                resource_type=rt,
                actions={Permission.READ}
            )
            for rt in [ResourceType.PROJECT, ResourceType.DRILLHOLE, ResourceType.SAMPLE,
                      ResourceType.BLOCK_MODEL, ResourceType.RESOURCE_ESTIMATE, 
                      ResourceType.REPORT, ResourceType.CROSS_SECTION]
        ]
        
        self.roles["viewer"] = Role(
            id="viewer",
            name="Viewer",
            description="Read-only access",
            permissions=viewer_permissions,
            is_system_role=True
        )
        
        qp_permissions = resource_geologist_permissions + [
            Permission_(
                name="approve_access",
                resource_type=ResourceType.RESOURCE_ESTIMATE,
                actions={Permission.APPROVE}
            ),
            Permission_(
                name="report_approve",
                resource_type=ResourceType.REPORT,
                actions={Permission.APPROVE, Permission.EXPORT}
            )
        ]
        
        self.roles["qualified_person"] = Role(
            id="qualified_person",
            name="Qualified Person",
            description="QP/CP with approval authority",
            permissions=qp_permissions,
            is_system_role=True
        )
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self.roles.get(role_id)
    
    def create_role(self, name: str, description: str,
                   permissions: List[Permission_],
                   tenant_id: Optional[str] = None,
                   created_by: Optional[str] = None) -> Role:
        """Create a new role."""
        role_id = str(uuid.uuid4())
        
        role = Role(
            id=role_id,
            name=name,
            description=description,
            permissions=permissions,
            is_system_role=False,
            tenant_id=tenant_id,
            created_by=created_by
        )
        
        self.roles[role_id] = role
        return role
    
    def update_role(self, role_id: str, **kwargs) -> Optional[Role]:
        """Update a role."""
        role = self.roles.get(role_id)
        if not role or role.is_system_role:
            return None
        
        for key, value in kwargs.items():
            if hasattr(role, key):
                setattr(role, key, value)
        
        role.updated_at = datetime.now()
        return role
    
    def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self.roles.get(role_id)
        if not role or role.is_system_role:
            return False
        
        del self.roles[role_id]
        return True
    
    def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """List all roles."""
        roles = list(self.roles.values())
        
        if tenant_id:
            roles = [r for r in roles if r.tenant_id is None or r.tenant_id == tenant_id]
        
        return roles


class UserManager:
    """Manage users."""
    
    def __init__(self, role_manager: RoleManager):
        self.users: Dict[str, User] = {}
        self.role_manager = role_manager
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    def create_user(self, username: str, email: str, password: str,
                   first_name: str = "", last_name: str = "",
                   roles: List[str] = None,
                   tenant_id: Optional[str] = None) -> User:
        """Create a new user."""
        if self.get_user_by_username(username):
            raise ValueError(f"Username '{username}' already exists")
        
        if self.get_user_by_email(email):
            raise ValueError(f"Email '{email}' already exists")
        
        user_id = str(uuid.uuid4())
        password_hash = PasswordHasher.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            roles=roles or ["viewer"],
            tenant_id=tenant_id,
            password_changed_at=datetime.now()
        )
        
        self.users[user_id] = user
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        for user in self.users.values():
            if user.email == email:
                return user
        return None
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update a user."""
        user = self.users.get(user_id)
        if not user:
            return None
        
        if 'password' in kwargs:
            kwargs['password_hash'] = PasswordHasher.hash_password(kwargs.pop('password'))
            kwargs['password_changed_at'] = datetime.now()
        
        for key, value in kwargs.items():
            if hasattr(user, key) and key not in ['id', 'created_at']:
                setattr(user, key, value)
        
        user.updated_at = datetime.now()
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        user = self.get_user_by_username(username)
        
        if not user:
            return None
        
        if user.is_locked:
            return None
        
        if not PasswordHasher.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.status = UserStatus.LOCKED
                user.locked_until = datetime.now() + self.lockout_duration
            
            return None
        
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        if user.status == UserStatus.LOCKED:
            user.status = UserStatus.ACTIVE
        
        return user
    
    def change_password(self, user_id: str, old_password: str, 
                       new_password: str) -> bool:
        """Change user password."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        if not PasswordHasher.verify_password(old_password, user.password_hash):
            return False
        
        user.password_hash = PasswordHasher.hash_password(new_password)
        user.password_changed_at = datetime.now()
        user.updated_at = datetime.now()
        
        return True
    
    def reset_password(self, user_id: str, new_password: str) -> bool:
        """Reset user password (admin function)."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.password_hash = PasswordHasher.hash_password(new_password)
        user.password_changed_at = datetime.now()
        user.updated_at = datetime.now()
        user.failed_login_attempts = 0
        user.locked_until = None
        
        if user.status == UserStatus.LOCKED:
            user.status = UserStatus.ACTIVE
        
        return True
    
    def list_users(self, tenant_id: Optional[str] = None) -> List[User]:
        """List all users."""
        users = list(self.users.values())
        
        if tenant_id:
            users = [u for u in users if u.tenant_id == tenant_id]
        
        return users
    
    def get_user_permissions(self, user_id: str) -> List[Permission_]:
        """Get all permissions for a user."""
        user = self.users.get(user_id)
        if not user:
            return []
        
        permissions = []
        for role_id in user.roles:
            role = self.role_manager.get_role(role_id)
            if role:
                permissions.extend(role.permissions)
        
        return permissions
    
    def check_permission(self, user_id: str, resource_type: ResourceType,
                        action: Permission, resource: Optional[Dict] = None) -> bool:
        """Check if user has permission."""
        user = self.users.get(user_id)
        if not user or user.status != UserStatus.ACTIVE:
            return False
        
        for role_id in user.roles:
            role = self.role_manager.get_role(role_id)
            if role and role.has_permission(resource_type, action, resource):
                return True
        
        return False


class SessionManager:
    """Manage user sessions."""
    
    def __init__(self, session_duration: timedelta = timedelta(hours=24)):
        self.sessions: Dict[str, Session] = {}
        self.session_duration = session_duration
    
    def create_session(self, user_id: str, ip_address: str = "",
                      user_agent: str = "") -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        token = TokenGenerator.generate_session_token()
        
        session = Session(
            id=session_id,
            user_id=user_id,
            token=token,
            expires_at=datetime.now() + self.session_duration,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def get_session_by_token(self, token: str) -> Optional[Session]:
        """Get session by token."""
        for session in self.sessions.values():
            if session.token == token and session.is_valid:
                return session
        return None
    
    def validate_session(self, token: str) -> Optional[Session]:
        """Validate a session token."""
        session = self.get_session_by_token(token)
        
        if session and session.is_valid:
            session.last_activity = datetime.now()
            return session
        
        return None
    
    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        session = self.sessions.get(session_id)
        if session:
            session.is_active = False
            return True
        return False
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user."""
        count = 0
        for session in self.sessions.values():
            if session.user_id == user_id and session.is_active:
                session.is_active = False
                count += 1
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions."""
        expired = [sid for sid, s in self.sessions.items() if s.is_expired]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)
    
    def list_user_sessions(self, user_id: str) -> List[Session]:
        """List all active sessions for a user."""
        return [s for s in self.sessions.values() 
                if s.user_id == user_id and s.is_valid]


class APIKeyManager:
    """Manage API keys."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
    
    def create_api_key(self, user_id: str, name: str,
                      permissions: List[Permission_],
                      expires_in_days: Optional[int] = None,
                      rate_limit: int = 1000,
                      allowed_ips: List[str] = None) -> Tuple[str, APIKey]:
        """Create a new API key. Returns (full_key, api_key_object)."""
        key_id = str(uuid.uuid4())
        full_key, prefix, key_hash = TokenGenerator.generate_api_key()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            permissions=permissions,
            expires_at=expires_at,
            rate_limit=rate_limit,
            allowed_ips=allowed_ips or []
        )
        
        self.api_keys[key_id] = api_key
        return full_key, api_key
    
    def validate_api_key(self, key: str, ip_address: str = "") -> Optional[APIKey]:
        """Validate an API key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash and api_key.is_valid:
                if api_key.allowed_ips and ip_address not in api_key.allowed_ips:
                    return None
                
                api_key.last_used = datetime.now()
                return api_key
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self.api_keys.get(key_id)
        if api_key:
            api_key.is_active = False
            return True
        return False
    
    def list_user_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [k for k in self.api_keys.values() if k.user_id == user_id]


class AuditLogger:
    """Audit logging system."""
    
    def __init__(self, max_entries: int = 100000):
        self.entries: List[AuditLogEntry] = []
        self.max_entries = max_entries
    
    def log(self, user_id: Optional[str], username: Optional[str],
           action: AuditAction, resource_type: ResourceType,
           resource_id: Optional[str] = None,
           tenant_id: Optional[str] = None,
           ip_address: str = "",
           user_agent: str = "",
           details: Dict[str, Any] = None,
           status: str = "success",
           error_message: Optional[str] = None) -> AuditLogEntry:
        """Log an audit entry."""
        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            username=username,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            status=status,
            error_message=error_message
        )
        
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        return entry
    
    def query(self, user_id: Optional[str] = None,
             action: Optional[AuditAction] = None,
             resource_type: Optional[ResourceType] = None,
             resource_id: Optional[str] = None,
             tenant_id: Optional[str] = None,
             start_date: Optional[datetime] = None,
             end_date: Optional[datetime] = None,
             limit: int = 100) -> List[AuditLogEntry]:
        """Query audit logs."""
        results = self.entries.copy()
        
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        
        if action:
            results = [e for e in results if e.action == action]
        
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        return results[:limit]
    
    def export_to_json(self, filepath: str, **query_params):
        """Export audit logs to JSON."""
        entries = self.query(**query_params)
        
        with open(filepath, 'w') as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)


class TenantManager:
    """Manage multi-tenant organizations."""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
    
    def create_tenant(self, name: str, slug: str,
                     settings: Dict[str, Any] = None,
                     quota: Dict[str, int] = None) -> Tenant:
        """Create a new tenant."""
        if self.get_tenant_by_slug(slug):
            raise ValueError(f"Tenant with slug '{slug}' already exists")
        
        tenant_id = str(uuid.uuid4())
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            slug=slug,
            settings=settings or {},
            quota=quota or {
                "max_users": 100,
                "max_projects": 50,
                "max_storage_gb": 100
            }
        )
        
        self.tenants[tenant_id] = tenant
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        for tenant in self.tenants.values():
            if tenant.slug == slug:
                return tenant
        return None
    
    def update_tenant(self, tenant_id: str, **kwargs) -> Optional[Tenant]:
        """Update a tenant."""
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return None
        
        for key, value in kwargs.items():
            if hasattr(tenant, key) and key not in ['id', 'created_at']:
                setattr(tenant, key, value)
        
        return tenant
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""
        if tenant_id in self.tenants:
            del self.tenants[tenant_id]
            return True
        return False
    
    def list_tenants(self) -> List[Tenant]:
        """List all tenants."""
        return list(self.tenants.values())


class RBACSystem:
    """
    Complete RBAC system integrating all components.
    """
    
    def __init__(self):
        self.role_manager = RoleManager()
        self.user_manager = UserManager(self.role_manager)
        self.session_manager = SessionManager()
        self.api_key_manager = APIKeyManager()
        self.audit_logger = AuditLogger()
        self.tenant_manager = TenantManager()
    
    def login(self, username: str, password: str,
             ip_address: str = "", user_agent: str = "") -> Optional[Dict[str, Any]]:
        """Authenticate user and create session."""
        user = self.user_manager.authenticate(username, password)
        
        if not user:
            self.audit_logger.log(
                user_id=None,
                username=username,
                action=AuditAction.FAILED_LOGIN,
                resource_type=ResourceType.USER,
                ip_address=ip_address,
                user_agent=user_agent,
                status="failure",
                error_message="Invalid credentials"
            )
            return None
        
        session = self.session_manager.create_session(
            user.id, ip_address, user_agent
        )
        
        self.audit_logger.log(
            user_id=user.id,
            username=user.username,
            action=AuditAction.LOGIN,
            resource_type=ResourceType.USER,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {
            "user": user.to_dict(),
            "session": {
                "id": session.id,
                "token": session.token,
                "expires_at": session.expires_at.isoformat()
            }
        }
    
    def logout(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        session = self.session_manager.get_session_by_token(session_token)
        
        if not session:
            return False
        
        user = self.user_manager.get_user(session.user_id)
        
        self.session_manager.invalidate_session(session.id)
        
        self.audit_logger.log(
            user_id=session.user_id,
            username=user.username if user else None,
            action=AuditAction.LOGOUT,
            resource_type=ResourceType.USER,
            ip_address=session.ip_address
        )
        
        return True
    
    def validate_request(self, token: str, resource_type: ResourceType,
                        action: Permission, resource: Optional[Dict] = None,
                        ip_address: str = "") -> Tuple[bool, Optional[User], str]:
        """Validate a request with token."""
        if token.startswith("mv_"):
            api_key = self.api_key_manager.validate_api_key(token, ip_address)
            if not api_key:
                return False, None, "Invalid API key"
            
            user = self.user_manager.get_user(api_key.user_id)
            if not user:
                return False, None, "User not found"
            
            for perm in api_key.permissions:
                if perm.resource_type == resource_type and perm.allows(action, resource):
                    return True, user, "OK"
            
            return False, user, "Permission denied"
        
        else:
            session = self.session_manager.validate_session(token)
            if not session:
                return False, None, "Invalid or expired session"
            
            user = self.user_manager.get_user(session.user_id)
            if not user:
                return False, None, "User not found"
            
            if self.user_manager.check_permission(user.id, resource_type, action, resource):
                return True, user, "OK"
            
            return False, user, "Permission denied"
    
    def create_user(self, admin_user_id: str, username: str, email: str,
                   password: str, roles: List[str] = None,
                   tenant_id: Optional[str] = None, **kwargs) -> Optional[User]:
        """Create a new user (admin function)."""
        if not self.user_manager.check_permission(admin_user_id, ResourceType.USER, Permission.ADMIN):
            return None
        
        user = self.user_manager.create_user(
            username=username,
            email=email,
            password=password,
            roles=roles,
            tenant_id=tenant_id,
            **kwargs
        )
        
        admin = self.user_manager.get_user(admin_user_id)
        
        self.audit_logger.log(
            user_id=admin_user_id,
            username=admin.username if admin else None,
            action=AuditAction.CREATE,
            resource_type=ResourceType.USER,
            resource_id=user.id,
            tenant_id=tenant_id,
            details={"created_user": username}
        )
        
        return user
    
    def get_summary(self) -> Dict[str, Any]:
        """Get system summary."""
        return {
            "users": len(self.user_manager.users),
            "roles": len(self.role_manager.roles),
            "active_sessions": len([s for s in self.session_manager.sessions.values() if s.is_valid]),
            "api_keys": len(self.api_key_manager.api_keys),
            "tenants": len(self.tenant_manager.tenants),
            "audit_entries": len(self.audit_logger.entries)
        }


def create_rbac_system() -> RBACSystem:
    """Factory function to create RBAC system."""
    return RBACSystem()
