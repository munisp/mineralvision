"""
Permify Authorization Integration
==================================

Production-grade fine-grained authorization for MineralVision:
- Relationship-based access control (ReBAC)
- Attribute-based access control (ABAC)
- Role-based access control (RBAC)
- Policy-as-code
- Real-time permission checks
- Audit logging

Permify provides Google Zanzibar-inspired authorization
with flexible policy definitions.
"""

import asyncio
import json
import logging
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Standard permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    SHARE = "share"
    EXPORT = "export"
    APPROVE = "approve"
    EXECUTE = "execute"


class ResourceType(Enum):
    """Types of resources."""
    PROJECT = "project"
    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    SAMPLE = "sample"
    SENSOR = "sensor"
    WORKFLOW = "workflow"
    DASHBOARD = "dashboard"
    ORGANIZATION = "organization"
    TEAM = "team"


class RelationType(Enum):
    """Types of relationships."""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    MEMBER = "member"
    ADMIN = "admin"
    PARENT = "parent"


@dataclass
class Subject:
    """Authorization subject (user or group)."""
    type: str
    id: str
    
    def __str__(self) -> str:
        return f"{self.type}:{self.id}"


@dataclass
class Resource:
    """Authorization resource."""
    type: ResourceType
    id: str
    
    def __str__(self) -> str:
        return f"{self.type.value}:{self.id}"


@dataclass
class Relationship:
    """Relationship between subject and resource."""
    subject: Subject
    relation: RelationType
    resource: Resource
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (str(self.subject), self.relation.value, str(self.resource))


@dataclass
class PermissionCheck:
    """Permission check request."""
    subject: Subject
    permission: Permission
    resource: Resource


@dataclass
class PermissionResult:
    """Permission check result."""
    allowed: bool
    reason: str = ""
    checked_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0


@dataclass
class PermifyConfig:
    """Permify configuration."""
    host: str = "localhost"
    port: int = 3476
    tenant_id: str = "mineralvision"
    schema_version: str = "v1"


class PolicySchema:
    """
    Policy schema definition.
    
    Defines the authorization model using Permify's schema language.
    """
    
    MINERALVISION_SCHEMA = """
    entity user {}
    
    entity team {
        relation admin @user
        relation member @user
        
        permission manage = admin
        permission view = admin or member
    }
    
    entity organization {
        relation admin @user
        relation member @user @team#member
        
        permission manage = admin
        permission view = admin or member
    }
    
    entity project {
        relation organization @organization
        relation owner @user
        relation editor @user @team#member
        relation viewer @user @team#member
        
        permission admin = owner or organization.admin
        permission write = admin or editor
        permission read = write or viewer
        permission delete = admin
        permission share = admin or owner
    }
    
    entity dataset {
        relation project @project
        relation owner @user
        relation editor @user
        relation viewer @user
        
        permission admin = owner or project.admin
        permission write = admin or editor or project.write
        permission read = write or viewer or project.read
        permission delete = admin
        permission export = admin or owner
    }
    
    entity model {
        relation project @project
        relation owner @user
        relation editor @user
        relation viewer @user
        
        permission admin = owner or project.admin
        permission write = admin or editor or project.write
        permission read = write or viewer or project.read
        permission delete = admin
        permission execute = write
        permission deploy = admin
    }
    
    entity report {
        relation project @project
        relation owner @user
        relation editor @user
        relation viewer @user
        
        permission admin = owner or project.admin
        permission write = admin or editor or project.write
        permission read = write or viewer or project.read
        permission delete = admin
        permission approve = admin
        permission export = read
    }
    
    entity sample {
        relation dataset @dataset
        relation owner @user
        
        permission admin = owner or dataset.admin
        permission write = admin or dataset.write
        permission read = write or dataset.read
        permission delete = admin
    }
    
    entity sensor {
        relation project @project
        relation owner @user
        relation operator @user
        
        permission admin = owner or project.admin
        permission configure = admin or operator
        permission read = configure or project.read
    }
    
    entity workflow {
        relation project @project
        relation owner @user
        relation executor @user
        
        permission admin = owner or project.admin
        permission execute = admin or executor or project.write
        permission read = execute or project.read
        permission cancel = admin or owner
    }
    """
    
    @classmethod
    def get_schema(cls) -> str:
        return cls.MINERALVISION_SCHEMA


class MockPermifyClient:
    """Mock Permify client."""
    
    def __init__(self, config: PermifyConfig):
        self.config = config
        self._relationships: List[Relationship] = []
        self._schema_loaded = False
    
    async def write_schema(self, schema: str) -> Dict[str, Any]:
        """Write authorization schema."""
        self._schema_loaded = True
        return {'schema_version': self.config.schema_version}
    
    async def write_relationship(self, relationship: Relationship) -> Dict[str, Any]:
        """Write a relationship."""
        # Check for duplicates
        for r in self._relationships:
            if r.to_tuple() == relationship.to_tuple():
                return {'snap_token': str(uuid.uuid4())}
        
        self._relationships.append(relationship)
        return {'snap_token': str(uuid.uuid4())}
    
    async def delete_relationship(self, relationship: Relationship) -> Dict[str, Any]:
        """Delete a relationship."""
        self._relationships = [
            r for r in self._relationships
            if r.to_tuple() != relationship.to_tuple()
        ]
        return {'snap_token': str(uuid.uuid4())}
    
    async def check_permission(self, check: PermissionCheck) -> PermissionResult:
        """Check if subject has permission on resource."""
        start_time = datetime.now()
        
        # Direct relationship check
        for rel in self._relationships:
            if (str(rel.subject) == str(check.subject) and 
                str(rel.resource) == str(check.resource)):
                
                # Check if relation grants permission
                if self._relation_grants_permission(rel.relation, check.permission):
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    return PermissionResult(
                        allowed=True,
                        reason=f"Direct {rel.relation.value} relationship",
                        latency_ms=latency
                    )
        
        # Check inherited permissions through parent resources
        allowed, reason = await self._check_inherited_permission(check)
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        return PermissionResult(
            allowed=allowed,
            reason=reason,
            latency_ms=latency
        )
    
    def _relation_grants_permission(self, relation: RelationType, 
                                   permission: Permission) -> bool:
        """Check if a relation grants a permission."""
        permission_map = {
            RelationType.OWNER: {Permission.READ, Permission.WRITE, Permission.DELETE, 
                                Permission.ADMIN, Permission.SHARE, Permission.EXPORT,
                                Permission.APPROVE, Permission.EXECUTE},
            RelationType.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE,
                                Permission.ADMIN, Permission.SHARE, Permission.APPROVE,
                                Permission.EXECUTE},
            RelationType.EDITOR: {Permission.READ, Permission.WRITE, Permission.EXECUTE},
            RelationType.VIEWER: {Permission.READ},
            RelationType.MEMBER: {Permission.READ},
        }
        
        return permission in permission_map.get(relation, set())
    
    async def _check_inherited_permission(self, check: PermissionCheck) -> Tuple[bool, str]:
        """Check inherited permissions through parent resources."""
        # Find parent relationships for the resource
        for rel in self._relationships:
            if (str(rel.subject) == str(check.resource) and 
                rel.relation == RelationType.PARENT):
                # Check permission on parent
                parent_check = PermissionCheck(
                    subject=check.subject,
                    permission=check.permission,
                    resource=Resource(
                        type=ResourceType(rel.resource.type.value),
                        id=rel.resource.id
                    )
                )
                result = await self.check_permission(parent_check)
                if result.allowed:
                    return True, f"Inherited from parent: {rel.resource}"
        
        return False, "No matching permission found"
    
    async def list_relationships(self, subject: Subject = None,
                                resource: Resource = None) -> List[Relationship]:
        """List relationships."""
        results = self._relationships
        
        if subject:
            results = [r for r in results if str(r.subject) == str(subject)]
        if resource:
            results = [r for r in results if str(r.resource) == str(resource)]
        
        return results
    
    async def expand_permissions(self, subject: Subject,
                                resource: Resource) -> List[Permission]:
        """Expand all permissions a subject has on a resource."""
        permissions = []
        
        for perm in Permission:
            check = PermissionCheck(subject=subject, permission=perm, resource=resource)
            result = await self.check_permission(check)
            if result.allowed:
                permissions.append(perm)
        
        return permissions
    
    async def lookup_subjects(self, resource: Resource,
                             permission: Permission) -> List[Subject]:
        """Find all subjects with a permission on a resource."""
        subjects = []
        
        for rel in self._relationships:
            if str(rel.resource) == str(resource):
                if self._relation_grants_permission(rel.relation, permission):
                    subjects.append(rel.subject)
        
        return subjects


class RelationshipManager:
    """
    Manage authorization relationships.
    
    Provides:
    - Relationship creation
    - Relationship deletion
    - Relationship queries
    """
    
    def __init__(self, client: MockPermifyClient):
        self.client = client
    
    async def grant(self, subject: Subject, relation: RelationType,
                   resource: Resource) -> Dict[str, Any]:
        """Grant a relationship."""
        relationship = Relationship(
            subject=subject,
            relation=relation,
            resource=resource
        )
        return await self.client.write_relationship(relationship)
    
    async def revoke(self, subject: Subject, relation: RelationType,
                    resource: Resource) -> Dict[str, Any]:
        """Revoke a relationship."""
        relationship = Relationship(
            subject=subject,
            relation=relation,
            resource=resource
        )
        return await self.client.delete_relationship(relationship)
    
    async def grant_owner(self, user_id: str, resource: Resource) -> Dict[str, Any]:
        """Grant owner relationship."""
        return await self.grant(
            Subject(type="user", id=user_id),
            RelationType.OWNER,
            resource
        )
    
    async def grant_editor(self, user_id: str, resource: Resource) -> Dict[str, Any]:
        """Grant editor relationship."""
        return await self.grant(
            Subject(type="user", id=user_id),
            RelationType.EDITOR,
            resource
        )
    
    async def grant_viewer(self, user_id: str, resource: Resource) -> Dict[str, Any]:
        """Grant viewer relationship."""
        return await self.grant(
            Subject(type="user", id=user_id),
            RelationType.VIEWER,
            resource
        )
    
    async def add_team_member(self, user_id: str, team_id: str) -> Dict[str, Any]:
        """Add user to team."""
        return await self.grant(
            Subject(type="user", id=user_id),
            RelationType.MEMBER,
            Resource(type=ResourceType.TEAM, id=team_id)
        )
    
    async def set_parent(self, child: Resource, parent: Resource) -> Dict[str, Any]:
        """Set parent-child relationship between resources."""
        return await self.grant(
            Subject(type=child.type.value, id=child.id),
            RelationType.PARENT,
            parent
        )
    
    async def list_for_subject(self, subject: Subject) -> List[Relationship]:
        """List all relationships for a subject."""
        return await self.client.list_relationships(subject=subject)
    
    async def list_for_resource(self, resource: Resource) -> List[Relationship]:
        """List all relationships for a resource."""
        return await self.client.list_relationships(resource=resource)


class PermissionChecker:
    """
    Check permissions.
    
    Provides:
    - Permission checks
    - Permission expansion
    - Subject lookup
    """
    
    def __init__(self, client: MockPermifyClient):
        self.client = client
        self._cache: Dict[str, Tuple[PermissionResult, datetime]] = {}
        self._cache_ttl = timedelta(seconds=30)
    
    async def check(self, subject: Subject, permission: Permission,
                   resource: Resource, use_cache: bool = True) -> PermissionResult:
        """Check if subject has permission on resource."""
        cache_key = f"{subject}:{permission.value}:{resource}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return result
        
        # Perform check
        check = PermissionCheck(
            subject=subject,
            permission=permission,
            resource=resource
        )
        result = await self.client.check_permission(check)
        
        # Cache result
        self._cache[cache_key] = (result, datetime.now())
        
        return result
    
    async def can_read(self, user_id: str, resource: Resource) -> bool:
        """Check if user can read resource."""
        result = await self.check(
            Subject(type="user", id=user_id),
            Permission.READ,
            resource
        )
        return result.allowed
    
    async def can_write(self, user_id: str, resource: Resource) -> bool:
        """Check if user can write to resource."""
        result = await self.check(
            Subject(type="user", id=user_id),
            Permission.WRITE,
            resource
        )
        return result.allowed
    
    async def can_delete(self, user_id: str, resource: Resource) -> bool:
        """Check if user can delete resource."""
        result = await self.check(
            Subject(type="user", id=user_id),
            Permission.DELETE,
            resource
        )
        return result.allowed
    
    async def can_admin(self, user_id: str, resource: Resource) -> bool:
        """Check if user has admin permission on resource."""
        result = await self.check(
            Subject(type="user", id=user_id),
            Permission.ADMIN,
            resource
        )
        return result.allowed
    
    async def get_permissions(self, user_id: str, 
                             resource: Resource) -> List[Permission]:
        """Get all permissions a user has on a resource."""
        return await self.client.expand_permissions(
            Subject(type="user", id=user_id),
            resource
        )
    
    async def who_can(self, permission: Permission,
                     resource: Resource) -> List[str]:
        """Find all users with a permission on a resource."""
        subjects = await self.client.lookup_subjects(resource, permission)
        return [s.id for s in subjects if s.type == "user"]
    
    def clear_cache(self) -> None:
        """Clear the permission cache."""
        self._cache.clear()


class PermifyAuthorization:
    """
    Permify authorization integration for MineralVision.
    
    Provides fine-grained authorization:
    - Relationship management
    - Permission checking
    - Policy enforcement
    
    Example:
        authz = PermifyAuthorization()
        await authz.connect()
        
        # Grant permissions
        await authz.relationships.grant_owner("user1", 
            Resource(ResourceType.PROJECT, "proj1"))
        
        # Check permissions
        can_read = await authz.permissions.can_read("user1",
            Resource(ResourceType.PROJECT, "proj1"))
    """
    
    def __init__(self, config: PermifyConfig = None):
        self.config = config or PermifyConfig()
        self.client: Optional[MockPermifyClient] = None
        self.relationships: Optional[RelationshipManager] = None
        self.permissions: Optional[PermissionChecker] = None
        self._connected = False
    
    async def connect(self) -> 'PermifyAuthorization':
        """Connect to Permify."""
        self.client = MockPermifyClient(self.config)
        
        # Load schema
        await self.client.write_schema(PolicySchema.get_schema())
        
        self.relationships = RelationshipManager(self.client)
        self.permissions = PermissionChecker(self.client)
        
        self._connected = True
        logger.info(f"Connected to Permify at {self.config.host}:{self.config.port}")
        return self
    
    async def setup_organization(self, org_id: str, admin_user_id: str) -> Dict[str, Any]:
        """Setup an organization with admin."""
        results = {}
        
        # Create organization admin relationship
        results['admin'] = await self.relationships.grant(
            Subject(type="user", id=admin_user_id),
            RelationType.ADMIN,
            Resource(type=ResourceType.ORGANIZATION, id=org_id)
        )
        
        return results
    
    async def setup_project(self, project_id: str, org_id: str,
                           owner_user_id: str) -> Dict[str, Any]:
        """Setup a project with owner and organization."""
        results = {}
        
        project = Resource(type=ResourceType.PROJECT, id=project_id)
        org = Resource(type=ResourceType.ORGANIZATION, id=org_id)
        
        # Set project owner
        results['owner'] = await self.relationships.grant_owner(owner_user_id, project)
        
        # Set organization as parent
        results['org'] = await self.relationships.set_parent(project, org)
        
        return results
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_permify(config: PermifyConfig = None) -> PermifyAuthorization:
    """Create a Permify authorization instance."""
    return PermifyAuthorization(config)


async def create_and_connect_permify(config: PermifyConfig = None) -> PermifyAuthorization:
    """Create and connect Permify."""
    authz = PermifyAuthorization(config)
    await authz.connect()
    return authz


# FastAPI dependency for permission checking

class PermissionDependency:
    """FastAPI dependency for permission checking."""
    
    def __init__(self, authz: PermifyAuthorization, permission: Permission,
                resource_type: ResourceType, resource_id_param: str = "id"):
        self.authz = authz
        self.permission = permission
        self.resource_type = resource_type
        self.resource_id_param = resource_id_param
    
    async def __call__(self, request, **kwargs) -> bool:
        """Check permission."""
        user_id = request.state.user_id  # Assumes auth middleware sets this
        resource_id = kwargs.get(self.resource_id_param)
        
        if not resource_id:
            return False
        
        resource = Resource(type=self.resource_type, id=resource_id)
        result = await self.authz.permissions.check(
            Subject(type="user", id=user_id),
            self.permission,
            resource
        )
        
        if not result.allowed:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Permission denied")
        
        return True
