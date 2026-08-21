"""Shared resource-ownership authorization for database-backed API routes.

Global authentication establishes *who* made a request; these helpers establish
whether that identity may access a project-scoped resource.  They intentionally
fail closed and do not infer ownership from client-supplied identifiers.
"""

from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .auth_middleware import TokenPayload
from .database import ProjectModel


ADMIN_ROLES = frozenset({"admin", "security_admin"})


def is_admin(user: TokenPayload) -> bool:
    """Return true only for an explicitly privileged role claim."""
    return bool(set(user.roles or [user.role]) & ADMIN_ROLES)


def require_project_access(db: Session, project_id: str, user: TokenPayload) -> ProjectModel:
    """Return a project only when the caller owns it or has an admin role.

    A 404 is used for a missing project and a 403 for an existing but inaccessible
    project.  This permits clients to distinguish invalid input from authorization
    failures while avoiding any accidental owner override by request payloads.
    """
    project = db.query(ProjectModel).filter(ProjectModel.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} not found")
    if not is_admin(user) and project.owner_id != user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for this project")
    return project


def project_scope_query(query, user: TokenPayload):
    """Restrict a ProjectModel query to the caller's tenant unless privileged."""
    if is_admin(user):
        return query
    return query.filter(ProjectModel.owner_id == user.user_id)
