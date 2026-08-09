"""Governance logic: access tiers, RBAC enforcement, consent/attribution, audit.

Tiers (ascending sensitivity): public < restricted < sacred.

Enforcement rules:
  - list / export NEVER include sacred records;
  - restricted records require one of RESTRICTED_ROLES;
  - sacred records are reachable only by direct id and only by SACRED_ROLES;
  - every successful read (get/list/export) writes an audit row.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from .models import AccessAuditModel, KnowledgeRecordModel


class AccessTier(str, Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    SACRED = "sacred"


# Roles allowed to read restricted-tier knowledge.
RESTRICTED_ROLES = ("researcher", "custodian", "admin")
# Roles allowed to read sacred-tier knowledge (direct id access only).
SACRED_ROLES = ("custodian", "admin")
# Roles allowed to create records.
CREATE_ROLES = ("custodian", "admin")
# Roles allowed to bulk-export (sacred excluded regardless).
EXPORT_ROLES = RESTRICTED_ROLES
# Roles allowed to read the audit trail.
AUDIT_ROLES = ("custodian", "admin")

TIER_RANK = {AccessTier.PUBLIC: 0, AccessTier.RESTRICTED: 1, AccessTier.SACRED: 2}


def tier_permits_role(tier: AccessTier, role: str) -> bool:
    """Role check for direct (by-id) reads of a record at a given tier."""
    if tier == AccessTier.PUBLIC:
        return True
    if tier == AccessTier.RESTRICTED:
        return role in RESTRICTED_ROLES
    return role in SACRED_ROLES  # sacred


def require_tier_access(tier: AccessTier, role: str) -> None:
    """Raise 403 when the role may not read this tier."""
    if not tier_permits_role(tier, role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: tier '{tier.value}' requires elevated role",
        )


def record_to_dict(record: KnowledgeRecordModel) -> Dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "community": record.community,
        "tier": record.tier,
        "content": record.content,
        "consent_reference": record.consent_reference,
        "attribution": record.attribution,
        "created_by": record.created_by,
        "created_at": record.created_at.isoformat(),
    }


class IndigenousGovernance:
    """Business logic over the knowledge store; every read audits."""

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ audit
    def _audit(self, record_id: int, actor: str, role: str, action: str, tier: str) -> None:
        self.db.add(
            AccessAuditModel(
                record_id=record_id,
                actor=actor,
                actor_role=role,
                action=action,
                tier=tier,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )
        self.db.commit()

    # ----------------------------------------------------------------- create
    def create_record(
        self,
        title: str,
        community: str,
        tier: AccessTier,
        content: str,
        consent_reference: str,
        attribution: str,
        actor: str,
    ) -> KnowledgeRecordModel:
        record = KnowledgeRecordModel(
            title=title,
            community=community,
            tier=tier.value,
            content=content,
            consent_reference=consent_reference,
            attribution=attribution,
            created_by=actor,
            created_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    # ------------------------------------------------------------------- read
    def get_record(self, record_id: int, actor: str, role: str) -> KnowledgeRecordModel:
        """Direct id access; tier-enforced; always audited on success."""
        record = self.db.get(KnowledgeRecordModel, record_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"knowledge record {record_id} not found")
        tier = AccessTier(record.tier)
        require_tier_access(tier, role)
        self._audit(record.id, actor, role, "read", record.tier)
        return record

    def list_records(self, actor: str, role: str) -> List[KnowledgeRecordModel]:
        """List view: sacred records are NEVER listed; restricted is role-gated."""
        query = self.db.query(KnowledgeRecordModel).filter(
            KnowledgeRecordModel.tier != AccessTier.SACRED.value
        )
        if role not in RESTRICTED_ROLES:
            query = query.filter(KnowledgeRecordModel.tier == AccessTier.PUBLIC.value)
        records = query.order_by(KnowledgeRecordModel.id.asc()).all()
        for record in records:
            self._audit(record.id, actor, role, "list", record.tier)
        return records

    def export_records(self, actor: str, role: str) -> List[KnowledgeRecordModel]:
        """Bulk export: same exclusions as list — sacred never leaves in bulk."""
        records = (
            self.db.query(KnowledgeRecordModel)
            .filter(KnowledgeRecordModel.tier != AccessTier.SACRED.value)
            .order_by(KnowledgeRecordModel.id.asc())
            .all()
        )
        for record in records:
            self._audit(record.id, actor, role, "export", record.tier)
        return records

    def audit_trail(self, record_id: Optional[int] = None) -> List[AccessAuditModel]:
        query = self.db.query(AccessAuditModel)
        if record_id is not None:
            query = query.filter(AccessAuditModel.record_id == record_id)
        return query.order_by(AccessAuditModel.id.asc()).all()


def audit_to_dict(row: AccessAuditModel) -> Dict[str, Any]:
    return {
        "id": row.id,
        "record_id": row.record_id,
        "actor": row.actor,
        "actor_role": row.actor_role,
        "action": row.action,
        "tier": row.tier,
        "timestamp": row.timestamp.isoformat(),
    }
