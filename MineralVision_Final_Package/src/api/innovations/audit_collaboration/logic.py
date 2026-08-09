"""Audit trail (JSON diff), threaded comments, settings versioning + revert.

The JSON diff is a real recursive structural diff producing a deterministic
list of change entries: {path, op: added|removed|changed, before, after}.
Settings snapshots are immutable; revert creates a NEW snapshot whose
content equals the target version (append-only semantics preserved).
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .models import AuditEventModel, CommentModel, SettingsSnapshotModel, _utcnow


# ---------------------------------------------------------------------------
# JSON diff
# ---------------------------------------------------------------------------

def json_diff(before: Any, after: Any, path: str = "") -> List[Dict[str, Any]]:
    """Recursive structural diff → deterministic change list (sorted paths)."""
    changes: List[Dict[str, Any]] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in before:
                changes.append({"path": sub, "op": "added", "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": sub, "op": "removed", "before": before[key], "after": None})
            else:
                changes.extend(json_diff(before[key], after[key], sub))
    elif isinstance(before, list) and isinstance(after, list):
        for i in range(max(len(before), len(after))):
            sub = f"{path}[{i}]"
            if i >= len(before):
                changes.append({"path": sub, "op": "added", "before": None, "after": after[i]})
            elif i >= len(after):
                changes.append({"path": sub, "op": "removed", "before": before[i], "after": None})
            else:
                changes.extend(json_diff(before[i], after[i], sub))
    elif before != after:
        changes.append({"path": path or "$", "op": "changed", "before": before, "after": after})
    return changes


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class AuditCollaboration:
    def __init__(self, db: Session):
        self.db = db

    # -------------------------------------------------------------- events
    def record_event(
        self,
        actor: str,
        action: str,
        entity_type: str,
        entity_id: str,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
    ) -> AuditEventModel:
        """Append an audit event; the diff is computed and stored."""
        before = before or {}
        after = after or {}
        event = AuditEventModel(
            actor=actor, action=action, entity_type=entity_type, entity_id=entity_id,
            before=before, after=after, diff=json_diff(before, after),
            created_at=_utcnow(),
        )
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def list_events(
        self, entity_type: Optional[str] = None, entity_id: Optional[str] = None,
    ) -> List[AuditEventModel]:
        query = self.db.query(AuditEventModel)
        if entity_type is not None:
            query = query.filter(AuditEventModel.entity_type == entity_type)
        if entity_id is not None:
            query = query.filter(AuditEventModel.entity_id == entity_id)
        return query.order_by(AuditEventModel.id).all()

    # ------------------------------------------------------------- comments
    def add_comment(
        self,
        entity_type: str,
        entity_id: str,
        author: str,
        body: str,
        parent_id: Optional[int] = None,
    ) -> CommentModel:
        if parent_id is not None:
            parent = self.db.get(CommentModel, parent_id)
            if parent is None:
                raise HTTPException(status_code=404, detail=f"parent comment {parent_id} not found")
            if parent.entity_type != entity_type or parent.entity_id != entity_id:
                raise HTTPException(status_code=422, detail="parent comment belongs to a different entity")
        comment = CommentModel(
            entity_type=entity_type, entity_id=entity_id, parent_id=parent_id,
            author=author, body=body, created_at=_utcnow(),
        )
        self.db.add(comment)
        self.db.commit()
        self.db.refresh(comment)
        return comment

    def get_comments(self, entity_type: str, entity_id: str) -> List[CommentModel]:
        return (
            self.db.query(CommentModel)
            .filter(CommentModel.entity_type == entity_type, CommentModel.entity_id == entity_id)
            .order_by(CommentModel.id)
            .all()
        )

    def get_threaded_comments(self, entity_type: str, entity_id: str) -> List[Dict[str, Any]]:
        """Depth-first thread tree: each top-level comment followed by its
        replies (recursively), stable by id within each level."""
        comments = self.get_comments(entity_type, entity_id)
        by_parent: Dict[Optional[int], List[CommentModel]] = {}
        for c in comments:
            by_parent.setdefault(c.parent_id, []).append(c)
        out: List[Dict[str, Any]] = []

        def walk(parent_id: Optional[int], depth: int) -> None:
            for c in by_parent.get(parent_id, []):
                out.append(comment_to_dict(c, depth))
                walk(c.id, depth + 1)

        walk(None, 0)
        return out

    # ------------------------------------------------------------- settings
    def set_settings(
        self, project_id: str, settings: Dict[str, Any], actor: str, note: str = "",
    ) -> SettingsSnapshotModel:
        """Store a new immutable settings snapshot (version = max + 1)."""
        latest = self._latest(project_id)
        version = 1 if latest is None else latest.version + 1
        snapshot = SettingsSnapshotModel(
            project_id=project_id, version=version, settings=settings,
            actor=actor, note=note, created_at=_utcnow(),
        )
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def revert_settings(self, project_id: str, version: int, actor: str) -> SettingsSnapshotModel:
        """Revert = new snapshot whose settings equal the target version's."""
        target = (
            self.db.query(SettingsSnapshotModel)
            .filter(SettingsSnapshotModel.project_id == project_id,
                    SettingsSnapshotModel.version == version)
            .first()
        )
        if target is None:
            raise HTTPException(status_code=404, detail=f"no settings version {version} for {project_id!r}")
        return self.set_settings(
            project_id, dict(target.settings), actor, note=f"revert to version {version}",
        )

    def _latest(self, project_id: str) -> Optional[SettingsSnapshotModel]:
        return (
            self.db.query(SettingsSnapshotModel)
            .filter(SettingsSnapshotModel.project_id == project_id)
            .order_by(SettingsSnapshotModel.version.desc())
            .first()
        )

    def get_current_settings(self, project_id: str) -> Optional[SettingsSnapshotModel]:
        return self._latest(project_id)

    def get_settings_history(self, project_id: str) -> List[SettingsSnapshotModel]:
        return (
            self.db.query(SettingsSnapshotModel)
            .filter(SettingsSnapshotModel.project_id == project_id)
            .order_by(SettingsSnapshotModel.version)
            .all()
        )


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------

def event_to_dict(event: AuditEventModel) -> Dict[str, Any]:
    return {
        "id": event.id, "actor": event.actor, "action": event.action,
        "entity_type": event.entity_type, "entity_id": event.entity_id,
        "before": event.before, "after": event.after, "diff": event.diff,
        "created_at": event.created_at.isoformat(),
    }


def comment_to_dict(comment: CommentModel, depth: int = 0) -> Dict[str, Any]:
    return {
        "id": comment.id, "entity_type": comment.entity_type, "entity_id": comment.entity_id,
        "parent_id": comment.parent_id, "author": comment.author, "body": comment.body,
        "depth": depth, "created_at": comment.created_at.isoformat(),
    }


def snapshot_to_dict(snapshot: SettingsSnapshotModel) -> Dict[str, Any]:
    return {
        "id": snapshot.id, "project_id": snapshot.project_id, "version": snapshot.version,
        "settings": snapshot.settings, "actor": snapshot.actor, "note": snapshot.note,
        "created_at": snapshot.created_at.isoformat(),
    }
