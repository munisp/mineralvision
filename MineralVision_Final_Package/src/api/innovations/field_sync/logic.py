"""Change-log sync protocol with optimistic versioning.

Rules:
  - each entity has a monotonically increasing version;
  - a client op carries the ``base_version`` it was made against; if that no
    longer matches the server version the op is rejected as a CONFLICT —
    server state wins, and a conflict record retains BOTH versions;
  - ``client_op_id`` is a unique idempotency key: retrying the same op
    returns the original outcome ("duplicate") without re-applying;
  - ``pull_since`` returns exactly the applied ops with version > N (delta).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from .models import ConflictModel, EntityStateModel, SyncOpModel, _utcnow

VALID_OPS = ("create", "update", "delete")


@dataclass
class OpResult:
    client_op_id: str
    entity_id: str
    status: str  # applied | conflict | duplicate
    version: int  # current server version after processing
    detail: str = ""


def _get_state(db: Session, entity_id: str) -> Optional[EntityStateModel]:
    return db.get(EntityStateModel, entity_id)


class FieldSync:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ apply
    def apply_op(
        self,
        client_op_id: str,
        entity_id: str,
        op: str,
        base_version: int,
        payload: Optional[Dict[str, Any]] = None,
        client_ts: Optional[datetime] = None,
    ) -> OpResult:
        if op not in VALID_OPS:
            raise ValueError(f"invalid op {op!r}; expected one of {VALID_OPS}")
        payload = payload or {}

        # Idempotent retry: same client_op_id → replay the recorded outcome.
        prior = (
            self.db.query(SyncOpModel)
            .filter(SyncOpModel.client_op_id == client_op_id)
            .first()
        )
        if prior is not None:
            state = _get_state(self.db, entity_id)
            return OpResult(
                client_op_id=client_op_id,
                entity_id=entity_id,
                status="duplicate",
                version=state.version if state else 0,
                detail=f"op already processed with status '{prior.status}'",
            )

        state = _get_state(self.db, entity_id)
        current_version = state.version if state else 0

        # Optimistic version check.
        if base_version != current_version:
            conflict = ConflictModel(
                entity_id=entity_id,
                client_op_id=client_op_id,
                op=op,
                base_version=base_version,
                server_version=current_version,
                client_payload=payload,
                server_payload=(state.data if state else {}),
                resolution="server_wins",
                created_at=_utcnow(),
            )
            self.db.add(conflict)
            self.db.add(SyncOpModel(
                client_op_id=client_op_id, entity_id=entity_id, op=op,
                base_version=base_version, applied_version=None,
                status="conflict", payload=payload, client_ts=client_ts,
                created_at=_utcnow(),
            ))
            self.db.commit()
            return OpResult(
                client_op_id=client_op_id, entity_id=entity_id, status="conflict",
                version=current_version,
                detail=f"base_version {base_version} != server version {current_version}; server wins",
            )

        # Entity-existence semantics.
        if op == "create" and state is not None and not state.deleted:
            raise ValueError(f"entity {entity_id!r} already exists at version {current_version}")
        if op in ("update", "delete") and (state is None or state.deleted):
            raise ValueError(f"entity {entity_id!r} does not exist")

        new_version = current_version + 1  # monotonic
        if op == "create":
            state = EntityStateModel(
                entity_id=entity_id, version=new_version, data=payload,
                deleted=False, updated_at=_utcnow(),
            )
            self.db.add(state)
        elif op == "update":
            state.data = {**state.data, **payload}
            state.version = new_version
            state.updated_at = _utcnow()
        else:  # delete — tombstone retained for delta propagation
            state.deleted = True
            state.version = new_version
            state.updated_at = _utcnow()

        self.db.add(SyncOpModel(
            client_op_id=client_op_id, entity_id=entity_id, op=op,
            base_version=base_version, applied_version=new_version,
            status="applied", payload=payload, client_ts=client_ts,
            created_at=_utcnow(),
        ))
        self.db.commit()
        return OpResult(
            client_op_id=client_op_id, entity_id=entity_id, status="applied",
            version=new_version, detail="",
        )

    # ------------------------------------------------------------------- read
    def get_state(self, entity_id: str) -> Optional[EntityStateModel]:
        return _get_state(self.db, entity_id)

    def pull_since(self, since: int = 0, entity_id: Optional[str] = None) -> List[SyncOpModel]:
        """Delta download: applied ops with applied_version > `since`."""
        query = self.db.query(SyncOpModel).filter(
            SyncOpModel.status == "applied",
            SyncOpModel.applied_version > since,
        )
        if entity_id is not None:
            query = query.filter(SyncOpModel.entity_id == entity_id)
        return query.order_by(SyncOpModel.entity_id, SyncOpModel.applied_version).all()

    def list_conflicts(self, entity_id: Optional[str] = None) -> List[ConflictModel]:
        query = self.db.query(ConflictModel)
        if entity_id is not None:
            query = query.filter(ConflictModel.entity_id == entity_id)
        return query.order_by(ConflictModel.id).all()


def state_to_dict(state: EntityStateModel) -> Dict[str, Any]:
    return {
        "entity_id": state.entity_id,
        "version": state.version,
        "data": state.data,
        "deleted": state.deleted,
        "updated_at": state.updated_at.isoformat(),
    }


def op_to_dict(op: SyncOpModel) -> Dict[str, Any]:
    return {
        "client_op_id": op.client_op_id,
        "entity_id": op.entity_id,
        "op": op.op,
        "base_version": op.base_version,
        "applied_version": op.applied_version,
        "status": op.status,
        "payload": op.payload,
        "client_ts": op.client_ts.isoformat() if op.client_ts else None,
    }


def conflict_to_dict(conflict: ConflictModel) -> Dict[str, Any]:
    return {
        "id": conflict.id,
        "entity_id": conflict.entity_id,
        "client_op_id": conflict.client_op_id,
        "op": conflict.op,
        "base_version": conflict.base_version,
        "server_version": conflict.server_version,
        "client_payload": conflict.client_payload,
        "server_payload": conflict.server_payload,
        "resolution": conflict.resolution,
        "created_at": conflict.created_at.isoformat(),
    }
