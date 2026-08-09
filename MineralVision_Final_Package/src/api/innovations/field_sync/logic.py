"""Change-log sync protocol with optimistic versioning.

Rules:
  - each entity has a monotonically increasing version;
  - a client op carries the ``base_version`` it was made against; if that no
    longer matches the server version the op is rejected as a CONFLICT —
    server state wins, and a conflict record retains BOTH versions;
  - ``client_op_id`` is a unique idempotency key: retrying the same op
    returns the original outcome ("duplicate") without re-applying;
  - ``pull_since`` returns exactly the applied ops with version > N (delta);
  - per-device cursors (``sync_device_cursors``) track how far each device
    has downloaded, so reconnecting devices pull only newer entities;
  - conflict resolution policy is selectable per op: ``occ`` (optimistic
    concurrency, server wins — default) or ``lww`` (last-writer-wins by
    timestamp: the op's ``client_ts`` vs the server state's ``updated_at``;
    ties go to the server). Either way a conflict record retaining BOTH
    versions is written and returned to the client.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .models import (
    ConflictModel,
    DeviceCursorModel,
    EntityStateModel,
    SyncOpModel,
    _utcnow,
)

VALID_OPS = ("create", "update", "delete")
VALID_RESOLUTIONS = ("occ", "lww")


@dataclass
class OpResult:
    client_op_id: str
    entity_id: str
    status: str  # applied | conflict | duplicate
    version: int  # current server version after processing
    detail: str = ""
    conflict: Optional[Dict[str, Any]] = None  # populated when a conflict was logged


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
        entity_type: Optional[str] = None,
        device_id: Optional[str] = None,
        resolution: str = "occ",
    ) -> OpResult:
        if op not in VALID_OPS:
            raise ValueError(f"invalid op {op!r}; expected one of {VALID_OPS}")
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"invalid resolution {resolution!r}; expected one of {VALID_RESOLUTIONS}"
            )
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
            if resolution == "lww":
                return self._resolve_lww(
                    client_op_id=client_op_id,
                    entity_id=entity_id,
                    op=op,
                    base_version=base_version,
                    payload=payload,
                    client_ts=client_ts,
                    entity_type=entity_type,
                    device_id=device_id,
                    state=state,
                    current_version=current_version,
                )

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
                entity_type=entity_type, device_id=device_id,
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
                entity_id=entity_id, entity_type=entity_type,
                version=new_version, data=payload,
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
            entity_type=entity_type, device_id=device_id,
            base_version=base_version, applied_version=new_version,
            status="applied", payload=payload, client_ts=client_ts,
            created_at=_utcnow(),
        ))
        self.db.commit()
        return OpResult(
            client_op_id=client_op_id, entity_id=entity_id, status="applied",
            version=new_version, detail="",
        )

    # ------------------------------------------------- last-writer-wins (LWW)
    def _resolve_lww(
        self,
        client_op_id: str,
        entity_id: str,
        op: str,
        base_version: int,
        payload: Dict[str, Any],
        client_ts: Optional[datetime],
        entity_type: Optional[str],
        device_id: Optional[str],
        state: Optional[EntityStateModel],
        current_version: int,
    ) -> OpResult:
        """Last-writer-wins conflict resolution by timestamp.

        Compares the op's ``client_ts`` against the server state's
        ``updated_at`` (server receive time as fallback when client_ts is
        missing). The newer write wins; ties go to the server. A conflict
        record retaining BOTH versions is always written.
        """
        # A client op without a timestamp is stamped at server receive time,
        # which can never beat an existing server state (ties → server).
        effective_client_ts = client_ts or _utcnow()
        server_ts = state.updated_at if state is not None else datetime.min
        client_wins = effective_client_ts > server_ts

        conflict = ConflictModel(
            entity_id=entity_id,
            client_op_id=client_op_id,
            op=op,
            base_version=base_version,
            server_version=current_version,
            client_payload=payload,
            server_payload=(state.data if state else {}),
            resolution="client_wins_lww" if client_wins else "server_wins_lww",
            created_at=_utcnow(),
        )
        self.db.add(conflict)

        if not client_wins:
            self.db.add(SyncOpModel(
                client_op_id=client_op_id, entity_id=entity_id, op=op,
                entity_type=entity_type, device_id=device_id,
                base_version=base_version, applied_version=None,
                status="conflict", payload=payload, client_ts=client_ts,
                created_at=_utcnow(),
            ))
            self.db.commit()
            self.db.refresh(conflict)
            return OpResult(
                client_op_id=client_op_id, entity_id=entity_id, status="conflict",
                version=current_version,
                detail=(
                    f"base_version {base_version} != server version {current_version}; "
                    f"LWW: server timestamp {server_ts.isoformat()} >= client "
                    f"{effective_client_ts.isoformat()}; server wins"
                ),
                conflict=conflict_to_dict(conflict),
            )

        # Client write is newer: apply it over the stale server state.
        new_version = current_version + 1
        if op == "delete":
            if state is not None:
                state.deleted = True
                state.version = new_version
                state.updated_at = _utcnow()
        elif state is None:
            state = EntityStateModel(
                entity_id=entity_id, entity_type=entity_type,
                version=new_version, data=payload,
                deleted=False, updated_at=_utcnow(),
            )
            self.db.add(state)
        else:
            state.data = {**state.data, **payload} if op == "update" else payload
            if entity_type is not None:
                state.entity_type = entity_type
            state.deleted = False
            state.version = new_version
            state.updated_at = _utcnow()

        self.db.add(SyncOpModel(
            client_op_id=client_op_id, entity_id=entity_id, op=op,
            entity_type=entity_type, device_id=device_id,
            base_version=base_version, applied_version=new_version,
            status="applied", payload=payload, client_ts=client_ts,
            created_at=_utcnow(),
        ))
        self.db.commit()
        self.db.refresh(conflict)
        return OpResult(
            client_op_id=client_op_id, entity_id=entity_id, status="applied",
            version=new_version,
            detail=(
                f"LWW: client timestamp {effective_client_ts.isoformat()} > "
                f"server {server_ts.isoformat()}; client wins"
            ),
            conflict=conflict_to_dict(conflict),
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

    # ------------------------------------------------------- device cursors
    def get_device_cursor(self, device_id: str) -> int:
        row = self.db.get(DeviceCursorModel, device_id)
        return row.cursor if row is not None else 0

    def pull_for_device(
        self, device_id: str, since: Optional[int] = None
    ) -> Tuple[List[SyncOpModel], int, int]:
        """Delta download for a device.

        Uses the device's stored cursor unless ``since`` is given explicitly.
        Advances the stored cursor to the highest applied version returned.
        Returns ``(ops, cursor_before, cursor_after)``.
        """
        cursor_before = self.get_device_cursor(device_id) if since is None else since
        ops = self.pull_since(since=cursor_before)
        cursor_after = cursor_before
        for op in ops:
            if op.applied_version is not None and op.applied_version > cursor_after:
                cursor_after = op.applied_version
        row = self.db.get(DeviceCursorModel, device_id)
        if row is None:
            row = DeviceCursorModel(device_id=device_id, cursor=cursor_after,
                                    updated_at=_utcnow())
            self.db.add(row)
        else:
            row.cursor = cursor_after
            row.updated_at = _utcnow()
        self.db.commit()
        return ops, cursor_before, cursor_after


def state_to_dict(state: EntityStateModel) -> Dict[str, Any]:
    return {
        "entity_id": state.entity_id,
        "entity_type": state.entity_type,
        "version": state.version,
        "data": state.data,
        "deleted": state.deleted,
        "updated_at": state.updated_at.isoformat(),
    }


def op_to_dict(op: SyncOpModel) -> Dict[str, Any]:
    return {
        "client_op_id": op.client_op_id,
        "entity_id": op.entity_id,
        "entity_type": op.entity_type,
        "device_id": op.device_id,
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
