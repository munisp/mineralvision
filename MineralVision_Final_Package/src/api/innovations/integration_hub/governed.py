"""Governed source-lineage and staged write-back controls.

This module deliberately keeps MineralVision in an evidence-and-decision role.
It records source-system lineage, produces canonical content hashes, discovers
ArcGIS Feature Service capabilities, and stages—not executes—write-back
proposals. A distinct MFA-attested reviewer must approve a proposal before an
external connector worker may attempt a destination write.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping

from sqlalchemy.orm import Session

from .models import EvidenceRecordModel, WritebackProposalModel


SUPPORTED_SOURCE_SYSTEMS = {
    "seequent_evo",
    "arcgis_enterprise",
    "file_export",
    "huggingface_dataset",
}
SUPPORTED_WRITEBACK_TARGETS = {"arcgis_enterprise"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return the stable representation used for record and request hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_source_system(source_system: str) -> str:
    if source_system not in SUPPORTED_SOURCE_SYSTEMS:
        raise ValueError(
            f"unsupported source_system {source_system!r}; "
            f"allowed: {sorted(SUPPORTED_SOURCE_SYSTEMS)}"
        )
    return source_system


def register_evidence(
    db: Session,
    *,
    tenant_id: str,
    source_system: str,
    source_ref: str,
    source_version: str,
    observed_at: datetime,
    geometry: Dict[str, Any],
    payload: Dict[str, Any],
    model_run: Dict[str, Any],
) -> EvidenceRecordModel:
    """Persist a tenant-bound source record with immutable canonical lineage.

    The source reference must be an incumbent object UUID/path, ArcGIS
    service/layer/object reference, or approved export manifest reference.
    The raw payload remains in the tenant-bound integration store; the hash
    gives reviewers and write-back workers a deterministic integrity anchor.
    """
    if not tenant_id or not source_ref or not source_version:
        raise ValueError("tenant_id, source_ref, and source_version are required")
    validate_source_system(source_system)
    if not isinstance(geometry, dict) or not isinstance(payload, dict) or not isinstance(model_run, dict):
        raise ValueError("geometry, payload, and model_run must be JSON objects")
    if observed_at.tzinfo is not None:
        observed_at = observed_at.astimezone(timezone.utc).replace(tzinfo=None)

    canonical = {
        "tenant_id": tenant_id,
        "source_system": source_system,
        "source_ref": source_ref,
        "source_version": source_version,
        "observed_at": observed_at.isoformat(),
        "geometry": geometry,
        "payload": payload,
        "model_run": model_run,
    }
    record = EvidenceRecordModel(
        evidence_id=f"ev_{uuid.uuid4().hex}",
        tenant_id=tenant_id,
        source_system=source_system,
        source_ref=source_ref,
        source_version=source_version,
        observed_at=observed_at,
        ingested_at=utcnow(),
        geometry=geometry,
        payload=payload,
        model_run=model_run,
        lineage_hash=sha256_json(canonical),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def discover_arcgis_capabilities(service_metadata: Mapping[str, Any]) -> Dict[str, bool]:
    """Reduce ArcGIS Feature Service metadata to the capabilities we require.

    The result is intentionally conservative. An integration cannot infer a
    write ability from credentials alone: it must be advertised by the target
    service and still be subjected to reviewer approval and destination
    schema/CRS preflight by a connector worker.
    """
    raw = str(service_metadata.get("capabilities", ""))
    capabilities = {item.strip().lower() for item in raw.split(",") if item.strip()}
    editing = bool(service_metadata.get("allowGeometryUpdates")) and {
        "create", "update", "delete"
    }.issubset(capabilities)
    return {
        "query": "query" in capabilities,
        "create": "create" in capabilities,
        "update": "update" in capabilities,
        "delete": "delete" in capabilities,
        "sync": "sync" in capabilities,
        "uploads": "uploads" in capabilities,
        "editing": editing,
        "supports_apply_edits_with_global_ids": bool(
            (service_metadata.get("advancedEditingCapabilities") or {}).get(
                "supportsApplyEditsWithGlobalIds"
            )
        ),
    }


def stage_writeback(
    db: Session,
    *,
    tenant_id: str,
    evidence_id: str,
    target_system: str,
    target_ref: str,
    candidate_payload: Dict[str, Any],
    submitted_by: str,
    dry_run: Dict[str, Any],
) -> WritebackProposalModel:
    """Create an immutable, approval-required destination write proposal."""
    if target_system not in SUPPORTED_WRITEBACK_TARGETS:
        raise ValueError("only arcgis_enterprise staged write-back is supported")
    if not all([tenant_id, evidence_id, target_ref, submitted_by]):
        raise ValueError("tenant_id, evidence_id, target_ref, and submitted_by are required")
    if not isinstance(candidate_payload, dict) or not isinstance(dry_run, dict):
        raise ValueError("candidate_payload and dry_run must be JSON objects")

    evidence = (
        db.query(EvidenceRecordModel)
        .filter(
            EvidenceRecordModel.evidence_id == evidence_id,
            EvidenceRecordModel.tenant_id == tenant_id,
        )
        .first()
    )
    if evidence is None:
        raise ValueError("evidence record not found in the requesting tenant")

    request = {
        "tenant_id": tenant_id,
        "evidence_id": evidence_id,
        "evidence_lineage_hash": evidence.lineage_hash,
        "target_system": target_system,
        "target_ref": target_ref,
        "candidate_payload": candidate_payload,
        "dry_run": dry_run,
    }
    proposal = WritebackProposalModel(
        proposal_id=f"wb_{uuid.uuid4().hex}",
        tenant_id=tenant_id,
        evidence_id=evidence_id,
        target_system=target_system,
        target_ref=target_ref,
        state="staged",
        request_hash=sha256_json(request),
        candidate_payload=candidate_payload,
        dry_run=dry_run,
        submitted_by=submitted_by,
        created_at=utcnow(),
    )
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


def approve_writeback(
    db: Session,
    *,
    tenant_id: str,
    proposal_id: str,
    reviewer_id: str,
    mfa_verified: bool,
    review_reason: str,
) -> WritebackProposalModel:
    """Approve a staged proposal only through four-eyes, MFA-backed review."""
    proposal = (
        db.query(WritebackProposalModel)
        .filter(
            WritebackProposalModel.proposal_id == proposal_id,
            WritebackProposalModel.tenant_id == tenant_id,
        )
        .first()
    )
    if proposal is None:
        raise ValueError("write-back proposal not found in the requesting tenant")
    if proposal.state != "staged":
        raise ValueError(f"proposal is not staged (state={proposal.state!r})")
    if not mfa_verified:
        raise ValueError("MFA verification is required to approve write-back")
    if reviewer_id == proposal.submitted_by:
        raise ValueError("submitter cannot approve their own write-back proposal")
    if not review_reason.strip():
        raise ValueError("review_reason is required")

    proposal.state = "approved"
    proposal.reviewer_id = reviewer_id
    proposal.mfa_verified = True
    proposal.review_reason = review_reason.strip()
    proposal.approved_at = utcnow()
    db.commit()
    db.refresh(proposal)
    return proposal


def evidence_to_dict(record: EvidenceRecordModel) -> Dict[str, Any]:
    return {
        "evidence_id": record.evidence_id,
        "tenant_id": record.tenant_id,
        "source_system": record.source_system,
        "source_ref": record.source_ref,
        "source_version": record.source_version,
        "observed_at": record.observed_at.isoformat(),
        "ingested_at": record.ingested_at.isoformat(),
        "geometry": record.geometry,
        "payload": record.payload,
        "model_run": record.model_run,
        "lineage_hash": record.lineage_hash,
    }


def proposal_to_dict(proposal: WritebackProposalModel) -> Dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "tenant_id": proposal.tenant_id,
        "evidence_id": proposal.evidence_id,
        "target_system": proposal.target_system,
        "target_ref": proposal.target_ref,
        "state": proposal.state,
        "request_hash": proposal.request_hash,
        "candidate_payload": proposal.candidate_payload,
        "dry_run": proposal.dry_run,
        "submitted_by": proposal.submitted_by,
        "reviewer_id": proposal.reviewer_id,
        "mfa_verified": proposal.mfa_verified,
        "review_reason": proposal.review_reason,
        "created_at": proposal.created_at.isoformat(),
        "approved_at": proposal.approved_at.isoformat() if proposal.approved_at else None,
    }
