# Governed Evidence, Four-Eyes Approval, and Lineage-Hash Specification

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**API base path:** `/api/innovations/integration_hub`  
**Implementation:** `src/api/innovations/integration_hub/governed.py` and `routes.py`

## Scope and Trust Boundary

This API creates a controlled evidence-to-write-back workflow. It does not directly mutate ArcGIS, Seequent/Evo, or another incumbent system. The workflow is deliberately split into four bounded stages: evidence registration, capability interpretation, staged write-back, and MFA-backed four-eyes approval. A separate connector worker—not this API—may process an approved proposal after repeating destination validation.

> **Security invariant:** Evidence registration and proposal staging are immutable internal records. External mutation requires a later worker, a distinct reviewer, MFA evidence, a destination allow-list, preflight validation, idempotency/reconciliation, and destination-result auditing.

## Authentication and Tenant Authorization

Every endpoint requires `X-API-Key` and the relevant hub scope. Governed endpoints also require a **tenant-bound** key: the persisted `ApiKeyModel.tenant_id` must be nonempty and exactly equal to the request `tenant_id`. Legacy/unbound keys remain usable for non-governed webhook functions but are rejected by governed evidence and write-back endpoints.

| Operation | Required hub scope | Additional enforcement |
|---|---|---|
| Register evidence | `write` | Key tenant equals request tenant; supported source system; canonical lineage payload validates |
| Discover ArcGIS capability | `read` | No remote URL accepted or fetched; caller supplies already-obtained service metadata |
| Stage write-back | `write` | Key tenant equals request tenant; referenced evidence exists in same tenant; target is allowed |
| Approve write-back | `write` | Key tenant equals request tenant; proposal is staged; reviewer differs from submitter; MFA is true; reason is nonempty |

The API key is a service-to-service boundary. Production deployments should add OIDC/OPA identity enforcement around the hub and source the reviewer/MFA assertions from verified identity claims, rather than accepting them from a browser payload.

## Data Types

| Type | Constraints | Meaning |
|---|---|---|
| `tenant_id` | Nonempty string, max 128 | Customer/tenant isolation boundary |
| `source_system` | `seequent_evo`, `arcgis_enterprise`, or `file_export` | Approved evidence origin |
| `source_ref` | Nonempty string, max 1024 | Object UUID/path, Feature Service/layer/object reference, or export-manifest reference |
| `source_version` | Nonempty string, max 256 | Source version, revision, edit timestamp, or ingestion cursor |
| `geometry` | JSON object | GeoJSON-like geometry; CRS policy is carried in payload or tenant mapping |
| `payload` | JSON object | Approved source attributes and contextual metadata |
| `model_run` | JSON object | Model ID, artifact hash, confidence/uncertainty, and associated provenance |
| `dry_run` | JSON object | Destination schema/CRS/candidate-layer preflight outcome and diff summary |

## API Contract

### Register Evidence

```http
POST /api/innovations/integration_hub/evidence
X-API-Key: mvk_<key_id>.<secret>
Content-Type: application/json
```

```json
{
  "tenant_id": "northstar-mining",
  "source_system": "arcgis_enterprise",
  "source_ref": "https://gis.example.com/arcgis/rest/services/Exploration/FeatureServer/7/42",
  "source_version": "edit-date:2026-08-22T10:14:00Z",
  "observed_at": "2026-08-20T07:32:00Z",
  "geometry": {"type": "Point", "coordinates": [120.04, -31.95]},
  "payload": {
    "crs": "EPSG:4326",
    "attributes": {"hole_id": "DH-042", "au_ppm": 1.2}
  },
  "model_run": {
    "model_id": "prospectivity-v1",
    "artifact_hash": "b7f9...",
    "confidence": 0.81,
    "uncertainty": "review-required"
  }
}
```

A successful `201 Created` returns:

```json
{
  "evidence_id": "ev_<uuidhex>",
  "tenant_id": "northstar-mining",
  "source_system": "arcgis_enterprise",
  "source_ref": "https://gis.example.com/.../7/42",
  "source_version": "edit-date:2026-08-22T10:14:00Z",
  "observed_at": "2026-08-20T07:32:00",
  "ingested_at": "2026-08-22T12:00:00",
  "geometry": {"type": "Point", "coordinates": [120.04, -31.95]},
  "payload": {"crs": "EPSG:4326", "attributes": {"hole_id": "DH-042", "au_ppm": 1.2}},
  "model_run": {"model_id": "prospectivity-v1", "artifact_hash": "b7f9...", "confidence": 0.81},
  "lineage_hash": "<64 lowercase hexadecimal SHA-256 characters>"
}
```

The response means the evidence was persisted in the tenant evidence store. It does not mean an AI result is approved, accurate, current, or written to an incumbent platform.

### Inspect ArcGIS Capability Metadata

```http
POST /api/innovations/integration_hub/arcgis/capabilities
X-API-Key: mvk_<key_id>.<secret>
Content-Type: application/json
```

```json
{
  "service_metadata": {
    "capabilities": "Query,Create,Update,Delete,Sync,Uploads",
    "allowGeometryUpdates": true,
    "advancedEditingCapabilities": {
      "supportsApplyEditsWithGlobalIds": true
    }
  }
}
```

A successful `200 OK` returns a conservative boolean map:

```json
{
  "capabilities": {
    "query": true,
    "create": true,
    "update": true,
    "delete": true,
    "sync": true,
    "uploads": true,
    "editing": true,
    "supports_apply_edits_with_global_ids": true
  }
}
```

The endpoint parses caller-supplied metadata only. It never retrieves a caller-supplied URL and cannot be used as an SSRF proxy. A dedicated connector worker must retrieve metadata using a deployment allow-list, TLS validation, and an approved service identity. Feature Service capability metadata is the appropriate source for checking whether query, editing, synchronization, upload, and global-ID workflows are supported. [1]

### Stage Candidate Write-Back

```http
POST /api/innovations/integration_hub/writebacks
X-API-Key: mvk_<key_id>.<secret>
Content-Type: application/json
```

```json
{
  "tenant_id": "northstar-mining",
  "evidence_id": "ev_<uuidhex>",
  "target_system": "arcgis_enterprise",
  "target_ref": "https://gis.example.com/arcgis/rest/services/MV_Candidates/FeatureServer/0",
  "candidate_payload": {
    "attributes": {"mv_status": "candidate", "mv_evidence_id": "ev_<uuidhex>"},
    "geometry": {"x": 120.04, "y": -31.95}
  },
  "submitted_by": "analyst@northstar.example",
  "dry_run": {
    "schema_valid": true,
    "crs_valid": true,
    "destination_layer_is_candidate": true,
    "diff": {"attributes_added": ["mv_status", "mv_evidence_id"]}
  }
}
```

A successful `201 Created` returns a proposal in `staged` state:

```json
{
  "proposal_id": "wb_<uuidhex>",
  "tenant_id": "northstar-mining",
  "evidence_id": "ev_<uuidhex>",
  "target_system": "arcgis_enterprise",
  "target_ref": "https://gis.example.com/.../MV_Candidates/FeatureServer/0",
  "state": "staged",
  "request_hash": "<64 lowercase hexadecimal SHA-256 characters>",
  "candidate_payload": {"attributes": {"mv_status": "candidate"}},
  "dry_run": {"schema_valid": true, "crs_valid": true},
  "submitted_by": "analyst@northstar.example",
  "reviewer_id": null,
  "mfa_verified": false,
  "review_reason": "",
  "created_at": "2026-08-22T12:05:00",
  "approved_at": null
}
```

Only `arcgis_enterprise` is currently accepted as a staged write-back target. The referenced evidence must exist under the same tenant. This endpoint does not perform a destination write.

### Approve a Staged Proposal: Four-Eyes Workflow

```http
POST /api/innovations/integration_hub/writebacks/{proposal_id}/approve
X-API-Key: mvk_<key_id>.<secret>
Content-Type: application/json
```

```json
{
  "tenant_id": "northstar-mining",
  "reviewer_id": "reviewer@northstar.example",
  "mfa_verified": true,
  "review_reason": "Source lineage, geometry, dry-run schema result, and candidate-layer diff reviewed."
}
```

A successful `200 OK` returns the original proposal with the following changes:

```json
{
  "state": "approved",
  "reviewer_id": "reviewer@northstar.example",
  "mfa_verified": true,
  "review_reason": "Source lineage, geometry, dry-run schema result, and candidate-layer diff reviewed.",
  "approved_at": "2026-08-22T12:10:00"
}
```

The approval is an authorization gate, not an external write receipt. A connector worker must atomically claim/reload an approved proposal, repeat destination preflight, use stable/idempotent destination identifiers where available, then record a final destination outcome.

## Error Specification

FastAPI request-schema validation errors return `422 Unprocessable Entity` with a `detail` array describing the field error. Domain validation failures return `422` with a string `detail`. Authentication failures use `401`; scope, tenant, or policy failures use `403`. Unknown routes return `404`. Unhandled faults are sanitized to `500` by the platform exception handler.

| HTTP status | Error condition | Example response | Client action |
|---:|---|---|---|
| 401 | Missing, malformed, inactive, or invalid API key | `{"detail":"API key required (X-API-Key header)"}` | Obtain a valid active key; do not retry with altered tenant data. |
| 403 | Valid key lacks required scope | `{"detail":"API key lacks required scope 'write'"}` | Request least-privilege scope elevation through the tenant administrator. |
| 403 | Key unbound or bound to another tenant | `{"detail":"API key is not bound to this tenant"}` | Use a tenant-bound integration key matching `tenant_id`; do not change tenant ID to bypass the check. |
| 422 | Pydantic field/type/length/date validation error | `{"detail":[{"loc":["body","observed_at"],"msg":"..."}]}` | Correct the request shape/data. |
| 422 | Unsupported evidence source | `{"detail":"unsupported source_system 'x'; allowed: [...]"}` | Use the approved adapter or submit an onboarding request. |
| 422 | Evidence absent or belongs to another tenant | `{"detail":"evidence record not found in the requesting tenant"}` | Reconcile source/evidence identifiers within the same tenant. |
| 422 | Unsupported write-back target | `{"detail":"only arcgis_enterprise staged write-back is supported"}` | Keep target in read-only/export mode or implement an approved adapter. |
| 422 | Proposal no longer staged | `{"detail":"proposal is not staged (state='approved')"}` | Retrieve the proposal/audit state; never re-approve by retry. |
| 422 | MFA not verified | `{"detail":"MFA verification is required to approve write-back"}` | Complete the identity-provider step-up flow; server integration must derive this from verified claims. |
| 422 | Submitter tries self-approval | `{"detail":"submitter cannot approve their own write-back proposal"}` | Use a distinct authorized reviewer. |
| 422 | Empty rationale | `{"detail":"review_reason is required"}` | Supply an evidence-based review rationale. |
| 500 | Unexpected server error | `{"detail":"Internal server error"}` | Preserve correlation ID, do not blindly retry a subsequent external write, and investigate server/audit records. |

## Canonical Lineage Hash

### Hash Input

Each evidence record derives `lineage_hash` from exactly these fields:

```json
{
  "tenant_id": "<tenant_id>",
  "source_system": "<source_system>",
  "source_ref": "<source_ref>",
  "source_version": "<source_version>",
  "observed_at": "<UTC ISO-8601 timestamp without offset>",
  "geometry": {"...": "..."},
  "payload": {"...": "..."},
  "model_run": {"...": "..."}
}
```

The implementation first converts `observed_at` to UTC and removes the offset, then serializes the exact mapping with Python `json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)`. Sorting object keys and using compact separators prevents inconsequential JSON formatting differences from changing the result. Finally it UTF-8 encodes the canonical string and computes SHA-256.

```python
canonical = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    default=str,
)
lineage_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

The stored hash is a 64-character lowercase hexadecimal SHA-256 digest. The same evidence payload produces the same hash even if JSON object keys arrive in another order. A material change to one of the bound fields produces a different hash.

### Security Characteristics and Limits

| Property | What it provides | What it does not provide |
|---|---|---|
| SHA-256 collision resistance | Detectable content/reference/version change when the expected canonical record is known | Proof of who supplied the data or that a privileged database operator never altered the record and hash together |
| Tenant inclusion | A record from a different tenant hashes differently even if source content matches | Tenant access control by itself; the API key/identity layer and database query filters enforce access |
| Source/version inclusion | An evidence hash binds the incumbent reference and declared version | Independent confirmation that the source system version is truthful; connector reconciliation is still required |
| Model-run inclusion | A candidate is tied to the declared model artifact/provenance fields | Model accuracy, calibration, approval, or permission to take action |
| Canonical serialization | Stable reproducibility across JSON key ordering/whitespace | Canonical equivalence for semantically identical arrays with a different order; array order is intentionally meaningful |

The hash is an **integrity anchor**, not a digital signature or a full tamper-evident audit chain. Higher-assurance deployments should additionally write immutable audit events to an append-only/WORM-capable store or trusted ledger, sign connector envelopes with tenant-managed keys, include a hash of original source export bytes where permitted, and export correlated audit events off-host to the SIEM. NIST describes hash functions as mechanisms for detecting changes to data and distinguishes them from digital signatures and authentication controls. [2]

## Request Hash for a Write-Back Proposal

`request_hash` uses the same canonical SHA-256 construction, but its payload contains the tenant, evidence ID, referenced evidence `lineage_hash`, destination target, candidate payload, and dry-run result. This binds a staged proposal to the exact source-evidence state and preflight report it was created from. It is not currently an HMAC; external worker auditing should bind it to an authenticated actor and a write-attempt audit record.

## References

[1] [ArcGIS Enterprise Feature Service REST API](https://developers.arcgis.com/rest/services-reference/enterprise/feature-service/)

[2] [NIST FIPS 180-4: Secure Hash Standard](https://csrc.nist.gov/pubs/fips/180-4/upd1/final)
