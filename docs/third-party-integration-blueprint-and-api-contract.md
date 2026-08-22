# Third-Party Integration Blueprint and API Contract

**Prepared by Manus AI**  
**Date:** 22 August 2026  
**Scope:** A four-week standard deployment for a single customer tenant, one approved source system, and a read-first evidence workflow. ArcGIS candidate write-back is staged and human-approved; Seequent/Evo integration is source-reference/object-link oriented until customer governance approves supported object-level operations.

## Architecture Principles

MineralVision is a **governed decision layer**, not the incumbent system of record. It receives a bounded source representation, normalizes it into a tenant-scoped evidence record, links any AI output to a registered model run, and allows a reviewer to approve or reject a candidate. An external write is always a separate worker operation that consumes a staged, approved proposal; it is never performed by evidence registration or review endpoints.

```text
                         Customer identity boundary
              ┌─────────────────────────────────────────────────┐
              │ OIDC user / scoped integration service account   │
              └─────────────────────────────────────────────────┘
                                      │
                                      ▼
 Source System ─► Connector adapter ─► MineralVision evidence API ─► reviewer workspace
 (Seequent/Evo,  (capability discovery,    │                          │
  ArcGIS, file)   source allow-list,       │                          ├─ reject / annotate
                  schema & CRS check)      │                          └─ approve with MFA
                                           ▼                                    │
                                tenant-bound evidence store                     ▼
                                source lineage hash                    staged write-back proposal
                                           │                                    │
                                           └─────────────── audit/SIEM ◄────────┘
                                                                                │
                                                                                ▼
                                                                    destination worker
                                                                    (dry-run/allow-list/idempotency)
                                                                                │
                                                                                ▼
                                                                    incumbent candidate layer
```

Each source reference carries the source-system identifier, immutable source ID/path, source version, observation time, ingestion time, geometry/CRS, canonical payload hash, model-run information, reviewer decision, and destination result. A source model or authoritative layer is never overwritten by the P0 connector.

## Four-Week Standard Deployment

| Week | Customer activity | MineralVision activity | Exit artifact |
|---|---|---|---|
| 1: Scope and security | Name workflow sponsor, reviewer, tenant, data steward, source-system administrator, and data classification. Grant sandbox/API access and source documentation. | Create tenant connector record, source allow-list, data processing map, OIDC group mapping, and connector threat model. | Signed integration scope and source-service capability snapshot. |
| 2: Read-only ingestion | Provide bounded test data/service layer and target CRS/schema. | Configure connector capability discovery, bounded query/import, source-reference mapping, CRS validation, lineage storage, and connector-health dashboard. | Evidence records with complete lineage and reconciliation report. |
| 3: Review and decision | Train reviewers and verify source links. | Configure candidate presentation, uncertainty/model provenance, reviewer roles, MFA policy, audit/SIEM events, and acceptance tests. | Reviewer acceptance test and customer-approved evidence bundle. |
| 4: Supervised hand-off | Approve non-production candidate destination layer, if write-back is in scope. | Enable staged write-back only; run dry-run, reviewer approval, idempotent worker simulation, destination reconciliation, credential-revocation test, and deployment retrospective. | Deployment packet, runbook, rollback evidence, and go/no-go scorecard. |

The standard deployment ends at read-only integration if the source system lacks a safe write capability, the customer declines candidate-layer creation, or any lineage/security test fails. This is a valid deployment outcome; it is not a project failure.

## Connector Requirements

| Category | Mandatory requirement |
|---|---|
| Tenant isolation | One connector record, source allow-list, credential scope, evidence namespace, audit stream, and destination mapping per tenant. |
| Authentication | Customer-managed OAuth/client credentials or a least-privilege service identity. No shared user password or unrestricted administrative token. |
| Network safety | Approved base URL/service ID, TLS validation, SSRF-safe destination allow-list, bounded pagination/attachment size, timeout/retry policy, and no remote URL ingestion from user payload. |
| Data contract | Explicit source schema mapping, CRS validation, unit/time-zone policy, record limit, ingestion cursor/version, error map, and source availability/SLA expectation. |
| Evidence | Canonical content hash and source version for every record; no untracked transformation or model run. |
| Write-back | Separate approved candidate layer or metadata field; dry-run report; human MFA approval; stable integration ID/idempotency key; returned destination result capture; compensating action. |
| Operations | Connector health, retry outcome, token-expiry event, schema-drift event, source outage state, SIEM/audit event, and credential revocation procedure. |

## API Versioning and Authentication

The governed integration API lives under:

```text
/api/innovations/integration_hub
```

The P0 endpoint contract uses `X-API-Key` scoped keys for service-to-service sandbox access. Production deployments should pair this with the platform’s OIDC/OPA identity controls and replace broad hub keys with tenant-specific credentials managed by the customer’s secret store. The existing hub exposes `read` and `write` scopes; deployment policy must grant only the required scope.

All timestamps are ISO-8601. Geometry is a GeoJSON-like JSON object paired with a documented CRS in either the source payload or a tenant mapping. Client implementations must treat unknown fields as forward-compatible and must fail visibly on a schema version they cannot process.

## Endpoint Contract

### 1. Register Source Evidence

```http
POST /api/innovations/integration_hub/evidence
X-API-Key: mvk_<id>.<secret>
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
    "artifact_hash": "<sha256>",
    "confidence": 0.81,
    "uncertainty": "review-required"
  }
}
```

A successful `201` returns an `evidence_id` and `lineage_hash`. The client must store both. The endpoint validates the source-system identifier, requires all source/tenant/time fields, and hashes a canonical representation. It **does not** write to ArcGIS, Seequent, or any other source system.

### 2. Discover ArcGIS Capability Contract

```http
POST /api/innovations/integration_hub/arcgis/capabilities
X-API-Key: mvk_<id>.<secret>
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

The response is a conservative capability map. The P0 endpoint never fetches a caller-supplied URL; the deployed connector worker must fetch ArcGIS metadata only through an approved service allow-list. ArcGIS Feature Service metadata advertises supported query, editing, synchronization, upload, and related capabilities, so this discovery result is a prerequisite for all connector behavior. [1]

### 3. Stage a Candidate Write-Back

```http
POST /api/innovations/integration_hub/writebacks
X-API-Key: mvk_<id>.<secret>
Content-Type: application/json
```

```json
{
  "tenant_id": "northstar-mining",
  "evidence_id": "ev_<id>",
  "target_system": "arcgis_enterprise",
  "target_ref": "https://gis.example.com/arcgis/rest/services/MV_Candidates/FeatureServer/0",
  "candidate_payload": {
    "attributes": {"mv_status": "candidate", "mv_evidence_id": "ev_<id>"},
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

The result is `state: "staged"`. The endpoint verifies that the evidence record is in the named tenant and stores a deterministic request hash. It does not make an external call.

### 4. Approve a Staged Proposal

```http
POST /api/innovations/integration_hub/writebacks/{proposal_id}/approve
X-API-Key: mvk_<id>.<secret>
Content-Type: application/json
```

```json
{
  "tenant_id": "northstar-mining",
  "reviewer_id": "reviewer@northstar.example",
  "mfa_verified": true,
  "review_reason": "Geometry, source lineage, and candidate-layer diff reviewed."
}
```

Approval requires MFA evidence, a nonempty reason, a proposal in `staged` state, and a reviewer distinct from `submitted_by`. The connector worker may process only `approved` proposals. It must repeat destination capability/schema/CRS validation immediately before a write and record the returned ArcGIS object/global ID or full failure result. ArcGIS editing responses expose per-feature success/error information that should become part of this destination record. [2]

## Seequent/Evo Mapping

Seequent’s Geoscience Object API describes object-level integration via common geoscience objects referenced by UUID or user-defined object path. [3] The P0 Seequent mapping therefore uses `source_system: "seequent_evo"`, `source_ref` set to the source UUID/path, and `source_version` set to a customer-authorized object version/checkpoint. MineralVision stores a link and review package; it does not alter the model artifact.

| MineralVision field | Seequent/Evo P0 mapping |
|---|---|
| `source_ref` | Geoscience Object UUID or user-defined object path |
| `source_version` | Customer-approved object revision/checkpoint or ingestion cursor |
| `payload` | Permitted metadata/measurements only; no proprietary artifact copied without authorization |
| `geometry` | Representative geometry or source spatial reference where permitted |
| `model_run` | MineralVision candidate model identity; separate from incumbent geological interpretation |
| destination | Hyperlink/object reference or approved metadata association, not direct mutation of authoritative geological model |

## Write-Back Worker Contract

The worker is deliberately not included in the general API request path. It must accept only an approved proposal ID and execute this sequence:

1. Reload the proposal and evidence by tenant; require `state == "approved"`, reviewer MFA evidence, and immutable request hash.
2. Verify source and destination allow-list, credential scope, service metadata/capabilities, destination candidate-layer policy, schema, CRS, geometry, and stable integration ID.
3. Record an outbound attempt audit event before delivery.
4. Call the approved destination operation with idempotency/global-ID semantics where supported.
5. Persist the exact destination outcome: success/error, returned object/global ID, response hash, timestamp, retry count, and connector version.
6. Never retry a mutation without reconciling the prior destination result through the stable integration ID.
7. Emit a tenant-scoped audit/SIEM event and present the final status to the reviewer.

## References

[1] [ArcGIS Enterprise Feature Service REST API](https://developers.arcgis.com/rest/services-reference/enterprise/feature-service/)

[2] [ArcGIS Enterprise Add Features REST API](https://developers.arcgis.com/rest/services-reference/enterprise/add-features/)

[3] [Seequent Geoscience Object API](https://developer.seequent.com/docs/api/geoscience-object/geoscience-object-api)
