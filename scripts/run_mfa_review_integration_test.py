#!/usr/bin/env python3
"""Exercise the deployed MFA-gated oil-spill review workflow end to end.

The script never creates or forges identity tokens. Obtain three short-lived tokens
from the configured Keycloak test realm and supply them by environment variable:

* OPERATOR_TOKEN:  `oil_spill_operator`; used only to create a controlled mask incident.
* REVIEWER_NO_MFA_TOKEN: `oil_spill_reviewer` but no accepted `amr`/`acr` MFA claim.
* REVIEWER_MFA_TOKEN: `oil_spill_reviewer` with an accepted MFA assurance claim.

A passing execution proves both deny and allow behavior through the deployed request
path: Caddy -> APISIX -> FastAPI OIDC -> OPA -> oil-spill review endpoint. It does
not test volumetric DDoS capacity, WAF learning quality, or identity-provider security.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

CONTROLLED_MASK_BASE64 = "AP//AA=="  # Four 8-bit pixels: 0,255,255,0.


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=os.getenv("MINERALVISION_API_URL", "https://app.example.com"))
    parser.add_argument("--reviewer", default=os.getenv("MFA_REVIEWER_ID", "integration-reviewer"))
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mfa_review_integration_report.json"))
    parser.add_argument("--keep-incident", action="store_true", help="Record the controlled incident id in the output; cleanup is manual and audited.")
    return parser.parse_args()


def token(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required; obtain a short-lived test token from Keycloak")
    return value


def request(method: str, url: str, bearer_token: str, *, payload: dict[str, Any] | None, timeout: float) -> requests.Response:
    return requests.request(
        method,
        url,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=timeout,
    )


def expect(response: requests.Response, expected_status: int, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"body_excerpt": response.text[:500]}
    if response.status_code != expected_status:
        raise RuntimeError(
            f"{context}: expected HTTP {expected_status}, received {response.status_code}; response={payload}"
        )
    return payload


def main() -> int:
    options = args()
    base_url = options.api_url.rstrip("/")
    operator = token("OPERATOR_TOKEN")
    reviewer_without_mfa = token("REVIEWER_NO_MFA_TOKEN")
    reviewer_with_mfa = token("REVIEWER_MFA_TOKEN")

    report: dict[str, Any] = {
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "api_url": base_url,
        "controls_tested": ["route reachability", "OIDC authentication", "OPA role permission", "OPA MFA gate", "review persistence"],
    }

    # A four-pixel deterministic annotation creates an auditable pending-review
    # incident without raw imagery, model execution, or a real response action.
    create = request(
        "POST",
        f"{base_url}/api/oil-spill/analyze/mask",
        operator,
        payload={
            "mask_base64": CONTROLLED_MASK_BASE64,
            "image_width_px": 2,
            "image_height_px": 2,
            "source": "manual_annotation",
            "model_id": "integration-test-annotation",
            "model_version": "non-production",
            "image_id": f"mfa-policy-test-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            "probability_threshold": 0.5,
            "min_component_area_px": 1,
            "metadata": {
                "test_only": True,
                "purpose": "OPA MFA authorization integration test",
                "no_operational_response": True,
            },
        },
        timeout=options.timeout,
    )
    created = expect(create, 201, "controlled incident creation")
    incident_id = created.get("incident_id")
    if not incident_id or created.get("review_status") != "pending_review":
        raise RuntimeError(f"controlled incident is malformed: {created}")
    report["incident_id"] = incident_id
    report["creation_status"] = created.get("review_status")

    review_payload = {
        "status": "confirmed",
        "reviewer": options.reviewer,
        "note": "Controlled MFA authorization integration test; no operational action authorized.",
    }

    # Negative control: a valid reviewer role that lacks assurance must never be
    # able to transition the review state.
    denied = request(
        "PATCH",
        f"{base_url}/api/oil-spill/incidents/{incident_id}/review",
        reviewer_without_mfa,
        payload=review_payload,
        timeout=options.timeout,
    )
    denied_payload = expect(denied, 403, "reviewer without MFA")
    report["without_mfa"] = {"status": denied.status_code, "reason": denied_payload.get("reason")}

    # Positive control: the Keycloak token must carry both reviewer role and a
    # verified MFA claim recognized by the API OIDC validator/OPA policy.
    allowed = request(
        "PATCH",
        f"{base_url}/api/oil-spill/incidents/{incident_id}/review",
        reviewer_with_mfa,
        payload=review_payload,
        timeout=options.timeout,
    )
    confirmed = expect(allowed, 200, "reviewer with MFA")
    if confirmed.get("review_status") != "confirmed":
        raise RuntimeError(f"MFA-authorized review did not confirm the incident: {confirmed}")
    report["with_mfa"] = {"status": allowed.status_code, "review_status": confirmed.get("review_status")}

    # Read-after-write confirms the persistent state via a policy-authorized GET.
    retrieved = request(
        "GET",
        f"{base_url}/api/oil-spill/incidents/{incident_id}",
        reviewer_with_mfa,
        payload=None,
        timeout=options.timeout,
    )
    persisted = expect(retrieved, 200, "review persistence readback")
    if persisted.get("review_status") != "confirmed":
        raise RuntimeError(f"review persistence check failed: {persisted}")
    report["readback"] = {"status": retrieved.status_code, "review_status": persisted.get("review_status")}
    report["result"] = "passed"
    if not options.keep_incident:
        report["cleanup_note"] = "The controlled incident is intentionally retained as an auditable record; remove it through an approved retention workflow, not by database mutation."

    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, RuntimeError) as exc:
        print(f"MFA review integration test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
