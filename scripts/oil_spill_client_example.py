#!/usr/bin/env python3
"""Exercise MineralVision oil-spill endpoints with versioned mask evidence.

Example:
  export MINERALVISION_API_URL=https://mineralvision.example
  export MINERALVISION_TOKEN=REPLACE_WITH_SHORT_LIVED_JWT
  python scripts/oil_spill_client_example.py --mask sample_probability_mask.png \
      --model-id oil-segmentation-aerial --model-version 2026.08.20 --gsd 0.15

The script does not create a segmentation model or bypass the model-governance gate.
It submits an independently generated probability mask and optionally records a human review,
requests an advisory coverage plan, and writes a GeoJSON evidence export.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


def api_request(api_url: str, token: str, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        f"{api_url.rstrip('/')}{path}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=30) as response:
            decoded = response.read().decode("utf-8")
            return json.loads(decoded) if decoded else {}
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Unable to reach {api_url}: {error.reason}") from error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="PNG probability mask encoded as 0–255 grayscale")
    parser.add_argument("--api-url", default=os.getenv("MINERALVISION_API_URL"), help="API base URL (or MINERALVISION_API_URL)")
    parser.add_argument("--token", default=os.getenv("MINERALVISION_TOKEN"), help="JWT token (or MINERALVISION_TOKEN)")
    parser.add_argument("--source", default="drone_rgb", choices=["drone_rgb", "satellite_rgb", "fluorosensor", "manual_annotation"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--gsd", type=float, default=None, help="Ground sampling distance in metres/pixel")
    parser.add_argument("--bounds", default=None, help="Optional JSON object: {west,south,east,north}")
    parser.add_argument("--review", choices=["confirmed", "needs_resurvey", "false_positive"], default=None)
    parser.add_argument("--reviewer", default="api_client_operator")
    parser.add_argument("--coverage", action="store_true", help="Request an advisory coverage plan after assessment")
    parser.add_argument("--geojson-output", type=Path, default=None, help="Write a GeoJSON evidence export")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.api_url or not args.token:
        raise SystemExit("--api-url/MINERALVISION_API_URL and --token/MINERALVISION_TOKEN are required")
    if not args.mask.is_file():
        raise SystemExit(f"Mask file does not exist: {args.mask}")

    with Image.open(args.mask) as image:
        width, height = image.size
    mask_base64 = base64.b64encode(args.mask.read_bytes()).decode("ascii")
    bounds = json.loads(args.bounds) if args.bounds else None

    incident = api_request(
        args.api_url,
        args.token,
        "POST",
        "/api/oil-spill/analyze/mask",
        {
            "mask_base64": mask_base64,
            "image_width_px": width,
            "image_height_px": height,
            "source": args.source,
            "model_id": args.model_id,
            "model_version": args.model_version,
            "ground_sampling_distance_m": args.gsd,
            "geographic_bounds": bounds,
            "metadata": {"client": "scripts/oil_spill_client_example.py", "mask_file": args.mask.name},
        },
    )
    incident_id = incident["incident_id"]
    print(json.dumps({"assessment": incident}, indent=2))

    if args.review:
        reviewed = api_request(
            args.api_url,
            args.token,
            "PATCH",
            f"/api/oil-spill/incidents/{incident_id}/review",
            {"status": args.review, "reviewer": args.reviewer, "note": "Recorded by sample client."},
        )
        print(json.dumps({"review": reviewed}, indent=2))

    if args.coverage:
        plan = api_request(
            args.api_url,
            args.token,
            "POST",
            f"/api/oil-spill/incidents/{incident_id}/coverage-plan",
            {"cell_size_m": 50, "drone_count": 2, "buffer_m": 100},
        )
        print(json.dumps({"coverage_plan": plan}, indent=2))

    if args.geojson_output:
        export = api_request(args.api_url, args.token, "GET", f"/api/oil-spill/incidents/{incident_id}/export.geojson")
        args.geojson_output.write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"Wrote GeoJSON evidence to {args.geojson_output}")

    print(f"Completed without triggering external operational actions. Incident: {incident_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
