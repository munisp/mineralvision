"""Jurisdiction templates for regulatory tenement submissions.

Each template declares named sections.  A section maps a key in the posted
project data to a file in the bundle via a renderer.  ``required`` sections
must be present and non-empty for validation to pass.
"""

import csv
import io
import json
from typing import Any, Dict, List


def _json_renderer(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _csv_renderer(columns: List[str]):
    def render(rows: List[Dict[str, Any]]) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: str(r.get(columns[0], ""))):
            writer.writerow(row)
        return buf.getvalue()

    return render


DRILLHOLE_COLUMNS = ["hole_id", "east", "north", "elevation", "depth", "dip", "azimuth"]
ASSAY_QAQC_COLUMNS = ["sample_id", "hole_id", "from_m", "to_m", "commodity", "value", "qaqc_flag"]
EXPENDITURE_COLUMNS = ["category", "description", "amount", "currency"]

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "wa_dmirs_annual": {
        "title": "WA DMIRS-style Annual Mineral Exploration Report",
        "sections": [
            {
                "key": "tenement_details",
                "filename": "tenement_details.json",
                "required": True,
                "renderer": _json_renderer,
                "description": "Tenement identifiers, holders, grant/expiry dates",
            },
            {
                "key": "activities_summary",
                "filename": "activities_summary.json",
                "required": True,
                "renderer": _json_renderer,
                "description": "Summary of exploration activities for the reporting year",
            },
            {
                "key": "drillholes",
                "filename": "drillholes.csv",
                "required": True,
                "renderer": _csv_renderer(DRILLHOLE_COLUMNS),
                "description": "Drillhole collar export",
            },
            {
                "key": "assay_qaqc",
                "filename": "assay_qaqc.csv",
                "required": True,
                "renderer": _csv_renderer(ASSAY_QAQC_COLUMNS),
                "description": "Assay results with QAQC flags",
            },
            {
                "key": "expenditure",
                "filename": "expenditure.csv",
                "required": True,
                "renderer": _csv_renderer(EXPENDITURE_COLUMNS),
                "description": "Expenditure statement by category",
            },
            {
                "key": "environmental_statement",
                "filename": "environmental_statement.json",
                "required": True,
                "renderer": _json_renderer,
                "description": "Environmental compliance and rehabilitation statement (WA-specific)",
            },
        ],
    },
    "generic": {
        "title": "Generic Tenement Annual Report",
        "sections": [
            {
                "key": "tenement_details",
                "filename": "tenement_details.json",
                "required": True,
                "renderer": _json_renderer,
                "description": "Tenement identifiers, holders, grant/expiry dates",
            },
            {
                "key": "activities_summary",
                "filename": "activities_summary.json",
                "required": True,
                "renderer": _json_renderer,
                "description": "Summary of exploration activities for the reporting year",
            },
            {
                "key": "drillholes",
                "filename": "drillholes.csv",
                "required": True,
                "renderer": _csv_renderer(DRILLHOLE_COLUMNS),
                "description": "Drillhole collar export",
            },
            {
                "key": "expenditure",
                "filename": "expenditure.csv",
                "required": True,
                "renderer": _csv_renderer(EXPENDITURE_COLUMNS),
                "description": "Expenditure statement by category",
            },
            {
                "key": "assay_qaqc",
                "filename": "assay_qaqc.csv",
                "required": False,
                "renderer": _csv_renderer(ASSAY_QAQC_COLUMNS),
                "description": "Assay results with QAQC flags (optional)",
            },
        ],
    },
}
