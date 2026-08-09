"""
JORC / NI 43-101 resource classification & report engine — pure logic.

Classification approach
-----------------------
Each block is classified Measured / Indicated / Inferred from two geometric
criteria evaluated against the drill-sample set:

* ``nearest`` — anisotropic (search-ellipsoid scaled) distance from the block
  centroid to the nearest drill sample, expressed as a fraction of the
  variogram range.
* ``n_samples`` — number of drill samples falling inside the search ellipsoid
  centred on the block.

Default thresholds (configurable; see :func:`default_rules`) follow common
JORC 2012 practice.  JORC 2012 is principles-based (Table 1: classification
reflects confidence in geological and grade continuity), so no universal
spacing is mandated; the defaults below mirror widely used industry guidance
for well-understood deposits:

* Measured  : nearest sample <= 0.25 x variogram range, >= 4 samples in ellipse
* Indicated : nearest sample <= 0.50 x variogram range, >= 3 samples in ellipse
* Inferred  : nearest sample <= 1.00 x variogram range, >= 2 samples in ellipse

Blocks failing all rules are reported as UNCLASSIFIED (not a JORC category —
excluded from any public resource statement).

Code reuse: the anisotropic search-ellipsoid geometry is reused from the
existing geostatistics core (``api.geostatistics.kriging.SearchEllipsoid``);
only numpy is used beyond that.  pandas-free by design.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from api.geostatistics.kriging import SearchEllipsoid


class ResourceClass(str, Enum):
    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"
    UNCLASSIFIED = "unclassified"


_CLASS_RANK = {
    ResourceClass.MEASURED: 3,
    ResourceClass.INDICATED: 2,
    ResourceClass.INFERRED: 1,
    ResourceClass.UNCLASSIFIED: 0,
}


@dataclass
class ClassificationRule:
    """Single class rule: max nearest-sample distance (fraction of variogram
    range) and minimum samples within the search ellipsoid."""
    max_range_fraction: float
    min_samples: int


@dataclass
class ClassificationRules:
    measured: ClassificationRule = field(
        default_factory=lambda: ClassificationRule(0.25, 4))
    indicated: ClassificationRule = field(
        default_factory=lambda: ClassificationRule(0.50, 3))
    inferred: ClassificationRule = field(
        default_factory=lambda: ClassificationRule(1.00, 2))


def default_rules() -> ClassificationRules:
    """Default JORC-2012-practice classification rules (see module docstring)."""
    return ClassificationRules()


@dataclass
class BlockClassification:
    x: float
    y: float
    z: float
    grade: float
    density: float
    resource_class: ResourceClass
    nearest_sample_distance: float
    n_samples_in_ellipse: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resource_class"] = self.resource_class.value
        return d


def _as_blocks_array(blocks: Sequence[Dict[str, float]]) -> np.ndarray:
    rows = []
    for b in blocks:
        rows.append((
            float(b["x"]), float(b["y"]), float(b["z"]),
            float(b["grade"]), float(b.get("density", 2.7)),
        ))
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError("blocks must yield an (N, 5) array")
    return arr


def _as_samples_array(samples: Sequence[Dict[str, float]]) -> np.ndarray:
    rows = [(float(s["x"]), float(s["y"]), float(s["z"])) for s in samples]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("samples must yield an (M, 3) array")
    return arr


def classify_blocks(
    blocks: Sequence[Dict[str, float]],
    samples: Sequence[Dict[str, float]],
    variogram_range: float,
    search: Optional[SearchEllipsoid] = None,
    rules: Optional[ClassificationRules] = None,
) -> List[BlockClassification]:
    """Classify every block Measured/Indicated/Inferred/Unclassified.

    Parameters
    ----------
    blocks : sequence of dicts with keys x, y, z, grade[, density]
    samples : sequence of dicts with keys x, y, z (drill composite points)
    variogram_range : float
        Practical variogram range (length units) used to normalise distances.
    search : SearchEllipsoid, optional
        Defaults to an isotropic sphere of radius ``variogram_range``.
    rules : ClassificationRules, optional
        Defaults to :func:`default_rules`.
    """
    if variogram_range <= 0:
        raise ValueError("variogram_range must be positive")
    if rules is None:
        rules = default_rules()
    if search is None:
        search = SearchEllipsoid(variogram_range, variogram_range, variogram_range)

    b = _as_blocks_array(blocks)
    s = _as_samples_array(samples)
    if len(s) == 0:
        raise ValueError("at least one drill sample is required")

    # scaled (ellipsoid) distances: (n_blocks, n_samples); per-sample loop
    # keeps memory modest and math explicit.
    scaled = np.empty((len(b), len(s)), dtype=float)
    for j in range(len(s)):
        dxyz = b[:, :3] - s[j]
        # vectorise the isotropic fast path; rotated path loops per block.
        if search.azimuth == 0.0 and search.dip == 0.0:
            r = np.sqrt(
                (dxyz[:, 0] / search.radius_major) ** 2
                + (dxyz[:, 1] / search.radius_minor) ** 2
                + (dxyz[:, 2] / search.radius_vertical) ** 2
            ) * search.radius_major
        else:
            r = np.array([search.scaled_distance(*d) for d in dxyz])
        scaled[:, j] = r

    nearest = scaled.min(axis=1)
    n_in_ellipse = (scaled <= search.radius_major).sum(axis=1)

    ordered = (
        (ResourceClass.MEASURED, rules.measured),
        (ResourceClass.INDICATED, rules.indicated),
        (ResourceClass.INFERRED, rules.inferred),
    )

    out: List[BlockClassification] = []
    for i in range(len(b)):
        cls = ResourceClass.UNCLASSIFIED
        for klass, rule in ordered:
            if (nearest[i] <= rule.max_range_fraction * variogram_range
                    and n_in_ellipse[i] >= rule.min_samples):
                cls = klass
                break
        out.append(BlockClassification(
            x=float(b[i, 0]), y=float(b[i, 1]), z=float(b[i, 2]),
            grade=float(b[i, 3]), density=float(b[i, 4]),
            resource_class=cls,
            nearest_sample_distance=float(nearest[i]),
            n_samples_in_ellipse=int(n_in_ellipse[i]),
        ))
    return out


def summarize_by_class(
    classified: Sequence[BlockClassification],
    block_volume: float,
) -> Dict[str, Dict[str, float]]:
    """Per-class block count, tonnage and tonnage-weighted average grade.

    ``block_volume`` is the (uniform) parent-block volume in cubic units;
    tonnage = volume x density.
    """
    if block_volume <= 0:
        raise ValueError("block_volume must be positive")
    acc: Dict[ResourceClass, Dict[str, float]] = {
        k: {"n_blocks": 0, "tonnage": 0.0, "metal": 0.0}
        for k in _CLASS_RANK
    }
    for c in classified:
        mass = block_volume * c.density
        a = acc[c.resource_class]
        a["n_blocks"] += 1
        a["tonnage"] += mass
        a["metal"] += mass * c.grade

    summary: Dict[str, Dict[str, float]] = {}
    for klass in (ResourceClass.MEASURED, ResourceClass.INDICATED,
                  ResourceClass.INFERRED, ResourceClass.UNCLASSIFIED):
        a = acc[klass]
        summary[klass.value] = {
            "n_blocks": int(a["n_blocks"]),
            "tonnage": a["tonnage"],
            "avg_grade": (a["metal"] / a["tonnage"]) if a["tonnage"] > 0 else 0.0,
            "metal_content": a["metal"],
        }
    return summary


def build_report(
    classified: Sequence[BlockClassification],
    samples: Sequence[Dict[str, float]],
    variogram_range: float,
    search: SearchEllipsoid,
    rules: ClassificationRules,
    block_volume: float,
    qaqc_summary: Optional[Dict[str, Any]] = None,
    project_name: str = "unnamed",
    element: str = "grade",
) -> Dict[str, Any]:
    """Build the structured JORC-style report JSON.

    Sections: data summary, QAQC statement (fields carried from the caller's
    QAQC summary — placeholder fields when not supplied), estimation params,
    classification table, grade-tonnage by class.
    """
    s = _as_samples_array(samples)
    b_arr = np.array([[c.x, c.y, c.z, c.grade, c.density] for c in classified],
                     dtype=float)
    class_table = summarize_by_class(classified, block_volume)

    data_summary = {
        "project": project_name,
        "element": element,
        "n_samples": int(len(s)),
        "n_blocks": int(len(classified)),
        "block_volume": float(block_volume),
        "sample_extent": {
            "x": [float(s[:, 0].min()), float(s[:, 0].max())],
            "y": [float(s[:, 1].min()), float(s[:, 1].max())],
            "z": [float(s[:, 2].min()), float(s[:, 2].max())],
        },
        "block_grade_mean": float(b_arr[:, 3].mean()) if len(b_arr) else 0.0,
        "block_grade_std": float(b_arr[:, 3].std(ddof=1)) if len(b_arr) > 1 else 0.0,
    }

    qaqc_statement = {
        "standards_insertion_rate_pct": None,
        "standards_within_control_pct": None,
        "blank_contamination_events": None,
        "duplicate_mean_hard_pct": None,
        "comment": (
            "QAQC fields populated from qaqc_analyzer summary; nulls indicate "
            "no QAQC data supplied with this report run."
        ),
    }
    if qaqc_summary:
        for key in qaqc_statement:
            if key in qaqc_summary:
                qaqc_statement[key] = qaqc_summary[key]
        qaqc_statement["comment"] = qaqc_summary.get(
            "comment", "QAQC summary supplied by caller.")

    estimation_params = {
        "variogram_range": float(variogram_range),
        "search_ellipsoid": asdict(search),
        "classification_rules": {
            "measured": asdict(rules.measured),
            "indicated": asdict(rules.indicated),
            "inferred": asdict(rules.inferred),
        },
        "method": (
            "distance-to-nearest-sample + sample-count within search "
            "ellipsoid, distances expressed as fractions of variogram range"
        ),
        "framework": "JORC 2012 (principles-based; thresholds configurable)",
    }

    n_total = len(classified)
    classification_table = {
        klass: {
            **stats,
            "pct_of_blocks": (100.0 * stats["n_blocks"] / n_total) if n_total else 0.0,
        }
        for klass, stats in class_table.items()
    }

    return {
        "report_type": "JORC 2012 mineral resource classification",
        "project": project_name,
        "element": element,
        "data_summary": data_summary,
        "qaqc_statement": qaqc_statement,
        "estimation_params": estimation_params,
        "classification_table": classification_table,
        "grade_tonnage_by_class": class_table,
    }
