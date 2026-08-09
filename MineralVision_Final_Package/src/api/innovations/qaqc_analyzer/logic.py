"""
QAQC anomaly detection — pure logic (numpy only).

Implements industry-standard assay QA/QC statistics:

* **Standards (CRMs)** — Shewhart control charts against the CRM certified
  value: |z| > 2 -> warning, |z| > 3 -> failure, plus bias % and Western
  Electric style run rules (2 consecutive points beyond 2SD on the same side;
  N consecutive points on the same side of the certified mean).
* **Blanks** — contamination flag when a blank assays above
  ``blank_multiplier`` (default 5) times the detection limit.
* **Field duplicates** — HARD (half absolute relative difference) per pair,
  ranked descending, and Thompson-Howarth precision data: for each pair the
  coefficient of variation CV = |x1 - x2| / (mean * sqrt(2)) * 100 % plotted
  against the pair mean (sorted by mean), which is the standard
  Thompson-Howarth precision-plot input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Sequence

import numpy as np


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class RowType(str, Enum):
    STANDARD = "standard"
    BLANK = "blank"
    DUPLICATE = "duplicate"
    SAMPLE = "sample"


@dataclass
class StandardResult:
    sample_id: str
    crm_id: str
    value: float
    expected_mean: float
    expected_sd: float
    z_score: float
    bias_pct: float
    status: str  # "pass" | "warn" | "fail"


@dataclass
class StandardAnalysis:
    crm_id: str
    n: int
    results: List[StandardResult]
    n_warnings: int
    n_failures: int
    mean_bias_pct: float
    run_rule_violations: List[str]

    @property
    def verdict(self) -> Verdict:
        if self.n_failures > 0 or self.run_rule_violations:
            return Verdict.FAIL
        if self.n_warnings > 0:
            return Verdict.WARN
        return Verdict.PASS


@dataclass
class BlankResult:
    sample_id: str
    value: float
    detection_limit: float
    multiple_of_dl: float
    contaminated: bool


@dataclass
class DuplicatePair:
    pair_id: str
    original_id: str
    duplicate_id: str
    original_value: float
    duplicate_value: float
    hard_pct: float          # half absolute relative difference, %
    mean: float
    cv_pct: float            # Thompson-Howarth pair CV, %


def analyze_standards(
    rows: Sequence[Dict[str, Any]],
    crm_library: Dict[str, Dict[str, float]],
    run_rule_consecutive_2sd: int = 2,
    run_rule_one_side: int = 8,
) -> List[StandardAnalysis]:
    """Control-chart analysis of CRM standard assays.

    ``crm_library`` maps crm_id -> {"mean": certified value, "sd": certified SD}.
    Rows with an unknown crm_id are skipped (reported as no-data by omission).
    Run rules: ``run_rule_consecutive_2sd`` consecutive points beyond +2SD or
    -2SD on the same side; ``run_rule_one_side`` consecutive points on one side
    of the certified mean.
    """
    by_crm: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        crm_id = str(r.get("crm_id", ""))
        if crm_id in crm_library:
            by_crm.setdefault(crm_id, []).append(r)

    out: List[StandardAnalysis] = []
    for crm_id, crm_rows in sorted(by_crm.items()):
        expected_mean = float(crm_library[crm_id]["mean"])
        expected_sd = float(crm_library[crm_id]["sd"])
        if expected_sd <= 0:
            raise ValueError(f"CRM {crm_id}: expected sd must be positive")

        results: List[StandardResult] = []
        for r in crm_rows:
            v = float(r["value"])
            z = (v - expected_mean) / expected_sd
            status = "pass"
            if abs(z) > 3.0:
                status = "fail"
            elif abs(z) > 2.0:
                status = "warn"
            results.append(StandardResult(
                sample_id=str(r.get("sample_id", "")),
                crm_id=crm_id,
                value=v,
                expected_mean=expected_mean,
                expected_sd=expected_sd,
                z_score=z,
                bias_pct=100.0 * (v - expected_mean) / expected_mean,
                status=status,
            ))

        n_warn = sum(1 for r in results if r.status == "warn")
        n_fail = sum(1 for r in results if r.status == "fail")
        mean_bias = float(np.mean([r.bias_pct for r in results]))

        violations: List[str] = []
        zs = [r.z_score for r in results]
        # Run rule: k consecutive beyond +/-2SD on the same side.
        for side, label in ((1.0, "+2SD"), (-1.0, "-2SD")):
            run = 0
            for idx, z in enumerate(zs):
                run = run + 1 if side * z > 2.0 else 0
                if run >= run_rule_consecutive_2sd:
                    violations.append(
                        f"{run_rule_consecutive_2sd} consecutive results beyond "
                        f"{label} (ending at result {idx})"
                    )
                    break
        # Run rule: n consecutive on one side of the mean.
        for side, label in ((1.0, "above"), (-1.0, "below")):
            run = 0
            for idx, z in enumerate(zs):
                run = run + 1 if side * z > 0 else 0
                if run >= run_rule_one_side:
                    violations.append(
                        f"{run_rule_one_side} consecutive results {label} the "
                        f"certified mean (bias drift, ending at result {idx})"
                    )
                    break

        out.append(StandardAnalysis(
            crm_id=crm_id,
            n=len(results),
            results=results,
            n_warnings=n_warn,
            n_failures=n_fail,
            mean_bias_pct=mean_bias,
            run_rule_violations=violations,
        ))
    return out


def analyze_blanks(
    rows: Sequence[Dict[str, Any]],
    detection_limit: float,
    multiplier: float = 5.0,
) -> List[BlankResult]:
    """Flag blank contamination: value > multiplier x detection limit."""
    if detection_limit <= 0:
        raise ValueError("detection_limit must be positive")
    out = []
    for r in rows:
        v = float(r["value"])
        dl = float(r.get("detection_limit", detection_limit))
        mult = v / dl if dl > 0 else math.inf
        out.append(BlankResult(
            sample_id=str(r.get("sample_id", "")),
            value=v,
            detection_limit=dl,
            multiple_of_dl=mult,
            contaminated=mult > multiplier,
        ))
    return out


def _pair_stats(a: float, b: float):
    mean = (a + b) / 2.0
    if mean == 0:
        return 0.0, 0.0, 0.0
    hard = 0.5 * abs(a - b) / mean * 100.0
    # Thompson-Howarth pair CV: sd/mean with sd = |a-b|/sqrt(2)
    cv = abs(a - b) / (mean * math.sqrt(2.0)) * 100.0
    return hard, cv, mean


def analyze_duplicates(
    pairs: Sequence[Dict[str, Any]],
) -> List[DuplicatePair]:
    """HARD ranking + Thompson-Howarth per-pair precision data.

    Each pair dict: {pair_id, original_id, duplicate_id,
                     original_value, duplicate_value}.
    """
    out = []
    for p in pairs:
        a = float(p["original_value"])
        b = float(p["duplicate_value"])
        hard, cv, mean = _pair_stats(a, b)
        out.append(DuplicatePair(
            pair_id=str(p.get("pair_id", "")),
            original_id=str(p.get("original_id", "")),
            duplicate_id=str(p.get("duplicate_id", "")),
            original_value=a,
            duplicate_value=b,
            hard_pct=hard,
            mean=mean,
            cv_pct=cv,
        ))
    return out


def thompson_howarth_data(pairs: Sequence[DuplicatePair]) -> Dict[str, List[float]]:
    """Thompson-Howarth precision-plot arrays: pairs sorted by mean."""
    ordered = sorted(pairs, key=lambda p: p.mean)
    return {
        "mean": [p.mean for p in ordered],
        "cv_pct": [p.cv_pct for p in ordered],
        "pair_id": [p.pair_id for p in ordered],
    }


def hard_ranking(pairs: Sequence[DuplicatePair]) -> List[DuplicatePair]:
    """Duplicate pairs ranked by HARD (worst first)."""
    return sorted(pairs, key=lambda p: p.hard_pct, reverse=True)


def analyze_batch(
    rows: Sequence[Dict[str, Any]],
    crm_library: Dict[str, Dict[str, float]],
    detection_limit: float = 0.01,
    blank_multiplier: float = 5.0,
) -> Dict[str, Any]:
    """Full QAQC analysis of one batch of rows.

    Row schema: {sample_id, row_type, value, crm_id?, detection_limit?,
                 pair_id?}
    Duplicate pairs are formed either from explicit ``original_value`` /
    ``duplicate_value`` fields on a duplicate row, or by matching two rows
    sharing the same ``pair_id``.
    """
    standards_rows = [r for r in rows if r.get("row_type") == RowType.STANDARD.value]
    blank_rows = [r for r in rows if r.get("row_type") == RowType.BLANK.value]
    dup_rows = [r for r in rows if r.get("row_type") == RowType.DUPLICATE.value]

    # Build duplicate pairs from explicit fields or from pair_id matching.
    pairs: List[Dict[str, Any]] = []
    pair_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in dup_rows:
        if "original_value" in r and "duplicate_value" in r:
            pairs.append(r)
        elif r.get("pair_id"):
            pair_groups.setdefault(str(r["pair_id"]), []).append(r)
    for pid, members in pair_groups.items():
        if len(members) >= 2:
            members = sorted(members, key=lambda m: str(m.get("sample_id", "")))
            pairs.append({
                "pair_id": pid,
                "original_id": members[0].get("sample_id", ""),
                "duplicate_id": members[1].get("sample_id", ""),
                "original_value": members[0]["value"],
                "duplicate_value": members[1]["value"],
            })

    std_analyses = analyze_standards(standards_rows, crm_library)
    blank_results = analyze_blanks(
        blank_rows, detection_limit, blank_multiplier)
    dup_pairs = analyze_duplicates(pairs)

    failures: List[str] = []
    warnings: List[str] = []

    for sa in std_analyses:
        for r in sa.results:
            if r.status == "fail":
                failures.append(
                    f"standard {r.sample_id} ({r.crm_id}): z={r.z_score:.2f} "
                    f"beyond +/-3SD")
            elif r.status == "warn":
                warnings.append(
                    f"standard {r.sample_id} ({r.crm_id}): z={r.z_score:.2f} "
                    f"beyond +/-2SD")
        for v in sa.run_rule_violations:
            failures.append(f"standard {sa.crm_id}: run rule — {v}")

    for b in blank_results:
        if b.contaminated:
            failures.append(
                f"blank {b.sample_id}: {b.value:g} = {b.multiple_of_dl:.1f}x "
                f"detection limit (>{blank_multiplier:g}x)")

    contaminated = [b for b in blank_results if b.contaminated]
    verdict = Verdict.PASS
    if failures:
        verdict = Verdict.FAIL
    elif warnings:
        verdict = Verdict.WARN

    summary_stats = {
        "n_standards": sum(sa.n for sa in std_analyses),
        "n_blanks": len(blank_results),
        "n_duplicate_pairs": len(dup_pairs),
        "standards_within_control_pct": (
            100.0
            * sum(1 for sa in std_analyses for r in sa.results if r.status == "pass")
            / max(1, sum(sa.n for sa in std_analyses))
        ),
        "blank_contamination_events": len(contaminated),
        "duplicate_mean_hard_pct": (
            float(np.mean([p.hard_pct for p in dup_pairs])) if dup_pairs else None
        ),
    }

    return {
        "verdict": verdict.value,
        "failures": failures,
        "warnings": warnings,
        "standards": [
            {
                "crm_id": sa.crm_id,
                "n": sa.n,
                "n_warnings": sa.n_warnings,
                "n_failures": sa.n_failures,
                "mean_bias_pct": sa.mean_bias_pct,
                "run_rule_violations": sa.run_rule_violations,
                "verdict": sa.verdict.value,
                "results": [r.__dict__ for r in sa.results],
            }
            for sa in std_analyses
        ],
        "blanks": [b.__dict__ for b in blank_results],
        "duplicates": {
            "hard_ranking": [p.__dict__ for p in hard_ranking(dup_pairs)],
            "thompson_howarth": thompson_howarth_data(dup_pairs),
        },
        "summary_stats": summary_stats,
    }
