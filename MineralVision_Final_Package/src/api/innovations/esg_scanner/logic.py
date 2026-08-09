"""Rule-pack engine for ESG / environmental compliance gap scanning.

Rule packs are JSON files shipped in ``rulepacks/``.  Each rule declares a
field check (dotted-path lookup into the posted project data) plus severity,
remediation text and a framework reference (GRI/IFC/GISTM-style).  The
scanner evaluates every rule deterministically and returns the gap list —
rules whose check FAILS against the supplied data.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

RULEPACK_DIR = os.path.join(os.path.dirname(__file__), "rulepacks")

SEVERITIES = ("critical", "major", "minor")
# Order used for deterministic output: most severe first, then rule id.
_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}

OPS = ("present", "missing", "lt", "lte", "gt", "gte", "between", "equals", "not_equals", "is_true", "is_false")

_MISSING = object()


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    description: str
    check_field: str
    op: str
    value: Any
    severity: str
    remediation: str
    framework_ref: str


@dataclass(frozen=True)
class Gap:
    rule_id: str
    category: str
    severity: str
    description: str
    remediation: str
    framework_ref: str
    observed: Any  # observed value at the field path ("<missing>" when absent)


@dataclass
class ScanResult:
    gaps: List[Gap] = field(default_factory=list)
    rules_evaluated: int = 0
    rules_passed: int = 0

    @property
    def compliant(self) -> bool:
        return not any(g.severity == "critical" for g in self.gaps)

    def severity_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in SEVERITIES}
        for gap in self.gaps:
            counts[gap.severity] += 1
        return counts


def load_rulepacks(rulepack_dir: str = RULEPACK_DIR) -> Dict[str, List[Rule]]:
    """Load and validate all rule packs. Deterministic order (sorted filenames)."""
    packs: Dict[str, List[Rule]] = {}
    for fname in sorted(os.listdir(rulepack_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(rulepack_dir, fname), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        category = raw["category"]
        rules = []
        for r in raw["rules"]:
            check = r["check"]
            if check["op"] not in OPS:
                raise ValueError(f"rule {r['id']}: unknown op {check['op']!r}")
            if r["severity"] not in SEVERITIES:
                raise ValueError(f"rule {r['id']}: unknown severity {r['severity']!r}")
            rules.append(
                Rule(
                    id=r["id"],
                    category=category,
                    description=r["description"],
                    check_field=check["field"],
                    op=check["op"],
                    value=check.get("value"),
                    severity=r["severity"],
                    remediation=r["remediation"],
                    framework_ref=r["framework_ref"],
                )
            )
        packs[category] = rules
    return packs


def resolve_field(data: Dict[str, Any], dotted_path: str) -> Any:
    """Dotted-path lookup; returns the _MISSING sentinel when absent."""
    node: Any = data
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def evaluate_check(rule: Rule, observed: Any) -> bool:
    """Return True when the rule PASSES (no gap)."""
    op = rule.op
    if op == "present":
        return observed is not _MISSING and observed is not None and observed != ""
    if op == "missing":
        return observed is _MISSING or observed is None
    if observed is _MISSING or observed is None:
        return False  # every value-based check fails when data is absent
    if op == "lt":
        return observed < rule.value
    if op == "lte":
        return observed <= rule.value
    if op == "gt":
        return observed > rule.value
    if op == "gte":
        return observed >= rule.value
    if op == "between":
        low, high = rule.value
        return low <= observed <= high
    if op == "equals":
        return observed == rule.value
    if op == "not_equals":
        return observed != rule.value
    if op == "is_true":
        return observed is True
    if op == "is_false":
        return observed is False
    raise ValueError(f"unknown op {op!r}")


class ESGScanner:
    """Evaluates project data against the rule packs."""

    def __init__(self, rulepack_dir: str = RULEPACK_DIR):
        self.packs = load_rulepacks(rulepack_dir)

    def scan(self, project_data: Dict[str, Any], categories: Optional[List[str]] = None) -> ScanResult:
        result = ScanResult()
        selected = categories if categories else sorted(self.packs.keys())
        for category in selected:
            if category not in self.packs:
                raise ValueError(f"unknown rule category {category!r}; have {sorted(self.packs)}")
            for rule in self.packs[category]:
                result.rules_evaluated += 1
                observed = resolve_field(project_data, rule.check_field)
                if evaluate_check(rule, observed):
                    result.rules_passed += 1
                else:
                    result.gaps.append(
                        Gap(
                            rule_id=rule.id,
                            category=rule.category,
                            severity=rule.severity,
                            description=rule.description,
                            remediation=rule.remediation,
                            framework_ref=rule.framework_ref,
                            observed="<missing>" if observed is _MISSING else observed,
                        )
                    )
        # Deterministic ordering: severity, then rule id.
        result.gaps.sort(key=lambda g: (_SEVERITY_ORDER[g.severity], g.rule_id))
        return result


def gap_to_dict(gap: Gap) -> Dict[str, Any]:
    return {
        "rule_id": gap.rule_id,
        "category": gap.category,
        "severity": gap.severity,
        "description": gap.description,
        "remediation": gap.remediation,
        "framework_ref": gap.framework_ref,
        "observed": gap.observed,
    }


def result_to_dict(result: ScanResult) -> Dict[str, Any]:
    return {
        "compliant": result.compliant,
        "rules_evaluated": result.rules_evaluated,
        "rules_passed": result.rules_passed,
        "severity_counts": result.severity_counts(),
        "gaps": [gap_to_dict(g) for g in result.gaps],
    }
