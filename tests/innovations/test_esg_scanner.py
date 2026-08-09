"""Deterministic tests for the esg_scanner innovation (B4-14)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.innovations.esg_scanner import router
from api.innovations.esg_scanner.logic import (
    OPS,
    SEVERITIES,
    ESGScanner,
    evaluate_check,
    load_rulepacks,
    resolve_field,
)

# Fully-compliant project data satisfying every rule in every pack.
COMPLIANT_PROJECT = {
    "water": {
        "discharge_permit": "LIC-1234",
        "monitoring_data_recorded": True,
        "tss_mg_l": 32.5,
        "discharge_ph": 7.1,
        "groundwater_baseline_study": "GW-BASELINE-2023",
        "annual_consumption_m3": 145000,
    },
    "emissions": {
        "scope1_tco2e": 12500,
        "scope2_tco2e": 3200,
        "pm10_ug_m3": 41,
        "reduction_target_adopted": True,
        "nox_compliant": True,
        "ods_register": "ODS-REG-2024",
    },
    "tailings": {
        "design_report": "TSF-DESIGN-REV-C",
        "engineer_of_record": "J. Engineer, PE",
        "last_independent_review_years": 2,
        "emergency_preparedness_plan": "EPRP-2024",
        "freeboard_above_minimum": True,
        "instrumentation_review_quarterly": True,
    },
    "rehabilitation": {
        "closure_plan": "MCP-2024",
        "closure_cost_estimate": {"total": 4200000, "currency": "AUD"},
        "provisioning_coverage_ratio": 1.15,
        "progressive_rehab_pct": 45,
        "topsoil_inventory_m3": 88000,
        "biodiversity_plan": "BMP-2024",
    },
    "community": {
        "consent_agreement": "LUAA-2022-014",
        "stakeholder_engagement_plan": "SEP-2024",
        "grievance_mechanism_active": True,
        "grievances_open_over_90d": 0,
        "heritage_survey_completed": True,
        "local_employment_pct": 34,
    },
}


class TestRulePacks:
    def test_five_categories_with_at_least_five_rules_each(self):
        packs = load_rulepacks()
        assert set(packs) == {"water", "emissions", "tailings", "rehabilitation", "community"}
        for category, rules in packs.items():
            assert len(rules) >= 5, f"{category} has only {len(rules)} rules"
            for rule in rules:
                assert rule.severity in SEVERITIES
                assert rule.op in OPS
                assert rule.remediation.strip()
                assert rule.framework_ref.strip()

    def test_rule_ids_unique(self):
        packs = load_rulepacks()
        ids = [r.id for rules in packs.values() for r in rules]
        assert len(ids) == len(set(ids))


class TestCheckOps:
    @pytest.mark.parametrize("op,observed,value,expected", [
        ("present", "x", None, True),
        ("present", "", None, False),
        ("lt", 4, 5, True),
        ("lt", 5, 5, False),
        ("lte", 50, 50, True),
        ("gte", 1.0, 1.0, True),
        ("between", 6.0, [6.0, 9.0], True),
        ("between", 9.5, [6.0, 9.0], False),
        ("is_true", True, None, True),
        ("is_true", 1, None, False),
        ("is_false", False, None, True),
        ("not_equals", "b", "a", True),
    ])
    def test_ops_matrix(self, op, observed, value, expected):
        from api.innovations.esg_scanner.logic import Rule
        rule = Rule("T", "t", "d", "f", op, value, "minor", "r", "ref")
        assert evaluate_check(rule, observed) is expected

    def test_missing_field_fails_value_checks_but_passes_missing_op(self):
        from api.innovations.esg_scanner.logic import Rule, _MISSING
        lte = Rule("T", "t", "d", "f", "lte", 3, "minor", "r", "ref")
        missing = Rule("T", "t", "d", "f", "missing", None, "minor", "r", "ref")
        assert evaluate_check(lte, _MISSING) is False
        assert evaluate_check(missing, _MISSING) is True

    def test_resolve_dotted_path(self):
        data = {"a": {"b": {"c": 42}}}
        assert resolve_field(data, "a.b.c") == 42
        assert resolve_field(data, "a.b.zzz") is not 42


class TestScanner:
    def test_compliant_project_has_zero_gaps(self):
        result = ESGScanner().scan(COMPLIANT_PROJECT)
        assert result.rules_evaluated == 30
        assert result.rules_passed == 30
        assert result.gaps == []
        assert result.compliant is True
        assert result.severity_counts() == {"critical": 0, "major": 0, "minor": 0}

    def test_planted_gaps_detected_with_correct_severity(self):
        data = {**{k: dict(v) for k, v in COMPLIANT_PROJECT.items()}}
        del data["water"]["discharge_permit"]          # WAT-001 critical
        data["water"]["discharge_ph"] = 5.2            # WAT-004 critical (outside 6.0-9.0)
        data["emissions"]["pm10_ug_m3"] = 87           # EMI-003 major
        data["community"]["grievance_mechanism_active"] = False  # COM-003 major
        data["rehabilitation"]["provisioning_coverage_ratio"] = 0.4  # REH-003 critical

        result = ESGScanner().scan(data)
        by_id = {g.rule_id: g for g in result.gaps}
        assert set(by_id) == {"WAT-001", "WAT-004", "EMI-003", "COM-003", "REH-003"}
        assert by_id["WAT-001"].severity == "critical"
        assert by_id["WAT-001"].observed == "<missing>"
        assert by_id["WAT-004"].observed == 5.2
        assert by_id["EMI-003"].severity == "major"
        assert by_id["REH-003"].severity == "critical"
        assert result.compliant is False
        assert result.severity_counts() == {"critical": 3, "major": 2, "minor": 0}
        assert result.rules_passed == 30 - 5
        # deterministic ordering: criticals first by rule id
        assert [g.rule_id for g in result.gaps[:3]] == ["REH-003", "WAT-001", "WAT-004"]
        # remediation + framework refs carried through
        assert "licence" in by_id["WAT-001"].remediation.lower()
        assert "GRI" in by_id["WAT-001"].framework_ref or "IFC" in by_id["WAT-001"].framework_ref

    def test_category_subset_scan(self):
        result = ESGScanner().scan({}, categories=["emissions"])
        assert result.rules_evaluated == 6
        assert all(g.category == "emissions" for g in result.gaps)
        with pytest.raises(ValueError):
            ESGScanner().scan({}, categories=["agriculture"])

    def test_empty_data_flags_every_rule(self):
        result = ESGScanner().scan({})
        assert result.rules_evaluated == 30
        assert len(result.gaps) == 30
        assert result.compliant is False


class TestAPI:
    @pytest.fixture()
    def client(self):
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_scan_endpoint(self, client):
        resp = client.post("/innovations/esg_scanner/scan", json={"project_data": COMPLIANT_PROJECT})
        assert resp.status_code == 200
        body = resp.json()
        assert body["compliant"] is True
        assert body["rules_evaluated"] == 30
        assert body["gaps"] == []

        bad = {"water": {"discharge_ph": 4.0}}
        resp = client.post("/innovations/esg_scanner/scan", json={"project_data": bad})
        body = resp.json()
        assert body["compliant"] is False
        ids = [g["rule_id"] for g in body["gaps"]]
        assert "WAT-004" in ids and "TAF-001" in ids

    def test_rules_endpoint_lists_packs(self, client):
        resp = client.get("/innovations/esg_scanner/rules")
        assert resp.status_code == 200
        packs = resp.json()
        assert set(packs) == {"water", "emissions", "tailings", "rehabilitation", "community"}
        assert all(len(rules) >= 5 for rules in packs.values())
