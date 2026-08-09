"""Deterministic tests for the QAQC analyzer (planted anomalies)."""

import math

import pytest
from api.innovations.qaqc_analyzer import logic, router
from fastapi import FastAPI
from fastapi.testclient import TestClient

CRM = {"OREAS-1": {"mean": 2.00, "sd": 0.05}}


def _std(sid, value, crm="OREAS-1"):
    return {"sample_id": sid, "row_type": "standard", "value": value, "crm_id": crm}


# ---------------------------------------------------------------- logic ----

def test_control_chart_exact_z_scores():
    rows = [
        _std("S1", 2.00),                  # z = 0
        _std("S2", 2.00 + 2.5 * 0.05),     # z = +2.5 -> warn
        _std("S3", 2.00 - 3.5 * 0.05),     # z = -3.5 -> fail
    ]
    analyses = logic.analyze_standards(rows, CRM)
    assert len(analyses) == 1
    res = analyses[0].results
    assert res[0].z_score == pytest.approx(0.0) and res[0].status == "pass"
    assert res[1].z_score == pytest.approx(2.5) and res[1].status == "warn"
    assert res[2].z_score == pytest.approx(-3.5) and res[2].status == "fail"
    assert analyses[0].n_warnings == 1
    assert analyses[0].n_failures == 1
    # bias % exact: (2.125 - 2.0)/2.0 = 6.25 %
    assert res[1].bias_pct == pytest.approx(6.25)


def test_run_rule_two_consecutive_beyond_2sd():
    # Two consecutive +2.2SD results (neither fails) -> run-rule violation -> FAIL.
    rows = [_std(f"S{i}", 2.00 + 2.2 * 0.05) for i in range(2)]
    analyses = logic.analyze_standards(rows, CRM)
    assert analyses[0].n_failures == 0
    assert len(analyses[0].run_rule_violations) == 1
    assert "beyond +2SD" in analyses[0].run_rule_violations[0]
    assert analyses[0].verdict == logic.Verdict.FAIL


def test_run_rule_eight_consecutive_one_side():
    # 8 results all slightly above mean (small bias drift) -> violation.
    rows = [_std(f"S{i}", 2.00 + 0.3 * 0.05) for i in range(8)]
    analyses = logic.analyze_standards(rows, CRM)
    assert any("above the certified mean" in v
               for v in analyses[0].run_rule_violations)
    # 7 in a row -> no violation.
    rows = rows[:7]
    analyses = logic.analyze_standards(rows, CRM)
    assert analyses[0].run_rule_violations == []
    assert analyses[0].verdict == logic.Verdict.PASS


def test_mean_bias_pct():
    rows = [_std("S1", 2.02), _std("S2", 2.04)]
    analyses = logic.analyze_standards(rows, CRM)
    # biases: 1.0 % and 2.0 % -> mean 1.5 %
    assert analyses[0].mean_bias_pct == pytest.approx(1.5)


def test_blank_contamination_threshold():
    rows = [
        {"sample_id": "B1", "row_type": "blank", "value": 0.049},  # 4.9x DL ok
        {"sample_id": "B2", "row_type": "blank", "value": 0.051},  # 5.1x DL bad
    ]
    res = logic.analyze_blanks(rows, detection_limit=0.01)
    assert res[0].multiple_of_dl == pytest.approx(4.9)
    assert not res[0].contaminated
    assert res[1].multiple_of_dl == pytest.approx(5.1)
    assert res[1].contaminated


def test_duplicate_hard_and_thompson_howarth():
    pairs = [
        {"pair_id": "P1", "original_id": "A1", "duplicate_id": "A2",
         "original_value": 2.0, "duplicate_value": 2.2},
        {"pair_id": "P2", "original_id": "B1", "duplicate_id": "B2",
         "original_value": 0.5, "duplicate_value": 0.4},
    ]
    out = logic.analyze_duplicates(pairs)
    # P1: mean 2.1, HARD = 0.5*0.2/2.1*100 = 4.7619 %
    assert out[0].hard_pct == pytest.approx(0.5 * 0.2 / 2.1 * 100)
    # CV = 0.2/(2.1*sqrt(2))*100
    assert out[0].cv_pct == pytest.approx(0.2 / (2.1 * math.sqrt(2)) * 100)
    # P2: mean 0.45, HARD = 0.5*0.1/0.45*100 = 11.111 %
    assert out[1].hard_pct == pytest.approx(0.5 * 0.1 / 0.45 * 100)

    ranking = logic.hard_ranking(out)
    assert [p.pair_id for p in ranking] == ["P2", "P1"]  # worst first

    th = logic.thompson_howarth_data(out)
    assert th["mean"] == sorted(th["mean"])
    assert th["pair_id"] == ["P2", "P1"]
    assert th["cv_pct"][1] == pytest.approx(out[0].cv_pct)


def test_batch_verdicts():
    # Clean batch -> PASS
    clean = [_std("S1", 2.01), _std("S2", 1.99),
             {"sample_id": "B1", "row_type": "blank", "value": 0.02}]
    r = logic.analyze_batch(clean, CRM, detection_limit=0.01)
    assert r["verdict"] == "pass"
    assert r["failures"] == []
    assert r["summary_stats"]["standards_within_control_pct"] == pytest.approx(100.0)

    # Failed standard -> FAIL with a failures entry
    bad = clean + [_std("S3", 2.00 + 4.0 * 0.05)]
    r = logic.analyze_batch(bad, CRM, detection_limit=0.01)
    assert r["verdict"] == "fail"
    assert any("S3" in f and "3SD" in f for f in r["failures"])

    # Warning only -> WARN
    warn = clean + [_std("S4", 2.00 + 2.4 * 0.05)]
    r = logic.analyze_batch(warn, CRM, detection_limit=0.01)
    assert r["verdict"] == "warn"
    assert r["failures"] == [] and len(r["warnings"]) == 1

    # Contaminated blank -> FAIL
    contam = clean + [{"sample_id": "B9", "row_type": "blank", "value": 0.06}]
    r = logic.analyze_batch(contam, CRM, detection_limit=0.01)
    assert r["verdict"] == "fail"
    assert r["summary_stats"]["blank_contamination_events"] == 1


def test_batch_duplicate_pairing_by_pair_id():
    rows = [
        {"sample_id": "D1", "row_type": "duplicate", "value": 1.0, "pair_id": "PX"},
        {"sample_id": "D2", "row_type": "duplicate", "value": 1.1, "pair_id": "PX"},
    ]
    r = logic.analyze_batch(rows, {}, detection_limit=0.01)
    th = r["duplicates"]["thompson_howarth"]
    assert th["pair_id"] == ["PX"]
    assert th["mean"] == [pytest.approx(1.05)]
    assert r["summary_stats"]["duplicate_mean_hard_pct"] == pytest.approx(
        0.5 * 0.1 / 1.05 * 100)


# ------------------------------------------------------------------ API ----

@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_api_analyze_planted_anomalies(client):
    payload = {
        "batch_id": "B-2024-01",
        "rows": [
            {"sample_id": "S1", "row_type": "standard", "value": 2.01,
             "crm_id": "OREAS-1"},
            {"sample_id": "S2", "row_type": "standard", "value": 2.18,
             "crm_id": "OREAS-1"},                      # z=+3.6 -> fail
            {"sample_id": "BL1", "row_type": "blank", "value": 0.07},  # 7x DL
            {"sample_id": "D1", "row_type": "duplicate", "value": 1.0,
             "pair_id": "P1"},
            {"sample_id": "D2", "row_type": "duplicate", "value": 1.1,
             "pair_id": "P1"},
        ],
        "crm_library": CRM,
        "detection_limit": 0.01,
    }
    r = client.post("/innovations/qaqc_analyzer/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["batch_id"] == "B-2024-01"
    assert body["verdict"] == "fail"
    assert len(body["failures"]) == 2   # 3SD standard + contaminated blank
    assert body["standards"][0]["n_failures"] == 1
    assert body["blanks"][0]["contaminated"] is True
    assert body["duplicates"]["hard_ranking"][0]["hard_pct"] == pytest.approx(
        0.5 * 0.1 / 1.05 * 100)


def test_api_analyze_batches(client):
    base = {"crm_library": CRM, "detection_limit": 0.01}
    batch_ok = {**base, "batch_id": "ok", "rows": [
        {"sample_id": "S1", "row_type": "standard", "value": 2.0,
         "crm_id": "OREAS-1"}]}
    batch_bad = {**base, "batch_id": "bad", "rows": [
        {"sample_id": "S9", "row_type": "standard", "value": 2.3,
         "crm_id": "OREAS-1"}]}                        # z=+6 -> fail
    r = client.post("/innovations/qaqc_analyzer/analyze-batches",
                    json=[batch_ok, batch_bad])
    assert r.status_code == 200
    assert r.json()["verdicts"] == {"ok": "pass", "bad": "fail"}
