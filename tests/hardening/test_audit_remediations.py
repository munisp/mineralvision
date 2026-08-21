"""Regression tests for security defects confirmed during the mission-critical audit."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MineralVision_Final_Package" / "src"))
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))

from api.auth_middleware import TokenPayload
from api.authz import require_project_access
from api.security.opa import OPAConfigurationError, OPAMiddleware
from middleware.financial.tigerbeetle_ledger import _validate_transfer_request


class _Project:
    def __init__(self, owner_id: str):
        self.owner_id = owner_id
        self.id = "project-1"


class _Query:
    def __init__(self, project):
        self.project = project

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self.project


class _Session:
    def __init__(self, project):
        self.project = project

    def query(self, _model):
        return _Query(self.project)


def _user(user_id: str, role: str = "user") -> TokenPayload:
    return TokenPayload(
        user_id=user_id,
        username=user_id,
        email=f"{user_id}@example.test",
        role=role,
        roles=[role],
        exp=__import__("datetime").datetime.now(),
    )


def test_project_access_blocks_cross_tenant_user():
    with pytest.raises(HTTPException) as error:
        require_project_access(_Session(_Project("owner")), "project-1", _user("attacker"))
    assert error.value.status_code == 403


def test_project_access_allows_owner_and_admin():
    project = _Project("owner")
    assert require_project_access(_Session(project), "project-1", _user("owner")) is project
    assert require_project_access(_Session(project), "project-1", _user("admin-user", "admin")) is project


@pytest.mark.parametrize("debit,credit,amount", [(1, 1, 10), (1, 2, 0), (1, 2, -1), (0, 2, 1), (1, 2, True)])
def test_financial_transfer_validation_rejects_invalid_value_requests(debit, credit, amount):
    with pytest.raises(ValueError):
        _validate_transfer_request(debit, credit, amount)


def test_financial_transfer_validation_accepts_positive_minor_units():
    _validate_transfer_request(1, 2, 100)


def test_opa_rejects_non_http_configuration(monkeypatch):
    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.setenv("OPA_URL", "file:///etc/passwd")
    monkeypatch.setenv("ENV", "development")
    with pytest.raises(OPAConfigurationError):
        OPAMiddleware(app=object())


def test_opa_production_rejects_non_private_policy_host(monkeypatch):
    monkeypatch.setenv("OPA_ENABLED", "true")
    monkeypatch.setenv("OPA_URL", "https://untrusted.example")
    monkeypatch.setenv("ENV", "production")
    with pytest.raises(OPAConfigurationError):
        OPAMiddleware(app=object())
