"""E2E tests for the stakeholder onboarding innovation (wave 6).

Covers: org creation, invitations (hashed tokens, console email capture),
token validation/expiry, acceptance -> real platform user -> real /auth/login,
password-reset round-trip, role validation, RBAC stakeholder roles, and the
journey manifest endpoint fix. No mocks of the platform auth path — tokens are
captured via an injected email backend only.
"""

import re
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.database import Base as PlatformBase
from api.database import UserModel
from api.database import get_db as platform_get_db
from api.endpoints.auth import router as auth_router
from api.endpoints.users import router as users_router
from api.innovations.onboarding import router as onboarding_router
from api.innovations.onboarding.db import Base as OnboardingBase
from api.innovations.onboarding.db import get_db as onboarding_get_db
from api.innovations.onboarding.email import get_email_service
from api.innovations.onboarding.models import InvitationModel, MembershipModel


class CapturingEmailBackend:
    """Test email backend: captures messages and extracts the token link."""

    delivery_mode = "console"

    def __init__(self):
        self.sent = []

    def send(self, to, subject, body):
        self.sent.append({"to": to, "subject": subject, "body": body})
        return self.delivery_mode

    def last_token(self):
        m = re.search(r"/(?:accept-invite|reset-password)/(\S+)", self.sent[-1]["body"])
        assert m, "no token link found in captured email"
        return m.group(1).strip()


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    OnboardingBase.metadata.create_all(eng)
    PlatformBase.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = factory()
    yield db
    db.close()


@pytest.fixture()
def email_backend():
    return CapturingEmailBackend()


@pytest.fixture()
def client(engine, email_backend):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(onboarding_router)
    app.include_router(auth_router, prefix="/auth")
    app.include_router(users_router, prefix="/api/users")
    app.dependency_overrides[onboarding_get_db] = override_db
    app.dependency_overrides[platform_get_db] = override_db
    app.dependency_overrides[get_email_service] = lambda: email_backend
    with TestClient(app) as c:
        yield c


def _register_and_login(client, username, email, password="Sup3rSecret!"):
    r = client.post("/auth/register", json={
        "username": username, "email": email, "password": password,
    })
    assert r.status_code == 201, r.text
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _make_org(client, token, name="Acme Minerals"):
    r = client.post("/innovations/onboarding/orgs", json={"name": name}, headers=_auth(token))
    assert r.status_code == 201, r.text
    return r.json()


class TestOrganizations:
    def test_create_org_makes_caller_org_admin(self, client, session):
        token = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, token)
        assert org["slug"] == "acme-minerals"
        assert org["status"] == "active"
        m = session.query(MembershipModel).filter_by(org_id=org["id"]).one()
        assert m.role == "org_admin" and m.status == "active"

    def test_orgs_mine_lists_membership(self, client):
        token = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, token)
        r = client.get("/innovations/onboarding/orgs-mine", headers=_auth(token))
        assert r.status_code == 200
        orgs = r.json()["organizations"]
        assert len(orgs) == 1 and orgs[0]["id"] == org["id"]
        assert orgs[0]["my_role"] == "org_admin"

    def test_org_detail_forbidden_for_non_member(self, client):
        owner = _register_and_login(client, "alice", "alice@example.com")
        outsider = _register_and_login(client, "bob", "bob@example.com")
        org = _make_org(client, owner)
        r = client.get(f"/innovations/onboarding/orgs/{org['id']}", headers=_auth(outsider))
        assert r.status_code == 403

    def test_duplicate_slug_conflict(self, client):
        token = _register_and_login(client, "alice", "alice@example.com")
        _make_org(client, token)
        r = client.post("/innovations/onboarding/orgs",
                        json={"name": "Acme Minerals"}, headers=_auth(token))
        assert r.status_code == 409


class TestInvitations:
    def _invite(self, client, token, org_id, email="new@example.com", role="viewer"):
        return client.post(
            f"/innovations/onboarding/orgs/{org_id}/invitations",
            json={"email": email, "role": role}, headers=_auth(token),
        )

    def test_full_onboarding_e2e(self, client, email_backend, session):
        owner = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, owner)

        r = self._invite(client, owner, org["id"], role="geologist")
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["email_delivery"] == "console"
        assert "token" not in body  # raw token never exposed by default
        assert len(email_backend.sent) == 1
        invite_token = email_backend.last_token()

        # token stored hashed at rest
        inv = session.query(InvitationModel).filter_by(email="new@example.com").one()
        assert invite_token not in inv.token_hash and len(inv.token_hash) == 64

        # validate
        r = client.get(f"/innovations/onboarding/invitations/{invite_token}")
        assert r.status_code == 200, r.text
        info = r.json()
        assert info["email"] == "new@example.com"
        assert info["role"] == "geologist"
        assert info["org"]["id"] == org["id"]

        # accept -> real platform user
        r = client.post(
            f"/innovations/onboarding/invitations/{invite_token}/accept",
            json={"password": "N3wPassw0rd!", "full_name": "New User"},
        )
        assert r.status_code == 201, r.text
        accepted = r.json()
        assert accepted["success"] is True and accepted["role"] == "geologist"

        # user can log in via the REAL /auth/login
        r = client.post("/auth/login",
                        json={"username": "new@example.com", "password": "N3wPassw0rd!"})
        assert r.status_code == 200, r.text
        assert r.json()["user"]["role"] == "geologist"

        # membership visible on org detail
        r = client.get(f"/innovations/onboarding/orgs/{org['id']}", headers=_auth(owner))
        members = r.json()["members"]
        assert any(m["role"] == "geologist" for m in members) and len(members) == 2

        # token now burned
        r = client.get(f"/innovations/onboarding/invitations/{invite_token}")
        assert r.status_code == 410

    def test_invite_unknown_role_422(self, client):
        owner = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, owner)
        r = self._invite(client, owner, org["id"], role="superuser")
        assert r.status_code == 422

    def test_invite_forbidden_for_non_admin_member(self, client, email_backend):
        owner = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, owner)
        # onboard a plain viewer member
        self._invite(client, owner, org["id"], email="v@example.com", role="viewer")
        token = email_backend.last_token()
        client.post(f"/innovations/onboarding/invitations/{token}/accept",
                    json={"password": "View3rPass!"})
        viewer_token = _login(client, "v@example.com", "View3rPass!")
        r = self._invite(client, viewer_token, org["id"], email="x@example.com")
        assert r.status_code == 403

    def test_invite_forbidden_for_non_member(self, client):
        owner = _register_and_login(client, "alice", "alice@example.com")
        outsider = _register_and_login(client, "bob", "bob@example.com")
        org = _make_org(client, owner)
        r = self._invite(client, outsider, org["id"])
        assert r.status_code == 403

    def test_expired_invitation_410(self, client, email_backend, session):
        owner = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, owner)
        self._invite(client, owner, org["id"])
        token = email_backend.last_token()
        inv = session.query(InvitationModel).filter_by(email="new@example.com").one()
        inv.expires_at = datetime.utcnow() - timedelta(hours=1)
        session.commit()
        r = client.get(f"/innovations/onboarding/invitations/{token}")
        assert r.status_code == 410

    def test_unknown_invitation_404(self, client):
        r = client.get("/innovations/onboarding/invitations/bogus-token")
        assert r.status_code == 404

    def test_all_stakeholder_roles_invitable(self, client, email_backend):
        owner = _register_and_login(client, "alice", "alice@example.com")
        org = _make_org(client, owner)
        for i, role in enumerate(["viewer", "geologist", "resource_geologist",
                                  "field_technician", "investor", "regulator",
                                  "custodian", "org_admin"]):
            r = self._invite(client, owner, org["id"],
                             email=f"user{i}@example.com", role=role)
            assert r.status_code == 201, f"{role}: {r.text}"
            assert r.json()["role"] == role


def _login(client, username, password):
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


class TestPasswordReset:
    def test_reset_round_trip(self, client, email_backend):
        _register_and_login(client, "alice", "alice@example.com", password="OldPassw0rd!")
        r = client.post("/innovations/onboarding/password-reset/request",
                        json={"email": "alice@example.com"})
        assert r.status_code == 200 and r.json()["email_delivery"] == "console"
        token = email_backend.last_token()

        r = client.post("/innovations/onboarding/password-reset/confirm",
                        json={"token": token, "new_password": "Br4ndNewPass!"})
        assert r.status_code == 200 and r.json()["success"] is True

        # new password works, old fails
        _login(client, "alice", "Br4ndNewPass!")
        r = client.post("/auth/login",
                        json={"username": "alice", "password": "OldPassw0rd!"})
        assert r.status_code == 401

        # token single-use
        r = client.post("/innovations/onboarding/password-reset/confirm",
                        json={"token": token, "new_password": "An0therPass!"})
        assert r.status_code == 410

    def test_reset_unknown_email_no_leak(self, client, email_backend):
        r = client.post("/innovations/onboarding/password-reset/request",
                        json={"email": "ghost@example.com"})
        assert r.status_code == 200 and r.json()["email_delivery"] == "none"
        assert email_backend.sent == []

    def test_reset_unknown_token_404(self, client):
        r = client.post("/innovations/onboarding/password-reset/confirm",
                        json={"token": "bogus-token-value", "new_password": "Whatever1!"})
        assert r.status_code == 404


class TestRoleCatalogue:
    def test_stakeholder_roles_in_role_manager(self):
        from api.auth.rbac import RoleManager
        rm = RoleManager()
        for role in ["field_technician", "investor", "regulator", "custodian",
                     "admin", "geologist", "resource_geologist", "viewer",
                     "qualified_person"]:
            assert role in rm.roles, role
            assert rm.roles[role].permissions, role

    def test_users_create_rejects_unknown_role(self, client, session):
        _register_and_login(client, "root", "root@example.com")
        admin = session.query(UserModel).filter_by(username="root").one()
        admin.role = "admin"
        session.commit()
        # role is embedded in the JWT — re-login after promotion
        admin_token = _login(client, "root", "Sup3rSecret!")
        r = client.post("/api/users", headers=_auth(admin_token), json={
            "name": "Bad Role", "email": "bad@example.com",
            "password": "Whatever1!", "roles": ["overlord"],
        })
        assert r.status_code == 422

    def test_users_create_accepts_stakeholder_role(self, client, session):
        _register_and_login(client, "root", "root@example.com")
        admin = session.query(UserModel).filter_by(username="root").one()
        admin.role = "admin"
        session.commit()
        # role is embedded in the JWT — re-login after promotion
        admin_token = _login(client, "root", "Sup3rSecret!")
        r = client.post("/api/users", headers=_auth(admin_token), json={
            "name": "Ina Investor", "email": "ina@example.com",
            "password": "Whatever1!", "roles": ["investor"],
        })
        assert r.status_code == 201, r.text
        assert r.json()["roles"] == ["investor"]


class TestJourneyManifest:
    def test_step_001_2_endpoint_is_a_real_route(self):
        from api.orchestration.journeys import get_journey_registry
        registry = get_journey_registry()
        step = None
        for journey in registry.list_all():
            for s in journey.steps:
                if s.id == "step-001-2":
                    step = s
                    break
        assert step is not None, "step-001-2 not found in journey manifests"
        assert step.endpoint != "/api/users/invite"
        route_paths = {r.path for r in onboarding_router.routes}
        assert step.endpoint in route_paths, (
            f"journey endpoint {step.endpoint} does not match any onboarding route: "
            f"{sorted(route_paths)}"
        )
