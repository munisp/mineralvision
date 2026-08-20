from api.security.oidc import OIDCValidator


def test_keycloak_role_claims_are_merged_without_duplicates():
    claims = {
        "realm_access": {"roles": ["oil_spill_operator", "offline_access"]},
        "resource_access": {"mineralvision-api": {"roles": ["oil_spill_operator", "oil_spill_reviewer"]}},
        "roles": ["security_admin"],
    }
    assert OIDCValidator._roles(claims) == [
        "offline_access",
        "oil_spill_operator",
        "oil_spill_reviewer",
        "security_admin",
    ]


def test_mfa_accepts_webauthn_or_assurance_claims():
    assert OIDCValidator._mfa_verified({"amr": ["pwd", "webauthn"]}) is True
    assert OIDCValidator._mfa_verified({"acr": "aal2"}) is True
    assert OIDCValidator._mfa_verified({"amr": ["pwd"]}) is False


def test_project_claims_require_a_string_or_list():
    assert OIDCValidator._project_ids({"project_ids": ["project-a", "project-a", "project-b"]}) == ["project-a", "project-b"]
    assert OIDCValidator._project_ids({"project_ids": "project-a"}) == ["project-a"]
    assert OIDCValidator._project_ids({"project_ids": {"unexpected": "shape"}}) == []
