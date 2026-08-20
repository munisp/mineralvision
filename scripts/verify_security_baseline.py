#!/usr/bin/env python3
"""Static contract checks for the MineralVision defense-in-depth baseline.

This verifier does not claim runtime safety; it detects accidental removal of key
repository controls before deployment. Runtime validation still requires a controlled
staging environment, real identity provider, gateway/WAF release, and external testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    compose = yaml.safe_load(text("docker-compose.secure.yml"))
    services = compose.get("services", {})
    required_services = {"caddy", "apisix", "appsec-agent", "api", "api-migrate", "keycloak", "opa", "postgres", "redis"}
    require(required_services.issubset(services), "secure compose is missing required defense services")
    require(set(services["caddy"].get("ports", [])) == {"80:80", "443:443"}, "Caddy must be the only public listener")
    for name in ("postgres", "redis", "api", "keycloak", "opa", "apisix"):
        require(not services[name].get("ports"), f"{name} must not publish a host port")
    networks = compose.get("networks", {})
    require(networks.get("application", {}).get("internal") is True, "application network must be internal")
    require(networks.get("data", {}).get("internal") is True, "data network must be internal")

    caddy = text("security/caddy/Caddyfile")
    for header in ("Strict-Transport-Security", "X-Content-Type-Options", "Content-Security-Policy"):
        require(header in caddy, f"Caddy security header missing: {header}")
    require("header_up -X-Forwarded-For" in caddy, "Caddy must strip spoofable forwarding headers")

    apisix = yaml.safe_load(text("security/apisix/apisix.yaml"))
    route_ids = {route["id"] for route in apisix.get("routes", [])}
    require("oil-spill-raw-image-inference" in route_ids, "raw image inference must have its own quota")
    require(all("limit-req" in route.get("plugins", {}) for route in apisix.get("routes", [])), "every public route requires a rate limit")

    realm = json.loads(text("security/identity/mineralvision-realm.json"))
    require(realm.get("bruteForceProtected") is True, "Keycloak brute-force protection must be enabled")
    require(realm.get("registrationAllowed") is False, "public self-registration must be disabled")
    require(realm.get("adminEventsEnabled") is True, "Keycloak administrative events must be enabled")
    roles = {role["name"] for role in realm["roles"]["realm"]}
    require({"oil_spill_approver", "oil_spill_reviewer", "security_admin"}.issubset(roles), "privileged realm roles missing")

    policy = text("security/opa/mineralvision.rego")
    require("default decision" in policy and "default action" in policy, "OPA policy must deny unknown paths")
    require("mfa_required" in policy and "oil_spill.model.approve" in policy, "OPA MFA gate missing")

    workflow = text(".github/workflows/security-and-benchmark.yml")
    for term in ("permissions:\n  contents: read", "sealed-benchmark", "mineralvision-sealed-benchmark", "pip-audit"):
        require(term in workflow, f"security workflow missing: {term}")
    require("/approve" not in workflow, "benchmark workflow must never approve a model")

    print("security baseline static contracts passed")


if __name__ == "__main__":
    main()
