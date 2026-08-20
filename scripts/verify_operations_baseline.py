#!/usr/bin/env python3
"""Verify repository contracts for secure Compose testing, log forwarding, and PITR.

This is a static preflight check. It does not prove a deployed collector, object
store, image digest, Keycloak realm, or backup repository is operational.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    base = yaml.safe_load(read("docker-compose.secure.yml"))
    overlay = yaml.safe_load(read("docker-compose.observability.yml"))
    require(base["networks"]["application"].get("internal") is True, "application network must remain private")
    require(base["networks"]["data"].get("internal") is True, "data network must remain private")
    require(set(base["services"]["caddy"].get("ports", [])) == {"80:80", "443:443"}, "Caddy must remain the only public service")
    require("fluent-bit" in overlay["services"], "observability overlay must define Fluent Bit")
    require("127.0.0.1:24224:24224" in overlay["services"]["fluent-bit"].get("ports", []), "Fluent Bit input must be host-loopback only")
    for service in ("caddy", "apisix", "appsec-agent", "keycloak", "api"):
        logging = overlay["services"].get(service, {}).get("logging", {})
        require(logging.get("driver") == "fluentd", f"{service} must ship Docker logs to Fluent Bit")

    fluent_bit = read("security/observability/fluent-bit.conf")
    for required in ("tls                   on", "tls.verify            on", "SIEM_FORWARD_SHARED_KEY", "mineralvision.appsec-file"):
        require(required in fluent_bit, f"Fluent Bit SIEM control missing: {required}")

    pitr = read("security/postgres/postgresql-pitr.conf")
    for required in ("wal_level = replica", "archive_mode = on", "archive_command = 'pgbackrest --stanza=mineralvision archive-push %p'"):
        require(required in pitr, f"PostgreSQL PITR setting missing: {required}")
    pgbackrest = read("security/postgres/pgbackrest.conf.template")
    for required in ("repo1-type=s3", "repo1-cipher-type=aes-256-cbc", "repo1-retention-full=2", "repo1-retention-diff=7"):
        require(required in pgbackrest, f"pgBackRest configuration missing: {required}")

    print("operations baseline static contracts passed")


if __name__ == "__main__":
    main()
