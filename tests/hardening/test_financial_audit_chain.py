"""Integration coverage for the read-only financial audit-chain verifier."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))
sys.path.insert(0, str(ROOT / "scripts"))

from middleware.financial.tigerbeetle_ledger import (
    Account, MockTigerBeetleClient, PostgresTransferControlStore,
    RegulatedTransferService, TransferApproval, TransferIntent,
    TransferManager, TransferPolicy, TigerBeetleConfig,
)
from verify_financial_audit_chain import verify


AUDIT_KEY = "local-test-audit-key-must-be-at-least-32-bytes-long"


def test_audit_verifier_accepts_chain_and_detects_tamper():
    database_url = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        pytest.skip("MV_TEST_DATABASE_URL/DATABASE_URL PostgreSQL integration database is not configured")

    async def scenario():
        key = f"audit-chain-{uuid.uuid4().hex}"
        store = PostgresTransferControlStore(database_url, AUDIT_KEY)
        client = MockTigerBeetleClient(TigerBeetleConfig())
        await client.create_accounts([Account(id=401, ledger=1, code=1000), Account(id=402, ledger=1, code=4000)])
        service = RegulatedTransferService(TransferManager(client), store, production=True)
        intent = TransferIntent(key, "maker", 401, 402, 750, "USD", 1, 4000, "audit test", f"ref-{key}")
        approvals = [TransferApproval("checker-a", "aal2", "challenge-a"), TransferApproval("checker-b", "aal3", "challenge-b")]
        await service.submit(intent, approvals, TransferPolicy("USD", 10_000))

        outcome = verify(database_url, AUDIT_KEY, key)
        assert outcome["ok"] is True
        assert outcome["event_count"] == 2

        import psycopg2
        url = database_url.replace("postgresql+psycopg2://", "postgresql://")
        with psycopg2.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE financial_transfer_audit_events SET previous_hash = 'tampered' WHERE idempotency_key = %s AND sequence = (SELECT max(sequence) FROM financial_transfer_audit_events WHERE idempotency_key = %s)",
                (key, key),
            )
        outcome = verify(database_url, AUDIT_KEY, key)
        assert outcome["ok"] is False
        assert "previous_hash_mismatch" in outcome["failures"][0]["reasons"]

        with psycopg2.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM financial_transfer_audit_events WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_approvals WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_intents WHERE idempotency_key = %s", (key,))

    asyncio.run(scenario())
