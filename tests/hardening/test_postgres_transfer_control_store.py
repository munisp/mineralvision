"""PostgreSQL integration test for durable ledger controls; no external funds are moved."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))

from middleware.financial.tigerbeetle_ledger import (
    Account,
    MockTigerBeetleClient,
    PostgresTransferControlStore,
    RegulatedTransferService,
    TransferApproval,
    TransferIntent,
    TransferManager,
    TransferPolicy,
    TigerBeetleConfig,
)


def test_postgres_store_replays_one_durable_receipt():
    database_url = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        pytest.skip("MV_TEST_DATABASE_URL/DATABASE_URL PostgreSQL integration database is not configured")

    async def scenario():
        key = f"test-transfer-{uuid.uuid4().hex}"
        store = PostgresTransferControlStore(database_url, "local-test-audit-key-must-be-at-least-32-bytes-long")
        client = MockTigerBeetleClient(TigerBeetleConfig())
        await client.create_accounts([Account(id=301, ledger=1, code=1000), Account(id=302, ledger=1, code=4000)])
        service = RegulatedTransferService(TransferManager(client), store, production=True)
        intent = TransferIntent(
            idempotency_key=key, actor_id="maker", debit_account_id=301, credit_account_id=302,
            amount=500, currency="USD", ledger=1, code=4000,
            purpose="integration test", external_reference=f"ref-{key}",
        )
        approvals = [
            TransferApproval("reviewer-a", "aal2", "challenge-a"),
            TransferApproval("reviewer-b", "aal3", "challenge-b"),
        ]
        policy = TransferPolicy(currency="USD", maximum_minor_amount=10_000)
        first = await service.submit(intent, approvals, policy)
        replay = await service.submit(intent, approvals, policy)
        assert first.tigerbeetle_transfer_id == replay.tigerbeetle_transfer_id
        assert first.audit_event_hash == replay.audit_event_hash

        import psycopg2
        url = database_url.replace("postgresql+psycopg2://", "postgresql://")
        with psycopg2.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT state FROM financial_transfer_intents WHERE idempotency_key = %s", (key,))
            assert cursor.fetchone()[0] == "posted"
            cursor.execute("SELECT count(*) FROM financial_transfer_audit_events WHERE idempotency_key = %s", (key,))
            assert cursor.fetchone()[0] == 2
            cursor.execute("DELETE FROM financial_transfer_audit_events WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_approvals WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_intents WHERE idempotency_key = %s", (key,))

    asyncio.run(scenario())
