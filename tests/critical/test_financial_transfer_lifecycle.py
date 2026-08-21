"""FIN-01 through FIN-06 controls for the future regulated-transfer boundary.

All tests use the in-memory TigerBeetle test double or a disposable PostgreSQL
control store.  They never submit a real payment or contact a payment network.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))

from middleware.financial.tigerbeetle_ledger import (
    Account,
    IdempotencyConflict,
    InMemoryTransferControlStore,
    MockTigerBeetleClient,
    PostgresTransferControlStore,
    RegulatedTransferService,
    TransferApproval,
    TransferControlError,
    TransferIntent,
    TransferManager,
    TransferPolicy,
    TigerBeetleConfig,
)
from scripts.verify_financial_audit_chain import verify


AUDIT_KEY = "test-only-audit-key-with-at-least-thirty-two-characters"


def _intent(*, key: str | None = None, amount: int = 5_000, reference: str | None = None) -> TransferIntent:
    key = key or f"fin-test-{uuid.uuid4().hex}"
    return TransferIntent(
        idempotency_key=key,
        actor_id="maker-01",
        debit_account_id=101,
        credit_account_id=202,
        amount=amount,
        currency="USD",
        ledger=1,
        code=4000,
        purpose="controlled integration test",
        external_reference=reference or f"invoice-{key}",
    )


def _approvals(*, count: int = 2, maker: bool = False, assurance: str = "aal2") -> list[TransferApproval]:
    values = [
        TransferApproval("maker-01" if maker else "checker-01", assurance, "challenge-01"),
        TransferApproval("checker-02", "aal3", "challenge-02"),
        TransferApproval("checker-03", "aal2", "challenge-03"),
    ]
    return values[:count]


def _policy(*, maximum: int = 10_000, approvals_required: int = 2) -> TransferPolicy:
    return TransferPolicy(
        currency="USD",
        maximum_minor_amount=maximum,
        approvals_required=approvals_required,
    )


async def _service(*, store=None, production: bool = False) -> tuple[RegulatedTransferService, MockTigerBeetleClient]:
    client = MockTigerBeetleClient(TigerBeetleConfig())
    await client.create_accounts([Account(id=101, ledger=1, code=1000), Account(id=202, ledger=1, code=4000)])
    return RegulatedTransferService(TransferManager(client), store or InMemoryTransferControlStore(AUDIT_KEY.encode()), production), client


def test_fin_01_concurrent_same_business_key_creates_one_ledger_effect():
    async def scenario() -> None:
        store = InMemoryTransferControlStore(AUDIT_KEY.encode())
        service, client = await _service(store=store)
        intent = _intent(key="fin-01-race")
        first, second = await asyncio.gather(
            service.submit(intent, _approvals(), _policy()),
            service.submit(intent, _approvals(), _policy()),
        )
        assert first.tigerbeetle_transfer_id == second.tigerbeetle_transfer_id == intent.tigerbeetle_id()
        # The stable TigerBeetle ID turns the second racing submission into an idempotent replay.
        assert client._accounts[202].credits_posted == intent.amount
        assert len(client._transfers) == 1

    asyncio.run(scenario())


def test_fin_02_same_idempotency_key_with_business_payload_change_is_rejected():
    async def scenario() -> None:
        service, _ = await _service()
        intent = _intent(key="fin-02-conflict")
        await service.submit(intent, _approvals(), _policy())
        with pytest.raises(IdempotencyConflict):
            await service.submit(
                _intent(key=intent.idempotency_key, amount=intent.amount + 1, reference=intent.external_reference),
                _approvals(),
                _policy(),
            )

    asyncio.run(scenario())


def test_fin_03_maker_self_approval_and_duplicate_checkers_are_denied():
    async def scenario() -> None:
        service, _ = await _service()
        with pytest.raises(TransferControlError, match="maker may not approve"):
            await service.submit(_intent(key="fin-03-maker"), _approvals(maker=True), _policy())
        duplicate = [
            TransferApproval("checker-01", "aal2", "challenge-a"),
            TransferApproval("checker-01", "aal3", "challenge-b"),
        ]
        with pytest.raises(TransferControlError, match="insufficient distinct"):
            await service.submit(_intent(key="fin-03-duplicate"), duplicate, _policy())

    asyncio.run(scenario())


def test_fin_04_high_value_policy_requires_distinct_step_up_approvals_and_limit():
    async def scenario() -> None:
        service, _ = await _service()
        high_value = _intent(key="fin-04-high-value", amount=9_000)
        with pytest.raises(TransferControlError, match="insufficient distinct"):
            await service.submit(high_value, _approvals(count=1), _policy(approvals_required=2))
        with pytest.raises(TransferControlError, match="step-up MFA"):
            await service.submit(high_value, _approvals(assurance="aal1"), _policy(approvals_required=2))
        receipt = await service.submit(high_value, _approvals(), _policy(approvals_required=2))
        assert receipt.status == "posted"
        with pytest.raises(TransferControlError, match="exceeds"):
            await service.submit(_intent(key="fin-04-limit", amount=10_001), _approvals(), _policy(maximum=10_000))

    asyncio.run(scenario())


def test_fin_05_postgres_hmac_chain_detects_payload_and_predecessor_tampering():
    database_url = os.environ.get("MV_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
        pytest.skip("requires disposable PostgreSQL transfer-control database")

    async def scenario() -> None:
        key = f"fin-05-{uuid.uuid4().hex}"
        store = PostgresTransferControlStore(database_url, AUDIT_KEY)
        service, _ = await _service(store=store, production=True)
        await service.submit(_intent(key=key), _approvals(), _policy())
        clean = verify(database_url, AUDIT_KEY, key)
        assert clean["ok"] is True and clean["event_count"] == 2

        import psycopg2

        url = database_url.replace("postgresql+psycopg2://", "postgresql://")
        with psycopg2.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE financial_transfer_audit_events SET previous_hash = 'tampered' "
                "WHERE idempotency_key = %s AND event_type = 'transfer_posted'",
                (key,),
            )
            assert cursor.rowcount == 1
        tampered = verify(database_url, AUDIT_KEY, key)
        assert tampered["ok"] is False
        assert "previous_hash_mismatch" in tampered["failures"][0]["reasons"]
        with psycopg2.connect(url) as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM financial_transfer_audit_events WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_approvals WHERE idempotency_key = %s", (key,))
            cursor.execute("DELETE FROM financial_transfer_intents WHERE idempotency_key = %s", (key,))

    asyncio.run(scenario())


def test_fin_06_lost_response_retry_reconciles_stable_tigerbeetle_identifier():
    async def scenario() -> None:
        # A fresh idempotency store represents recovery after a response was lost.
        # The ledger client retains the original stable ID and replies `exists`.
        client = MockTigerBeetleClient(TigerBeetleConfig())
        await client.create_accounts([Account(id=101, ledger=1, code=1000), Account(id=202, ledger=1, code=4000)])
        intent = _intent(key="fin-06-reconcile")
        first = RegulatedTransferService(TransferManager(client), InMemoryTransferControlStore(AUDIT_KEY.encode()))
        first_receipt = await first.submit(intent, _approvals(), _policy())
        recovered = RegulatedTransferService(TransferManager(client), InMemoryTransferControlStore(AUDIT_KEY.encode()))
        recovered_receipt = await recovered.submit(intent, _approvals(), _policy())
        assert recovered_receipt.result.error_code == "idempotent_replay"
        assert recovered_receipt.tigerbeetle_transfer_id == first_receipt.tigerbeetle_transfer_id
        assert len(client._transfers) == 1
        with pytest.raises(TransferControlError):
            RegulatedTransferService(TransferManager(client), InMemoryTransferControlStore(), production=True)

    asyncio.run(scenario())
