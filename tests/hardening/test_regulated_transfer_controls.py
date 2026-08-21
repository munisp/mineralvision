"""High-assurance ledger control tests; no real value transfer is performed."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MineralVision_Enhanced"))

from middleware.financial.tigerbeetle_ledger import (
    Account,
    InMemoryTransferControlStore,
    IdempotencyConflict,
    MockTigerBeetleClient,
    RegulatedTransferService,
    TransferApproval,
    TransferControlError,
    TransferIntent,
    TransferManager,
    TransferPolicy,
    TigerBeetleConfig,
)


async def _service(production: bool = False):
    client = MockTigerBeetleClient(TigerBeetleConfig())
    await client.create_accounts([
        Account(id=101, ledger=1, code=1000),
        Account(id=202, ledger=1, code=4000),
    ])
    return RegulatedTransferService(
        TransferManager(client), InMemoryTransferControlStore(), production=production
    )


def _intent(key: str = "payment-2026-0001", amount: int = 10_000) -> TransferIntent:
    return TransferIntent(
        idempotency_key=key,
        actor_id="maker-1",
        debit_account_id=101,
        credit_account_id=202,
        amount=amount,
        currency="USD",
        ledger=1,
        code=4000,
        purpose="approved service settlement",
        external_reference="invoice-0001",
    )


def _policy() -> TransferPolicy:
    return TransferPolicy(currency="USD", maximum_minor_amount=50_000, approvals_required=2)


def _approvals():
    return [
        TransferApproval(approver_id="reviewer-1", assurance="aal2", challenge_id="challenge-a"),
        TransferApproval(approver_id="reviewer-2", assurance="aal3", challenge_id="challenge-b"),
    ]


def test_retry_with_same_idempotency_key_is_single_stable_transfer():
    async def scenario():
        service = await _service()
        first = await service.submit(_intent(), _approvals(), _policy())
        replay = await service.submit(_intent(), _approvals(), _policy())
        assert first.tigerbeetle_transfer_id == replay.tigerbeetle_transfer_id
        assert first.request_hash == replay.request_hash
        assert replay.status == "posted"
    asyncio.run(scenario())


def test_same_idempotency_key_with_changed_amount_is_rejected():
    async def scenario():
        service = await _service()
        await service.submit(_intent(), _approvals(), _policy())
        with pytest.raises(IdempotencyConflict):
            await service.submit(_intent(amount=10_001), _approvals(), _policy())
    asyncio.run(scenario())


def test_maker_cannot_approve_and_two_distinct_step_up_approvers_are_required():
    async def scenario():
        service = await _service()
        invalid = [
            TransferApproval(approver_id="maker-1", assurance="aal2", challenge_id="challenge-a"),
            TransferApproval(approver_id="reviewer-1", assurance="aal2", challenge_id="challenge-b"),
        ]
        with pytest.raises(TransferControlError):
            await service.submit(_intent(), invalid, _policy())
    asyncio.run(scenario())


def test_production_rejects_ephemeral_audit_store():
    async def scenario():
        with pytest.raises(TransferControlError):
            await _service(production=True)
    asyncio.run(scenario())


def test_non_step_up_approval_is_rejected():
    async def scenario():
        service = await _service()
        weak = [
            TransferApproval(approver_id="reviewer-1", assurance="aal1", challenge_id="challenge-a"),
            TransferApproval(approver_id="reviewer-2", assurance="aal2", challenge_id="challenge-b"),
        ]
        with pytest.raises(TransferControlError):
            await service.submit(_intent(), weak, _policy())
    asyncio.run(scenario())
