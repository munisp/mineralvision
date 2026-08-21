"""
TigerBeetle Financial Ledger Integration
=========================================

Production-grade financial ledger for MineralVision:
- ACID-compliant transactions
- Double-entry bookkeeping
- High-throughput transfers
- Audit trail
- Multi-currency support
- Batch processing

TigerBeetle provides a purpose-built financial
accounting database with safety guarantees.
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import struct
import hashlib

logger = logging.getLogger(__name__)

try:
    import tigerbeetle
    TIGERBEETLE_AVAILABLE = True
except ImportError:
    TIGERBEETLE_AVAILABLE = False

from .._mock_fallback import real_client_unavailable


class AccountFlags(IntFlag):
    """Account flags."""
    NONE = 0
    LINKED = 1
    DEBITS_MUST_NOT_EXCEED_CREDITS = 2
    CREDITS_MUST_NOT_EXCEED_DEBITS = 4
    HISTORY = 8


class TransferFlags(IntFlag):
    """Transfer flags."""
    NONE = 0
    LINKED = 1
    PENDING = 2
    POST_PENDING_TRANSFER = 4
    VOID_PENDING_TRANSFER = 8


class AccountType(Enum):
    """Types of accounts."""
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"


class LedgerCode(Enum):
    """Ledger codes for MineralVision."""
    CASH = 1000
    ACCOUNTS_RECEIVABLE = 1100
    INVENTORY = 1200
    EQUIPMENT = 1500
    ACCOUNTS_PAYABLE = 2000
    REVENUE = 4000
    COST_OF_GOODS = 5000
    OPERATING_EXPENSES = 6000
    EXPLORATION_COSTS = 6100
    ANALYSIS_COSTS = 6200
    SUBSCRIPTION_REVENUE = 4100
    SERVICE_REVENUE = 4200


@dataclass
class Account:
    """Financial account."""
    id: int
    ledger: int
    code: int
    user_data_128: int = 0
    user_data_64: int = 0
    user_data_32: int = 0
    flags: AccountFlags = AccountFlags.NONE
    debits_pending: int = 0
    debits_posted: int = 0
    credits_pending: int = 0
    credits_posted: int = 0
    timestamp: int = 0
    
    @property
    def balance(self) -> int:
        """Get account balance (credits - debits)."""
        return (self.credits_posted - self.debits_posted)
    
    @property
    def available_balance(self) -> int:
        """Get available balance."""
        return (self.credits_posted - self.credits_pending - 
                self.debits_posted - self.debits_pending)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'ledger': self.ledger,
            'code': self.code,
            'flags': self.flags.value,
            'debits_pending': self.debits_pending,
            'debits_posted': self.debits_posted,
            'credits_pending': self.credits_pending,
            'credits_posted': self.credits_posted,
            'balance': self.balance,
            'available_balance': self.available_balance,
            'timestamp': self.timestamp
        }


@dataclass
class Transfer:
    """Financial transfer."""
    id: int
    debit_account_id: int
    credit_account_id: int
    amount: int
    ledger: int
    code: int
    user_data_128: int = 0
    user_data_64: int = 0
    user_data_32: int = 0
    pending_id: int = 0
    flags: TransferFlags = TransferFlags.NONE
    timestamp: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'debit_account_id': self.debit_account_id,
            'credit_account_id': self.credit_account_id,
            'amount': self.amount,
            'ledger': self.ledger,
            'code': self.code,
            'flags': self.flags.value,
            'pending_id': self.pending_id,
            'timestamp': self.timestamp
        }


@dataclass
class TransferResult:
    """Result of a transfer operation."""
    transfer_id: int
    success: bool
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TigerBeetleConfig:
    """TigerBeetle configuration."""
    cluster_id: int = 0
    addresses: List[str] = field(default_factory=lambda: ["127.0.0.1:3000"])
    max_concurrency: int = 32


class MockTigerBeetleClient:
    """Mock TigerBeetle client."""
    
    def __init__(self, config: TigerBeetleConfig):
        self.config = config
        self._accounts: Dict[int, Account] = {}
        self._transfers: Dict[int, Transfer] = {}
        self._pending_transfers: Dict[int, Transfer] = {}
        self._next_timestamp = 1
    
    def _get_timestamp(self) -> int:
        """Get next timestamp."""
        ts = self._next_timestamp
        self._next_timestamp += 1
        return ts
    
    async def create_accounts(self, accounts: List[Account]) -> List[Optional[str]]:
        """Create accounts."""
        results = []
        
        for account in accounts:
            if account.id in self._accounts:
                results.append("exists")
            else:
                account.timestamp = self._get_timestamp()
                self._accounts[account.id] = account
                results.append(None)
        
        return results
    
    async def lookup_accounts(self, ids: List[int]) -> List[Optional[Account]]:
        """Lookup accounts by ID."""
        return [self._accounts.get(id) for id in ids]
    
    async def create_transfers(self, transfers: List[Transfer]) -> List[TransferResult]:
        """Create transfers."""
        results = []
        
        for transfer in transfers:
            # Validate accounts exist
            debit_account = self._accounts.get(transfer.debit_account_id)
            credit_account = self._accounts.get(transfer.credit_account_id)
            
            if not debit_account:
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=False,
                    error_code="debit_account_not_found"
                ))
                continue
            
            if not credit_account:
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=False,
                    error_code="credit_account_not_found"
                ))
                continue
            
            # Check if transfer already exists
            if transfer.id in self._transfers:
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=False,
                    error_code="exists"
                ))
                continue
            
            # Handle pending transfers
            if transfer.flags & TransferFlags.PENDING:
                transfer.timestamp = self._get_timestamp()
                self._pending_transfers[transfer.id] = transfer
                
                # Update pending balances
                debit_account.debits_pending += transfer.amount
                credit_account.credits_pending += transfer.amount
                
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=True
                ))
                continue
            
            # Handle post pending
            if transfer.flags & TransferFlags.POST_PENDING_TRANSFER:
                pending = self._pending_transfers.get(transfer.pending_id)
                if not pending:
                    results.append(TransferResult(
                        transfer_id=transfer.id,
                        success=False,
                        error_code="pending_transfer_not_found"
                    ))
                    continue
                
                # Move from pending to posted
                debit_account.debits_pending -= pending.amount
                debit_account.debits_posted += pending.amount
                credit_account.credits_pending -= pending.amount
                credit_account.credits_posted += pending.amount
                
                del self._pending_transfers[transfer.pending_id]
                transfer.timestamp = self._get_timestamp()
                self._transfers[transfer.id] = transfer
                
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=True
                ))
                continue
            
            # Handle void pending
            if transfer.flags & TransferFlags.VOID_PENDING_TRANSFER:
                pending = self._pending_transfers.get(transfer.pending_id)
                if not pending:
                    results.append(TransferResult(
                        transfer_id=transfer.id,
                        success=False,
                        error_code="pending_transfer_not_found"
                    ))
                    continue
                
                # Remove pending amounts
                debit_account.debits_pending -= pending.amount
                credit_account.credits_pending -= pending.amount
                
                del self._pending_transfers[transfer.pending_id]
                
                results.append(TransferResult(
                    transfer_id=transfer.id,
                    success=True
                ))
                continue
            
            # Regular transfer
            # Check balance constraints
            if debit_account.flags & AccountFlags.CREDITS_MUST_NOT_EXCEED_DEBITS:
                if debit_account.balance - transfer.amount < 0:
                    results.append(TransferResult(
                        transfer_id=transfer.id,
                        success=False,
                        error_code="exceeds_credits"
                    ))
                    continue
            
            # Execute transfer
            debit_account.debits_posted += transfer.amount
            credit_account.credits_posted += transfer.amount
            
            transfer.timestamp = self._get_timestamp()
            self._transfers[transfer.id] = transfer
            
            results.append(TransferResult(
                transfer_id=transfer.id,
                success=True
            ))
        
        return results
    
    async def lookup_transfers(self, ids: List[int]) -> List[Optional[Transfer]]:
        """Lookup transfers by ID."""
        return [self._transfers.get(id) for id in ids]
    
    async def get_account_transfers(self, account_id: int,
                                   limit: int = 100) -> List[Transfer]:
        """Get transfers for an account."""
        transfers = []
        for transfer in self._transfers.values():
            if (transfer.debit_account_id == account_id or 
                transfer.credit_account_id == account_id):
                transfers.append(transfer)
        
        return sorted(transfers, key=lambda t: t.timestamp, reverse=True)[:limit]
    
    async def get_account_balances(self, account_id: int) -> Dict[str, int]:
        """Get account balances."""
        account = self._accounts.get(account_id)
        if not account:
            return {}
        
        return {
            'debits_pending': account.debits_pending,
            'debits_posted': account.debits_posted,
            'credits_pending': account.credits_pending,
            'credits_posted': account.credits_posted,
            'balance': account.balance,
            'available_balance': account.available_balance
        }


class AccountManager:
    """
    Account management for TigerBeetle.
    
    Provides:
    - Account creation
    - Account lookup
    - Balance queries
    """
    
    def __init__(self, client: MockTigerBeetleClient):
        self.client = client
        self._id_counter = 1
    
    def _generate_id(self) -> int:
        """Generate unique account ID."""
        id = self._id_counter
        self._id_counter += 1
        return id
    
    async def create(self, ledger: int, code: int,
                    flags: AccountFlags = AccountFlags.NONE,
                    user_data: int = 0) -> Account:
        """Create an account."""
        account = Account(
            id=self._generate_id(),
            ledger=ledger,
            code=code,
            flags=flags,
            user_data_128=user_data
        )
        
        results = await self.client.create_accounts([account])
        if results[0]:
            raise ValueError(f"Failed to create account: {results[0]}")
        
        return account
    
    async def create_asset_account(self, code: LedgerCode,
                                  user_id: int = 0) -> Account:
        """Create an asset account."""
        return await self.create(
            ledger=1,
            code=code.value,
            flags=AccountFlags.DEBITS_MUST_NOT_EXCEED_CREDITS,
            user_data=user_id
        )
    
    async def create_liability_account(self, code: LedgerCode,
                                       user_id: int = 0) -> Account:
        """Create a liability account."""
        return await self.create(
            ledger=2,
            code=code.value,
            user_data=user_id
        )
    
    async def create_revenue_account(self, code: LedgerCode,
                                    user_id: int = 0) -> Account:
        """Create a revenue account."""
        return await self.create(
            ledger=4,
            code=code.value,
            user_data=user_id
        )
    
    async def create_expense_account(self, code: LedgerCode,
                                    user_id: int = 0) -> Account:
        """Create an expense account."""
        return await self.create(
            ledger=5,
            code=code.value,
            user_data=user_id
        )
    
    async def get(self, account_id: int) -> Optional[Account]:
        """Get an account by ID."""
        results = await self.client.lookup_accounts([account_id])
        return results[0] if results else None
    
    async def get_balance(self, account_id: int) -> Dict[str, int]:
        """Get account balance."""
        return await self.client.get_account_balances(account_id)
    
    async def get_history(self, account_id: int,
                         limit: int = 100) -> List[Transfer]:
        """Get account transaction history."""
        return await self.client.get_account_transfers(account_id, limit)


def _validate_transfer_request(debit_account_id: int, credit_account_id: int, amount: int) -> None:
    """Reject malformed value transfers before a ledger client observes them."""
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        raise ValueError("transfer amount must be a positive integer in minor units")
    if debit_account_id <= 0 or credit_account_id <= 0:
        raise ValueError("transfer account IDs must be positive")
    if debit_account_id == credit_account_id:
        raise ValueError("debit and credit accounts must differ")


class TransferManager:
    """
    Transfer management for TigerBeetle.
    
    Provides:
    - Transfer creation
    - Pending transfers
    - Batch transfers
    """
    
    def __init__(self, client: MockTigerBeetleClient):
        self.client = client
        self._id_counter = 1
    
    def _generate_id(self) -> int:
        """Generate unique transfer ID."""
        id = self._id_counter
        self._id_counter += 1
        return id
    
    async def transfer(self, debit_account_id: int, credit_account_id: int,
                      amount: int, ledger: int = 1, code: int = 0,
                      user_data: int = 0) -> TransferResult:
        """Execute a validated transfer."""
        _validate_transfer_request(debit_account_id, credit_account_id, amount)
        transfer = Transfer(
            id=self._generate_id(),
            debit_account_id=debit_account_id,
            credit_account_id=credit_account_id,
            amount=amount,
            ledger=ledger,
            code=code,
            user_data_128=user_data
        )
        
        results = await self.client.create_transfers([transfer])
        return results[0]
    
    async def create_pending(self, debit_account_id: int, credit_account_id: int,
                            amount: int, ledger: int = 1, code: int = 0) -> TransferResult:
        """Create a validated pending transfer."""
        _validate_transfer_request(debit_account_id, credit_account_id, amount)
        transfer = Transfer(
            id=self._generate_id(),
            debit_account_id=debit_account_id,
            credit_account_id=credit_account_id,
            amount=amount,
            ledger=ledger,
            code=code,
            flags=TransferFlags.PENDING
        )
        
        results = await self.client.create_transfers([transfer])
        return results[0]
    
    async def post_pending(self, pending_id: int) -> TransferResult:
        """Post a pending transfer."""
        transfer = Transfer(
            id=self._generate_id(),
            debit_account_id=0,
            credit_account_id=0,
            amount=0,
            ledger=0,
            code=0,
            pending_id=pending_id,
            flags=TransferFlags.POST_PENDING_TRANSFER
        )
        
        results = await self.client.create_transfers([transfer])
        return results[0]
    
    async def void_pending(self, pending_id: int) -> TransferResult:
        """Void a pending transfer."""
        transfer = Transfer(
            id=self._generate_id(),
            debit_account_id=0,
            credit_account_id=0,
            amount=0,
            ledger=0,
            code=0,
            pending_id=pending_id,
            flags=TransferFlags.VOID_PENDING_TRANSFER
        )
        
        results = await self.client.create_transfers([transfer])
        return results[0]
    
    async def batch_transfer(self, transfers: List[Dict[str, Any]]) -> List[TransferResult]:
        """Execute batch transfers."""
        transfer_objects = []
        
        for t in transfers:
            _validate_transfer_request(t['debit_account_id'], t['credit_account_id'], t['amount'])
            transfer_objects.append(Transfer(
                id=self._generate_id(),
                debit_account_id=t['debit_account_id'],
                credit_account_id=t['credit_account_id'],
                amount=t['amount'],
                ledger=t.get('ledger', 1),
                code=t.get('code', 0),
                flags=TransferFlags(t.get('flags', 0))
            ))
        
        return await self.client.create_transfers(transfer_objects)
    
    async def get(self, transfer_id: int) -> Optional[Transfer]:
        """Get a transfer by ID."""
        results = await self.client.lookup_transfers([transfer_id])
        return results[0] if results else None


class TigerBeetleLedger:
    """
    TigerBeetle ledger integration for MineralVision.
    
    Provides financial accounting:
    - Account management
    - Transfer processing
    - Double-entry bookkeeping
    - Audit trail
    
    Example:
        ledger = TigerBeetleLedger()
        await ledger.connect()
        
        # Create accounts
        cash = await ledger.accounts.create_asset_account(LedgerCode.CASH)
        revenue = await ledger.accounts.create_revenue_account(LedgerCode.SUBSCRIPTION_REVENUE)
        
        # Record revenue
        result = await ledger.transfers.transfer(
            debit_account_id=cash.id,
            credit_account_id=revenue.id,
            amount=10000  # $100.00 in cents
        )
        
        # Get balance
        balance = await ledger.accounts.get_balance(cash.id)
    """
    
    def __init__(self, config: TigerBeetleConfig = None):
        self.config = config or TigerBeetleConfig()
        self.client: Optional[MockTigerBeetleClient] = None
        self.accounts: Optional[AccountManager] = None
        self.transfers: Optional[TransferManager] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'TigerBeetleLedger':
        """
        Connect to TigerBeetle (real client first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if TIGERBEETLE_AVAILABLE:
            try:
                self.client = tigerbeetle.Client(
                    cluster_id=self.config.cluster_id,
                    addresses=self.config.addresses,
                    max_concurrency=self.config.max_concurrency
                )
                logger.info(f"Connected to TigerBeetle cluster {self.config.cluster_id}")
            except Exception as e:
                if real_client_unavailable("TigerBeetle", "cluster connection failed", e):
                    self._degraded = True
                    self.client = MockTigerBeetleClient(self.config)
        else:
            if real_client_unavailable("TigerBeetle", "tigerbeetle package not installed"):
                self._degraded = True
                self.client = MockTigerBeetleClient(self.config)
        
        self.accounts = AccountManager(self.client)
        self.transfers = TransferManager(self.client)
        
        self._connected = True
        return self
    
    async def setup_chart_of_accounts(self) -> Dict[str, Account]:
        """Setup standard chart of accounts."""
        accounts = {}
        
        # Asset accounts
        accounts['cash'] = await self.accounts.create_asset_account(LedgerCode.CASH)
        accounts['accounts_receivable'] = await self.accounts.create_asset_account(
            LedgerCode.ACCOUNTS_RECEIVABLE
        )
        accounts['inventory'] = await self.accounts.create_asset_account(LedgerCode.INVENTORY)
        accounts['equipment'] = await self.accounts.create_asset_account(LedgerCode.EQUIPMENT)
        
        # Liability accounts
        accounts['accounts_payable'] = await self.accounts.create_liability_account(
            LedgerCode.ACCOUNTS_PAYABLE
        )
        
        # Revenue accounts
        accounts['subscription_revenue'] = await self.accounts.create_revenue_account(
            LedgerCode.SUBSCRIPTION_REVENUE
        )
        accounts['service_revenue'] = await self.accounts.create_revenue_account(
            LedgerCode.SERVICE_REVENUE
        )
        
        # Expense accounts
        accounts['exploration_costs'] = await self.accounts.create_expense_account(
            LedgerCode.EXPLORATION_COSTS
        )
        accounts['analysis_costs'] = await self.accounts.create_expense_account(
            LedgerCode.ANALYSIS_COSTS
        )
        accounts['operating_expenses'] = await self.accounts.create_expense_account(
            LedgerCode.OPERATING_EXPENSES
        )
        
        return accounts
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_tigerbeetle(config: TigerBeetleConfig = None) -> TigerBeetleLedger:
    """Create a TigerBeetle ledger instance."""
    return TigerBeetleLedger(config)


async def create_and_connect_tigerbeetle(config: TigerBeetleConfig = None) -> TigerBeetleLedger:
    """Create and connect TigerBeetle."""
    ledger = TigerBeetleLedger(config)
    await ledger.connect()
    return ledger


# ---------------------------------------------------------------------------
# High-assurance real-value transfer controls
# ---------------------------------------------------------------------------
# This layer is intentionally separate from the exploration platform API.  It
# must be enabled only after the PostgreSQL migration, KMS-backed audit key,
# regulated payment partner review, and an independent security assessment.

class TransferControlError(RuntimeError):
    """Raised when a transfer-control invariant is not satisfied."""


class IdempotencyConflict(TransferControlError):
    """The same idempotency key was reused with different business input."""


@dataclass(frozen=True)
class TransferIntent:
    """Immutable business request for a real-value transfer in minor units."""
    idempotency_key: str
    actor_id: str
    debit_account_id: int
    credit_account_id: int
    amount: int
    currency: str
    ledger: int
    code: int
    purpose: str
    external_reference: str

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "idempotency_key": self.idempotency_key,
            "actor_id": self.actor_id,
            "debit_account_id": self.debit_account_id,
            "credit_account_id": self.credit_account_id,
            "amount": self.amount,
            "currency": self.currency,
            "ledger": self.ledger,
            "code": self.code,
            "purpose": self.purpose,
            "external_reference": self.external_reference,
        }

    def payload_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def tigerbeetle_id(self) -> int:
        """Derive the retry-stable non-zero 128-bit TigerBeetle object ID."""
        digest = hashlib.sha256(f"mineralvision-transfer:{self.idempotency_key}".encode("utf-8")).digest()[:16]
        value = int.from_bytes(digest, "big")
        return value if value else 1


@dataclass(frozen=True)
class TransferApproval:
    """Authenticated, step-up verified maker-checker approval evidence."""
    approver_id: str
    assurance: str
    challenge_id: str
    approved_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class TransferPolicy:
    """Static limits; fetch dynamically from an independently managed policy store in production."""
    currency: str
    maximum_minor_amount: int
    approvals_required: int = 2
    accepted_assurance: Tuple[str, ...] = ("aal2", "aal3")


@dataclass
class ControlledTransferReceipt:
    idempotency_key: str
    tigerbeetle_transfer_id: int
    status: str
    request_hash: str
    audit_event_hash: str
    result: TransferResult


class TransferControlStore(ABC):
    """Durable state required to make business idempotency observable and auditable."""

    durable: bool = False

    @abstractmethod
    async def reserve(self, intent: TransferIntent) -> Optional[ControlledTransferReceipt]:
        """Atomically reserve an idempotency key or return the prior receipt."""

    @abstractmethod
    async def record_approvals(self, intent: TransferIntent, approvals: List[TransferApproval]) -> None:
        """Persist independent maker-checker approval evidence before submission."""

    @abstractmethod
    async def complete(self, receipt: ControlledTransferReceipt) -> None:
        """Persist the immutable completion outcome."""

    @abstractmethod
    async def append_audit(self, event_type: str, intent: TransferIntent, actor_id: str, details: Dict[str, Any]) -> str:
        """Append a tamper-evident event and return its hash."""


class InMemoryTransferControlStore(TransferControlStore):
    """Test double only.  It is deliberately rejected for production transfers."""

    durable = False

    def __init__(self, audit_key: bytes = b"test-only-audit-key"):
        self._receipts: Dict[str, ControlledTransferReceipt] = {}
        self._hashes: Dict[str, str] = {}
        self._approvals: Dict[str, List[TransferApproval]] = {}
        self._audit_key = audit_key
        self._lock = asyncio.Lock()

    async def reserve(self, intent: TransferIntent) -> Optional[ControlledTransferReceipt]:
        async with self._lock:
            previous = self._receipts.get(intent.idempotency_key)
            if previous and previous.request_hash != intent.payload_hash():
                raise IdempotencyConflict("idempotency key cannot be reused with different transfer input")
            return previous

    async def record_approvals(self, intent: TransferIntent, approvals: List[TransferApproval]) -> None:
        async with self._lock:
            self._approvals[intent.idempotency_key] = list(approvals)

    async def complete(self, receipt: ControlledTransferReceipt) -> None:
        async with self._lock:
            previous = self._receipts.get(receipt.idempotency_key)
            if previous and previous.request_hash != receipt.request_hash:
                raise IdempotencyConflict("completion conflicts with original idempotent request")
            self._receipts[receipt.idempotency_key] = receipt

    async def append_audit(self, event_type: str, intent: TransferIntent, actor_id: str, details: Dict[str, Any]) -> str:
        async with self._lock:
            prior_hash = self._hashes.get(intent.idempotency_key, "")
            event = {
                "event_type": event_type,
                "intent": intent.canonical_payload(),
                "actor_id": actor_id,
                "details": details,
                "previous_hash": prior_hash,
            }
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            event_hash = hashlib.sha256(self._audit_key + encoded).hexdigest()
            self._hashes[intent.idempotency_key] = event_hash
            return event_hash


def _validate_controlled_intent(intent: TransferIntent, policy: TransferPolicy) -> None:
    _validate_transfer_request(intent.debit_account_id, intent.credit_account_id, intent.amount)
    if not intent.idempotency_key or len(intent.idempotency_key) > 128:
        raise TransferControlError("idempotency_key is required and must not exceed 128 characters")
    if not intent.actor_id or not intent.external_reference or not intent.purpose:
        raise TransferControlError("actor_id, external_reference, and purpose are required")
    if intent.currency != policy.currency:
        raise TransferControlError("transfer currency does not match the applicable policy")
    if intent.amount > policy.maximum_minor_amount:
        raise TransferControlError("transfer exceeds the approved minor-unit limit")
    if intent.ledger <= 0 or intent.code <= 0:
        raise TransferControlError("ledger and transfer code must be positive and explicitly assigned")


def _validate_approvals(intent: TransferIntent, approvals: List[TransferApproval], policy: TransferPolicy) -> None:
    unique_approvers = {approval.approver_id for approval in approvals}
    if intent.actor_id in unique_approvers:
        raise TransferControlError("maker may not approve their own transfer")
    if len(unique_approvers) < policy.approvals_required:
        raise TransferControlError("insufficient distinct maker-checker approvals")
    for approval in approvals:
        if approval.assurance not in policy.accepted_assurance or not approval.challenge_id:
            raise TransferControlError("approval lacks verified step-up MFA assurance evidence")


async def _transfer_with_explicit_id(
    manager: TransferManager, intent: TransferIntent
) -> TransferResult:
    """Submit a retry-stable TigerBeetle transfer ID rather than a process-local counter."""
    transfer = Transfer(
        id=intent.tigerbeetle_id(),
        debit_account_id=intent.debit_account_id,
        credit_account_id=intent.credit_account_id,
        amount=intent.amount,
        ledger=intent.ledger,
        code=intent.code,
        user_data_128=int.from_bytes(hashlib.sha256(intent.external_reference.encode("utf-8")).digest()[:16], "big"),
    )
    results = await manager.client.create_transfers([transfer])
    result = results[0]
    # TigerBeetle's `exists` is the expected reply after a lost response/retry
    # for the same stable transfer ID. It is reconciled as an idempotent success.
    if not result.success and result.error_code == "exists":
        return TransferResult(transfer_id=transfer.id, success=True, error_code="idempotent_replay")
    return result


class RegulatedTransferService:
    """Policy-gated transfer coordinator for future real-value integration.

    It intentionally has no HTTP endpoint.  A future payments service must call
    this only after OIDC/OPA authorization, sanctions/KYC checks, a durable
    PostgreSQL control store, and external reconciliation are configured.
    """

    def __init__(self, manager: TransferManager, store: TransferControlStore, production: bool = False):
        if production and not store.durable:
            raise TransferControlError("production transfers require a durable PostgreSQL control and audit store")
        self.manager = manager
        self.store = store
        self.production = production

    async def submit(
        self, intent: TransferIntent, approvals: List[TransferApproval], policy: TransferPolicy
    ) -> ControlledTransferReceipt:
        _validate_controlled_intent(intent, policy)
        _validate_approvals(intent, approvals, policy)
        existing = await self.store.reserve(intent)
        if existing:
            return existing

        await self.store.record_approvals(intent, approvals)
        await self.store.append_audit(
            "transfer_requested", intent, intent.actor_id,
            {"approvers": [approval.approver_id for approval in approvals], "request_hash": intent.payload_hash()},
        )
        result = await _transfer_with_explicit_id(self.manager, intent)
        if not result.success:
            audit_hash = await self.store.append_audit(
                "transfer_rejected", intent, intent.actor_id, {"error_code": result.error_code}
            )
            raise TransferControlError(f"ledger rejected transfer: {result.error_code}; audit={audit_hash}")
        audit_hash = await self.store.append_audit(
            "transfer_posted", intent, intent.actor_id,
            {"tigerbeetle_transfer_id": result.transfer_id, "replay": result.error_code == "idempotent_replay"},
        )
        receipt = ControlledTransferReceipt(
            idempotency_key=intent.idempotency_key,
            tigerbeetle_transfer_id=result.transfer_id,
            status="posted",
            request_hash=intent.payload_hash(),
            audit_event_hash=audit_hash,
            result=result,
        )
        await self.store.complete(receipt)
        return receipt


class TransferInProgress(TransferControlError):
    """A matching idempotency key is already being reconciled; caller must retry."""


class PostgresTransferControlStore(TransferControlStore):
    """PostgreSQL-backed idempotency and tamper-evident audit store.

    The schema is installed by Alembic revision ``0003_financial_transfer_controls``.
    This store never creates tables at runtime.  The HMAC audit key must be loaded
    from a secret manager and rotated with a documented key-version procedure.
    """

    durable = True

    def __init__(self, database_url: str, audit_hmac_key: str, key_version: str = "v1"):
        if not database_url.startswith(("postgres://", "postgresql://", "postgresql+")):
            raise TransferControlError("durable transfer controls require a PostgreSQL URL")
        if len(audit_hmac_key) < 32:
            raise TransferControlError("LEDGER_AUDIT_HMAC_KEY must be at least 32 characters")
        self.database_url = database_url.replace("postgresql+psycopg2://", "postgresql://")
        self.audit_key = audit_hmac_key.encode("utf-8")
        self.key_version = key_version

    def _connect(self):
        try:
            import psycopg2
        except ImportError as exc:
            raise TransferControlError("psycopg2 is required for the PostgreSQL transfer-control store") from exc
        return psycopg2.connect(self.database_url)

    @staticmethod
    def _receipt_from_json(data: Dict[str, Any]) -> ControlledTransferReceipt:
        result = TransferResult(
            transfer_id=int(data["result"]["transfer_id"]),
            success=bool(data["result"]["success"]),
            error_code=data["result"].get("error_code"),
        )
        return ControlledTransferReceipt(
            idempotency_key=data["idempotency_key"],
            tigerbeetle_transfer_id=int(data["tigerbeetle_transfer_id"]),
            status=data["status"],
            request_hash=data["request_hash"],
            audit_event_hash=data["audit_event_hash"],
            result=result,
        )

    @staticmethod
    def _receipt_json(receipt: ControlledTransferReceipt) -> str:
        return json.dumps({
            "idempotency_key": receipt.idempotency_key,
            "tigerbeetle_transfer_id": str(receipt.tigerbeetle_transfer_id),
            "status": receipt.status,
            "request_hash": receipt.request_hash,
            "audit_event_hash": receipt.audit_event_hash,
            "result": {
                "transfer_id": str(receipt.result.transfer_id),
                "success": receipt.result.success,
                "error_code": receipt.result.error_code,
            },
        }, sort_keys=True, separators=(",", ":"))

    async def reserve(self, intent: TransferIntent) -> Optional[ControlledTransferReceipt]:
        return await asyncio.to_thread(self._reserve_sync, intent)

    def _reserve_sync(self, intent: TransferIntent) -> Optional[ControlledTransferReceipt]:
        request_hash = intent.payload_hash()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO financial_transfer_intents
                    (idempotency_key, request_hash, actor_id, state, intent_payload)
                VALUES (%s, %s, %s, 'in_progress', %s::jsonb)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                (intent.idempotency_key, request_hash, intent.actor_id,
                 json.dumps(intent.canonical_payload(), sort_keys=True)),
            )
            if cursor.rowcount == 1:
                return None
            cursor.execute(
                "SELECT request_hash, state, receipt FROM financial_transfer_intents WHERE idempotency_key = %s FOR UPDATE",
                (intent.idempotency_key,),
            )
            row = cursor.fetchone()
            if row is None:
                raise TransferControlError("idempotency reservation disappeared")
            existing_hash, state, receipt = row
            if existing_hash != request_hash:
                raise IdempotencyConflict("idempotency key cannot be reused with different transfer input")
            if state == "posted" and receipt:
                return self._receipt_from_json(receipt)
            raise TransferInProgress("matching transfer is in progress or requires reconciliation; retry with the same key")

    async def record_approvals(self, intent: TransferIntent, approvals: List[TransferApproval]) -> None:
        await asyncio.to_thread(self._record_approvals_sync, intent, approvals)

    def _record_approvals_sync(self, intent: TransferIntent, approvals: List[TransferApproval]) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            for approval in approvals:
                cursor.execute(
                    """
                    INSERT INTO financial_transfer_approvals
                        (idempotency_key, approver_id, assurance, challenge_id, decision)
                    VALUES (%s, %s, %s, %s, 'approved')
                    ON CONFLICT (idempotency_key, approver_id) DO NOTHING
                    """,
                    (intent.idempotency_key, approval.approver_id, approval.assurance, approval.challenge_id),
                )

    async def complete(self, receipt: ControlledTransferReceipt) -> None:
        await asyncio.to_thread(self._complete_sync, receipt)

    def _complete_sync(self, receipt: ControlledTransferReceipt) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE financial_transfer_intents
                SET state = 'posted', receipt = %s::jsonb, completed_at = NOW()
                WHERE idempotency_key = %s AND request_hash = %s AND state = 'in_progress'
                """,
                (self._receipt_json(receipt), receipt.idempotency_key, receipt.request_hash),
            )
            if cursor.rowcount != 1:
                raise TransferControlError("transfer completion state changed; reconcile TigerBeetle before retrying")

    async def append_audit(self, event_type: str, intent: TransferIntent, actor_id: str, details: Dict[str, Any]) -> str:
        return await asyncio.to_thread(self._append_audit_sync, event_type, intent, actor_id, details)

    def _append_audit_sync(self, event_type: str, intent: TransferIntent, actor_id: str, details: Dict[str, Any]) -> str:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT event_hash FROM financial_transfer_audit_events WHERE idempotency_key = %s ORDER BY sequence DESC LIMIT 1 FOR UPDATE",
                (intent.idempotency_key,),
            )
            previous_hash = (cursor.fetchone() or [""])[0]
            event = {
                "event_type": event_type,
                "intent": intent.canonical_payload(),
                "actor_id": actor_id,
                "details": details,
                "previous_hash": previous_hash,
                "key_version": self.key_version,
            }
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            event_hash = hashlib.sha256(self.audit_key + encoded).hexdigest()
            cursor.execute(
                """
                INSERT INTO financial_transfer_audit_events
                    (idempotency_key, event_type, actor_id, details, previous_hash, event_hash, key_version)
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                """,
                (intent.idempotency_key, event_type, actor_id,
                 json.dumps(details, sort_keys=True), previous_hash, event_hash, self.key_version),
            )
            return event_hash
