"""
Blockchain Data Provenance Module for MineralVision.

This module provides comprehensive blockchain capabilities including:
- Smart contract source code (Solidity)
- Multi-signature support for high-value transactions
- Gas optimization and batching
- Event listening and subscriptions
- Layer-2 integration for scalability (Optimism, Arbitrum, Polygon)
"""

# BlockchainDataProvenance requires optional web3/ipfs backends; keep the
# package importable without them (endpoints degrade with HTTP 503).
try:
    from .blockchain_data_provenance import BlockchainDataProvenance
except ImportError:  # pragma: no cover - depends on optional deps
    BlockchainDataProvenance = None

from .local_ledger import (
    LedgerSigner,
    LocalCryptoLedger,
    Block,
    merkle_root,
)
from .advanced_blockchain import (
    TransactionStatus,
    EventType,
    Transaction,
    MultisigProposal,
    MINERAL_PROVENANCE_CONTRACT,
    GasOptimizer,
    EventSubscriber,
    MultisigManager,
    Layer2Bridge,
    AdvancedBlockchainManager,
    create_advanced_blockchain_manager,
    create_gas_optimizer,
    create_event_subscriber,
    create_multisig_manager,
    create_l2_bridge
)

__all__ = [
    'BlockchainDataProvenance',
    'LedgerSigner',
    'LocalCryptoLedger',
    'Block',
    'merkle_root',
    'TransactionStatus',
    'EventType',
    'Transaction',
    'MultisigProposal',
    'MINERAL_PROVENANCE_CONTRACT',
    'GasOptimizer',
    'EventSubscriber',
    'MultisigManager',
    'Layer2Bridge',
    'AdvancedBlockchainManager',
    'create_advanced_blockchain_manager',
    'create_gas_optimizer',
    'create_event_subscriber',
    'create_multisig_manager',
    'create_l2_bridge'
]
