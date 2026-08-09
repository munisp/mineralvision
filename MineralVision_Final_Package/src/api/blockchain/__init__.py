"""
Blockchain Data Provenance Module for MineralVision.

This module provides comprehensive blockchain capabilities including:
- Smart contract source code (Solidity)
- Multi-signature support for high-value transactions
- Gas optimization and batching
- Event listening and subscriptions
- Layer-2 integration for scalability (Optimism, Arbitrum, Polygon)
"""

from .blockchain_data_provenance import BlockchainDataProvenance
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
