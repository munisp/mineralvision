"""
Advanced Blockchain Data Provenance Module for MineralVision.

This module provides enhanced blockchain capabilities including:
- Smart contract source code (Solidity)
- Multi-signature support for high-value transactions
- Gas optimization and batching
- Event listening and subscriptions
- Layer-2 integration for scalability
"""

import hashlib
import json
import time
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

from .local_ledger import LocalCryptoLedger


class TransactionStatus(Enum):
    """Transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    """Blockchain event types."""
    DATA_REGISTERED = "DataRegistered"
    DATA_UPDATED = "DataUpdated"
    MINERAL_RIGHT_REGISTERED = "MineralRightRegistered"
    MINERAL_RIGHT_TRANSFERRED = "MineralRightTransferred"
    MULTISIG_PROPOSED = "MultisigProposed"
    MULTISIG_APPROVED = "MultisigApproved"
    MULTISIG_EXECUTED = "MultisigExecuted"


@dataclass
class Transaction:
    """Blockchain transaction."""
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    status: TransactionStatus
    block_number: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_hash': self.tx_hash,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'value': self.value,
            'gas_used': self.gas_used,
            'gas_price': self.gas_price,
            'status': self.status.value,
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data
        }


@dataclass
class MultisigProposal:
    """Multi-signature proposal."""
    proposal_id: str
    proposer: str
    action: str
    params: Dict[str, Any]
    required_signatures: int
    signatures: List[str] = field(default_factory=list)
    executed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def is_approved(self) -> bool:
        return len(self.signatures) >= self.required_signatures
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'proposal_id': self.proposal_id,
            'proposer': self.proposer,
            'action': self.action,
            'params': self.params,
            'required_signatures': self.required_signatures,
            'signatures': self.signatures,
            'executed': self.executed,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


# Solidity Smart Contract Source Code
MINERAL_PROVENANCE_CONTRACT = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

/**
 * @title MineralProvenance
 * @dev Smart contract for mineral exploration data provenance and rights management
 * @author MineralVision Team
 */
contract MineralProvenance is AccessControl, ReentrancyGuard, Pausable {
    
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant REGISTRAR_ROLE = keccak256("REGISTRAR_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    
    struct DataRecord {
        bytes32 dataHash;
        string ipfsHash;
        address owner;
        uint256 timestamp;
        string dataType;
        string metadata;
        bool verified;
        address verifier;
    }
    
    struct MineralRight {
        string rightId;
        address owner;
        string location;
        string mineralType;
        uint256 registrationDate;
        uint256 expirationDate;
        bool active;
        string metadata;
    }
    
    struct MultisigTransaction {
        address proposer;
        bytes data;
        uint256 requiredApprovals;
        uint256 approvalCount;
        bool executed;
        mapping(address => bool) approvals;
    }
    
    // Storage
    mapping(bytes32 => DataRecord) public dataRecords;
    mapping(bytes32 => DataRecord[]) public dataHistory;
    mapping(string => MineralRight) public mineralRights;
    mapping(uint256 => MultisigTransaction) public multisigTransactions;
    
    bytes32[] public allDataHashes;
    string[] public allRightIds;
    uint256 public multisigNonce;
    uint256 public requiredMultisigApprovals;
    address[] public multisigSigners;
    
    // Events
    event DataRegistered(
        bytes32 indexed dataHash,
        string ipfsHash,
        address indexed owner,
        string dataType,
        uint256 timestamp
    );
    
    event DataUpdated(
        bytes32 indexed dataHash,
        string newIpfsHash,
        address indexed updater,
        uint256 timestamp
    );
    
    event DataVerified(
        bytes32 indexed dataHash,
        address indexed verifier,
        uint256 timestamp
    );
    
    event MineralRightRegistered(
        string indexed rightId,
        address indexed owner,
        string location,
        string mineralType,
        uint256 timestamp
    );
    
    event MineralRightTransferred(
        string indexed rightId,
        address indexed from,
        address indexed to,
        uint256 timestamp
    );
    
    event MultisigProposed(
        uint256 indexed txId,
        address indexed proposer,
        bytes data
    );
    
    event MultisigApproved(
        uint256 indexed txId,
        address indexed approver
    );
    
    event MultisigExecuted(
        uint256 indexed txId,
        bool success
    );
    
    constructor(address[] memory _signers, uint256 _requiredApprovals) {
        require(_signers.length >= _requiredApprovals, "Invalid signer count");
        require(_requiredApprovals > 0, "Required approvals must be > 0");
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(REGISTRAR_ROLE, msg.sender);
        
        multisigSigners = _signers;
        requiredMultisigApprovals = _requiredApprovals;
        
        for (uint256 i = 0; i < _signers.length; i++) {
            _grantRole(ADMIN_ROLE, _signers[i]);
        }
    }
    
    // Data Registration Functions
    
    function registerData(
        bytes32 _dataHash,
        string calldata _ipfsHash,
        string calldata _dataType,
        string calldata _metadata
    ) external onlyRole(REGISTRAR_ROLE) whenNotPaused nonReentrant returns (bool) {
        require(dataRecords[_dataHash].timestamp == 0, "Data already registered");
        
        DataRecord memory record = DataRecord({
            dataHash: _dataHash,
            ipfsHash: _ipfsHash,
            owner: msg.sender,
            timestamp: block.timestamp,
            dataType: _dataType,
            metadata: _metadata,
            verified: false,
            verifier: address(0)
        });
        
        dataRecords[_dataHash] = record;
        dataHistory[_dataHash].push(record);
        allDataHashes.push(_dataHash);
        
        emit DataRegistered(_dataHash, _ipfsHash, msg.sender, _dataType, block.timestamp);
        
        return true;
    }
    
    function updateData(
        bytes32 _dataHash,
        string calldata _newIpfsHash,
        string calldata _metadata
    ) external whenNotPaused nonReentrant returns (bool) {
        DataRecord storage record = dataRecords[_dataHash];
        require(record.timestamp > 0, "Data not found");
        require(record.owner == msg.sender || hasRole(ADMIN_ROLE, msg.sender), "Not authorized");
        
        record.ipfsHash = _newIpfsHash;
        record.metadata = _metadata;
        record.timestamp = block.timestamp;
        record.verified = false;
        record.verifier = address(0);
        
        dataHistory[_dataHash].push(record);
        
        emit DataUpdated(_dataHash, _newIpfsHash, msg.sender, block.timestamp);
        
        return true;
    }
    
    function verifyData(bytes32 _dataHash) external onlyRole(VERIFIER_ROLE) returns (bool) {
        DataRecord storage record = dataRecords[_dataHash];
        require(record.timestamp > 0, "Data not found");
        require(!record.verified, "Already verified");
        
        record.verified = true;
        record.verifier = msg.sender;
        
        emit DataVerified(_dataHash, msg.sender, block.timestamp);
        
        return true;
    }
    
    function getDataRecord(bytes32 _dataHash) external view returns (
        string memory ipfsHash,
        address owner,
        uint256 timestamp,
        string memory dataType,
        bool verified
    ) {
        DataRecord memory record = dataRecords[_dataHash];
        return (record.ipfsHash, record.owner, record.timestamp, record.dataType, record.verified);
    }
    
    function getDataHistory(bytes32 _dataHash) external view returns (DataRecord[] memory) {
        return dataHistory[_dataHash];
    }
    
    // Mineral Rights Functions
    
    function registerMineralRight(
        string calldata _rightId,
        string calldata _location,
        string calldata _mineralType,
        uint256 _expirationDate,
        string calldata _metadata
    ) external onlyRole(REGISTRAR_ROLE) whenNotPaused nonReentrant returns (bool) {
        require(mineralRights[_rightId].registrationDate == 0, "Right already registered");
        require(_expirationDate > block.timestamp, "Invalid expiration date");
        
        mineralRights[_rightId] = MineralRight({
            rightId: _rightId,
            owner: msg.sender,
            location: _location,
            mineralType: _mineralType,
            registrationDate: block.timestamp,
            expirationDate: _expirationDate,
            active: true,
            metadata: _metadata
        });
        
        allRightIds.push(_rightId);
        
        emit MineralRightRegistered(_rightId, msg.sender, _location, _mineralType, block.timestamp);
        
        return true;
    }
    
    function transferMineralRight(
        string calldata _rightId,
        address _newOwner
    ) external whenNotPaused nonReentrant returns (bool) {
        MineralRight storage right = mineralRights[_rightId];
        require(right.registrationDate > 0, "Right not found");
        require(right.owner == msg.sender, "Not the owner");
        require(right.active, "Right not active");
        require(_newOwner != address(0), "Invalid new owner");
        
        address previousOwner = right.owner;
        right.owner = _newOwner;
        
        emit MineralRightTransferred(_rightId, previousOwner, _newOwner, block.timestamp);
        
        return true;
    }
    
    function getMineralRight(string calldata _rightId) external view returns (
        address owner,
        string memory location,
        string memory mineralType,
        uint256 registrationDate,
        uint256 expirationDate,
        bool active
    ) {
        MineralRight memory right = mineralRights[_rightId];
        return (right.owner, right.location, right.mineralType, 
                right.registrationDate, right.expirationDate, right.active);
    }
    
    // Multisig Functions
    
    function proposeMultisig(bytes calldata _data) external returns (uint256) {
        require(hasRole(ADMIN_ROLE, msg.sender), "Not a signer");
        
        uint256 txId = multisigNonce++;
        MultisigTransaction storage txn = multisigTransactions[txId];
        txn.proposer = msg.sender;
        txn.data = _data;
        txn.requiredApprovals = requiredMultisigApprovals;
        txn.approvalCount = 1;
        txn.approvals[msg.sender] = true;
        
        emit MultisigProposed(txId, msg.sender, _data);
        emit MultisigApproved(txId, msg.sender);
        
        return txId;
    }
    
    function approveMultisig(uint256 _txId) external {
        require(hasRole(ADMIN_ROLE, msg.sender), "Not a signer");
        MultisigTransaction storage txn = multisigTransactions[_txId];
        require(!txn.executed, "Already executed");
        require(!txn.approvals[msg.sender], "Already approved");
        
        txn.approvals[msg.sender] = true;
        txn.approvalCount++;
        
        emit MultisigApproved(_txId, msg.sender);
    }
    
    function executeMultisig(uint256 _txId) external nonReentrant {
        MultisigTransaction storage txn = multisigTransactions[_txId];
        require(!txn.executed, "Already executed");
        require(txn.approvalCount >= txn.requiredApprovals, "Not enough approvals");
        
        txn.executed = true;
        
        (bool success,) = address(this).call(txn.data);
        
        emit MultisigExecuted(_txId, success);
    }
    
    // Admin Functions
    
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
    
    function addSigner(address _signer) external onlyRole(DEFAULT_ADMIN_ROLE) {
        multisigSigners.push(_signer);
        _grantRole(ADMIN_ROLE, _signer);
    }
    
    function removeSigner(address _signer) external onlyRole(DEFAULT_ADMIN_ROLE) {
        for (uint256 i = 0; i < multisigSigners.length; i++) {
            if (multisigSigners[i] == _signer) {
                multisigSigners[i] = multisigSigners[multisigSigners.length - 1];
                multisigSigners.pop();
                _revokeRole(ADMIN_ROLE, _signer);
                break;
            }
        }
    }
    
    function updateRequiredApprovals(uint256 _required) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_required > 0 && _required <= multisigSigners.length, "Invalid requirement");
        requiredMultisigApprovals = _required;
    }
    
    // View Functions
    
    function getAllDataHashes() external view returns (bytes32[] memory) {
        return allDataHashes;
    }
    
    function getAllRightIds() external view returns (string[] memory) {
        return allRightIds;
    }
    
    function getSigners() external view returns (address[] memory) {
        return multisigSigners;
    }
    
    function isApproved(uint256 _txId, address _signer) external view returns (bool) {
        return multisigTransactions[_txId].approvals[_signer];
    }
}
'''


class GasOptimizer:
    """
    Gas optimization for blockchain transactions.
    
    Provides batching, gas estimation, and optimization strategies.
    """
    
    def __init__(self, max_batch_size: int = 50,
                 max_gas_price_gwei: float = 100.0):
        self.max_batch_size = max_batch_size
        self.max_gas_price_gwei = max_gas_price_gwei
        self.pending_transactions: List[Dict] = []
        self._lock = threading.Lock()
        
    def estimate_gas(self, transaction_type: str, params: Dict) -> int:
        """
        Estimate gas for a transaction.
        
        Args:
            transaction_type: Type of transaction
            params: Transaction parameters
            
        Returns:
            Estimated gas units
        """
        # Base gas costs
        base_costs = {
            'register_data': 150000,
            'update_data': 80000,
            'verify_data': 50000,
            'register_mineral_right': 200000,
            'transfer_mineral_right': 70000,
            'propose_multisig': 100000,
            'approve_multisig': 50000,
            'execute_multisig': 150000
        }
        
        base = base_costs.get(transaction_type, 100000)
        
        # Add cost for data size
        data_size = len(json.dumps(params))
        data_cost = (data_size // 32) * 68  # 68 gas per 32 bytes
        
        return base + data_cost
        
    def get_optimal_gas_price(self, urgency: str = 'normal') -> int:
        """
        Get optimal gas price based on network conditions.
        
        Args:
            urgency: 'slow', 'normal', 'fast'
            
        Returns:
            Gas price in wei
        """
        # Simulated gas prices (in production, fetch from gas oracle)
        base_prices = {
            'slow': 20,
            'normal': 35,
            'fast': 50
        }
        
        gwei = min(base_prices.get(urgency, 35), self.max_gas_price_gwei)
        return int(gwei * 1e9)
        
    def add_to_batch(self, transaction: Dict) -> int:
        """
        Add transaction to batch queue.
        
        Args:
            transaction: Transaction to batch
            
        Returns:
            Position in batch
        """
        with self._lock:
            self.pending_transactions.append(transaction)
            return len(self.pending_transactions) - 1
            
    def get_batch(self) -> List[Dict]:
        """Get current batch of transactions."""
        with self._lock:
            batch = self.pending_transactions[:self.max_batch_size]
            self.pending_transactions = self.pending_transactions[self.max_batch_size:]
            return batch
            
    def optimize_batch(self, transactions: List[Dict]) -> List[Dict]:
        """
        Optimize a batch of transactions.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Optimized transaction list
        """
        # Sort by gas price (higher first for faster confirmation)
        sorted_txs = sorted(
            transactions,
            key=lambda x: x.get('gas_price', 0),
            reverse=True
        )
        
        # Group similar transactions
        grouped: Dict[str, List[Dict]] = {}
        for tx in sorted_txs:
            tx_type = tx.get('type', 'unknown')
            if tx_type not in grouped:
                grouped[tx_type] = []
            grouped[tx_type].append(tx)
            
        # Merge where possible
        optimized = []
        for tx_type, txs in grouped.items():
            if tx_type == 'register_data' and len(txs) > 1:
                # Batch data registrations
                optimized.append({
                    'type': 'batch_register_data',
                    'transactions': txs,
                    'gas_estimate': sum(tx.get('gas_estimate', 100000) for tx in txs) * 0.8
                })
            else:
                optimized.extend(txs)
                
        return optimized
        
    def calculate_batch_savings(self, transactions: List[Dict]) -> Dict[str, float]:
        """Calculate gas savings from batching."""
        individual_gas = sum(tx.get('gas_estimate', 100000) for tx in transactions)
        
        optimized = self.optimize_batch(transactions)
        batched_gas = sum(
            tx.get('gas_estimate', 100000) 
            for tx in optimized
        )
        
        savings = individual_gas - batched_gas
        savings_percent = (savings / individual_gas) * 100 if individual_gas > 0 else 0
        
        return {
            'individual_gas': individual_gas,
            'batched_gas': batched_gas,
            'savings': savings,
            'savings_percent': savings_percent
        }


class EventSubscriber:
    """
    Blockchain event subscription system.
    
    Listens for and processes blockchain events in real-time.
    """
    
    def __init__(self):
        self.subscriptions: Dict[EventType, List[Callable]] = {}
        self.event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
        self._listener_thread: Optional[threading.Thread] = None
        
    def subscribe(self, event_type: EventType,
                 callback: Callable[[Dict], None]) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Callback function for events
            
        Returns:
            Subscription ID
        """
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
            
        self.subscriptions[event_type].append(callback)
        
        subscription_id = f"{event_type.value}_{len(self.subscriptions[event_type])}"
        logger.info(f"Subscribed to {event_type.value}: {subscription_id}")
        
        return subscription_id
        
    def unsubscribe(self, event_type: EventType, callback: Callable) -> bool:
        """Unsubscribe from an event type."""
        if event_type in self.subscriptions:
            try:
                self.subscriptions[event_type].remove(callback)
                return True
            except ValueError:
                pass
        return False
        
    def start(self, poll_interval: float = 2.0) -> None:
        """Start event listening."""
        if self._running:
            return
            
        self._running = True
        
        self._processor_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self._processor_thread.start()
        
        self._listener_thread = threading.Thread(
            target=self._listen_for_events,
            args=(poll_interval,),
            daemon=True
        )
        self._listener_thread.start()
        
        logger.info("Event subscriber started")
        
    def stop(self) -> None:
        """Stop event listening."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join(timeout=2)
        if self._listener_thread:
            self._listener_thread.join(timeout=2)
        logger.info("Event subscriber stopped")
        
    def emit_event(self, event_type: EventType, data: Dict) -> None:
        """Emit an event (for testing or local events)."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'block_number': None
        }
        self.event_queue.put(event)
        
    def _listen_for_events(self, poll_interval: float) -> None:
        """Listen for blockchain events."""
        last_block = 0
        
        while self._running:
            try:
                # Simulate fetching events from blockchain
                # In production, use web3.py event filters
                events = self._fetch_events(last_block)
                
                for event in events:
                    self.event_queue.put(event)
                    if event.get('block_number'):
                        last_block = max(last_block, event['block_number'])
                        
            except Exception as e:
                logger.error(f"Event listener error: {e}")
                
            time.sleep(poll_interval)
            
    def _fetch_events(self, from_block: int) -> List[Dict]:
        """Fetch events from an external chain.

        No external chain is connected in this deployment (offline ledger);
        all events originate locally via emit_event(). Returns empty — it
        never fabricates events.
        """
        return []
        
    def _process_events(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = self.event_queue.get(timeout=1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                
    def _dispatch_event(self, event: Dict) -> None:
        """Dispatch event to subscribers."""
        event_type = event.get('type')
        
        if event_type in self.subscriptions:
            for callback in self.subscriptions[event_type]:
                try:
                    callback(event['data'])
                except Exception as e:
                    logger.error(f"Event callback error: {e}")


class MultisigManager:
    """
    Multi-signature transaction manager.
    
    Handles proposal, approval, and execution of multi-sig transactions.
    """
    
    def __init__(self, required_signatures: int = 2,
                 signers: List[str] = None):
        self.required_signatures = required_signatures
        self.signers: Set[str] = set(signers or [])
        self.proposals: Dict[str, MultisigProposal] = {}
        self.callbacks: List[Callable[[MultisigProposal], None]] = []
        self._lock = threading.Lock()
        
    def add_signer(self, address: str) -> None:
        """Add a signer."""
        with self._lock:
            self.signers.add(address)
            
    def remove_signer(self, address: str) -> None:
        """Remove a signer."""
        with self._lock:
            self.signers.discard(address)
            
    def set_required_signatures(self, count: int) -> None:
        """Set required signature count."""
        if count > len(self.signers):
            raise ValueError("Required signatures cannot exceed signer count")
        self.required_signatures = count
        
    def register_callback(self, callback: Callable[[MultisigProposal], None]) -> None:
        """Register callback for proposal events."""
        self.callbacks.append(callback)
        
    def propose(self, proposer: str, action: str, params: Dict,
               expiry_hours: int = 72) -> MultisigProposal:
        """
        Create a new multi-sig proposal.
        
        Args:
            proposer: Address of proposer
            action: Action to execute
            params: Action parameters
            expiry_hours: Hours until proposal expires
            
        Returns:
            Created proposal
        """
        if proposer not in self.signers:
            raise ValueError("Proposer is not a signer")
            
        proposal_id = hashlib.sha256(
            f"{proposer}{action}{json.dumps(params)}{time.time()}".encode()
        ).hexdigest()[:16]
        
        proposal = MultisigProposal(
            proposal_id=proposal_id,
            proposer=proposer,
            action=action,
            params=params,
            required_signatures=self.required_signatures,
            signatures=[proposer],  # Proposer auto-signs
            expires_at=datetime.now() + timedelta(hours=expiry_hours)
        )
        
        with self._lock:
            self.proposals[proposal_id] = proposal
            
        self._notify_callbacks(proposal)
        logger.info(f"Multisig proposal created: {proposal_id}")
        
        return proposal
        
    def approve(self, proposal_id: str, signer: str) -> bool:
        """
        Approve a proposal.
        
        Args:
            proposal_id: Proposal to approve
            signer: Approving signer address
            
        Returns:
            True if approval successful
        """
        with self._lock:
            proposal = self.proposals.get(proposal_id)
            
            if not proposal:
                raise ValueError("Proposal not found")
                
            if proposal.executed:
                raise ValueError("Proposal already executed")
                
            if proposal.is_expired():
                raise ValueError("Proposal expired")
                
            if signer not in self.signers:
                raise ValueError("Not a valid signer")
                
            if signer in proposal.signatures:
                raise ValueError("Already signed")
                
            proposal.signatures.append(signer)
            
        self._notify_callbacks(proposal)
        logger.info(f"Proposal {proposal_id} approved by {signer}")
        
        return True
        
    def execute(self, proposal_id: str) -> Dict[str, Any]:
        """
        Execute an approved proposal.
        
        Args:
            proposal_id: Proposal to execute
            
        Returns:
            Execution result
        """
        with self._lock:
            proposal = self.proposals.get(proposal_id)
            
            if not proposal:
                raise ValueError("Proposal not found")
                
            if proposal.executed:
                raise ValueError("Proposal already executed")
                
            if proposal.is_expired():
                raise ValueError("Proposal expired")
                
            if not proposal.is_approved():
                raise ValueError("Not enough signatures")
                
            proposal.executed = True
            
        # Execute the action
        result = self._execute_action(proposal.action, proposal.params)
        
        self._notify_callbacks(proposal)
        logger.info(f"Proposal {proposal_id} executed")
        
        return result
        
    def get_proposal(self, proposal_id: str) -> Optional[MultisigProposal]:
        """Get a proposal by ID."""
        return self.proposals.get(proposal_id)
        
    def get_pending_proposals(self) -> List[MultisigProposal]:
        """Get all pending proposals."""
        return [
            p for p in self.proposals.values()
            if not p.executed and not p.is_expired()
        ]
        
    def _execute_action(self, action: str, params: Dict) -> Dict[str, Any]:
        """Execute the proposal action."""
        # In production, this would call the smart contract
        return {
            'action': action,
            'params': params,
            'status': 'executed',
            'timestamp': datetime.now().isoformat()
        }
        
    def _notify_callbacks(self, proposal: MultisigProposal) -> None:
        """Notify registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(proposal)
            except Exception as e:
                logger.error(f"Multisig callback error: {e}")


class Layer2Bridge:
    """
    Layer-2 scaling solution bridge.
    
    Supports Optimism, Arbitrum, and Polygon for scalability.
    """
    
    def __init__(self, l1_provider: str = None, l2_provider: str = None):
        self.l1_provider = l1_provider
        self.l2_provider = l2_provider
        self.supported_networks = {
            'optimism': {
                'chain_id': 10,
                'bridge_address': '0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1'
            },
            'arbitrum': {
                'chain_id': 42161,
                'bridge_address': '0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a'
            },
            'polygon': {
                'chain_id': 137,
                'bridge_address': '0xA0c68C638235ee32657e8f720a23ceC1bFc77C77'
            }
        }
        self.pending_deposits: List[Dict] = []
        self.pending_withdrawals: List[Dict] = []
        
    def deposit_to_l2(self, network: str, amount: float,
                     recipient: str) -> Dict[str, Any]:
        """
        Deposit funds to Layer-2.
        
        Args:
            network: L2 network name
            amount: Amount to deposit
            recipient: Recipient address on L2
            
        Returns:
            Deposit transaction details
        """
        if network not in self.supported_networks:
            raise ValueError(f"Unsupported network: {network}")
            
        deposit = {
            'id': hashlib.sha256(f"{network}{amount}{recipient}{time.time()}".encode()).hexdigest()[:16],
            'network': network,
            'amount': amount,
            'recipient': recipient,
            'status': 'pending',
            'l1_tx_hash': None,
            'l2_tx_hash': None,
            'created_at': datetime.now().isoformat()
        }
        
        self.pending_deposits.append(deposit)
        
        # Simulate L1 transaction
        deposit['l1_tx_hash'] = f"0x{hashlib.sha256(deposit['id'].encode()).hexdigest()}"
        deposit['status'] = 'l1_confirmed'
        
        logger.info(f"L2 deposit initiated: {deposit['id']}")
        
        return deposit
        
    def withdraw_from_l2(self, network: str, amount: float,
                        recipient: str) -> Dict[str, Any]:
        """
        Withdraw funds from Layer-2.
        
        Args:
            network: L2 network name
            amount: Amount to withdraw
            recipient: Recipient address on L1
            
        Returns:
            Withdrawal transaction details
        """
        if network not in self.supported_networks:
            raise ValueError(f"Unsupported network: {network}")
            
        withdrawal = {
            'id': hashlib.sha256(f"{network}{amount}{recipient}{time.time()}".encode()).hexdigest()[:16],
            'network': network,
            'amount': amount,
            'recipient': recipient,
            'status': 'pending',
            'l2_tx_hash': None,
            'l1_tx_hash': None,
            'challenge_period_ends': (datetime.now() + timedelta(days=7)).isoformat(),
            'created_at': datetime.now().isoformat()
        }
        
        self.pending_withdrawals.append(withdrawal)
        
        # Simulate L2 transaction
        withdrawal['l2_tx_hash'] = f"0x{hashlib.sha256(withdrawal['id'].encode()).hexdigest()}"
        withdrawal['status'] = 'l2_confirmed'
        
        logger.info(f"L2 withdrawal initiated: {withdrawal['id']}")
        
        return withdrawal
        
    def get_deposit_status(self, deposit_id: str) -> Optional[Dict]:
        """Get deposit status."""
        for deposit in self.pending_deposits:
            if deposit['id'] == deposit_id:
                return deposit
        return None
        
    def get_withdrawal_status(self, withdrawal_id: str) -> Optional[Dict]:
        """Get withdrawal status."""
        for withdrawal in self.pending_withdrawals:
            if withdrawal['id'] == withdrawal_id:
                return withdrawal
        return None
        
    def finalize_withdrawal(self, withdrawal_id: str) -> Dict[str, Any]:
        """
        Finalize a withdrawal after challenge period.
        
        Args:
            withdrawal_id: Withdrawal to finalize
            
        Returns:
            Finalization result
        """
        for withdrawal in self.pending_withdrawals:
            if withdrawal['id'] == withdrawal_id:
                if withdrawal['status'] != 'l2_confirmed':
                    raise ValueError("Withdrawal not ready for finalization")
                    
                # Check challenge period
                challenge_end = datetime.fromisoformat(withdrawal['challenge_period_ends'])
                if datetime.now() < challenge_end:
                    raise ValueError("Challenge period not ended")
                    
                withdrawal['l1_tx_hash'] = f"0x{hashlib.sha256(f'finalize_{withdrawal_id}'.encode()).hexdigest()}"
                withdrawal['status'] = 'finalized'
                
                logger.info(f"Withdrawal finalized: {withdrawal_id}")
                
                return withdrawal
                
        raise ValueError("Withdrawal not found")


class AdvancedBlockchainManager:
    """
    Advanced blockchain management combining all enhanced features.
    """
    
    def __init__(self, signers: List[str] = None,
                 required_signatures: int = 2):
        self.gas_optimizer = GasOptimizer()
        self.event_subscriber = EventSubscriber()
        self.multisig_manager = MultisigManager(
            required_signatures=required_signatures,
            signers=signers or []
        )
        self.l2_bridge = Layer2Bridge()
        
        # Real local cryptographic ledger (ed25519 / hmac-sha256 signed)
        self.ledger = LocalCryptoLedger()

        # Transaction history
        self.transactions: List[Transaction] = []
        self._lock = threading.Lock()
        
    def start(self) -> None:
        """Start all services."""
        self.event_subscriber.start()
        logger.info("Advanced blockchain manager started")
        
    def stop(self) -> None:
        """Stop all services."""
        self.event_subscriber.stop()
        logger.info("Advanced blockchain manager stopped")
        
    def register_data(self, data_hash: str, ipfs_hash: str,
                     data_type: str, metadata: Dict,
                     use_multisig: bool = False) -> Dict[str, Any]:
        """
        Register data with optional multi-sig.
        
        Args:
            data_hash: Hash of the data
            ipfs_hash: IPFS content hash
            data_type: Type of data
            metadata: Additional metadata
            use_multisig: Whether to use multi-sig
            
        Returns:
            Registration result
        """
        params = {
            'data_hash': data_hash,
            'ipfs_hash': ipfs_hash,
            'data_type': data_type,
            'metadata': metadata
        }
        
        if use_multisig:
            # Create multi-sig proposal
            proposal = self.multisig_manager.propose(
                proposer=next(iter(self.multisig_manager.signers)) if self.multisig_manager.signers else 'default',
                action='register_data',
                params=params
            )
            return {'proposal': proposal.to_dict()}
            
        # Direct registration — anchor in the real signed ledger
        gas_estimate = self.gas_optimizer.estimate_gas('register_data', params)
        gas_price = self.gas_optimizer.get_optimal_gas_price('normal')

        ledger_tx = {
            'action': 'register_data',
            **params,
        }
        block = self.ledger.add_block([ledger_tx])

        tx = Transaction(
            tx_hash=self.ledger.transaction_hash(ledger_tx),
            from_address=self.ledger.signer.address,
            to_address=self.ledger.CONTRACT_ADDRESS,
            value=0,
            gas_used=gas_estimate,
            gas_price=gas_price,
            status=TransactionStatus.CONFIRMED,
            block_number=block.index,
            data=params
        )

        with self._lock:
            self.transactions.append(tx)

        # Emit event
        self.event_subscriber.emit_event(
            EventType.DATA_REGISTERED,
            {'data_hash': data_hash, 'ipfs_hash': ipfs_hash}
        )

        return {
            'transaction': tx.to_dict(),
            'ledger': {
                'block_index': block.index,
                'block_hash': block.hash,
                'merkle_root': block.merkle_root,
                'signature': block.signature,
                'scheme': self.ledger.scheme,
            }
        }

    def verify_ledger(self) -> Dict[str, Any]:
        """Verify the full provenance chain (links, Merkle roots, signatures)."""
        return self.ledger.verify_chain()

    def get_ledger(self) -> Dict[str, Any]:
        """Export the full ledger."""
        return self.ledger.to_dict()

    def record_multisig_execution(self, proposal: MultisigProposal) -> Dict[str, Any]:
        """Anchor an executed multi-sig proposal in the ledger."""
        block = self.ledger.add_block([{
            'action': 'multisig_executed',
            'proposal_id': proposal.proposal_id,
            'proposer': proposal.proposer,
            'params': proposal.params,
            'signatures': proposal.signatures,
        }])
        return {'block_index': block.index, 'block_hash': block.hash}
        
    def get_contract_source(self) -> str:
        """Get the Solidity smart contract source code."""
        return MINERAL_PROVENANCE_CONTRACT
        
    def subscribe_to_events(self, event_type: EventType,
                           callback: Callable[[Dict], None]) -> str:
        """Subscribe to blockchain events."""
        return self.event_subscriber.subscribe(event_type, callback)
        
    def get_gas_estimate(self, transaction_type: str, params: Dict) -> Dict[str, Any]:
        """Get gas estimate for a transaction."""
        gas = self.gas_optimizer.estimate_gas(transaction_type, params)
        gas_price = self.gas_optimizer.get_optimal_gas_price('normal')
        
        return {
            'gas_estimate': gas,
            'gas_price_wei': gas_price,
            'gas_price_gwei': gas_price / 1e9,
            'estimated_cost_eth': (gas * gas_price) / 1e18
        }
        
    def batch_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Batch multiple transactions for gas savings."""
        for tx in transactions:
            self.gas_optimizer.add_to_batch(tx)
            
        batch = self.gas_optimizer.get_batch()
        savings = self.gas_optimizer.calculate_batch_savings(batch)
        optimized = self.gas_optimizer.optimize_batch(batch)
        
        return {
            'batch_size': len(batch),
            'optimized_count': len(optimized),
            'savings': savings
        }


def create_advanced_blockchain_manager(signers: List[str] = None,
                                       required_signatures: int = 2) -> AdvancedBlockchainManager:
    """Factory function to create advanced blockchain manager."""
    return AdvancedBlockchainManager(signers, required_signatures)


def create_gas_optimizer(max_batch_size: int = 50) -> GasOptimizer:
    """Factory function to create gas optimizer."""
    return GasOptimizer(max_batch_size=max_batch_size)


def create_event_subscriber() -> EventSubscriber:
    """Factory function to create event subscriber."""
    return EventSubscriber()


def create_multisig_manager(required_signatures: int = 2,
                           signers: List[str] = None) -> MultisigManager:
    """Factory function to create multisig manager."""
    return MultisigManager(required_signatures, signers)


def create_l2_bridge() -> Layer2Bridge:
    """Factory function to create L2 bridge."""
    return Layer2Bridge()
