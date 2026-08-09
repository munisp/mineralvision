"""
Blockchain Data Provenance System for MineralVision

This module implements a blockchain-based data provenance system that provides
an immutable record of data collection, processing, and analysis.

It uses Ethereum-compatible smart contracts for mineral rights management and
IPFS for decentralized storage of large datasets.
"""

import os
import json
import hashlib
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union
import requests
import ipfshttpclient
from web3 import Web3, HTTPProvider
from eth_account import Account
from eth_account.signers.local import LocalAccount
from eth_typing import ChecksumAddress
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ETHEREUM_NODE_URL = os.environ.get("ETHEREUM_NODE_URL", "http://localhost:8545")
IPFS_API_URL = os.environ.get("IPFS_API_URL", "/ip4/127.0.0.1/tcp/5001")
CONTRACT_ADDRESS = os.environ.get("DATA_PROVENANCE_CONTRACT_ADDRESS", "")
PRIVATE_KEY = os.environ.get("ETHEREUM_PRIVATE_KEY", "")

# Smart contract ABI (Application Binary Interface)
CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "owner",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "ipfsHash",
                "type": "string"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256"
            }
        ],
        "name": "DataRegistered",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "owner",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "ipfsHash",
                "type": "string"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256"
            }
        ],
        "name": "DataUpdated",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "rightId",
                "type": "bytes32"
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "owner",
                "type": "address"
            },
            {
                "indexed": True,
                "internalType": "address",
                "name": "recipient",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256"
            }
        ],
        "name": "RightTransferred",
        "type": "event"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            }
        ],
        "name": "getDataProvenance",
        "outputs": [
            {
                "components": [
                    {
                        "internalType": "address",
                        "name": "owner",
                        "type": "address"
                    },
                    {
                        "internalType": "string",
                        "name": "ipfsHash",
                        "type": "string"
                    },
                    {
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "dataType",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "metadata",
                        "type": "string"
                    }
                ],
                "internalType": "struct DataProvenanceContract.DataRecord",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            }
        ],
        "name": "getDataProvenanceHistory",
        "outputs": [
            {
                "components": [
                    {
                        "internalType": "address",
                        "name": "owner",
                        "type": "address"
                    },
                    {
                        "internalType": "string",
                        "name": "ipfsHash",
                        "type": "string"
                    },
                    {
                        "internalType": "uint256",
                        "name": "timestamp",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "operation",
                        "type": "string"
                    }
                ],
                "internalType": "struct DataProvenanceContract.ProvenanceRecord[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "rightId",
                "type": "bytes32"
            }
        ],
        "name": "getMineralRight",
        "outputs": [
            {
                "components": [
                    {
                        "internalType": "address",
                        "name": "owner",
                        "type": "address"
                    },
                    {
                        "internalType": "string",
                        "name": "geographicBoundary",
                        "type": "string"
                    },
                    {
                        "internalType": "uint256",
                        "name": "validUntil",
                        "type": "uint256"
                    },
                    {
                        "internalType": "string",
                        "name": "mineralTypes",
                        "type": "string"
                    },
                    {
                        "internalType": "string",
                        "name": "metadata",
                        "type": "string"
                    }
                ],
                "internalType": "struct DataProvenanceContract.MineralRight",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            },
            {
                "internalType": "string",
                "name": "ipfsHash",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "dataType",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "metadata",
                "type": "string"
            }
        ],
        "name": "registerData",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "rightId",
                "type": "bytes32"
            },
            {
                "internalType": "string",
                "name": "geographicBoundary",
                "type": "string"
            },
            {
                "internalType": "uint256",
                "name": "validUntil",
                "type": "uint256"
            },
            {
                "internalType": "string",
                "name": "mineralTypes",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "metadata",
                "type": "string"
            }
        ],
        "name": "registerMineralRight",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "rightId",
                "type": "bytes32"
            },
            {
                "internalType": "address",
                "name": "newOwner",
                "type": "address"
            }
        ],
        "name": "transferMineralRight",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "dataId",
                "type": "bytes32"
            },
            {
                "internalType": "string",
                "name": "ipfsHash",
                "type": "string"
            },
            {
                "internalType": "string",
                "name": "metadata",
                "type": "string"
            }
        ],
        "name": "updateData",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

class BlockchainDataProvenance:
    """
    Blockchain-based data provenance system for MineralVision.
    
    This class provides functionality to:
    1. Register data on the blockchain with IPFS storage
    2. Verify data integrity and provenance
    3. Manage mineral rights through smart contracts
    4. Track data lineage and processing history
    """
    
    def __init__(
        self, 
        ethereum_node_url: str = ETHEREUM_NODE_URL,
        ipfs_api_url: str = IPFS_API_URL,
        contract_address: str = CONTRACT_ADDRESS,
        private_key: str = PRIVATE_KEY
    ):
        """
        Initialize the blockchain data provenance system.
        
        Args:
            ethereum_node_url: URL of the Ethereum node
            ipfs_api_url: URL of the IPFS API
            contract_address: Address of the deployed smart contract
            private_key: Ethereum private key for signing transactions
        """
        # Initialize Web3 connection
        self.w3 = Web3(HTTPProvider(ethereum_node_url))
        if not self.w3.is_connected():
            logger.warning(f"Could not connect to Ethereum node at {ethereum_node_url}")
            self.ethereum_available = False
        else:
            self.ethereum_available = True
            
            # Set up account from private key if provided
            if private_key:
                self.account = Account.from_key(private_key)
                logger.info(f"Using account: {self.account.address}")
            else:
                self.account = None
                logger.warning("No private key provided, read-only mode enabled")
            
            # Initialize contract
            if contract_address:
                self.contract_address = Web3.to_checksum_address(contract_address)
                self.contract = self.w3.eth.contract(
                    address=self.contract_address, 
                    abi=CONTRACT_ABI
                )
                logger.info(f"Contract initialized at {contract_address}")
            else:
                self.contract = None
                logger.warning("No contract address provided, contract functionality disabled")
        
        # Initialize IPFS connection
        try:
            self.ipfs = ipfshttpclient.connect(ipfs_api_url)
            self.ipfs_available = True
            logger.info("Connected to IPFS")
        except Exception as e:
            logger.warning(f"Could not connect to IPFS: {e}")
            self.ipfs_available = False
            self.ipfs = None
        
        # Local storage for offline operation
        self.local_storage_path = os.environ.get(
            "LOCAL_STORAGE_PATH", 
            "/tmp/mineralvision/blockchain"
        )
        os.makedirs(self.local_storage_path, exist_ok=True)
    
    def register_data(
        self, 
        data: Union[Dict, List, str, bytes],
        data_type: str,
        metadata: Dict[str, Any] = None,
        offline_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Register data in the blockchain provenance system.
        
        Args:
            data: The data to register (dict, list, string, or bytes)
            data_type: Type of data (e.g., 'geological', 'geophysical')
            metadata: Additional metadata about the data
            offline_mode: Whether to operate in offline mode
            
        Returns:
            Dictionary with registration details including data_id and ipfs_hash
        """
        # Generate a unique ID for the data if not in metadata
        data_id = metadata.get('data_id', str(uuid.uuid4())) if metadata else str(uuid.uuid4())
        
        # Prepare data for storage
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Calculate data hash
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'data_id': data_id,
            'data_type': data_type,
            'data_hash': data_hash,
            'timestamp': datetime.datetime.now().isoformat(),
            'version': metadata.get('version', '1.0')
        })
        
        result = {
            'data_id': data_id,
            'data_hash': data_hash,
            'timestamp': metadata['timestamp'],
            'data_type': data_type,
            'metadata': metadata
        }
        
        # Store in IPFS if available
        if self.ipfs_available and not offline_mode:
            try:
                # Store data
                ipfs_data_result = self.ipfs.add_bytes(data_bytes)
                data_ipfs_hash = ipfs_data_result
                
                # Store metadata
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                ipfs_metadata_result = self.ipfs.add_bytes(metadata_bytes)
                metadata_ipfs_hash = ipfs_metadata_result
                
                result['ipfs_hash'] = data_ipfs_hash
                result['metadata_ipfs_hash'] = metadata_ipfs_hash
                
                logger.info(f"Data stored in IPFS with hash: {data_ipfs_hash}")
                logger.info(f"Metadata stored in IPFS with hash: {metadata_ipfs_hash}")
            except Exception as e:
                logger.error(f"Failed to store data in IPFS: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Register on blockchain if available
        if self.ethereum_available and self.contract and self.account and not offline_mode:
            try:
                # Convert data_id to bytes32
                data_id_bytes32 = self.w3.to_bytes(hexstr=data_id) if data_id.startswith('0x') else self.w3.to_bytes(text=data_id)
                
                # Prepare transaction
                tx = self.contract.functions.registerData(
                    data_id_bytes32,
                    result.get('ipfs_hash', ''),
                    data_type,
                    json.dumps(metadata)
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.eth.gas_price
                })
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                result['transaction_hash'] = tx_hash.hex()
                result['block_number'] = tx_receipt['blockNumber']
                
                logger.info(f"Data registered on blockchain with tx hash: {tx_hash.hex()}")
            except Exception as e:
                logger.error(f"Failed to register data on blockchain: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Store locally if in offline mode or as backup
        if offline_mode or not self.ethereum_available:
            local_file_path = os.path.join(self.local_storage_path, f"{data_id}.json")
            with open(local_file_path, 'w') as f:
                json.dump({
                    'data': data if isinstance(data, (dict, list)) else data_bytes.hex(),
                    'metadata': metadata
                }, f)
            
            result['local_file_path'] = local_file_path
            logger.info(f"Data stored locally at: {local_file_path}")
        
        return result
    
    def verify_data(
        self, 
        data: Union[Dict, List, str, bytes],
        data_id: str = None,
        ipfs_hash: str = None
    ) -> Dict[str, Any]:
        """
        Verify the integrity and provenance of data.
        
        Args:
            data: The data to verify
            data_id: ID of the data in the blockchain
            ipfs_hash: IPFS hash of the data
            
        Returns:
            Dictionary with verification results
        """
        # Calculate hash of provided data
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        calculated_hash = hashlib.sha256(data_bytes).hexdigest()
        
        result = {
            'calculated_hash': calculated_hash,
            'verified': False,
            'provenance': None
        }
        
        # Verify against IPFS if hash provided
        if self.ipfs_available and ipfs_hash:
            try:
                ipfs_data = self.ipfs.cat(ipfs_hash)
                ipfs_hash_calculated = hashlib.sha256(ipfs_data).hexdigest()
                
                result['ipfs_verified'] = calculated_hash == ipfs_hash_calculated
                result['ipfs_hash_calculated'] = ipfs_hash_calculated
                
                if result['ipfs_verified']:
                    logger.info(f"Data verified against IPFS hash: {ipfs_hash}")
                else:
                    logger.warning(f"Data verification against IPFS failed")
            except Exception as e:
                logger.error(f"Failed to verify data against IPFS: {e}")
                result['ipfs_error'] = str(e)
        
        # Verify against blockchain if data_id provided
        if self.ethereum_available and self.contract and data_id:
            try:
                # Convert data_id to bytes32
                data_id_bytes32 = self.w3.to_bytes(hexstr=data_id) if data_id.startswith('0x') else self.w3.to_bytes(text=data_id)
                
                # Get data record from blockchain
                data_record = self.contract.functions.getDataProvenance(data_id_bytes32).call()
                
                # Get provenance history
                provenance_history = self.contract.functions.getDataProvenanceHistory(data_id_bytes32).call()
                
                # Format provenance history
                formatted_history = []
                for record in provenance_history:
                    formatted_history.append({
                        'owner': record[0],
                        'ipfs_hash': record[1],
                        'timestamp': datetime.datetime.fromtimestamp(record[2]).isoformat(),
                        'operation': record[3]
                    })
                
                result['blockchain_record'] = {
                    'owner': data_record[0],
                    'ipfs_hash': data_record[1],
                    'timestamp': datetime.datetime.fromtimestamp(data_record[2]).isoformat(),
                    'data_type': data_record[3],
                    'metadata': json.loads(data_record[4]) if data_record[4] else {}
                }
                
                result['provenance_history'] = formatted_history
                
                # If IPFS hash is in the blockchain record, verify against it
                if result['blockchain_record']['ipfs_hash']:
                    if self.ipfs_available:
                        try:
                            blockchain_ipfs_hash = result['blockchain_record']['ipfs_hash']
                            ipfs_data = self.ipfs.cat(blockchain_ipfs_hash)
                            blockchain_hash_calculated = hashlib.sha256(ipfs_data).hexdigest()
                            
                            result['blockchain_verified'] = calculated_hash == blockchain_hash_calculated
                            result['blockchain_hash_calculated'] = blockchain_hash_calculated
                            
                            if result['blockchain_verified']:
                                logger.info(f"Data verified against blockchain record")
                            else:
                                logger.warning(f"Data verification against blockchain failed")
                        except Exception as e:
                            logger.error(f"Failed to verify data against blockchain IPFS hash: {e}")
                            result['blockchain_ipfs_error'] = str(e)
                
                # Set overall verification result
                result['verified'] = result.get('blockchain_verified', False) or result.get('ipfs_verified', False)
                result['provenance'] = result['blockchain_record']
                
                logger.info(f"Retrieved provenance information from blockchain for data_id: {data_id}")
            except Exception as e:
                logger.error(f"Failed to verify data against blockchain: {e}")
                result['blockchain_error'] = str(e)
        
        # Check local storage if blockchain verification failed or not available
        if (not result['verified'] or 'blockchain_error' in result) and data_id:
            local_file_path = os.path.join(self.local_storage_path, f"{data_id}.json")
            if os.path.exists(local_file_path):
                try:
                    with open(local_file_path, 'r') as f:
                        local_data = json.load(f)
                    
                    # Extract metadata
                    local_metadata = local_data.get('metadata', {})
                    
                    # Verify hash if present in metadata
                    if 'data_hash' in local_metadata:
                        result['local_verified'] = calculated_hash == local_metadata['data_hash']
                        result['local_hash'] = local_metadata['data_hash']
                        
                        if result['local_verified']:
                            logger.info(f"Data verified against local storage")
                        else:
                            logger.warning(f"Data verification against local storage failed")
                    
                    result['local_metadata'] = local_metadata
                    
                    # Update overall verification result
                    result['verified'] = result['verified'] or result.get('local_verified', False)
                    if not result['provenance'] and 'local_metadata' in result:
                        result['provenance'] = result['local_metadata']
                    
                    logger.info(f"Retrieved provenance information from local storage for data_id: {data_id}")
                except Exception as e:
                    logger.error(f"Failed to verify data against local storage: {e}")
                    result['local_error'] = str(e)
        
        return result
    
    def register_mineral_right(
        self,
        geographic_boundary: Dict[str, Any],
        valid_until: datetime.datetime,
        mineral_types: List[str],
        metadata: Dict[str, Any] = None,
        offline_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Register a mineral right on the blockchain.
        
        Args:
            geographic_boundary: GeoJSON representation of the geographic boundary
            valid_until: Datetime until which the right is valid
            mineral_types: List of mineral types covered by the right
            metadata: Additional metadata about the mineral right
            offline_mode: Whether to operate in offline mode
            
        Returns:
            Dictionary with registration details
        """
        if not self.ethereum_available or not self.contract or not self.account:
            if not offline_mode:
                logger.warning("Ethereum not available, falling back to offline mode")
                offline_mode = True
        
        # Generate a unique ID for the mineral right
        right_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'right_id': right_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'owner': self.account.address if self.account else "unknown"
        })
        
        # Convert geographic boundary to string
        geographic_boundary_str = json.dumps(geographic_boundary)
        
        # Convert mineral types to string
        mineral_types_str = json.dumps(mineral_types)
        
        # Convert valid_until to timestamp
        valid_until_timestamp = int(valid_until.timestamp())
        
        result = {
            'right_id': right_id,
            'geographic_boundary': geographic_boundary,
            'valid_until': valid_until.isoformat(),
            'mineral_types': mineral_types,
            'metadata': metadata
        }
        
        # Register on blockchain if available
        if self.ethereum_available and self.contract and self.account and not offline_mode:
            try:
                # Convert right_id to bytes32
                right_id_bytes32 = self.w3.to_bytes(hexstr=right_id) if right_id.startswith('0x') else self.w3.to_bytes(text=right_id)
                
                # Prepare transaction
                tx = self.contract.functions.registerMineralRight(
                    right_id_bytes32,
                    geographic_boundary_str,
                    valid_until_timestamp,
                    mineral_types_str,
                    json.dumps(metadata)
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.eth.gas_price
                })
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                result['transaction_hash'] = tx_hash.hex()
                result['block_number'] = tx_receipt['blockNumber']
                
                logger.info(f"Mineral right registered on blockchain with tx hash: {tx_hash.hex()}")
            except Exception as e:
                logger.error(f"Failed to register mineral right on blockchain: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Store locally if in offline mode or as backup
        if offline_mode or not self.ethereum_available:
            local_file_path = os.path.join(self.local_storage_path, f"right_{right_id}.json")
            with open(local_file_path, 'w') as f:
                json.dump({
                    'right_id': right_id,
                    'geographic_boundary': geographic_boundary,
                    'valid_until': valid_until.isoformat(),
                    'mineral_types': mineral_types,
                    'metadata': metadata
                }, f)
            
            result['local_file_path'] = local_file_path
            logger.info(f"Mineral right stored locally at: {local_file_path}")
        
        return result
    
    def transfer_mineral_right(
        self,
        right_id: str,
        new_owner_address: str,
        offline_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Transfer a mineral right to a new owner.
        
        Args:
            right_id: ID of the mineral right
            new_owner_address: Ethereum address of the new owner
            offline_mode: Whether to operate in offline mode
            
        Returns:
            Dictionary with transfer details
        """
        if not self.ethereum_available or not self.contract or not self.account:
            if not offline_mode:
                logger.warning("Ethereum not available, falling back to offline mode")
                offline_mode = True
        
        result = {
            'right_id': right_id,
            'new_owner': new_owner_address,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Transfer on blockchain if available
        if self.ethereum_available and self.contract and self.account and not offline_mode:
            try:
                # Convert right_id to bytes32
                right_id_bytes32 = self.w3.to_bytes(hexstr=right_id) if right_id.startswith('0x') else self.w3.to_bytes(text=right_id)
                
                # Convert new owner address to checksum address
                new_owner_checksum = Web3.to_checksum_address(new_owner_address)
                
                # Prepare transaction
                tx = self.contract.functions.transferMineralRight(
                    right_id_bytes32,
                    new_owner_checksum
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.eth.gas_price
                })
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                result['transaction_hash'] = tx_hash.hex()
                result['block_number'] = tx_receipt['blockNumber']
                
                logger.info(f"Mineral right transferred on blockchain with tx hash: {tx_hash.hex()}")
            except Exception as e:
                logger.error(f"Failed to transfer mineral right on blockchain: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Update locally if in offline mode or as backup
        if offline_mode or not self.ethereum_available:
            local_file_path = os.path.join(self.local_storage_path, f"right_{right_id}.json")
            if os.path.exists(local_file_path):
                try:
                    with open(local_file_path, 'r') as f:
                        right_data = json.load(f)
                    
                    # Update owner
                    right_data['metadata']['previous_owner'] = right_data['metadata'].get('owner', 'unknown')
                    right_data['metadata']['owner'] = new_owner_address
                    right_data['metadata']['transfer_timestamp'] = result['timestamp']
                    
                    # Save updated data
                    with open(local_file_path, 'w') as f:
                        json.dump(right_data, f)
                    
                    result['local_file_path'] = local_file_path
                    logger.info(f"Mineral right transfer recorded locally at: {local_file_path}")
                except Exception as e:
                    logger.error(f"Failed to update mineral right locally: {e}")
                    result['local_error'] = str(e)
            else:
                logger.warning(f"Mineral right not found locally: {right_id}")
                result['local_error'] = "Mineral right not found locally"
        
        return result
    
    def get_mineral_right(self, right_id: str) -> Dict[str, Any]:
        """
        Get information about a mineral right.
        
        Args:
            right_id: ID of the mineral right
            
        Returns:
            Dictionary with mineral right details
        """
        result = {
            'right_id': right_id,
            'found': False
        }
        
        # Check blockchain if available
        if self.ethereum_available and self.contract:
            try:
                # Convert right_id to bytes32
                right_id_bytes32 = self.w3.to_bytes(hexstr=right_id) if right_id.startswith('0x') else self.w3.to_bytes(text=right_id)
                
                # Get mineral right from blockchain
                right_record = self.contract.functions.getMineralRight(right_id_bytes32).call()
                
                result['blockchain_record'] = {
                    'owner': right_record[0],
                    'geographic_boundary': json.loads(right_record[1]),
                    'valid_until': datetime.datetime.fromtimestamp(right_record[2]).isoformat(),
                    'mineral_types': json.loads(right_record[3]),
                    'metadata': json.loads(right_record[4]) if right_record[4] else {}
                }
                
                result['found'] = True
                logger.info(f"Retrieved mineral right information from blockchain for right_id: {right_id}")
            except Exception as e:
                logger.error(f"Failed to get mineral right from blockchain: {e}")
                result['blockchain_error'] = str(e)
        
        # Check local storage if blockchain retrieval failed or not available
        if not result['found'] or 'blockchain_error' in result:
            local_file_path = os.path.join(self.local_storage_path, f"right_{right_id}.json")
            if os.path.exists(local_file_path):
                try:
                    with open(local_file_path, 'r') as f:
                        local_data = json.load(f)
                    
                    result['local_record'] = local_data
                    result['found'] = True
                    
                    logger.info(f"Retrieved mineral right information from local storage for right_id: {right_id}")
                except Exception as e:
                    logger.error(f"Failed to get mineral right from local storage: {e}")
                    result['local_error'] = str(e)
        
        return result
    
    def update_data(
        self,
        data_id: str,
        updated_data: Union[Dict, List, str, bytes],
        metadata_updates: Dict[str, Any] = None,
        offline_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Update existing data in the blockchain provenance system.
        
        Args:
            data_id: ID of the data to update
            updated_data: The updated data
            metadata_updates: Updates to the metadata
            offline_mode: Whether to operate in offline mode
            
        Returns:
            Dictionary with update details
        """
        # Prepare data for storage
        if isinstance(updated_data, (dict, list)):
            data_bytes = json.dumps(updated_data).encode('utf-8')
        elif isinstance(updated_data, str):
            data_bytes = updated_data.encode('utf-8')
        elif isinstance(updated_data, bytes):
            data_bytes = updated_data
        else:
            raise ValueError(f"Unsupported data type: {type(updated_data)}")
        
        # Calculate data hash
        data_hash = hashlib.sha256(data_bytes).hexdigest()
        
        # Prepare metadata updates
        if metadata_updates is None:
            metadata_updates = {}
        
        metadata_updates.update({
            'data_hash': data_hash,
            'update_timestamp': datetime.datetime.now().isoformat(),
        })
        
        result = {
            'data_id': data_id,
            'data_hash': data_hash,
            'update_timestamp': metadata_updates['update_timestamp'],
            'metadata_updates': metadata_updates
        }
        
        # Store in IPFS if available
        if self.ipfs_available and not offline_mode:
            try:
                # Store updated data
                ipfs_data_result = self.ipfs.add_bytes(data_bytes)
                data_ipfs_hash = ipfs_data_result
                
                # Store updated metadata
                metadata_bytes = json.dumps(metadata_updates).encode('utf-8')
                ipfs_metadata_result = self.ipfs.add_bytes(metadata_bytes)
                metadata_ipfs_hash = ipfs_metadata_result
                
                result['ipfs_hash'] = data_ipfs_hash
                result['metadata_ipfs_hash'] = metadata_ipfs_hash
                
                logger.info(f"Updated data stored in IPFS with hash: {data_ipfs_hash}")
                logger.info(f"Updated metadata stored in IPFS with hash: {metadata_ipfs_hash}")
            except Exception as e:
                logger.error(f"Failed to store updated data in IPFS: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Update on blockchain if available
        if self.ethereum_available and self.contract and self.account and not offline_mode:
            try:
                # Convert data_id to bytes32
                data_id_bytes32 = self.w3.to_bytes(hexstr=data_id) if data_id.startswith('0x') else self.w3.to_bytes(text=data_id)
                
                # Prepare transaction
                tx = self.contract.functions.updateData(
                    data_id_bytes32,
                    result.get('ipfs_hash', ''),
                    json.dumps(metadata_updates)
                ).build_transaction({
                    'from': self.account.address,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                    'gas': 2000000,
                    'gasPrice': self.w3.eth.gas_price
                })
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction receipt
                tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                result['transaction_hash'] = tx_hash.hex()
                result['block_number'] = tx_receipt['blockNumber']
                
                logger.info(f"Data updated on blockchain with tx hash: {tx_hash.hex()}")
            except Exception as e:
                logger.error(f"Failed to update data on blockchain: {e}")
                if not offline_mode:
                    offline_mode = True
                    logger.info("Falling back to offline mode")
        
        # Update locally if in offline mode or as backup
        if offline_mode or not self.ethereum_available:
            local_file_path = os.path.join(self.local_storage_path, f"{data_id}.json")
            
            # Read existing data if available
            existing_data = {}
            if os.path.exists(local_file_path):
                try:
                    with open(local_file_path, 'r') as f:
                        existing_data = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read existing data from local storage: {e}")
            
            # Update data and metadata
            updated_record = {
                'data': updated_data if isinstance(updated_data, (dict, list)) else data_bytes.hex(),
                'metadata': {**(existing_data.get('metadata', {})), **metadata_updates}
            }
            
            # Save updated data
            with open(local_file_path, 'w') as f:
                json.dump(updated_record, f)
            
            result['local_file_path'] = local_file_path
            logger.info(f"Updated data stored locally at: {local_file_path}")
        
        return result
