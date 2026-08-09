"""
API endpoints for Blockchain Data Provenance System

This module provides FastAPI endpoints for the blockchain-based data provenance system,
allowing registration, verification, and management of data provenance and mineral rights.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import json
import datetime
import uuid
from pydantic import BaseModel

# Blockchain backend dependencies (web3, ipfshttpclient, eth_account) are
# optional. The API boots without them; these endpoints degrade with HTTP 503.
try:
    from ..blockchain.blockchain_data_provenance import BlockchainDataProvenance
    BLOCKCHAIN_AVAILABLE = True
    _BLOCKCHAIN_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - depends on optional deps
    BlockchainDataProvenance = None
    BLOCKCHAIN_AVAILABLE = False
    _BLOCKCHAIN_ERROR = str(exc)


def _require_blockchain():
    """Raise HTTP 503 when the blockchain backend is not installed."""
    if not BLOCKCHAIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Blockchain provenance is unavailable: optional blockchain "
                f"dependencies are not installed ({_BLOCKCHAIN_ERROR}). "
                "Install the optional blockchain requirements to enable this feature."
            )
        )


# Create router
router = APIRouter(
    prefix="/api/blockchain",
    tags=["blockchain"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(_require_blockchain)],
)

# Initialize blockchain data provenance service
blockchain_service = BlockchainDataProvenance() if BLOCKCHAIN_AVAILABLE else None

# Pydantic models for request/response validation
class DataRegistrationRequest(BaseModel):
    data_type: str
    metadata: Optional[Dict[str, Any]] = None
    offline_mode: bool = False

class DataRegistrationResponse(BaseModel):
    data_id: str
    data_hash: str
    timestamp: str
    ipfs_hash: Optional[str] = None
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    local_file_path: Optional[str] = None

class DataVerificationRequest(BaseModel):
    data_id: Optional[str] = None
    ipfs_hash: Optional[str] = None

class DataVerificationResponse(BaseModel):
    verified: bool
    calculated_hash: str
    ipfs_verified: Optional[bool] = None
    blockchain_verified: Optional[bool] = None
    local_verified: Optional[bool] = None
    provenance: Optional[Dict[str, Any]] = None
    provenance_history: Optional[List[Dict[str, Any]]] = None

class MineralRightRequest(BaseModel):
    geographic_boundary: Dict[str, Any]
    valid_until: str
    mineral_types: List[str]
    metadata: Optional[Dict[str, Any]] = None
    offline_mode: bool = False

class MineralRightResponse(BaseModel):
    right_id: str
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    local_file_path: Optional[str] = None

class MineralRightTransferRequest(BaseModel):
    right_id: str
    new_owner_address: str
    offline_mode: bool = False

class MineralRightTransferResponse(BaseModel):
    right_id: str
    new_owner: str
    timestamp: str
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None

class DataUpdateRequest(BaseModel):
    data_id: str
    metadata_updates: Optional[Dict[str, Any]] = None
    offline_mode: bool = False

class DataUpdateResponse(BaseModel):
    data_id: str
    data_hash: str
    update_timestamp: str
    ipfs_hash: Optional[str] = None
    transaction_hash: Optional[str] = None
    block_number: Optional[int] = None
    local_file_path: Optional[str] = None

@router.post("/register-data", response_model=DataRegistrationResponse)
async def register_data(
    file: UploadFile = File(...),
    data_type: str = Form(...),
    metadata: Optional[str] = Form(None),
    offline_mode: bool = Form(False)
):
    """
    Register data in the blockchain provenance system.
    
    Args:
        file: The data file to register
        data_type: Type of data (e.g., 'geological', 'geophysical')
        metadata: Additional metadata about the data (JSON string)
        offline_mode: Whether to operate in offline mode
        
    Returns:
        Registration details including data_id and ipfs_hash
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse metadata if provided
        parsed_metadata = json.loads(metadata) if metadata else {}
        
        # Add file information to metadata
        parsed_metadata.update({
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(file_content)
        })
        
        # Register data
        result = blockchain_service.register_data(
            data=file_content,
            data_type=data_type,
            metadata=parsed_metadata,
            offline_mode=offline_mode
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-data", response_model=DataVerificationResponse)
async def verify_data(
    file: UploadFile = File(...),
    data_id: Optional[str] = Form(None),
    ipfs_hash: Optional[str] = Form(None)
):
    """
    Verify the integrity and provenance of data.
    
    Args:
        file: The data file to verify
        data_id: ID of the data in the blockchain
        ipfs_hash: IPFS hash of the data
        
    Returns:
        Verification results
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Verify data
        result = blockchain_service.verify_data(
            data=file_content,
            data_id=data_id,
            ipfs_hash=ipfs_hash
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register-mineral-right", response_model=MineralRightResponse)
async def register_mineral_right(request: MineralRightRequest):
    """
    Register a mineral right on the blockchain.
    
    Args:
        request: Mineral right registration request
        
    Returns:
        Registration details
    """
    try:
        # Parse valid_until date
        valid_until = datetime.datetime.fromisoformat(request.valid_until)
        
        # Register mineral right
        result = blockchain_service.register_mineral_right(
            geographic_boundary=request.geographic_boundary,
            valid_until=valid_until,
            mineral_types=request.mineral_types,
            metadata=request.metadata,
            offline_mode=request.offline_mode
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/transfer-mineral-right", response_model=MineralRightTransferResponse)
async def transfer_mineral_right(request: MineralRightTransferRequest):
    """
    Transfer a mineral right to a new owner.
    
    Args:
        request: Mineral right transfer request
        
    Returns:
        Transfer details
    """
    try:
        # Transfer mineral right
        result = blockchain_service.transfer_mineral_right(
            right_id=request.right_id,
            new_owner_address=request.new_owner_address,
            offline_mode=request.offline_mode
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mineral-right/{right_id}")
async def get_mineral_right(right_id: str):
    """
    Get information about a mineral right.
    
    Args:
        right_id: ID of the mineral right
        
    Returns:
        Mineral right details
    """
    try:
        # Get mineral right
        result = blockchain_service.get_mineral_right(right_id=right_id)
        
        if not result['found']:
            raise HTTPException(status_code=404, detail=f"Mineral right with ID {right_id} not found")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-data", response_model=DataUpdateResponse)
async def update_data(
    file: UploadFile = File(...),
    data_id: str = Form(...),
    metadata_updates: Optional[str] = Form(None),
    offline_mode: bool = Form(False)
):
    """
    Update existing data in the blockchain provenance system.
    
    Args:
        file: The updated data file
        data_id: ID of the data to update
        metadata_updates: Updates to the metadata (JSON string)
        offline_mode: Whether to operate in offline mode
        
    Returns:
        Update details
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse metadata updates if provided
        parsed_metadata_updates = json.loads(metadata_updates) if metadata_updates else {}
        
        # Add file information to metadata updates
        parsed_metadata_updates.update({
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(file_content)
        })
        
        # Update data
        result = blockchain_service.update_data(
            data_id=data_id,
            updated_data=file_content,
            metadata_updates=parsed_metadata_updates,
            offline_mode=offline_mode
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_blockchain_status():
    """
    Get the status of the blockchain data provenance system.
    
    Returns:
        Status information
    """
    return {
        "ethereum_available": blockchain_service.ethereum_available,
        "ipfs_available": blockchain_service.ipfs_available,
        "contract_available": blockchain_service.contract is not None,
        "account_available": blockchain_service.account is not None,
        "local_storage_path": blockchain_service.local_storage_path
    }
