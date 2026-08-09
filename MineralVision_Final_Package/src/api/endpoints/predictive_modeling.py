"""
API endpoints for AI-Powered Predictive Modeling

This module provides FastAPI endpoints for mineral deposit prediction using
deep learning models with uncertainty quantification.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel
import uuid
import shutil
import tempfile
from datetime import datetime

# Heavy ML dependencies (torch, geopandas, mlflow, ...) are optional.
# The API boots without them; ML endpoints degrade with HTTP 503.
try:
    import geopandas as gpd
    import torch
    from ..ml.predictive_modeling.mineral_deposit_prediction import (
        MineralDepositPredictionService,
        MineralDepositDataset
    )
    PREDICTIVE_MODELING_AVAILABLE = True
    _PREDICTIVE_MODELING_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - depends on optional deps
    gpd = None
    torch = None
    MineralDepositPredictionService = None
    MineralDepositDataset = None
    PREDICTIVE_MODELING_AVAILABLE = False
    _PREDICTIVE_MODELING_ERROR = str(exc)

# Create router
router = APIRouter(
    prefix="/api/predictive-modeling",
    tags=["predictive-modeling"],
    responses={404: {"description": "Not found"}},
)

# Model storage directory
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/mineralvision/models")
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/mineralvision/data")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize prediction service (None when heavy ML deps are not installed)
prediction_service = (
    MineralDepositPredictionService(
        model_path=os.path.join(MODEL_DIR, "latest_model"),
        data_dir=DATA_DIR,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        uncertainty_estimation=True
    )
    if PREDICTIVE_MODELING_AVAILABLE else None
)


def _require_prediction_service():
    """Raise HTTP 503 when the ML stack is not installed."""
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Predictive modeling is unavailable: optional ML dependencies "
                f"are not installed ({_PREDICTIVE_MODELING_ERROR}). "
                "Install requirements-ml.txt to enable this feature."
            )
        )
    return prediction_service

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    features: List[float]
    with_uncertainty: bool = True

class PredictionResponse(BaseModel):
    prediction: float
    uncertainty: Optional[float] = None
    prediction_id: str

class TrainingRequest(BaseModel):
    geological_data: Optional[str] = None
    geophysical_data: Optional[str] = None
    geochemical_data: Optional[str] = None
    remote_sensing_data: Optional[str] = None
    historical_data: Optional[str] = None
    model_name: str
    hidden_dims: Optional[List[int]] = None
    uncertainty_estimation: bool = True

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    model_name: str

class TrainingStatus(BaseModel):
    job_id: str
    status: str
    model_name: str
    metrics: Optional[Dict[str, float]] = None
    start_time: str
    end_time: Optional[str] = None

class ValidationRequest(BaseModel):
    model_name: str
    new_data: Dict[str, Any]
    validation_results: Dict[str, Any]

class ValidationResponse(BaseModel):
    job_id: str
    status: str
    model_name: str

# Runtime storage for API operations
training_jobs = {}

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for mineral deposit likelihood.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with prediction value and optional uncertainty
    """
    _require_prediction_service()
    try:
        features = np.array(request.features, dtype=np.float32).reshape(1, -1)
        
        if request.with_uncertainty:
            prediction, uncertainty = prediction_service.predict(
                features, with_uncertainty=True
            )
            prediction_value = float(prediction[0][0])
            uncertainty_value = float(uncertainty[0][0])
        else:
            prediction = prediction_service.predict(
                features, with_uncertainty=False
            )
            prediction_value = float(prediction[0][0])
            uncertainty_value = None
        
        # Generate a unique ID for this prediction
        prediction_id = str(uuid.uuid4())
        
        return {
            "prediction": prediction_value,
            "uncertainty": uncertainty_value,
            "prediction_id": prediction_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train", response_model=TrainingResponse)
async def train_model(
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """
    Train a new predictive model.
    
    Args:
        background_tasks: FastAPI background tasks
        request: Training request with data paths and model configuration
        
    Returns:
        Training response with job ID and status
    """
    _require_prediction_service()
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Create a job record
    job = {
        "job_id": job_id,
        "status": "queued",
        "model_name": request.model_name,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "metrics": None
    }
    
    # Store the job
    training_jobs[job_id] = job
    
    # Define the training task
    def train_model_task():
        try:
            # Update job status
            training_jobs[job_id]["status"] = "running"
            
            # Train the model
            model = prediction_service.train_model(
                geological_data=request.geological_data,
                geophysical_data=request.geophysical_data,
                geochemical_data=request.geochemical_data,
                remote_sensing_data=request.remote_sensing_data,
                historical_data=request.historical_data,
                hidden_dims=request.hidden_dims
            )
            
            # Save the model with the requested name
            model_path = os.path.join(MODEL_DIR, request.model_name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            # Evaluate the model to get actual metrics
            try:
                eval_metrics = prediction_service.evaluate()
                training_jobs[job_id]["metrics"] = {
                    "test_loss": float(eval_metrics.get("test_loss", 0.0)),
                    "test_acc": float(eval_metrics.get("test_accuracy", 0.0)),
                    "auc_roc": float(eval_metrics.get("auc_roc", 0.0)),
                    "precision": float(eval_metrics.get("precision", 0.0)),
                    "recall": float(eval_metrics.get("recall", 0.0))
                }
            except Exception as eval_error:
                training_jobs[job_id]["metrics"] = {
                    "test_loss": 0.0,
                    "test_acc": 0.0,
                    "evaluation_error": str(eval_error)
                }
            
            # Update job status
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["end_time"] = datetime.now().isoformat()
        except Exception as e:
            # Update job status on failure
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["end_time"] = datetime.now().isoformat()
            training_jobs[job_id]["error"] = str(e)
    
    # Start the training task in the background
    background_tasks.add_task(train_model_task)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "model_name": request.model_name
    }

@router.get("/training-status/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """
    Get the status of a training job.
    
    Args:
        job_id: ID of the training job
        
    Returns:
        Training status
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[job_id]

@router.post("/upload-data", response_model=Dict[str, str])
async def upload_data(
    file: UploadFile = File(...),
    data_type: str = Form(...),
    description: Optional[str] = Form(None)
):
    """
    Upload data for model training.
    
    Args:
        file: Data file
        data_type: Type of data (geological, geophysical, etc.)
        description: Optional description of the data
        
    Returns:
        Dictionary with file path
    """
    try:
        # Create a unique filename
        filename = f"{data_type}_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(DATA_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create metadata
        metadata = {
            "original_filename": file.filename,
            "data_type": data_type,
            "description": description,
            "upload_time": datetime.now().isoformat(),
            "file_path": file_path
        }
        
        # Save metadata
        metadata_path = f"{file_path}.meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        return {"file_path": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-from-validation", response_model=ValidationResponse)
async def update_from_validation(
    background_tasks: BackgroundTasks,
    request: ValidationRequest
):
    """
    Update a model based on field validation results.

    Args:
        background_tasks: FastAPI background tasks
        request: Validation request with new data and validation results

    Returns:
        Validation response with job ID and status
    """
    _require_prediction_service()
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Create a job record
    job = {
        "job_id": job_id,
        "status": "queued",
        "model_name": request.model_name,
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    # Store the job
    training_jobs[job_id] = job
    
    # Define the update task
    def update_model_task():
        try:
            # Update job status
            training_jobs[job_id]["status"] = "running"
            
            # Load the model
            model_path = os.path.join(MODEL_DIR, request.model_name)
            prediction_service.model_path = model_path
            prediction_service._load_model()
            
            # Update the model
            prediction_service.update_from_validation(
                request.new_data,
                request.validation_results
            )
            
            # Update job status
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["end_time"] = datetime.now().isoformat()
        except Exception as e:
            # Update job status on failure
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["end_time"] = datetime.now().isoformat()
            training_jobs[job_id]["error"] = str(e)
    
    # Start the update task in the background
    background_tasks.add_task(update_model_task)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "model_name": request.model_name
    }

@router.get("/models", response_model=List[str])
async def list_models():
    """
    List available predictive models.
    
    Returns:
        List of model names
    """
    try:
        models = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data", response_model=Dict[str, List[str]])
async def list_data():
    """
    List available data files by type.
    
    Returns:
        Dictionary of data files by type
    """
    try:
        # Get all files in the data directory
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and not f.endswith('.meta.json')]
        
        # Group by data type
        data_by_type = {}
        
        for file in files:
            # Try to load metadata
            metadata_path = os.path.join(DATA_DIR, f"{file}.meta.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                data_type = metadata.get("data_type", "unknown")
            else:
                data_type = "unknown"
            
            # Add to the appropriate group
            if data_type not in data_by_type:
                data_by_type[data_type] = []
            data_by_type[data_type].append(file)
        
        return data_by_type
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
