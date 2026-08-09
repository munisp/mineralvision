"""
API endpoints for File Upload.

This module provides endpoints for uploading various file types
used in mineral exploration (CSV, Excel, LAS, shapefiles, etc.).
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import os
import tempfile
import shutil

router = APIRouter()

# Runtime storage for API operations
uploads_db: Dict[str, dict] = {}

# Supported file types
SUPPORTED_EXTENSIONS = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
    "las": "application/octet-stream",
    "laz": "application/octet-stream",
    "shp": "application/octet-stream",
    "geojson": "application/geo+json",
    "json": "application/json",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "sgy": "application/octet-stream",
    "segy": "application/octet-stream",
    "grd": "application/octet-stream",
    "xyz": "text/plain"
}


class UploadResponse(BaseModel):
    """Schema for upload response."""
    id: str
    filename: str
    originalFilename: str
    fileType: str
    size: int
    projectId: Optional[str]
    uploadedAt: str


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    projectId: str = Form(...)
):
    """Upload a file."""
    try:
        # Get file extension
        filename = file.filename or "unknown"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported types: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Generate unique filename
        upload_id = str(uuid.uuid4())
        stored_filename = f"{upload_id}.{ext}"
        
        # Store file metadata
        upload_data = {
            "id": upload_id,
            "filename": stored_filename,
            "originalFilename": filename,
            "fileType": ext,
            "mimeType": SUPPORTED_EXTENSIONS[ext],
            "size": file_size,
            "projectId": projectId,
            "uploadedAt": datetime.utcnow().isoformat(),
            "content": content  # In production, store to disk/S3
        }
        
        uploads_db[upload_id] = upload_data
        
        return UploadResponse(
            id=upload_id,
            filename=stored_filename,
            originalFilename=filename,
            fileType=ext,
            size=file_size,
            projectId=projectId,
            uploadedAt=upload_data["uploadedAt"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=List[UploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...),
    projectId: str = Form(...)
):
    """Upload multiple files."""
    results = []
    errors = []
    
    for file in files:
        try:
            filename = file.filename or "unknown"
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append({"filename": filename, "error": f"Unsupported file type: {ext}"})
                continue
            
            content = await file.read()
            file_size = len(content)
            
            upload_id = str(uuid.uuid4())
            stored_filename = f"{upload_id}.{ext}"
            
            upload_data = {
                "id": upload_id,
                "filename": stored_filename,
                "originalFilename": filename,
                "fileType": ext,
                "mimeType": SUPPORTED_EXTENSIONS[ext],
                "size": file_size,
                "projectId": projectId,
                "uploadedAt": datetime.utcnow().isoformat(),
                "content": content
            }
            
            uploads_db[upload_id] = upload_data
            
            results.append(UploadResponse(
                id=upload_id,
                filename=stored_filename,
                originalFilename=filename,
                fileType=ext,
                size=file_size,
                projectId=projectId,
                uploadedAt=upload_data["uploadedAt"]
            ))
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    if errors and not results:
        raise HTTPException(status_code=400, detail={"errors": errors})
    
    return results


@router.get("/{upload_id}")
async def get_upload(upload_id: str):
    """Get upload metadata by ID."""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    
    upload = uploads_db[upload_id]
    return UploadResponse(
        id=upload["id"],
        filename=upload["filename"],
        originalFilename=upload["originalFilename"],
        fileType=upload["fileType"],
        size=upload["size"],
        projectId=upload["projectId"],
        uploadedAt=upload["uploadedAt"]
    )


@router.get("")
async def list_uploads(
    projectId: Optional[str] = None,
    fileType: Optional[str] = None
):
    """List all uploads with optional filtering."""
    uploads = list(uploads_db.values())
    
    if projectId:
        uploads = [u for u in uploads if u.get("projectId") == projectId]
    if fileType:
        uploads = [u for u in uploads if u.get("fileType") == fileType]
    
    return [
        UploadResponse(
            id=u["id"],
            filename=u["filename"],
            originalFilename=u["originalFilename"],
            fileType=u["fileType"],
            size=u["size"],
            projectId=u["projectId"],
            uploadedAt=u["uploadedAt"]
        )
        for u in uploads
    ]


@router.delete("/{upload_id}")
async def delete_upload(upload_id: str):
    """Delete an upload."""
    if upload_id not in uploads_db:
        raise HTTPException(status_code=404, detail=f"Upload {upload_id} not found")
    
    del uploads_db[upload_id]
    return {"status": "deleted", "upload_id": upload_id}
