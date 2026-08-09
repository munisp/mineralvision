"""
Indigenous Knowledge API Endpoints

This module provides the API endpoints for the indigenous knowledge integration system,
enabling the MineralVision platform to respectfully incorporate traditional knowledge
into the mineral exploration process.
"""

from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import os
import datetime
import uuid
# Heavy geospatial dependencies (shapely, geopandas) are optional.
# The API boots without them; these endpoints degrade with HTTP 503.
try:
    from shapely.geometry import Point, Polygon, shape
    from ..indigenous_knowledge.core import (
        KnowledgeHolder,
        TraditionalKnowledge,
        CulturalHeritageSite,
        ResourceArea,
        ConsultationRecord,
        BenefitSharingAgreement,
        IndigenousKnowledgeManager
    )
    INDIGENOUS_KNOWLEDGE_AVAILABLE = True
    _INDIGENOUS_KNOWLEDGE_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover - depends on optional deps
    Point = Polygon = shape = None
    KnowledgeHolder = TraditionalKnowledge = CulturalHeritageSite = None
    ResourceArea = ConsultationRecord = BenefitSharingAgreement = None
    IndigenousKnowledgeManager = None
    INDIGENOUS_KNOWLEDGE_AVAILABLE = False
    _INDIGENOUS_KNOWLEDGE_ERROR = str(exc)


def _require_indigenous_knowledge():
    """Raise HTTP 503 when the geospatial stack is not installed."""
    if not INDIGENOUS_KNOWLEDGE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Indigenous knowledge features are unavailable: optional "
                "geospatial dependencies are not installed "
                f"({_INDIGENOUS_KNOWLEDGE_ERROR}). Install the optional "
                "geospatial requirements to enable this feature."
            )
        )


# Initialize router
router = APIRouter(
    prefix="/indigenous-knowledge",
    tags=["indigenous-knowledge"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(_require_indigenous_knowledge)],
)

# Initialize indigenous knowledge manager
knowledge_manager = IndigenousKnowledgeManager() if INDIGENOUS_KNOWLEDGE_AVAILABLE else None

# Knowledge Holder Endpoints

@router.post("/holders", response_model=Dict[str, Any])
async def create_knowledge_holder(
    name: str = Form(...),
    community: str = Form(None),
    role: str = Form(None),
    contact_info: str = Form(None),
    metadata: str = Form(None)
):
    """
    Create a new knowledge holder.
    """
    try:
        contact_info_dict = json.loads(contact_info) if contact_info else {}
        metadata_dict = json.loads(metadata) if metadata else {}
        
        holder = KnowledgeHolder(
            name=name,
            community=community,
            role=role,
            contact_info=contact_info_dict,
            metadata=metadata_dict
        )
        
        holder_id = knowledge_manager.add_knowledge_holder(holder)
        
        return {
            "status": "success",
            "message": "Knowledge holder created successfully",
            "holder_id": holder_id,
            "holder": holder.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating knowledge holder: {str(e)}")

@router.get("/holders", response_model=Dict[str, Any])
async def list_knowledge_holders(
    community: str = Query(None),
    role: str = Query(None)
):
    """
    List knowledge holders with optional filtering.
    """
    try:
        holders = knowledge_manager.knowledge_holders.values()
        
        if community:
            holders = [h for h in holders if h.community == community]
            
        if role:
            holders = [h for h in holders if h.role == role]
            
        return {
            "status": "success",
            "count": len(holders),
            "holders": [h.to_dict() for h in holders]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing knowledge holders: {str(e)}")

@router.get("/holders/{holder_id}", response_model=Dict[str, Any])
async def get_knowledge_holder(holder_id: str):
    """
    Get a specific knowledge holder.
    """
    holder = knowledge_manager.get_knowledge_holder(holder_id)
    
    if not holder:
        raise HTTPException(status_code=404, detail=f"Knowledge holder with ID {holder_id} not found")
        
    return {
        "status": "success",
        "holder": holder.to_dict()
    }

@router.put("/holders/{holder_id}", response_model=Dict[str, Any])
async def update_knowledge_holder(
    holder_id: str,
    name: str = Form(None),
    community: str = Form(None),
    role: str = Form(None),
    contact_info: str = Form(None),
    metadata: str = Form(None)
):
    """
    Update a knowledge holder.
    """
    holder = knowledge_manager.get_knowledge_holder(holder_id)
    
    if not holder:
        raise HTTPException(status_code=404, detail=f"Knowledge holder with ID {holder_id} not found")
        
    try:
        if name:
            holder.name = name
            
        if community:
            holder.community = community
            
        if role:
            holder.role = role
            
        if contact_info:
            holder.contact_info = json.loads(contact_info)
            
        if metadata:
            holder.metadata = json.loads(metadata)
            
        holder.updated_at = datetime.datetime.now()
        
        return {
            "status": "success",
            "message": "Knowledge holder updated successfully",
            "holder": holder.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating knowledge holder: {str(e)}")

# Traditional Knowledge Endpoints

@router.post("/knowledge", response_model=Dict[str, Any])
async def create_traditional_knowledge(
    title: str = Form(...),
    description: str = Form(None),
    knowledge_type: str = Form(...),
    holder_id: str = Form(None),
    community: str = Form(None),
    geometry: str = Form(None),
    coordinate_system: str = Form("EPSG:4326"),
    access_level: str = Form("public"),
    metadata: str = Form(None),
    # Cultural heritage site specific fields
    site_type: str = Form(None),
    significance: str = Form(None),
    protection_level: str = Form(None),
    seasonal_restrictions: str = Form(None),
    # Resource area specific fields
    resource_type: str = Form(None),
    seasonal_use: str = Form(None),
    current_use: bool = Form(None),
    sustainability_practices: str = Form(None)
):
    """
    Create a new traditional knowledge entry.
    """
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        
        # Parse geometry if provided
        geometry_obj = None
        if geometry:
            geometry_dict = json.loads(geometry)
            geometry_obj = shape(geometry_dict)
            
        # Create the appropriate knowledge object based on type
        if knowledge_type == "cultural_heritage_site":
            seasonal_restrictions_list = json.loads(seasonal_restrictions) if seasonal_restrictions else []
            
            knowledge = CulturalHeritageSite(
                title=title,
                description=description,
                holder_id=holder_id,
                community=community,
                geometry=geometry_obj,
                coordinate_system=coordinate_system,
                access_level=access_level,
                metadata=metadata_dict,
                site_type=site_type,
                significance=significance,
                protection_level=protection_level or "standard",
                seasonal_restrictions=seasonal_restrictions_list
            )
        elif knowledge_type == "resource_area":
            seasonal_use_dict = json.loads(seasonal_use) if seasonal_use else {}
            sustainability_practices_list = json.loads(sustainability_practices) if sustainability_practices else []
            
            knowledge = ResourceArea(
                title=title,
                description=description,
                holder_id=holder_id,
                community=community,
                geometry=geometry_obj,
                coordinate_system=coordinate_system,
                access_level=access_level,
                metadata=metadata_dict,
                resource_type=resource_type,
                seasonal_use=seasonal_use_dict,
                current_use=current_use if current_use is not None else True,
                sustainability_practices=sustainability_practices_list
            )
        else:
            knowledge = TraditionalKnowledge(
                title=title,
                description=description,
                knowledge_type=knowledge_type,
                holder_id=holder_id,
                community=community,
                geometry=geometry_obj,
                coordinate_system=coordinate_system,
                access_level=access_level,
                metadata=metadata_dict
            )
            
        knowledge_id = knowledge_manager.add_traditional_knowledge(knowledge)
        
        return {
            "status": "success",
            "message": "Traditional knowledge created successfully",
            "knowledge_id": knowledge_id,
            "knowledge": knowledge.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating traditional knowledge: {str(e)}")

@router.get("/knowledge", response_model=Dict[str, Any])
async def list_traditional_knowledge(
    knowledge_type: str = Query(None),
    community: str = Query(None),
    holder_id: str = Query(None),
    access_level: str = Query(None),
    latitude: float = Query(None),
    longitude: float = Query(None),
    buffer_distance: float = Query(0.01)
):
    """
    List traditional knowledge with optional filtering.
    """
    try:
        knowledge_entries = list(knowledge_manager.traditional_knowledge.values())
        
        if knowledge_type:
            knowledge_entries = [k for k in knowledge_entries if k.knowledge_type == knowledge_type]
            
        if community:
            knowledge_entries = [k for k in knowledge_entries if k.community == community]
            
        if holder_id:
            knowledge_entries = [k for k in knowledge_entries if k.holder_id == holder_id]
            
        if access_level:
            knowledge_entries = [k for k in knowledge_entries if k.access_level == access_level]
            
        if latitude is not None and longitude is not None:
            knowledge_entries = knowledge_manager.get_knowledge_by_area(latitude, longitude, buffer_distance)
            
        return {
            "status": "success",
            "count": len(knowledge_entries),
            "knowledge": [k.to_dict() for k in knowledge_entries]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing traditional knowledge: {str(e)}")

@router.get("/knowledge/{knowledge_id}", response_model=Dict[str, Any])
async def get_traditional_knowledge(knowledge_id: str):
    """
    Get a specific traditional knowledge entry.
    """
    knowledge = knowledge_manager.get_traditional_knowledge(knowledge_id)
    
    if not knowledge:
        raise HTTPException(status_code=404, detail=f"Traditional knowledge with ID {knowledge_id} not found")
        
    return {
        "status": "success",
        "knowledge": knowledge.to_dict()
    }

@router.post("/knowledge/{knowledge_id}/attachments", response_model=Dict[str, Any])
async def add_knowledge_attachment(
    knowledge_id: str,
    file: UploadFile = File(...),
    attachment_type: str = Form(...),
    description: str = Form(None)
):
    """
    Add an attachment to a traditional knowledge entry.
    """
    knowledge = knowledge_manager.get_traditional_knowledge(knowledge_id)
    
    if not knowledge:
        raise HTTPException(status_code=404, detail=f"Traditional knowledge with ID {knowledge_id} not found")
        
    try:
        # Create attachments directory if it doesn't exist
        attachments_dir = os.path.join(knowledge_manager.data_dir, "attachments", "knowledge", knowledge_id)
        os.makedirs(attachments_dir, exist_ok=True)
        
        # Generate a unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(attachments_dir, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(await file.read())
            
        # Add the attachment to the knowledge
        knowledge.add_attachment(file_path, attachment_type, description)
        
        return {
            "status": "success",
            "message": "Attachment added successfully",
            "knowledge": knowledge.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding attachment: {str(e)}")

# Consultation Endpoints

@router.post("/consultations", response_model=Dict[str, Any])
async def create_consultation(
    title: str = Form(...),
    description: str = Form(None),
    community: str = Form(...),
    consultation_date: str = Form(...),
    status: str = Form("planned"),
    metadata: str = Form(None)
):
    """
    Create a new consultation record.
    """
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        consultation_date_obj = datetime.date.fromisoformat(consultation_date)
        
        consultation = ConsultationRecord(
            title=title,
            description=description,
            community=community,
            consultation_date=consultation_date_obj,
            status=status,
            metadata=metadata_dict
        )
        
        consultation_id = knowledge_manager.add_consultation(consultation)
        
        return {
            "status": "success",
            "message": "Consultation created successfully",
            "consultation_id": consultation_id,
            "consultation": consultation.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating consultation: {str(e)}")

@router.get("/consultations", response_model=Dict[str, Any])
async def list_consultations(
    community: str = Query(None),
    status: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None)
):
    """
    List consultations with optional filtering.
    """
    try:
        consultations = list(knowledge_manager.consultations.values())
        
        if community:
            consultations = [c for c in consultations if c.community == community]
            
        if status:
            consultations = [c for c in consultations if c.status == status]
            
        if start_date:
            start_date_obj = datetime.date.fromisoformat(start_date)
            consultations = [c for c in consultations if c.consultation_date >= start_date_obj]
            
        if end_date:
            end_date_obj = datetime.date.fromisoformat(end_date)
            consultations = [c for c in consultations if c.consultation_date <= end_date_obj]
            
        return {
            "status": "success",
            "count": len(consultations),
            "consultations": [c.to_dict() for c in consultations]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing consultations: {str(e)}")

@router.get("/consultations/{consultation_id}", response_model=Dict[str, Any])
async def get_consultation(consultation_id: str):
    """
    Get a specific consultation record.
    """
    consultation = knowledge_manager.get_consultation(consultation_id)
    
    if not consultation:
        raise HTTPException(status_code=404, detail=f"Consultation with ID {consultation_id} not found")
        
    return {
        "status": "success",
        "consultation": consultation.to_dict()
    }

@router.post("/consultations/{consultation_id}/participants", response_model=Dict[str, Any])
async def add_consultation_participant(
    consultation_id: str,
    name: str = Form(...),
    role: str = Form(...),
    organization: str = Form(None),
    contact_info: str = Form(None)
):
    """
    Add a participant to a consultation.
    """
    consultation = knowledge_manager.get_consultation(consultation_id)
    
    if not consultation:
        raise HTTPException(status_code=404, detail=f"Consultation with ID {consultation_id} not found")
        
    try:
        contact_info_dict = json.loads(contact_info) if contact_info else {}
        
        consultation.add_participant(name, role, organization, contact_info_dict)
        
        return {
            "status": "success",
            "message": "Participant added successfully",
            "consultation": consultation.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding participant: {str(e)}")

@router.post("/consultations/{consultation_id}/outcomes", response_model=Dict[str, Any])
async def add_consultation_outcome(
    consultation_id: str,
    outcome: str = Form(...)
):
    """
    Add an outcome to a consultation.
    """
    consultation = knowledge_manager.get_consultation(consultation_id)
    
    if not consultation:
        raise HTTPException(status_code=404, detail=f"Consultation with ID {consultation_id} not found")
        
    try:
        consultation.add_outcome(outcome)
        
        return {
            "status": "success",
            "message": "Outcome added successfully",
            "consultation": consultation.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding outcome: {str(e)}")

# Benefit Sharing Agreement Endpoints

@router.post("/agreements", response_model=Dict[str, Any])
async def create_agreement(
    title: str = Form(...),
    description: str = Form(None),
    community: str = Form(...),
    start_date: str = Form(None),
    end_date: str = Form(None),
    status: str = Form("draft"),
    agreement_type: str = Form(None),
    metadata: str = Form(None)
):
    """
    Create a new benefit sharing agreement.
    """
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        start_date_obj = datetime.date.fromisoformat(start_date) if start_date else None
        end_date_obj = datetime.date.fromisoformat(end_date) if end_date else None
        
        agreement = BenefitSharingAgreement(
            title=title,
            description=description,
            community=community,
            start_date=start_date_obj,
            end_date=end_date_obj,
            status=status,
            agreement_type=agreement_type,
            metadata=metadata_dict
        )
        
        agreement_id = knowledge_manager.add_agreement(agreement)
        
        return {
            "status": "success",
            "message": "Agreement created successfully",
            "agreement_id": agreement_id,
            "agreement": agreement.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating agreement: {str(e)}")

@router.get("/agreements", response_model=Dict[str, Any])
async def list_agreements(
    community: str = Query(None),
    status: str = Query(None),
    agreement_type: str = Query(None)
):
    """
    List benefit sharing agreements with optional filtering.
    """
    try:
        agreements = list(knowledge_manager.agreements.values())
        
        if community:
            agreements = [a for a in agreements if a.community == community]
            
        if status:
            agreements = [a for a in agreements if a.status == status]
            
        if agreement_type:
            agreements = [a for a in agreements if a.agreement_type == agreement_type]
            
        return {
            "status": "success",
            "count": len(agreements),
            "agreements": [a.to_dict() for a in agreements]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agreements: {str(e)}")

@router.get("/agreements/{agreement_id}", response_model=Dict[str, Any])
async def get_agreement(agreement_id: str):
    """
    Get a specific benefit sharing agreement.
    """
    agreement = knowledge_manager.get_agreement(agreement_id)
    
    if not agreement:
        raise HTTPException(status_code=404, detail=f"Agreement with ID {agreement_id} not found")
        
    return {
        "status": "success",
        "agreement": agreement.to_dict()
    }

@router.post("/agreements/{agreement_id}/benefits", response_model=Dict[str, Any])
async def add_agreement_benefit(
    agreement_id: str,
    benefit_type: str = Form(...),
    description: str = Form(...),
    value: float = Form(None),
    schedule: str = Form(None)
):
    """
    Add a benefit to an agreement.
    """
    agreement = knowledge_manager.get_agreement(agreement_id)
    
    if not agreement:
        raise HTTPException(status_code=404, detail=f"Agreement with ID {agreement_id} not found")
        
    try:
        agreement.add_benefit(benefit_type, description, value, schedule)
        
        return {
            "status": "success",
            "message": "Benefit added successfully",
            "agreement": agreement.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding benefit: {str(e)}")

@router.post("/agreements/{agreement_id}/commitments", response_model=Dict[str, Any])
async def add_agreement_commitment(
    agreement_id: str,
    commitment_type: str = Form(...),
    description: str = Form(...),
    responsible: str = Form(...),
    due_date: str = Form(None),
    status: str = Form("pending")
):
    """
    Add a commitment to an agreement.
    """
    agreement = knowledge_manager.get_agreement(agreement_id)
    
    if not agreement:
        raise HTTPException(status_code=404, detail=f"Agreement with ID {agreement_id} not found")
        
    try:
        due_date_obj = datetime.date.fromisoformat(due_date) if due_date else None
        
        agreement.add_commitment(commitment_type, description, responsible, due_date_obj, status)
        
        return {
            "status": "success",
            "message": "Commitment added successfully",
            "agreement": agreement.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding commitment: {str(e)}")

# Data Import/Export Endpoints

@router.post("/export", response_model=Dict[str, Any])
async def export_data():
    """
    Export all indigenous knowledge data to a JSON file.
    """
    try:
        export_path = os.path.join(knowledge_manager.data_dir, f"export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        knowledge_manager.export_to_json(export_path)
        
        return {
            "status": "success",
            "message": "Data exported successfully",
            "export_path": export_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@router.post("/import", response_model=Dict[str, Any])
async def import_data(
    file: UploadFile = File(...)
):
    """
    Import indigenous knowledge data from a JSON file.
    """
    try:
        # Save the uploaded file
        import_path = os.path.join(knowledge_manager.data_dir, f"import_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(import_path, "wb") as f:
            f.write(await file.read())
            
        # Import the data
        knowledge_manager.import_from_json(import_path)
        
        return {
            "status": "success",
            "message": "Data imported successfully",
            "knowledge_holders_count": len(knowledge_manager.knowledge_holders),
            "traditional_knowledge_count": len(knowledge_manager.traditional_knowledge),
            "consultations_count": len(knowledge_manager.consultations),
            "agreements_count": len(knowledge_manager.agreements)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing data: {str(e)}")
