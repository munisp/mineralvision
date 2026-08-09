"""
Digital Twin API Endpoints

This module provides FastAPI endpoints for the MineralVision digital twin system,
enabling interaction with the digital twin through a RESTful API.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path
from typing import Dict, List, Any, Optional
import json
import datetime
from pydantic import BaseModel, Field

from ..digital_twin.core import (
    DigitalTwinManager, 
    DigitalTwinEntity, 
    SpatialEntity,
    MineralDeposit,
    ExplorationArea,
    Equipment,
    DigitalTwinSimulation,
    ExtractionSimulation,
    EnvironmentalImpactSimulation
)

# Initialize router
router = APIRouter(
    prefix="/digital-twin",
    tags=["digital-twin"],
    responses={404: {"description": "Not found"}},
)

# Initialize digital twin manager
digital_twin_manager = DigitalTwinManager()

# Pydantic models for request/response validation
class EntityBase(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = {}
    
class SpatialEntityCreate(EntityBase):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    polygon_coordinates: Optional[List[List[float]]] = None
    coordinate_system: Optional[str] = "EPSG:4326"
    
class MineralDepositCreate(SpatialEntityCreate):
    mineral_type: str
    probability: float
    volume_estimate: float
    depth: float
    
class ExplorationAreaCreate(SpatialEntityCreate):
    status: str
    priority: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
class EquipmentCreate(EntityBase):
    equipment_type: str
    status: str
    location_id: Optional[str] = None
    
class SimulationBase(BaseModel):
    name: str
    description: Optional[str] = ""
    
class ExtractionSimulationCreate(SimulationBase):
    deposit_id: str
    extraction_rate: float
    extraction_duration: int
    extraction_method: str
    efficiency: float
    mineral_price: float
    
class EnvironmentalSimulationCreate(SimulationBase):
    extraction_simulation_id: str
    area_id: str
    environmental_factors: Dict[str, Any]
    
class EntityResponse(BaseModel):
    entity_id: str
    name: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    properties: Dict[str, Any]
    relationships: Dict[str, List[Dict[str, Any]]]
    
class SimulationResponse(BaseModel):
    simulation_id: str
    name: str
    description: str
    created_at: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    status: str

# Entity endpoints
@router.post("/entities/mineral-deposit", response_model=EntityResponse)
async def create_mineral_deposit(deposit: MineralDepositCreate):
    """Create a new mineral deposit entity in the digital twin."""
    try:
        entity = MineralDeposit(
            name=deposit.name,
            metadata=deposit.metadata,
            mineral_type=deposit.mineral_type,
            probability=deposit.probability,
            volume_estimate=deposit.volume_estimate,
            depth=deposit.depth
        )
        
        if deposit.polygon_coordinates:
            entity.set_polygon_geometry(deposit.polygon_coordinates)
        elif deposit.latitude is not None and deposit.longitude is not None:
            entity.set_point_geometry(deposit.latitude, deposit.longitude, deposit.altitude)
            
        digital_twin_manager.add_entity(entity)
        return entity.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/entities/exploration-area", response_model=EntityResponse)
async def create_exploration_area(area: ExplorationAreaCreate):
    """Create a new exploration area entity in the digital twin."""
    try:
        entity = ExplorationArea(
            name=area.name,
            metadata=area.metadata,
            status=area.status,
            priority=area.priority
        )
        
        if area.start_date:
            entity.update_property("start_date", area.start_date)
            
        if area.end_date:
            entity.update_property("end_date", area.end_date)
        
        if area.polygon_coordinates:
            entity.set_polygon_geometry(area.polygon_coordinates)
        elif area.latitude is not None and area.longitude is not None:
            entity.set_point_geometry(area.latitude, area.longitude, area.altitude)
            
        digital_twin_manager.add_entity(entity)
        return entity.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/entities/equipment", response_model=EntityResponse)
async def create_equipment(equipment: EquipmentCreate):
    """Create a new equipment entity in the digital twin."""
    try:
        entity = Equipment(
            name=equipment.name,
            metadata=equipment.metadata,
            equipment_type=equipment.equipment_type,
            status=equipment.status
        )
        
        if equipment.location_id:
            location = digital_twin_manager.get_entity(equipment.location_id)
            if location:
                entity.add_relationship("located_at", location.entity_id)
            else:
                raise HTTPException(status_code=404, detail=f"Location entity {equipment.location_id} not found")
            
        digital_twin_manager.add_entity(entity)
        return entity.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(entity_id: str):
    """Get an entity from the digital twin by ID."""
    entity = digital_twin_manager.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return entity.to_dict()

@router.put("/entities/{entity_id}", response_model=EntityResponse)
async def update_entity(entity_id: str, properties: Dict[str, Any]):
    """Update an entity in the digital twin."""
    success = digital_twin_manager.update_entity(entity_id, properties)
    if not success:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return digital_twin_manager.get_entity(entity_id).to_dict()

@router.delete("/entities/{entity_id}")
async def delete_entity(entity_id: str):
    """Delete an entity from the digital twin."""
    success = digital_twin_manager.delete_entity(entity_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    return {"status": "success", "message": f"Entity {entity_id} deleted"}

@router.get("/entities", response_model=List[EntityResponse])
async def list_entities(
    entity_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List entities in the digital twin, with optional filtering by type."""
    entities = list(digital_twin_manager.entities.values())
    
    if entity_type:
        if entity_type == "MineralDeposit":
            entities = [e for e in entities if isinstance(e, MineralDeposit)]
        elif entity_type == "ExplorationArea":
            entities = [e for e in entities if isinstance(e, ExplorationArea)]
        elif entity_type == "Equipment":
            entities = [e for e in entities if isinstance(e, Equipment)]
        elif entity_type == "SpatialEntity":
            entities = [e for e in entities if isinstance(e, SpatialEntity)]
    
    # Apply pagination
    paginated_entities = entities[offset:offset + limit]
    
    return [entity.to_dict() for entity in paginated_entities]

# Simulation endpoints
@router.post("/simulations/extraction", response_model=SimulationResponse)
async def create_extraction_simulation(simulation: ExtractionSimulationCreate):
    """Create a new extraction simulation in the digital twin."""
    try:
        # Get the deposit entity
        deposit = digital_twin_manager.get_entity(simulation.deposit_id)
        if not deposit:
            raise HTTPException(status_code=404, detail=f"Deposit {simulation.deposit_id} not found")
        
        # Create the simulation
        sim = ExtractionSimulation(
            name=simulation.name,
            description=simulation.description
        )
        
        # Set simulation parameters
        sim.set_parameters({
            "deposit_data": deposit.to_dict(),
            "extraction_rate": simulation.extraction_rate,
            "extraction_duration": simulation.extraction_duration,
            "extraction_method": simulation.extraction_method,
            "efficiency": simulation.efficiency,
            "mineral_price": simulation.mineral_price
        })
        
        # Add the simulation to the manager
        digital_twin_manager.add_simulation(sim)
        
        return sim.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/simulations/environmental", response_model=SimulationResponse)
async def create_environmental_simulation(simulation: EnvironmentalSimulationCreate):
    """Create a new environmental impact simulation in the digital twin."""
    try:
        # Get the extraction simulation
        extraction_sim = digital_twin_manager.get_simulation(simulation.extraction_simulation_id)
        if not extraction_sim:
            raise HTTPException(status_code=404, detail=f"Extraction simulation {simulation.extraction_simulation_id} not found")
        
        # Get the area entity
        area = digital_twin_manager.get_entity(simulation.area_id)
        if not area:
            raise HTTPException(status_code=404, detail=f"Area {simulation.area_id} not found")
        
        # Create the simulation
        sim = EnvironmentalImpactSimulation(
            name=simulation.name,
            description=simulation.description
        )
        
        # Set simulation parameters
        sim.set_parameters({
            "extraction_data": extraction_sim.get_results(),
            "area_data": area.to_dict(),
            "environmental_factors": simulation.environmental_factors
        })
        
        # Add the simulation to the manager
        digital_twin_manager.add_simulation(sim)
        
        return sim.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/simulations/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: str):
    """Get a simulation from the digital twin by ID."""
    simulation = digital_twin_manager.get_simulation(simulation_id)
    if not simulation:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    return simulation.to_dict()

@router.post("/simulations/{simulation_id}/run", response_model=SimulationResponse)
async def run_simulation(simulation_id: str):
    """Run a simulation in the digital twin."""
    success = digital_twin_manager.run_simulation(simulation_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    return digital_twin_manager.get_simulation(simulation_id).to_dict()

@router.get("/simulations/{simulation_id}/results")
async def get_simulation_results(simulation_id: str):
    """Get the results of a simulation."""
    results = digital_twin_manager.get_simulation_results(simulation_id)
    if results is None:
        raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found")
    return results

@router.get("/simulations", response_model=List[SimulationResponse])
async def list_simulations(
    simulation_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List simulations in the digital twin, with optional filtering."""
    simulations = list(digital_twin_manager.simulations.values())
    
    if simulation_type:
        if simulation_type == "ExtractionSimulation":
            simulations = [s for s in simulations if isinstance(s, ExtractionSimulation)]
        elif simulation_type == "EnvironmentalImpactSimulation":
            simulations = [s for s in simulations if isinstance(s, EnvironmentalImpactSimulation)]
    
    if status:
        simulations = [s for s in simulations if s.status == status]
    
    # Apply pagination
    paginated_simulations = simulations[offset:offset + limit]
    
    return [simulation.to_dict() for simulation in paginated_simulations]

# Export/Import endpoints
@router.post("/export")
async def export_digital_twin(file_path: str = Body(..., embed=True)):
    """Export the digital twin to a JSON file."""
    try:
        digital_twin_manager.export_to_json(file_path)
        return {"status": "success", "message": f"Digital twin exported to {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/import")
async def import_digital_twin(file_path: str = Body(..., embed=True)):
    """Import the digital twin from a JSON file."""
    try:
        digital_twin_manager.import_from_json(file_path)
        return {"status": "success", "message": f"Digital twin imported from {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
