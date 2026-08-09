"""
Digital Twin Core Module

This module provides the core functionality for the MineralVision digital twin system,
which creates a virtual replica of exploration and mining operations.

The digital twin system enables:
1. Real-time monitoring of exploration activities
2. Simulation of different extraction scenarios
3. Environmental impact prediction
4. Operational optimization
"""

import uuid
import datetime
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Any, Optional, Union, Tuple

class DigitalTwinEntity:
    """Base class for all entities in the digital twin system."""
    
    def __init__(self, entity_id: str = None, name: str = None, metadata: Dict = None):
        """
        Initialize a digital twin entity.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name for the entity
            metadata: Additional metadata for the entity
        """
        self.entity_id = entity_id or str(uuid.uuid4())
        self.name = name or f"Entity-{self.entity_id[:8]}"
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        self.properties = {}
        self.relationships = {}
        
    def update_property(self, key: str, value: Any) -> None:
        """
        Update a property of the entity.
        
        Args:
            key: Property name
            value: Property value
        """
        self.properties[key] = value
        self.updated_at = datetime.datetime.now()
        
    def update_properties(self, properties: Dict[str, Any]) -> None:
        """
        Update multiple properties of the entity.
        
        Args:
            properties: Dictionary of property names and values
        """
        self.properties.update(properties)
        self.updated_at = datetime.datetime.now()
        
    def add_relationship(self, relation_type: str, target_entity_id: str, properties: Dict = None) -> None:
        """
        Add a relationship to another entity.
        
        Args:
            relation_type: Type of relationship
            target_entity_id: ID of the target entity
            properties: Additional properties for the relationship
        """
        if relation_type not in self.relationships:
            self.relationships[relation_type] = []
            
        self.relationships[relation_type].append({
            "target_id": target_entity_id,
            "properties": properties or {}
        })
        self.updated_at = datetime.datetime.now()
        
    def to_dict(self) -> Dict:
        """
        Convert the entity to a dictionary.
        
        Returns:
            Dictionary representation of the entity
        """
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "properties": self.properties,
            "relationships": self.relationships
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'DigitalTwinEntity':
        """
        Create an entity from a dictionary.
        
        Args:
            data: Dictionary representation of the entity
            
        Returns:
            New entity instance
        """
        entity = cls(
            entity_id=data.get("entity_id"),
            name=data.get("name"),
            metadata=data.get("metadata")
        )
        
        entity.properties = data.get("properties", {})
        entity.relationships = data.get("relationships", {})
        
        if "created_at" in data:
            entity.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            entity.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return entity


class SpatialEntity(DigitalTwinEntity):
    """Entity with spatial properties in the digital twin system."""
    
    def __init__(
        self, 
        entity_id: str = None, 
        name: str = None, 
        metadata: Dict = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326"
    ):
        """
        Initialize a spatial entity.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name for the entity
            metadata: Additional metadata for the entity
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
        """
        super().__init__(entity_id, name, metadata)
        self.geometry = geometry
        self.coordinate_system = coordinate_system
        
    def set_point_geometry(self, latitude: float, longitude: float, altitude: float = None) -> None:
        """
        Set the geometry to a point.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Optional altitude in meters
        """
        self.geometry = Point(longitude, latitude)
        if altitude is not None:
            self.update_property("altitude", altitude)
        
    def set_polygon_geometry(self, coordinates: List[Tuple[float, float]]) -> None:
        """
        Set the geometry to a polygon.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
        """
        self.geometry = Polygon(coordinates)
        
    def to_dict(self) -> Dict:
        """
        Convert the spatial entity to a dictionary.
        
        Returns:
            Dictionary representation of the spatial entity
        """
        data = super().to_dict()
        
        if self.geometry:
            data["geometry"] = json.loads(gpd.GeoSeries([self.geometry]).to_json())["features"][0]["geometry"]
            data["coordinate_system"] = self.coordinate_system
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'SpatialEntity':
        """
        Create a spatial entity from a dictionary.
        
        Args:
            data: Dictionary representation of the spatial entity
            
        Returns:
            New spatial entity instance
        """
        entity = super().from_dict(data)
        
        if "geometry" in data:
            from shapely.geometry import shape
            entity.geometry = shape(data["geometry"])
            entity.coordinate_system = data.get("coordinate_system", "EPSG:4326")
            
        return entity


class MineralDeposit(SpatialEntity):
    """Representation of a mineral deposit in the digital twin system."""
    
    def __init__(
        self, 
        entity_id: str = None, 
        name: str = None, 
        metadata: Dict = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326",
        mineral_type: str = None,
        probability: float = None,
        volume_estimate: float = None,
        depth: float = None
    ):
        """
        Initialize a mineral deposit.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name for the entity
            metadata: Additional metadata for the entity
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
            mineral_type: Type of mineral
            probability: Probability of the deposit
            volume_estimate: Estimated volume in cubic meters
            depth: Depth in meters
        """
        super().__init__(entity_id, name, metadata, geometry, coordinate_system)
        
        if mineral_type:
            self.update_property("mineral_type", mineral_type)
        
        if probability is not None:
            self.update_property("probability", probability)
            
        if volume_estimate is not None:
            self.update_property("volume_estimate", volume_estimate)
            
        if depth is not None:
            self.update_property("depth", depth)


class ExplorationArea(SpatialEntity):
    """Representation of an exploration area in the digital twin system."""
    
    def __init__(
        self, 
        entity_id: str = None, 
        name: str = None, 
        metadata: Dict = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326",
        status: str = None,
        priority: str = None,
        start_date: datetime.date = None,
        end_date: datetime.date = None
    ):
        """
        Initialize an exploration area.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name for the entity
            metadata: Additional metadata for the entity
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
            status: Status of the exploration area
            priority: Priority of the exploration area
            start_date: Start date of exploration
            end_date: End date of exploration
        """
        super().__init__(entity_id, name, metadata, geometry, coordinate_system)
        
        if status:
            self.update_property("status", status)
        
        if priority:
            self.update_property("priority", priority)
            
        if start_date:
            self.update_property("start_date", start_date.isoformat())
            
        if end_date:
            self.update_property("end_date", end_date.isoformat())


class Equipment(DigitalTwinEntity):
    """Representation of equipment in the digital twin system."""
    
    def __init__(
        self, 
        entity_id: str = None, 
        name: str = None, 
        metadata: Dict = None,
        equipment_type: str = None,
        status: str = None,
        location: SpatialEntity = None
    ):
        """
        Initialize an equipment entity.
        
        Args:
            entity_id: Unique identifier for the entity
            name: Human-readable name for the entity
            metadata: Additional metadata for the entity
            equipment_type: Type of equipment
            status: Status of the equipment
            location: Location of the equipment
        """
        super().__init__(entity_id, name, metadata)
        
        if equipment_type:
            self.update_property("equipment_type", equipment_type)
        
        if status:
            self.update_property("status", status)
            
        if location:
            self.add_relationship("located_at", location.entity_id)


class DigitalTwinSimulation:
    """Simulation capabilities for the digital twin system."""
    
    def __init__(self, name: str, description: str = None):
        """
        Initialize a digital twin simulation.
        
        Args:
            name: Name of the simulation
            description: Description of the simulation
        """
        self.simulation_id = str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.created_at = datetime.datetime.now()
        self.parameters = {}
        self.results = {}
        self.status = "created"
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set simulation parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.parameters = parameters
        self.status = "configured"
        
    def run(self) -> None:
        """
        Run the simulation.
        
        This is a placeholder method that should be overridden by subclasses.
        """
        self.status = "running"
        # Simulation logic would go here
        self.status = "completed"
        
    def get_results(self) -> Dict[str, Any]:
        """
        Get the simulation results.
        
        Returns:
            Dictionary of simulation results
        """
        return self.results
        
    def to_dict(self) -> Dict:
        """
        Convert the simulation to a dictionary.
        
        Returns:
            Dictionary representation of the simulation
        """
        return {
            "simulation_id": self.simulation_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "parameters": self.parameters,
            "results": self.results,
            "status": self.status
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'DigitalTwinSimulation':
        """
        Create a simulation from a dictionary.
        
        Args:
            data: Dictionary representation of the simulation
            
        Returns:
            New simulation instance
        """
        simulation = cls(
            name=data.get("name", "Unnamed Simulation"),
            description=data.get("description", "")
        )
        
        simulation.simulation_id = data.get("simulation_id", str(uuid.uuid4()))
        simulation.parameters = data.get("parameters", {})
        simulation.results = data.get("results", {})
        simulation.status = data.get("status", "created")
        
        if "created_at" in data:
            simulation.created_at = datetime.datetime.fromisoformat(data["created_at"])
            
        return simulation


class ExtractionSimulation(DigitalTwinSimulation):
    """Simulation of mineral extraction scenarios."""
    
    def __init__(self, name: str, description: str = None):
        """
        Initialize an extraction simulation.
        
        Args:
            name: Name of the simulation
            description: Description of the simulation
        """
        super().__init__(name, description)
        
    def run(self) -> None:
        """
        Run the extraction simulation.
        """
        self.status = "running"
        
        # Get required parameters
        deposit_data = self.parameters.get("deposit_data", {})
        extraction_rate = self.parameters.get("extraction_rate", 1000)  # tons per day
        extraction_duration = self.parameters.get("extraction_duration", 365)  # days
        extraction_method = self.parameters.get("extraction_method", "open_pit")
        
        # Simple simulation logic
        if not deposit_data:
            self.results = {
                "error": "No deposit data provided"
            }
            self.status = "failed"
            return
        
        # Calculate total extraction
        total_volume = deposit_data.get("volume_estimate", 0)
        mineral_density = deposit_data.get("density", 2.7)  # g/cm³
        total_mass = total_volume * mineral_density
        
        daily_extraction = min(extraction_rate, total_mass / extraction_duration)
        total_extracted = min(daily_extraction * extraction_duration, total_mass)
        extraction_efficiency = self.parameters.get("efficiency", 0.8)
        actual_extracted = total_extracted * extraction_efficiency
        
        # Calculate costs
        unit_cost = 0
        if extraction_method == "open_pit":
            unit_cost = 10  # $ per ton
        elif extraction_method == "underground":
            unit_cost = 25  # $ per ton
        elif extraction_method == "in_situ":
            unit_cost = 15  # $ per ton
            
        total_cost = actual_extracted * unit_cost
        
        # Calculate revenue
        mineral_price = self.parameters.get("mineral_price", 100)  # $ per ton
        total_revenue = actual_extracted * mineral_price
        
        # Calculate profit
        profit = total_revenue - total_cost
        
        # Store results
        self.results = {
            "total_volume": total_volume,
            "total_mass": total_mass,
            "daily_extraction": daily_extraction,
            "total_extracted": total_extracted,
            "actual_extracted": actual_extracted,
            "extraction_efficiency": extraction_efficiency,
            "total_cost": total_cost,
            "total_revenue": total_revenue,
            "profit": profit,
            "extraction_duration": extraction_duration,
            "extraction_method": extraction_method,
            "extraction_complete": total_extracted >= total_mass,
            "remaining_mass": max(0, total_mass - total_extracted)
        }
        
        self.status = "completed"


class EnvironmentalImpactSimulation(DigitalTwinSimulation):
    """Simulation of environmental impacts of mining operations."""
    
    def __init__(self, name: str, description: str = None):
        """
        Initialize an environmental impact simulation.
        
        Args:
            name: Name of the simulation
            description: Description of the simulation
        """
        super().__init__(name, description)
        
    def run(self) -> None:
        """
        Run the environmental impact simulation.
        """
        self.status = "running"
        
        # Get required parameters
        extraction_data = self.parameters.get("extraction_data", {})
        area_data = self.parameters.get("area_data", {})
        environmental_factors = self.parameters.get("environmental_factors", {})
        
        # Simple simulation logic
        if not extraction_data or not area_data:
            self.results = {
                "error": "Missing required data"
            }
            self.status = "failed"
            return
        
        # Calculate land disturbance
        extraction_method = extraction_data.get("extraction_method", "open_pit")
        extraction_volume = extraction_data.get("total_volume", 0)
        
        land_disturbance = 0
        if extraction_method == "open_pit":
            land_disturbance = extraction_volume / 20  # m²
        elif extraction_method == "underground":
            land_disturbance = extraction_volume / 100  # m²
        elif extraction_method == "in_situ":
            land_disturbance = extraction_volume / 200  # m²
            
        # Calculate water usage
        water_usage_factor = 0
        if extraction_method == "open_pit":
            water_usage_factor = 0.5  # m³ per ton
        elif extraction_method == "underground":
            water_usage_factor = 0.3  # m³ per ton
        elif extraction_method == "in_situ":
            water_usage_factor = 0.8  # m³ per ton
            
        water_usage = extraction_data.get("actual_extracted", 0) * water_usage_factor
        
        # Calculate carbon emissions
        energy_usage_factor = 0
        if extraction_method == "open_pit":
            energy_usage_factor = 20  # kWh per ton
        elif extraction_method == "underground":
            energy_usage_factor = 40  # kWh per ton
        elif extraction_method == "in_situ":
            energy_usage_factor = 30  # kWh per ton
            
        energy_usage = extraction_data.get("actual_extracted", 0) * energy_usage_factor
        carbon_emissions = energy_usage * 0.5  # kg CO2 per kWh
        
        # Calculate biodiversity impact
        area_size = area_data.get("size", 0)
        biodiversity_index = environmental_factors.get("biodiversity_index", 0.5)
        biodiversity_impact = land_disturbance / area_size * biodiversity_index
        
        # Calculate rehabilitation potential
        rehabilitation_factor = 0
        if extraction_method == "open_pit":
            rehabilitation_factor = 0.6
        elif extraction_method == "underground":
            rehabilitation_factor = 0.8
        elif extraction_method == "in_situ":
            rehabilitation_factor = 0.9
            
        rehabilitation_potential = land_disturbance * rehabilitation_factor
        
        # Store results
        self.results = {
            "land_disturbance": land_disturbance,
            "water_usage": water_usage,
            "energy_usage": energy_usage,
            "carbon_emissions": carbon_emissions,
            "biodiversity_impact": biodiversity_impact,
            "rehabilitation_potential": rehabilitation_potential,
            "impact_score": (land_disturbance / 1000 + water_usage / 10000 + carbon_emissions / 100000) / 3,
            "mitigation_recommendations": [
                "Implement water recycling systems",
                "Use renewable energy sources",
                "Progressive rehabilitation during operations",
                "Minimize footprint through efficient planning"
            ]
        }
        
        self.status = "completed"


class DigitalTwinManager:
    """Manager for the digital twin system."""
    
    def __init__(self):
        """Initialize the digital twin manager."""
        self.entities = {}
        self.simulations = {}
        
    def add_entity(self, entity: DigitalTwinEntity) -> str:
        """
        Add an entity to the digital twin system.
        
        Args:
            entity: Entity to add
            
        Returns:
            Entity ID
        """
        self.entities[entity.entity_id] = entity
        return entity.entity_id
        
    def get_entity(self, entity_id: str) -> Optional[DigitalTwinEntity]:
        """
        Get an entity from the digital twin system.
        
        Args:
            entity_id: ID of the entity to get
            
        Returns:
            Entity if found, None otherwise
        """
        return self.entities.get(entity_id)
        
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity in the digital twin system.
        
        Args:
            entity_id: ID of the entity to update
            properties: Properties to update
            
        Returns:
            True if the entity was updated, False otherwise
        """
        entity = self.get_entity(entity_id)
        if entity:
            entity.update_properties(properties)
            return True
        return False
        
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the digital twin system.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if the entity was deleted, False otherwise
        """
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False
        
    def add_simulation(self, simulation: DigitalTwinSimulation) -> str:
        """
        Add a simulation to the digital twin system.
        
        Args:
            simulation: Simulation to add
            
        Returns:
            Simulation ID
        """
        self.simulations[simulation.simulation_id] = simulation
        return simulation.simulation_id
        
    def get_simulation(self, simulation_id: str) -> Optional[DigitalTwinSimulation]:
        """
        Get a simulation from the digital twin system.
        
        Args:
            simulation_id: ID of the simulation to get
            
        Returns:
            Simulation if found, None otherwise
        """
        return self.simulations.get(simulation_id)
        
    def run_simulation(self, simulation_id: str) -> bool:
        """
        Run a simulation in the digital twin system.
        
        Args:
            simulation_id: ID of the simulation to run
            
        Returns:
            True if the simulation was run, False otherwise
        """
        simulation = self.get_simulation(simulation_id)
        if simulation:
            simulation.run()
            return True
        return False
        
    def get_simulation_results(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the results of a simulation.
        
        Args:
            simulation_id: ID of the simulation
            
        Returns:
            Simulation results if found, None otherwise
        """
        simulation = self.get_simulation(simulation_id)
        if simulation:
            return simulation.get_results()
        return None
        
    def export_to_json(self, file_path: str) -> None:
        """
        Export the digital twin system to a JSON file.
        
        Args:
            file_path: Path to the output file
        """
        data = {
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "simulations": {sim_id: sim.to_dict() for sim_id, sim in self.simulations.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_from_json(self, file_path: str) -> None:
        """
        Import the digital twin system from a JSON file.
        
        Args:
            file_path: Path to the input file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Import entities
        for entity_id, entity_data in data.get("entities", {}).items():
            entity_type = entity_data.get("metadata", {}).get("entity_type", "DigitalTwinEntity")
            
            if entity_type == "MineralDeposit":
                entity = MineralDeposit.from_dict(entity_data)
            elif entity_type == "ExplorationArea":
                entity = ExplorationArea.from_dict(entity_data)
            elif entity_type == "Equipment":
                entity = Equipment.from_dict(entity_data)
            elif entity_type == "SpatialEntity":
                entity = SpatialEntity.from_dict(entity_data)
            else:
                entity = DigitalTwinEntity.from_dict(entity_data)
                
            self.add_entity(entity)
            
        # Import simulations
        for sim_id, sim_data in data.get("simulations", {}).items():
            sim_type = sim_data.get("metadata", {}).get("simulation_type", "DigitalTwinSimulation")
            
            if sim_type == "ExtractionSimulation":
                simulation = ExtractionSimulation.from_dict(sim_data)
            elif sim_type == "EnvironmentalImpactSimulation":
                simulation = EnvironmentalImpactSimulation.from_dict(sim_data)
            else:
                simulation = DigitalTwinSimulation.from_dict(sim_data)
                
            self.add_simulation(simulation)
