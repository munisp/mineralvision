import os
import uuid
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from shapely.geometry import Point, Polygon, shape
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class DroneSpecification(BaseModel):
    """Drone specification model."""
    drone_id: str
    model: str
    battery_capacity_mah: int
    max_flight_time_minutes: int
    max_speed_mps: float
    max_payload_grams: int
    camera_resolution: str
    sensors: List[str]
    communication_range_meters: int
    
class DroneState(BaseModel):
    """Current state of a drone."""
    drone_id: str
    position: Dict[str, float]  # lat, lon, altitude
    battery_percentage: float
    status: str  # 'idle', 'mission', 'returning', 'charging', 'error'
    payload_status: str  # 'empty', 'sample_collected'
    current_mission_id: Optional[str] = None
    
class SamplingPoint(BaseModel):
    """A point of interest for sampling."""
    point_id: str
    position: Dict[str, float]  # lat, lon
    priority: int  # 1-10, higher is more important
    sample_type: str  # 'soil', 'rock', 'water', etc.
    estimated_time_seconds: int
    status: str  # 'pending', 'assigned', 'completed', 'failed'
    assigned_drone_id: Optional[str] = None
    completion_time: Optional[datetime] = None
    
class MissionPlan(BaseModel):
    """A mission plan for a drone."""
    mission_id: str
    drone_id: str
    waypoints: List[Dict[str, float]]  # list of lat, lon, altitude
    sampling_points: List[str]  # list of sampling_point_ids
    estimated_duration_minutes: float
    battery_requirement_percentage: float
    status: str  # 'planned', 'in_progress', 'completed', 'failed'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class ExplorationArea(BaseModel):
    """An area for exploration."""
    area_id: str
    name: str
    geometry: Dict[str, Any]  # GeoJSON geometry
    priority: int  # 1-10, higher is more important
    exploration_status: str  # 'pending', 'in_progress', 'completed'
    completion_percentage: float = 0.0
    
class AutonomousExplorationSystem:
    """Core system for autonomous exploration."""
    
    def __init__(self, data_dir: str = None):
        data_dir = data_dir or os.getenv(
            "MINERALVISION_DATA_DIR",
            os.path.join(os.path.expanduser("~"), ".mineralvision", "data", "autonomous_exploration"),
        )
        """Initialize the autonomous exploration system."""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data stores
        self.drones: Dict[str, DroneSpecification] = {}
        self.drone_states: Dict[str, DroneState] = {}
        self.sampling_points: Dict[str, SamplingPoint] = {}
        self.mission_plans: Dict[str, MissionPlan] = {}
        self.exploration_areas: Dict[str, ExplorationArea] = {}
        
        # Load existing data if available
        self._load_data()
    
    def _load_data(self):
        """Load data from disk."""
        try:
            if os.path.exists(os.path.join(self.data_dir, "drones.json")):
                with open(os.path.join(self.data_dir, "drones.json"), "r") as f:
                    drones_data = json.load(f)
                    for drone_data in drones_data:
                        drone = DroneSpecification(**drone_data)
                        self.drones[drone.drone_id] = drone
            
            if os.path.exists(os.path.join(self.data_dir, "drone_states.json")):
                with open(os.path.join(self.data_dir, "drone_states.json"), "r") as f:
                    drone_states_data = json.load(f)
                    for state_data in drone_states_data:
                        state = DroneState(**state_data)
                        self.drone_states[state.drone_id] = state
            
            if os.path.exists(os.path.join(self.data_dir, "sampling_points.json")):
                with open(os.path.join(self.data_dir, "sampling_points.json"), "r") as f:
                    sampling_points_data = json.load(f)
                    for point_data in sampling_points_data:
                        point = SamplingPoint(**point_data)
                        self.sampling_points[point.point_id] = point
            
            if os.path.exists(os.path.join(self.data_dir, "mission_plans.json")):
                with open(os.path.join(self.data_dir, "mission_plans.json"), "r") as f:
                    mission_plans_data = json.load(f)
                    for plan_data in mission_plans_data:
                        # Convert datetime strings to datetime objects
                        if plan_data.get("start_time"):
                            plan_data["start_time"] = datetime.fromisoformat(plan_data["start_time"])
                        if plan_data.get("end_time"):
                            plan_data["end_time"] = datetime.fromisoformat(plan_data["end_time"])
                        plan = MissionPlan(**plan_data)
                        self.mission_plans[plan.mission_id] = plan
            
            if os.path.exists(os.path.join(self.data_dir, "exploration_areas.json")):
                with open(os.path.join(self.data_dir, "exploration_areas.json"), "r") as f:
                    exploration_areas_data = json.load(f)
                    for area_data in exploration_areas_data:
                        area = ExplorationArea(**area_data)
                        self.exploration_areas[area.area_id] = area
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to disk."""
        try:
            with open(os.path.join(self.data_dir, "drones.json"), "w") as f:
                json.dump([drone.dict() for drone in self.drones.values()], f, indent=2)
            
            with open(os.path.join(self.data_dir, "drone_states.json"), "w") as f:
                json.dump([state.dict() for state in self.drone_states.values()], f, indent=2)
            
            with open(os.path.join(self.data_dir, "sampling_points.json"), "w") as f:
                json.dump([point.dict() for point in self.sampling_points.values()], f, indent=2)
            
            with open(os.path.join(self.data_dir, "mission_plans.json"), "w") as f:
                # Convert datetime objects to ISO format strings for JSON serialization
                mission_plans_data = []
                for plan in self.mission_plans.values():
                    plan_dict = plan.dict()
                    if plan_dict.get("start_time"):
                        plan_dict["start_time"] = plan_dict["start_time"].isoformat()
                    if plan_dict.get("end_time"):
                        plan_dict["end_time"] = plan_dict["end_time"].isoformat()
                    mission_plans_data.append(plan_dict)
                json.dump(mission_plans_data, f, indent=2)
            
            with open(os.path.join(self.data_dir, "exploration_areas.json"), "w") as f:
                json.dump([area.dict() for area in self.exploration_areas.values()], f, indent=2)
        
        except Exception as e:
            print(f"Error saving data: {e}")
    
    # Drone management methods
    def register_drone(self, drone_spec: DroneSpecification) -> str:
        """Register a new drone in the system."""
        if not drone_spec.drone_id:
            drone_spec.drone_id = str(uuid.uuid4())
        
        self.drones[drone_spec.drone_id] = drone_spec
        
        # Initialize drone state if not exists
        if drone_spec.drone_id not in self.drone_states:
            self.drone_states[drone_spec.drone_id] = DroneState(
                drone_id=drone_spec.drone_id,
                position={"lat": 0.0, "lon": 0.0, "altitude": 0.0},
                battery_percentage=100.0,
                status="idle",
                payload_status="empty"
            )
        
        self._save_data()
        return drone_spec.drone_id
    
    def update_drone_state(self, state: DroneState) -> bool:
        """Update the state of a drone."""
        if state.drone_id not in self.drones:
            return False
        
        self.drone_states[state.drone_id] = state
        self._save_data()
        return True
    
    def get_drone(self, drone_id: str) -> Optional[DroneSpecification]:
        """Get drone specification by ID."""
        return self.drones.get(drone_id)
    
    def get_drone_state(self, drone_id: str) -> Optional[DroneState]:
        """Get current state of a drone by ID."""
        return self.drone_states.get(drone_id)
    
    def get_all_drones(self) -> List[DroneSpecification]:
        """Get all registered drones."""
        return list(self.drones.values())
    
    def get_all_drone_states(self) -> List[DroneState]:
        """Get states of all drones."""
        return list(self.drone_states.values())
    
    def get_available_drones(self) -> List[str]:
        """Get IDs of drones available for missions."""
        available_drones = []
        for drone_id, state in self.drone_states.items():
            if state.status == "idle" and state.battery_percentage >= 30.0:
                available_drones.append(drone_id)
        return available_drones
    
    # Sampling point methods
    def add_sampling_point(self, point: SamplingPoint) -> str:
        """Add a new sampling point."""
        if not point.point_id:
            point.point_id = str(uuid.uuid4())
        
        self.sampling_points[point.point_id] = point
        self._save_data()
        return point.point_id
    
    def update_sampling_point(self, point: SamplingPoint) -> bool:
        """Update a sampling point."""
        if point.point_id not in self.sampling_points:
            return False
        
        self.sampling_points[point.point_id] = point
        self._save_data()
        return True
    
    def get_sampling_point(self, point_id: str) -> Optional[SamplingPoint]:
        """Get a sampling point by ID."""
        return self.sampling_points.get(point_id)
    
    def get_all_sampling_points(self) -> List[SamplingPoint]:
        """Get all sampling points."""
        return list(self.sampling_points.values())
    
    def get_pending_sampling_points(self) -> List[SamplingPoint]:
        """Get sampling points that are pending assignment."""
        return [point for point in self.sampling_points.values() if point.status == "pending"]
    
    # Mission planning methods
    def create_mission_plan(self, drone_id: str, sampling_point_ids: List[str]) -> Optional[str]:
        """Create a mission plan for a drone to visit sampling points."""
        if drone_id not in self.drones or drone_id not in self.drone_states:
            return None
        
        drone_spec = self.drones[drone_id]
        drone_state = self.drone_states[drone_id]
        
        if drone_state.status != "idle":
            return None
        
        # Validate sampling points
        valid_sampling_points = []
        for point_id in sampling_point_ids:
            if point_id in self.sampling_points and self.sampling_points[point_id].status == "pending":
                valid_sampling_points.append(self.sampling_points[point_id])
        
        if not valid_sampling_points:
            return None
        
        # Generate waypoints from sampling points
        waypoints = []
        for point in valid_sampling_points:
            waypoints.append({
                "lat": point.position["lat"],
                "lon": point.position["lon"],
                "altitude": 50.0  # Default altitude in meters
            })
        
        # Calculate estimated duration and battery requirement
        total_distance_meters = self._calculate_mission_distance(drone_state.position, waypoints)
        avg_speed_mps = drone_spec.max_speed_mps * 0.7  # 70% of max speed for conservative estimate
        flight_time_minutes = (total_distance_meters / avg_speed_mps) / 60
        
        # Add time for sampling
        sampling_time_minutes = sum([point.estimated_time_seconds for point in valid_sampling_points]) / 60
        total_time_minutes = flight_time_minutes + sampling_time_minutes
        
        # Calculate battery requirement
        battery_requirement = (total_time_minutes / drone_spec.max_flight_time_minutes) * 100
        
        # Create mission plan
        mission_id = str(uuid.uuid4())
        mission_plan = MissionPlan(
            mission_id=mission_id,
            drone_id=drone_id,
            waypoints=waypoints,
            sampling_points=[point.point_id for point in valid_sampling_points],
            estimated_duration_minutes=total_time_minutes,
            battery_requirement_percentage=battery_requirement,
            status="planned"
        )
        
        self.mission_plans[mission_id] = mission_plan
        
        # Update sampling points
        for point in valid_sampling_points:
            point.status = "assigned"
            point.assigned_drone_id = drone_id
            self.sampling_points[point.point_id] = point
        
        self._save_data()
        return mission_id
    
    def start_mission(self, mission_id: str) -> bool:
        """Start a planned mission."""
        if mission_id not in self.mission_plans:
            return False
        
        mission = self.mission_plans[mission_id]
        if mission.status != "planned":
            return False
        
        drone_id = mission.drone_id
        if drone_id not in self.drone_states:
            return False
        
        drone_state = self.drone_states[drone_id]
        if drone_state.status != "idle":
            return False
        
        if drone_state.battery_percentage < mission.battery_requirement_percentage:
            return False
        
        # Update mission status
        mission.status = "in_progress"
        mission.start_time = datetime.now()
        self.mission_plans[mission_id] = mission
        
        # Update drone state
        drone_state.status = "mission"
        drone_state.current_mission_id = mission_id
        self.drone_states[drone_id] = drone_state
        
        self._save_data()
        return True
    
    def complete_mission(self, mission_id: str, success: bool = True) -> bool:
        """Mark a mission as completed or failed."""
        if mission_id not in self.mission_plans:
            return False
        
        mission = self.mission_plans[mission_id]
        if mission.status != "in_progress":
            return False
        
        drone_id = mission.drone_id
        if drone_id not in self.drone_states:
            return False
        
        # Update mission status
        mission.status = "completed" if success else "failed"
        mission.end_time = datetime.now()
        self.mission_plans[mission_id] = mission
        
        # Update drone state
        drone_state = self.drone_states[drone_id]
        drone_state.status = "idle"
        drone_state.current_mission_id = None
        self.drone_states[drone_id] = drone_state
        
        # Update sampling points
        for point_id in mission.sampling_points:
            if point_id in self.sampling_points:
                point = self.sampling_points[point_id]
                if success:
                    point.status = "completed"
                    point.completion_time = datetime.now()
                else:
                    point.status = "pending"
                    point.assigned_drone_id = None
                self.sampling_points[point_id] = point
        
        self._save_data()
        return True
    
    def get_mission_plan(self, mission_id: str) -> Optional[MissionPlan]:
        """Get a mission plan by ID."""
        return self.mission_plans.get(mission_id)
    
    def get_all_mission_plans(self) -> List[MissionPlan]:
        """Get all mission plans."""
        return list(self.mission_plans.values())
    
    def get_drone_missions(self, drone_id: str) -> List[MissionPlan]:
        """Get all missions for a specific drone."""
        return [mission for mission in self.mission_plans.values() if mission.drone_id == drone_id]
    
    # Exploration area methods
    def add_exploration_area(self, area: ExplorationArea) -> str:
        """Add a new exploration area."""
        if not area.area_id:
            area.area_id = str(uuid.uuid4())
        
        self.exploration_areas[area.area_id] = area
        self._save_data()
        return area.area_id
    
    def update_exploration_area(self, area: ExplorationArea) -> bool:
        """Update an exploration area."""
        if area.area_id not in self.exploration_areas:
            return False
        
        self.exploration_areas[area.area_id] = area
        self._save_data()
        return True
    
    def get_exploration_area(self, area_id: str) -> Optional[ExplorationArea]:
        """Get an exploration area by ID."""
        return self.exploration_areas.get(area_id)
    
    def get_all_exploration_areas(self) -> List[ExplorationArea]:
        """Get all exploration areas."""
        return list(self.exploration_areas.values())
    
    # Advanced planning methods
    def generate_sampling_points_for_area(self, area_id: str, density: float = 0.0001) -> List[str]:
        """Generate sampling points for an exploration area based on density."""
        if area_id not in self.exploration_areas:
            return []
        
        area = self.exploration_areas[area_id]
        geometry = shape(area.geometry)
        
        if geometry.geom_type != "Polygon":
            return []
        
        # Get bounds of the polygon
        minx, miny, maxx, maxy = geometry.bounds
        
        # Generate grid points
        x_coords = np.arange(minx, maxx, density)
        y_coords = np.arange(miny, maxy, density)
        
        sampling_point_ids = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                if geometry.contains(point):
                    # Create sampling point
                    sampling_point = SamplingPoint(
                        point_id=str(uuid.uuid4()),
                        position={"lat": y, "lon": x},
                        priority=5,  # Default medium priority
                        sample_type="soil",  # Default sample type
                        estimated_time_seconds=300,  # Default 5 minutes
                        status="pending",
                        assigned_drone_id=None
                    )
                    
                    # Add to system
                    self.sampling_points[sampling_point.point_id] = sampling_point
                    sampling_point_ids.append(sampling_point.point_id)
        
        self._save_data()
        return sampling_point_ids
    
    def optimize_drone_assignments(self) -> Dict[str, List[str]]:
        """Optimize assignment of drones to sampling points using Hungarian algorithm."""
        available_drones = self.get_available_drones()
        pending_points = self.get_pending_sampling_points()
        
        if not available_drones or not pending_points:
            return {}
        
        # Create cost matrix (distance between drones and sampling points)
        cost_matrix = np.zeros((len(available_drones), len(pending_points)))
        
        for i, drone_id in enumerate(available_drones):
            drone_state = self.drone_states[drone_id]
            drone_pos = (drone_state.position["lat"], drone_state.position["lon"])
            
            for j, point in enumerate(pending_points):
                point_pos = (point.position["lat"], point.position["lon"])
                # Calculate Haversine distance
                cost_matrix[i, j] = self._haversine_distance(drone_pos, point_pos)
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create assignment dictionary
        assignments = {}
        for i, drone_idx in enumerate(row_ind):
            point_idx = col_ind[i]
            drone_id = available_drones[drone_idx]
            point_id = pending_points[point_idx].point_id
            
            if drone_id not in assignments:
                assignments[drone_id] = []
            
            assignments[drone_id].append(point_id)
        
        return assignments
    
    def generate_swarm_mission(self, area_id: str, num_drones: int = 3) -> List[str]:
        """Generate coordinated missions for a swarm of drones to explore an area."""
        if area_id not in self.exploration_areas:
            return []
        
        # Get available drones
        available_drones = self.get_available_drones()
        if len(available_drones) < num_drones:
            return []
        
        # Select drones for the swarm
        swarm_drones = available_drones[:num_drones]
        
        # Generate sampling points if needed
        area_points = [p for p in self.sampling_points.values() 
                      if p.status == "pending" and self._point_in_area(p.position, area_id)]
        
        if not area_points:
            # Generate new sampling points
            self.generate_sampling_points_for_area(area_id)
            area_points = [p for p in self.sampling_points.values() 
                          if p.status == "pending" and self._point_in_area(p.position, area_id)]
        
        if not area_points:
            return []
        
        # Divide area into sectors for each drone
        area = self.exploration_areas[area_id]
        geometry = shape(area.geometry)
        
        if geometry.geom_type != "Polygon":
            return []
        
        # Get bounds of the polygon
        minx, miny, maxx, maxy = geometry.bounds
        
        # Divide area horizontally into sectors
        sector_width = (maxx - minx) / num_drones
        
        # Assign points to sectors
        sectors = [[] for _ in range(num_drones)]
        for point in area_points:
            lon = point.position["lon"]
            sector_idx = min(num_drones - 1, int((lon - minx) / sector_width))
            sectors[sector_idx].append(point.point_id)
        
        # Create mission plans for each drone
        mission_ids = []
        for i, drone_id in enumerate(swarm_drones):
            if sectors[i]:  # Only create mission if there are points in the sector
                mission_id = self.create_mission_plan(drone_id, sectors[i])
                if mission_id:
                    mission_ids.append(mission_id)
        
        return mission_ids
    
    def adaptive_sampling_strategy(self, area_id: str, initial_results: Dict[str, float]) -> List[str]:
        """Generate adaptive sampling points based on initial results."""
        if area_id not in self.exploration_areas:
            return []
        
        # Get completed sampling points in the area
        completed_points = [p for p in self.sampling_points.values() 
                           if p.status == "completed" and self._point_in_area(p.position, area_id)]
        
        if not completed_points or not initial_results:
            return []
        
        # Find high-value regions based on initial results
        high_value_points = []
        for point in completed_points:
            if point.point_id in initial_results and initial_results[point.point_id] > 0.7:  # Threshold for high value
                high_value_points.append(point)
        
        if not high_value_points:
            return []
        
        # Generate new sampling points around high-value regions
        new_point_ids = []
        for point in high_value_points:
            # Generate 4 points around the high-value point
            offsets = [(0.0001, 0), (0, 0.0001), (-0.0001, 0), (0, -0.0001)]
            for dx, dy in offsets:
                new_pos = {
                    "lat": point.position["lat"] + dy,
                    "lon": point.position["lon"] + dx
                }
                
                # Check if point is in the exploration area
                if not self._point_in_area(new_pos, area_id):
                    continue
                
                # Create new sampling point with higher priority
                new_point = SamplingPoint(
                    point_id=str(uuid.uuid4()),
                    position=new_pos,
                    priority=8,  # Higher priority
                    sample_type=point.sample_type,
                    estimated_time_seconds=point.estimated_time_seconds,
                    status="pending",
                    assigned_drone_id=None
                )
                
                # Add to system
                self.sampling_points[new_point.point_id] = new_point
                new_point_ids.append(new_point.point_id)
        
        self._save_data()
        return new_point_ids
    
    # Helper methods
    def _calculate_mission_distance(self, start_pos: Dict[str, float], waypoints: List[Dict[str, float]]) -> float:
        """Calculate the total distance of a mission in meters."""
        if not waypoints:
            return 0.0
        
        total_distance = 0.0
        current_pos = (start_pos["lat"], start_pos["lon"])
        
        for waypoint in waypoints:
            waypoint_pos = (waypoint["lat"], waypoint["lon"])
            total_distance += self._haversine_distance(current_pos, waypoint_pos)
            current_pos = waypoint_pos
        
        return total_distance
    
    def _haversine_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate the Haversine distance between two points in meters."""
        # Earth radius in meters
        R = 6371000.0
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1 = np.radians(pos1)
        lat2, lon2 = np.radians(pos2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _point_in_area(self, position: Dict[str, float], area_id: str) -> bool:
        """Check if a point is within an exploration area."""
        if area_id not in self.exploration_areas:
            return False
        
        area = self.exploration_areas[area_id]
        geometry = shape(area.geometry)
        point = Point(position["lon"], position["lat"])
        
        return geometry.contains(point)
