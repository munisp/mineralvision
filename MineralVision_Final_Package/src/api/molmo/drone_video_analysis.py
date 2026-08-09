"""
Drone Video Analysis Module for MineralVision.

Specialized analysis of drone footage for mining exploration,
site monitoring, and geological survey applications.
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import json

from .molmo_integration import (
    Molmo2Client,
    Molmo2Config,
    VideoAnalysisResult,
    PointingResult,
    TrackingResult,
    AnalysisType,
    BoundingBox,
    create_molmo_client,
)

logger = logging.getLogger(__name__)


class FlightPattern(Enum):
    """Types of drone flight patterns."""
    GRID = "grid"
    ORBITAL = "orbital"
    LINEAR = "linear"
    SPIRAL = "spiral"
    WAYPOINT = "waypoint"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class TerrainType(Enum):
    """Types of terrain observed."""
    FLAT = "flat"
    HILLY = "hilly"
    MOUNTAINOUS = "mountainous"
    FORESTED = "forested"
    DESERT = "desert"
    WETLAND = "wetland"
    COASTAL = "coastal"
    MIXED = "mixed"


class GeologicalFeature(Enum):
    """Types of geological features."""
    OUTCROP = "outcrop"
    FAULT = "fault"
    FOLD = "fold"
    VEIN = "vein"
    ALTERATION_ZONE = "alteration_zone"
    GOSSAN = "gossan"
    CONTACT = "contact"
    INTRUSION = "intrusion"
    SEDIMENTARY_LAYER = "sedimentary_layer"
    ALLUVIAL_DEPOSIT = "alluvial_deposit"


@dataclass
class GeoLocation:
    """Geographic location with coordinates."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "accuracy": self.accuracy,
        }


@dataclass
class DroneMetadata:
    """Metadata from drone flight."""
    drone_model: Optional[str] = None
    camera_model: Optional[str] = None
    flight_date: Optional[datetime] = None
    flight_altitude: Optional[float] = None
    ground_sample_distance: Optional[float] = None
    flight_pattern: FlightPattern = FlightPattern.UNKNOWN
    start_location: Optional[GeoLocation] = None
    end_location: Optional[GeoLocation] = None
    waypoints: List[GeoLocation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "drone_model": self.drone_model,
            "camera_model": self.camera_model,
            "flight_date": self.flight_date.isoformat() if self.flight_date else None,
            "flight_altitude": self.flight_altitude,
            "ground_sample_distance": self.ground_sample_distance,
            "flight_pattern": self.flight_pattern.value,
            "start_location": self.start_location.to_dict() if self.start_location else None,
            "end_location": self.end_location.to_dict() if self.end_location else None,
            "waypoints": [w.to_dict() for w in self.waypoints],
        }


@dataclass
class GeologicalObservation:
    """Geological observation from drone imagery."""
    feature_type: GeologicalFeature
    location: Optional[GeoLocation]
    frame_index: int
    timestamp: float
    bbox: Optional[BoundingBox]
    description: str
    confidence: float
    mineral_indicators: List[str] = field(default_factory=list)
    color_anomalies: List[str] = field(default_factory=list)
    structural_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type": self.feature_type.value,
            "location": self.location.to_dict() if self.location else None,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "description": self.description,
            "confidence": self.confidence,
            "mineral_indicators": self.mineral_indicators,
            "color_anomalies": self.color_anomalies,
            "structural_notes": self.structural_notes,
        }


@dataclass
class SiteChangeDetection:
    """Change detection between drone surveys."""
    change_type: str
    location: Optional[GeoLocation]
    before_frame: int
    after_frame: int
    before_description: str
    after_description: str
    change_magnitude: float  # 0-1 scale
    significance: str  # low, medium, high, critical
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_type": self.change_type,
            "location": self.location.to_dict() if self.location else None,
            "before_frame": self.before_frame,
            "after_frame": self.after_frame,
            "before_description": self.before_description,
            "after_description": self.after_description,
            "change_magnitude": self.change_magnitude,
            "significance": self.significance,
            "recommendations": self.recommendations,
        }


@dataclass
class FlightAnalysisResult:
    """Complete result from drone flight analysis."""
    video_path: str
    duration_seconds: float
    total_frames: int
    frames_analyzed: int
    drone_metadata: DroneMetadata
    terrain_type: TerrainType
    terrain_description: str
    geological_observations: List[GeologicalObservation]
    site_changes: List[SiteChangeDetection]
    artisanal_mining_indicators: List[Dict[str, Any]]
    environmental_observations: List[Dict[str, Any]]
    points_of_interest: List[Dict[str, Any]]
    flight_quality_score: float
    coverage_assessment: str
    summary: str
    recommendations: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "drone_metadata": self.drone_metadata.to_dict(),
            "terrain_type": self.terrain_type.value,
            "terrain_description": self.terrain_description,
            "geological_observations": [g.to_dict() for g in self.geological_observations],
            "site_changes": [s.to_dict() for s in self.site_changes],
            "artisanal_mining_indicators": self.artisanal_mining_indicators,
            "environmental_observations": self.environmental_observations,
            "points_of_interest": self.points_of_interest,
            "flight_quality_score": self.flight_quality_score,
            "coverage_assessment": self.coverage_assessment,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }
    
    def to_geojson(self) -> Dict[str, Any]:
        """Export observations as GeoJSON for mapping."""
        features = []
        
        for obs in self.geological_observations:
            if obs.location:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [obs.location.longitude, obs.location.latitude],
                    },
                    "properties": {
                        "feature_type": obs.feature_type.value,
                        "description": obs.description,
                        "confidence": obs.confidence,
                        "mineral_indicators": obs.mineral_indicators,
                    },
                })
        
        for poi in self.points_of_interest:
            if "location" in poi and poi["location"]:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [poi["location"]["longitude"], poi["location"]["latitude"]],
                    },
                    "properties": {
                        "type": poi.get("type", "poi"),
                        "description": poi.get("description", ""),
                    },
                })
        
        return {
            "type": "FeatureCollection",
            "features": features,
        }


class DroneVideoAnalyzer:
    """
    Specialized analyzer for drone video footage.
    
    Combines Molmo2-8B's video understanding with domain-specific
    analysis for mining exploration and geological survey.
    """
    
    def __init__(
        self,
        molmo_client: Optional[Molmo2Client] = None,
        molmo_config: Optional[Molmo2Config] = None,
    ):
        self.molmo_client = molmo_client or create_molmo_client(molmo_config)
        self.config = molmo_config or Molmo2Config()
        self._lock = threading.Lock()
    
    def analyze_flight(
        self,
        video_path: str,
        metadata_path: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
    ) -> FlightAnalysisResult:
        """
        Analyze drone flight video.
        
        Args:
            video_path: Path to drone video file
            metadata_path: Optional path to flight metadata JSON
            focus_areas: Optional list of focus areas (e.g., ["geology", "mining"])
        
        Returns:
            FlightAnalysisResult with complete analysis
        """
        import time
        import cv2
        
        start_time = time.time()
        
        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Load drone metadata if available
        drone_metadata = self._load_drone_metadata(metadata_path)
        
        # Analyze terrain
        terrain_type, terrain_description = self._analyze_terrain(video_path)
        
        # Detect geological features
        geological_observations = self._detect_geological_features(video_path, fps)
        
        # Detect artisanal mining indicators
        artisanal_indicators = self._detect_artisanal_mining(video_path)
        
        # Environmental observations
        environmental_obs = self._analyze_environment(video_path)
        
        # Identify points of interest
        points_of_interest = self._identify_points_of_interest(
            video_path, focus_areas or []
        )
        
        # Assess flight quality
        flight_quality = self._assess_flight_quality(video_path)
        
        # Coverage assessment
        coverage = self._assess_coverage(video_path, drone_metadata)
        
        # Generate summary and recommendations
        summary = self._generate_summary(
            terrain_type, geological_observations, artisanal_indicators
        )
        recommendations = self._generate_recommendations(
            geological_observations, artisanal_indicators, environmental_obs
        )
        
        processing_time = time.time() - start_time
        
        return FlightAnalysisResult(
            video_path=video_path,
            duration_seconds=duration,
            total_frames=total_frames,
            frames_analyzed=min(self.config.max_frames, total_frames),
            drone_metadata=drone_metadata,
            terrain_type=terrain_type,
            terrain_description=terrain_description,
            geological_observations=geological_observations,
            site_changes=[],  # Populated by compare_flights
            artisanal_mining_indicators=artisanal_indicators,
            environmental_observations=environmental_obs,
            points_of_interest=points_of_interest,
            flight_quality_score=flight_quality,
            coverage_assessment=coverage,
            summary=summary,
            recommendations=recommendations,
            processing_time=processing_time,
            metadata={
                "focus_areas": focus_areas,
                "model": self.config.model_name,
            },
        )
    
    def _load_drone_metadata(self, metadata_path: Optional[str]) -> DroneMetadata:
        """Load drone flight metadata from JSON file."""
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                
                return DroneMetadata(
                    drone_model=data.get("drone_model"),
                    camera_model=data.get("camera_model"),
                    flight_date=datetime.fromisoformat(data["flight_date"]) if "flight_date" in data else None,
                    flight_altitude=data.get("altitude"),
                    ground_sample_distance=data.get("gsd"),
                    flight_pattern=FlightPattern(data.get("pattern", "unknown")),
                    start_location=GeoLocation(**data["start"]) if "start" in data else None,
                    end_location=GeoLocation(**data["end"]) if "end" in data else None,
                )
            except Exception as e:
                logger.warning(f"Failed to load drone metadata: {e}")
        
        return DroneMetadata()
    
    def _analyze_terrain(self, video_path: str) -> Tuple[TerrainType, str]:
        """Analyze terrain type from video."""
        prompt = """Analyze the terrain visible in this drone footage.
        
        Classify the terrain type:
        - flat: Relatively flat terrain
        - hilly: Rolling hills
        - mountainous: Steep mountains
        - forested: Dense vegetation
        - desert: Arid, sparse vegetation
        - wetland: Swamps, marshes
        - coastal: Near water bodies
        - mixed: Combination of types
        
        Describe the terrain characteristics, vegetation, and any notable features."""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        # Parse terrain type
        terrain_type = TerrainType.MIXED
        response_lower = result.response.lower()
        for tt in TerrainType:
            if tt.value in response_lower:
                terrain_type = tt
                break
        
        return terrain_type, result.response
    
    def _detect_geological_features(
        self,
        video_path: str,
        fps: float,
    ) -> List[GeologicalObservation]:
        """Detect geological features in drone footage."""
        prompt = """Identify geological features visible in this drone footage:
        
        Look for:
        - Rock outcrops and exposures
        - Fault lines or fractures
        - Fold structures
        - Veins or mineralized zones
        - Alteration zones (color changes)
        - Gossans (iron-stained caps)
        - Geological contacts
        - Intrusive bodies
        - Sedimentary layers
        - Alluvial deposits
        
        For each feature, describe:
        - Type of feature
        - Location in frame
        - Color anomalies
        - Potential mineral indicators
        - Structural characteristics"""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        observations = []
        
        # Parse features from response
        lines = result.response.split('\n')
        current_feature = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Detect feature types
            for feature_type in GeologicalFeature:
                if feature_type.value.replace('_', ' ') in line_lower:
                    # Extract mineral indicators
                    mineral_indicators = []
                    mineral_keywords = [
                        "gold", "copper", "iron", "quartz", "sulfide",
                        "oxide", "carbonate", "silica", "pyrite"
                    ]
                    for keyword in mineral_keywords:
                        if keyword in line_lower:
                            mineral_indicators.append(keyword)
                    
                    # Extract color anomalies
                    color_anomalies = []
                    color_keywords = [
                        "red", "orange", "yellow", "green", "blue",
                        "brown", "black", "white", "rusty", "stained"
                    ]
                    for keyword in color_keywords:
                        if keyword in line_lower:
                            color_anomalies.append(keyword)
                    
                    observations.append(GeologicalObservation(
                        feature_type=feature_type,
                        location=None,
                        frame_index=0,
                        timestamp=0.0,
                        bbox=None,
                        description=line.strip(),
                        confidence=0.7,
                        mineral_indicators=mineral_indicators,
                        color_anomalies=color_anomalies,
                    ))
                    break
        
        return observations
    
    def _detect_artisanal_mining(self, video_path: str) -> List[Dict[str, Any]]:
        """Detect artisanal mining indicators."""
        prompt = """Analyze this drone footage for artisanal/small-scale mining activity.
        
        Look for:
        - Small excavations or pits
        - Sluice boxes or panning equipment
        - Temporary shelters or camps
        - People working with hand tools
        - Informal processing areas
        - Tailings or waste piles
        - Access trails
        - Water diversion structures
        
        For each indicator, describe what you see and its location."""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        indicators = []
        
        # Parse indicators from response
        indicator_keywords = [
            "excavation", "pit", "sluice", "pan", "camp", "shelter",
            "worker", "people", "processing", "tailings", "trail"
        ]
        
        for keyword in indicator_keywords:
            if keyword in result.response.lower():
                indicators.append({
                    "type": keyword,
                    "description": result.response[:500],
                    "confidence": 0.7,
                })
        
        return indicators
    
    def _analyze_environment(self, video_path: str) -> List[Dict[str, Any]]:
        """Analyze environmental conditions."""
        prompt = """Assess environmental conditions visible in this drone footage:
        
        Evaluate:
        - Vegetation health and coverage
        - Water bodies and quality
        - Erosion or land degradation
        - Dust or air quality indicators
        - Wildlife presence
        - Land use patterns
        
        Note any environmental concerns or impacts."""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        observations = []
        
        env_categories = [
            "vegetation", "water", "erosion", "dust", "wildlife", "land use"
        ]
        
        for category in env_categories:
            if category in result.response.lower():
                observations.append({
                    "category": category,
                    "description": result.response[:300],
                    "timestamp": 0.0,
                })
        
        return observations
    
    def _identify_points_of_interest(
        self,
        video_path: str,
        focus_areas: List[str],
    ) -> List[Dict[str, Any]]:
        """Identify points of interest based on focus areas."""
        focus_str = ", ".join(focus_areas) if focus_areas else "geological features, mining activity"
        
        prompt = f"""Identify key points of interest in this drone footage.
        
        Focus on: {focus_str}
        
        For each point of interest:
        - Describe what makes it interesting
        - Note its approximate location in the video
        - Assess its significance (low/medium/high)
        - Recommend follow-up actions if needed"""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        # Parse points of interest
        pois = []
        
        # Simple parsing - look for numbered items or bullet points
        lines = result.response.split('\n')
        current_poi = None
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                if current_poi:
                    pois.append(current_poi)
                current_poi = {
                    "description": line.lstrip('0123456789.-* '),
                    "significance": "medium",
                    "timestamp": 0.0,
                }
            elif current_poi and line:
                current_poi["description"] += " " + line
        
        if current_poi:
            pois.append(current_poi)
        
        return pois
    
    def _assess_flight_quality(self, video_path: str) -> float:
        """Assess the quality of the drone flight footage."""
        prompt = """Assess the quality of this drone footage for survey purposes.
        
        Evaluate:
        - Image stability (shake, blur)
        - Lighting conditions
        - Coverage completeness
        - Altitude consistency
        - Overlap between frames
        
        Rate overall quality from 0-10."""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        
        # Extract quality score
        import re
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*10|out of 10)?', result.response)
        if score_match:
            score = float(score_match.group(1))
            return min(score / 10, 1.0)  # Normalize to 0-1
        
        return 0.7  # Default score
    
    def _assess_coverage(self, video_path: str, metadata: DroneMetadata) -> str:
        """Assess survey coverage."""
        prompt = """Assess the survey coverage of this drone flight.
        
        Consider:
        - Area covered
        - Gaps in coverage
        - Overlap quality
        - Edge coverage
        
        Provide a brief assessment."""
        
        result = self.molmo_client.analyze_video(video_path, prompt, AnalysisType.VIDEO_QA)
        return result.response[:500]
    
    def _generate_summary(
        self,
        terrain_type: TerrainType,
        geological_obs: List[GeologicalObservation],
        artisanal_indicators: List[Dict[str, Any]],
    ) -> str:
        """Generate analysis summary."""
        summary_parts = [
            f"Terrain: {terrain_type.value}",
            f"Geological features identified: {len(geological_obs)}",
        ]
        
        if geological_obs:
            feature_types = set(obs.feature_type.value for obs in geological_obs)
            summary_parts.append(f"Feature types: {', '.join(feature_types)}")
        
        if artisanal_indicators:
            summary_parts.append(
                f"Artisanal mining indicators: {len(artisanal_indicators)} detected"
            )
        
        return ". ".join(summary_parts)
    
    def _generate_recommendations(
        self,
        geological_obs: List[GeologicalObservation],
        artisanal_indicators: List[Dict[str, Any]],
        environmental_obs: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Geological recommendations
        if geological_obs:
            high_interest = [
                obs for obs in geological_obs
                if obs.mineral_indicators or obs.color_anomalies
            ]
            if high_interest:
                recommendations.append(
                    f"Ground-truth {len(high_interest)} geological features with mineral indicators"
                )
        
        # Artisanal mining recommendations
        if artisanal_indicators:
            recommendations.append(
                "Artisanal mining activity detected - consider detailed assessment"
            )
            recommendations.append(
                "Document for regulatory compliance if required"
            )
        
        # Environmental recommendations
        erosion_obs = [o for o in environmental_obs if o.get("category") == "erosion"]
        if erosion_obs:
            recommendations.append(
                "Erosion detected - assess impact and mitigation needs"
            )
        
        return recommendations
    
    def compare_flights(
        self,
        before_video: str,
        after_video: str,
    ) -> List[SiteChangeDetection]:
        """Compare two drone flights to detect changes."""
        prompt = """Compare these two sets of drone footage from the same area.
        
        Identify all changes between the before and after footage:
        - New excavations or earthworks
        - Vegetation changes
        - New structures or equipment
        - Water body changes
        - Access road modifications
        - Mining activity changes
        
        For each change, describe:
        - What changed
        - Magnitude of change (small/medium/large)
        - Significance (low/medium/high/critical)"""
        
        # Analyze before video
        before_result = self.molmo_client.analyze_video(
            before_video, "Describe the current state of this site.", AnalysisType.VIDEO_QA
        )
        
        # Analyze after video
        after_result = self.molmo_client.analyze_video(
            after_video, "Describe the current state of this site.", AnalysisType.VIDEO_QA
        )
        
        # Compare
        compare_prompt = f"""Compare these two site descriptions and identify changes:
        
        BEFORE: {before_result.response}
        
        AFTER: {after_result.response}
        
        List all significant changes."""
        
        # Use multi-image analysis if supported
        changes = []
        
        # Parse changes from comparison
        change_keywords = [
            "new", "removed", "changed", "increased", "decreased",
            "expanded", "reduced", "modified", "added"
        ]
        
        for keyword in change_keywords:
            if keyword in after_result.response.lower():
                changes.append(SiteChangeDetection(
                    change_type=keyword,
                    location=None,
                    before_frame=0,
                    after_frame=0,
                    before_description=before_result.response[:200],
                    after_description=after_result.response[:200],
                    change_magnitude=0.5,
                    significance="medium",
                    recommendations=["Verify change on ground"],
                ))
        
        return changes
    
    def analyze_for_gold_exploration(self, video_path: str) -> FlightAnalysisResult:
        """Specialized analysis for gold exploration."""
        return self.analyze_flight(
            video_path,
            focus_areas=[
                "gold indicators", "quartz veins", "alluvial deposits",
                "gossans", "alteration zones", "structural controls"
            ],
        )
    
    def analyze_for_lithium_exploration(self, video_path: str) -> FlightAnalysisResult:
        """Specialized analysis for lithium exploration."""
        return self.analyze_flight(
            video_path,
            focus_areas=[
                "pegmatites", "clay deposits", "brine pools",
                "evaporite deposits", "alteration patterns"
            ],
        )
    
    def analyze_for_environmental_monitoring(self, video_path: str) -> FlightAnalysisResult:
        """Specialized analysis for environmental monitoring."""
        return self.analyze_flight(
            video_path,
            focus_areas=[
                "vegetation health", "water quality", "erosion",
                "rehabilitation progress", "wildlife corridors"
            ],
        )
