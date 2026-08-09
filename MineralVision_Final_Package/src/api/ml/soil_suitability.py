"""
Agricultural Soil Suitability Module for MineralVision Platform.

Comprehensive soil assessment for agricultural planning:
- Oil Palm (Elaeis guineensis)
- Cocoa (Theobroma cacao)
- Ginger (Zingiber officinale)

Includes soil sample analysis, nutrient assessment, suitability scoring,
and remediation recommendations based on FAO land evaluation framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime


class CropType(Enum):
    """Supported crop types for suitability analysis."""
    OIL_PALM = "oil_palm"
    COCOA = "cocoa"
    GINGER = "ginger"
    RUBBER = "rubber"
    COFFEE_ARABICA = "coffee_arabica"
    COFFEE_ROBUSTA = "coffee_robusta"
    CASSAVA = "cassava"
    MAIZE = "maize"
    RICE_PADDY = "rice_paddy"
    RICE_UPLAND = "rice_upland"


class SuitabilityClass(Enum):
    """FAO land suitability classification with severity ranking."""
    S1 = "highly_suitable"  # No significant limitations
    S2 = "moderately_suitable"  # Moderate limitations
    S3 = "marginally_suitable"  # Severe limitations
    N1 = "currently_not_suitable"  # Limitations can be corrected
    N2 = "permanently_not_suitable"  # Cannot be corrected
    
    @property
    def severity(self) -> int:
        """Return severity ranking (0=best, 4=worst) for proper comparison."""
        severity_map = {
            "S1": 0,  # Best - highly suitable
            "S2": 1,  # Moderate limitations
            "S3": 2,  # Severe limitations
            "N1": 3,  # Currently not suitable
            "N2": 4   # Worst - permanently not suitable
        }
        return severity_map.get(self.name, 2)
    
    def __lt__(self, other: "SuitabilityClass") -> bool:
        """Compare suitability classes by severity (lower severity = better)."""
        if isinstance(other, SuitabilityClass):
            return self.severity < other.severity
        return NotImplemented
    
    def __gt__(self, other: "SuitabilityClass") -> bool:
        """Compare suitability classes by severity."""
        if isinstance(other, SuitabilityClass):
            return self.severity > other.severity
        return NotImplemented


class SoilTextureClass(Enum):
    """USDA soil texture classification."""
    SAND = "sand"
    LOAMY_SAND = "loamy_sand"
    SANDY_LOAM = "sandy_loam"
    LOAM = "loam"
    SILT_LOAM = "silt_loam"
    SILT = "silt"
    SANDY_CLAY_LOAM = "sandy_clay_loam"
    CLAY_LOAM = "clay_loam"
    SILTY_CLAY_LOAM = "silty_clay_loam"
    SANDY_CLAY = "sandy_clay"
    SILTY_CLAY = "silty_clay"
    CLAY = "clay"


class DrainageClass(Enum):
    """Soil drainage classification."""
    EXCESSIVELY_DRAINED = "excessively_drained"
    WELL_DRAINED = "well_drained"
    MODERATELY_WELL_DRAINED = "moderately_well_drained"
    SOMEWHAT_POORLY_DRAINED = "somewhat_poorly_drained"
    POORLY_DRAINED = "poorly_drained"
    VERY_POORLY_DRAINED = "very_poorly_drained"


@dataclass
class SoilSample:
    """Comprehensive soil sample data for agricultural assessment."""
    sample_id: str
    x: float  # Easting/longitude
    y: float  # Northing/latitude
    sampling_date: datetime
    sampling_depth_cm: Tuple[float, float]  # (top, bottom)
    
    # Physical properties
    texture_class: Optional[SoilTextureClass] = None
    sand_percent: float = 0.0
    silt_percent: float = 0.0
    clay_percent: float = 0.0
    bulk_density: float = 1.3  # g/cm3
    porosity: float = 0.5  # fraction
    water_holding_capacity: float = 0.0  # mm/m
    drainage_class: Optional[DrainageClass] = None
    
    # Chemical properties
    ph_water: float = 7.0  # pH in water (1:2.5)
    ph_kcl: Optional[float] = None  # pH in KCl
    ec: float = 0.0  # Electrical conductivity (dS/m)
    cec: float = 0.0  # Cation exchange capacity (cmol/kg)
    base_saturation: float = 0.0  # Percent
    organic_matter: float = 0.0  # Percent
    organic_carbon: float = 0.0  # Percent
    
    # Macronutrients (mg/kg or ppm unless noted)
    nitrogen_total: float = 0.0  # Total N (%)
    nitrogen_available: float = 0.0  # Available N (ppm)
    phosphorus_available: float = 0.0  # Available P (Bray-1 or Olsen)
    phosphorus_method: str = "bray1"  # bray1, olsen, mehlich3
    potassium_exchangeable: float = 0.0  # Exchangeable K (cmol/kg)
    potassium_available: float = 0.0  # Available K (ppm)
    
    # Secondary nutrients (cmol/kg for exchangeable, ppm for available)
    calcium_exchangeable: float = 0.0
    magnesium_exchangeable: float = 0.0
    sulfur_available: float = 0.0
    
    # Micronutrients (ppm)
    iron: float = 0.0
    manganese: float = 0.0
    zinc: float = 0.0
    copper: float = 0.0
    boron: float = 0.0
    molybdenum: float = 0.0
    
    # Problematic elements
    aluminum_exchangeable: float = 0.0  # cmol/kg
    aluminum_saturation: float = 0.0  # Percent
    sodium_exchangeable: float = 0.0  # cmol/kg
    esp: float = 0.0  # Exchangeable sodium percentage
    
    # Quality flags
    quality_flags: Dict[str, str] = field(default_factory=dict)
    lab_id: str = ""
    analysis_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClimateData:
    """Climate data for crop suitability assessment."""
    location_id: str
    x: float
    y: float
    
    # Temperature (Celsius)
    mean_annual_temp: float = 25.0
    mean_temp_warmest_month: float = 28.0
    mean_temp_coldest_month: float = 22.0
    absolute_min_temp: float = 15.0
    absolute_max_temp: float = 38.0
    
    # Rainfall (mm)
    annual_rainfall: float = 2000.0
    rainfall_wettest_month: float = 300.0
    rainfall_driest_month: float = 50.0
    dry_months: int = 2  # Months with <60mm rainfall
    
    # Other
    relative_humidity: float = 80.0  # Percent
    sunshine_hours: float = 2000.0  # Hours/year
    potential_evapotranspiration: float = 1500.0  # mm/year
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopographyData:
    """Topographic data for suitability assessment."""
    location_id: str
    x: float
    y: float
    
    elevation_m: float = 0.0
    slope_percent: float = 0.0
    aspect_degrees: float = 0.0  # 0-360
    curvature: float = 0.0
    twi: float = 0.0  # Topographic wetness index
    
    # Erosion risk factors
    slope_length_m: float = 100.0
    ls_factor: float = 1.0  # USLE LS factor
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class CropRequirements:
    """Crop-specific soil and climate requirements."""
    
    # Comprehensive requirements for each crop
    REQUIREMENTS: Dict[CropType, Dict[str, Any]] = {
        CropType.OIL_PALM: {
            "name": "Oil Palm (Elaeis guineensis)",
            "soil": {
                "ph": {"optimal": (4.0, 6.0), "suitable": (3.5, 7.0), "weight": 0.15},
                "texture": {
                    "optimal": [SoilTextureClass.CLAY_LOAM, SoilTextureClass.LOAM, SoilTextureClass.SANDY_CLAY_LOAM],
                    "suitable": [SoilTextureClass.SANDY_LOAM, SoilTextureClass.SILTY_CLAY_LOAM, SoilTextureClass.CLAY],
                    "weight": 0.10
                },
                "drainage": {
                    "optimal": [DrainageClass.WELL_DRAINED, DrainageClass.MODERATELY_WELL_DRAINED],
                    "suitable": [DrainageClass.SOMEWHAT_POORLY_DRAINED],
                    "weight": 0.15
                },
                "depth_cm": {"optimal": (100, 999), "suitable": (75, 100), "weight": 0.10},
                "organic_matter": {"optimal": (3.0, 10.0), "suitable": (1.5, 3.0), "weight": 0.05},
                "cec": {"optimal": (15, 50), "suitable": (8, 15), "weight": 0.05},
                "base_saturation": {"optimal": (50, 100), "suitable": (35, 50), "weight": 0.05},
                "n_available": {"optimal": (40, 100), "suitable": (20, 40), "weight": 0.05},
                "p_available": {"optimal": (15, 50), "suitable": (8, 15), "weight": 0.05},
                "k_available": {"optimal": (150, 400), "suitable": (80, 150), "weight": 0.05},
                "mg_exchangeable": {"optimal": (1.0, 5.0), "suitable": (0.5, 1.0), "weight": 0.03},
                "al_saturation": {"optimal": (0, 30), "suitable": (30, 60), "weight": 0.07}
            },
            "climate": {
                "annual_rainfall": {"optimal": (2000, 2500), "suitable": (1500, 3000), "weight": 0.15},
                "dry_months": {"optimal": (0, 2), "suitable": (2, 4), "weight": 0.10},
                "mean_temp": {"optimal": (24, 28), "suitable": (22, 32), "weight": 0.10},
                "min_temp": {"optimal": (18, 999), "suitable": (15, 18), "weight": 0.05},
                "sunshine_hours": {"optimal": (1800, 2500), "suitable": (1500, 1800), "weight": 0.05}
            },
            "topography": {
                "slope": {"optimal": (0, 12), "suitable": (12, 25), "weight": 0.10},
                "elevation": {"optimal": (0, 500), "suitable": (500, 800), "weight": 0.05}
            }
        },
        CropType.COCOA: {
            "name": "Cocoa (Theobroma cacao)",
            "soil": {
                "ph": {"optimal": (6.0, 7.5), "suitable": (5.0, 8.0), "weight": 0.15},
                "texture": {
                    "optimal": [SoilTextureClass.LOAM, SoilTextureClass.CLAY_LOAM, SoilTextureClass.SILT_LOAM],
                    "suitable": [SoilTextureClass.SANDY_LOAM, SoilTextureClass.SILTY_CLAY_LOAM],
                    "weight": 0.10
                },
                "drainage": {
                    "optimal": [DrainageClass.WELL_DRAINED],
                    "suitable": [DrainageClass.MODERATELY_WELL_DRAINED],
                    "weight": 0.15
                },
                "depth_cm": {"optimal": (150, 999), "suitable": (100, 150), "weight": 0.10},
                "organic_matter": {"optimal": (3.5, 10.0), "suitable": (2.0, 3.5), "weight": 0.08},
                "cec": {"optimal": (12, 50), "suitable": (8, 12), "weight": 0.05},
                "base_saturation": {"optimal": (60, 100), "suitable": (40, 60), "weight": 0.05},
                "n_available": {"optimal": (30, 80), "suitable": (15, 30), "weight": 0.05},
                "p_available": {"optimal": (12, 40), "suitable": (6, 12), "weight": 0.05},
                "k_available": {"optimal": (120, 350), "suitable": (60, 120), "weight": 0.05},
                "ca_exchangeable": {"optimal": (8, 20), "suitable": (4, 8), "weight": 0.04},
                "mg_exchangeable": {"optimal": (2.0, 8.0), "suitable": (1.0, 2.0), "weight": 0.03},
                "al_saturation": {"optimal": (0, 20), "suitable": (20, 40), "weight": 0.05}
            },
            "climate": {
                "annual_rainfall": {"optimal": (1500, 2000), "suitable": (1250, 2500), "weight": 0.15},
                "dry_months": {"optimal": (0, 3), "suitable": (3, 4), "weight": 0.10},
                "mean_temp": {"optimal": (21, 27), "suitable": (18, 32), "weight": 0.10},
                "min_temp": {"optimal": (15, 999), "suitable": (10, 15), "weight": 0.08},
                "humidity": {"optimal": (70, 90), "suitable": (60, 95), "weight": 0.05}
            },
            "topography": {
                "slope": {"optimal": (0, 8), "suitable": (8, 16), "weight": 0.08},
                "elevation": {"optimal": (0, 800), "suitable": (800, 1200), "weight": 0.05}
            }
        },
        CropType.GINGER: {
            "name": "Ginger (Zingiber officinale)",
            "soil": {
                "ph": {"optimal": (5.5, 6.5), "suitable": (5.0, 7.0), "weight": 0.15},
                "texture": {
                    "optimal": [SoilTextureClass.SANDY_LOAM, SoilTextureClass.LOAM, SoilTextureClass.SILT_LOAM],
                    "suitable": [SoilTextureClass.LOAMY_SAND, SoilTextureClass.CLAY_LOAM],
                    "weight": 0.12
                },
                "drainage": {
                    "optimal": [DrainageClass.WELL_DRAINED],
                    "suitable": [DrainageClass.MODERATELY_WELL_DRAINED, DrainageClass.EXCESSIVELY_DRAINED],
                    "weight": 0.15
                },
                "depth_cm": {"optimal": (45, 999), "suitable": (30, 45), "weight": 0.08},
                "organic_matter": {"optimal": (3.0, 8.0), "suitable": (2.0, 3.0), "weight": 0.10},
                "cec": {"optimal": (10, 30), "suitable": (5, 10), "weight": 0.05},
                "n_available": {"optimal": (50, 120), "suitable": (30, 50), "weight": 0.08},
                "p_available": {"optimal": (20, 60), "suitable": (10, 20), "weight": 0.08},
                "k_available": {"optimal": (200, 500), "suitable": (100, 200), "weight": 0.08},
                "zn": {"optimal": (2.0, 10.0), "suitable": (1.0, 2.0), "weight": 0.03},
                "b": {"optimal": (0.5, 2.0), "suitable": (0.2, 0.5), "weight": 0.03}
            },
            "climate": {
                "annual_rainfall": {"optimal": (1500, 3000), "suitable": (1200, 3500), "weight": 0.12},
                "dry_months": {"optimal": (0, 2), "suitable": (2, 4), "weight": 0.08},
                "mean_temp": {"optimal": (20, 30), "suitable": (15, 35), "weight": 0.10},
                "min_temp": {"optimal": (15, 999), "suitable": (10, 15), "weight": 0.05},
                "humidity": {"optimal": (70, 90), "suitable": (60, 95), "weight": 0.05}
            },
            "topography": {
                "slope": {"optimal": (0, 15), "suitable": (15, 30), "weight": 0.08},
                "elevation": {"optimal": (300, 1500), "suitable": (0, 300), "weight": 0.05}
            }
        }
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.requirements = self.REQUIREMENTS.get(crop_type, {})
    
    def get_soil_requirements(self) -> Dict[str, Any]:
        """Get soil requirements for the crop."""
        return self.requirements.get("soil", {})
    
    def get_climate_requirements(self) -> Dict[str, Any]:
        """Get climate requirements for the crop."""
        return self.requirements.get("climate", {})
    
    def get_topography_requirements(self) -> Dict[str, Any]:
        """Get topography requirements for the crop."""
        return self.requirements.get("topography", {})


class SoilSuitabilityScorer:
    """Score soil suitability for specific crops."""
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.requirements = CropRequirements(crop_type)
    
    def score_parameter(
        self,
        value: float,
        optimal_range: Tuple[float, float],
        suitable_range: Tuple[float, float],
        inverse: bool = False
    ) -> Tuple[float, SuitabilityClass]:
        """
        Score a single parameter against requirements.
        
        Returns (score 0-100, suitability class)
        """
        opt_min, opt_max = optimal_range
        suit_min, suit_max = suitable_range
        
        if inverse:
            # For parameters where lower is better (e.g., Al saturation)
            if value <= opt_max:
                score = 100
                suitability = SuitabilityClass.S1
            elif value <= suit_max:
                score = 70 - 30 * (value - opt_max) / (suit_max - opt_max)
                suitability = SuitabilityClass.S2
            else:
                score = max(0, 40 - 40 * (value - suit_max) / suit_max)
                suitability = SuitabilityClass.S3 if score > 20 else SuitabilityClass.N1
        else:
            if opt_min <= value <= opt_max:
                score = 100
                suitability = SuitabilityClass.S1
            elif suit_min <= value <= suit_max:
                if value < opt_min:
                    score = 70 + 30 * (value - suit_min) / (opt_min - suit_min)
                else:
                    score = 70 + 30 * (suit_max - value) / (suit_max - opt_max)
                suitability = SuitabilityClass.S2
            elif value < suit_min:
                score = max(0, 40 * value / suit_min)
                suitability = SuitabilityClass.S3 if score > 20 else SuitabilityClass.N1
            else:
                score = max(0, 40 * (1 - (value - suit_max) / suit_max))
                suitability = SuitabilityClass.S3 if score > 20 else SuitabilityClass.N1
        
        return score, suitability
    
    def score_texture(
        self,
        texture: SoilTextureClass,
        optimal: List[SoilTextureClass],
        suitable: List[SoilTextureClass]
    ) -> Tuple[float, SuitabilityClass]:
        """Score soil texture."""
        if texture in optimal:
            return 100, SuitabilityClass.S1
        elif texture in suitable:
            return 70, SuitabilityClass.S2
        else:
            return 40, SuitabilityClass.S3
    
    def score_drainage(
        self,
        drainage: DrainageClass,
        optimal: List[DrainageClass],
        suitable: List[DrainageClass]
    ) -> Tuple[float, SuitabilityClass]:
        """Score drainage class."""
        if drainage in optimal:
            return 100, SuitabilityClass.S1
        elif drainage in suitable:
            return 70, SuitabilityClass.S2
        else:
            return 30, SuitabilityClass.N1
    
    def score_soil_sample(
        self,
        sample: SoilSample,
        effective_depth_cm: float = 100
    ) -> Dict[str, Any]:
        """
        Score a soil sample for crop suitability.
        
        Returns detailed scoring breakdown.
        """
        soil_req = self.requirements.get_soil_requirements()
        scores = {}
        weights = {}
        suitabilities = {}
        
        # pH
        if "ph" in soil_req:
            req = soil_req["ph"]
            score, suit = self.score_parameter(
                sample.ph_water,
                req["optimal"],
                req["suitable"]
            )
            scores["ph"] = score
            weights["ph"] = req["weight"]
            suitabilities["ph"] = suit
        
        # Texture
        if "texture" in soil_req and sample.texture_class:
            req = soil_req["texture"]
            score, suit = self.score_texture(
                sample.texture_class,
                req["optimal"],
                req["suitable"]
            )
            scores["texture"] = score
            weights["texture"] = req["weight"]
            suitabilities["texture"] = suit
        
        # Drainage
        if "drainage" in soil_req and sample.drainage_class:
            req = soil_req["drainage"]
            score, suit = self.score_drainage(
                sample.drainage_class,
                req["optimal"],
                req["suitable"]
            )
            scores["drainage"] = score
            weights["drainage"] = req["weight"]
            suitabilities["drainage"] = suit
        
        # Effective depth
        if "depth_cm" in soil_req:
            req = soil_req["depth_cm"]
            score, suit = self.score_parameter(
                effective_depth_cm,
                req["optimal"],
                req["suitable"]
            )
            scores["depth"] = score
            weights["depth"] = req["weight"]
            suitabilities["depth"] = suit
        
        # Organic matter
        if "organic_matter" in soil_req:
            req = soil_req["organic_matter"]
            score, suit = self.score_parameter(
                sample.organic_matter,
                req["optimal"],
                req["suitable"]
            )
            scores["organic_matter"] = score
            weights["organic_matter"] = req["weight"]
            suitabilities["organic_matter"] = suit
        
        # CEC
        if "cec" in soil_req:
            req = soil_req["cec"]
            score, suit = self.score_parameter(
                sample.cec,
                req["optimal"],
                req["suitable"]
            )
            scores["cec"] = score
            weights["cec"] = req["weight"]
            suitabilities["cec"] = suit
        
        # Base saturation
        if "base_saturation" in soil_req:
            req = soil_req["base_saturation"]
            score, suit = self.score_parameter(
                sample.base_saturation,
                req["optimal"],
                req["suitable"]
            )
            scores["base_saturation"] = score
            weights["base_saturation"] = req["weight"]
            suitabilities["base_saturation"] = suit
        
        # Nitrogen
        if "n_available" in soil_req:
            req = soil_req["n_available"]
            score, suit = self.score_parameter(
                sample.nitrogen_available,
                req["optimal"],
                req["suitable"]
            )
            scores["nitrogen"] = score
            weights["nitrogen"] = req["weight"]
            suitabilities["nitrogen"] = suit
        
        # Phosphorus
        if "p_available" in soil_req:
            req = soil_req["p_available"]
            score, suit = self.score_parameter(
                sample.phosphorus_available,
                req["optimal"],
                req["suitable"]
            )
            scores["phosphorus"] = score
            weights["phosphorus"] = req["weight"]
            suitabilities["phosphorus"] = suit
        
        # Potassium
        if "k_available" in soil_req:
            req = soil_req["k_available"]
            score, suit = self.score_parameter(
                sample.potassium_available,
                req["optimal"],
                req["suitable"]
            )
            scores["potassium"] = score
            weights["potassium"] = req["weight"]
            suitabilities["potassium"] = suit
        
        # Calcium
        if "ca_exchangeable" in soil_req:
            req = soil_req["ca_exchangeable"]
            score, suit = self.score_parameter(
                sample.calcium_exchangeable,
                req["optimal"],
                req["suitable"]
            )
            scores["calcium"] = score
            weights["calcium"] = req["weight"]
            suitabilities["calcium"] = suit
        
        # Magnesium
        if "mg_exchangeable" in soil_req:
            req = soil_req["mg_exchangeable"]
            score, suit = self.score_parameter(
                sample.magnesium_exchangeable,
                req["optimal"],
                req["suitable"]
            )
            scores["magnesium"] = score
            weights["magnesium"] = req["weight"]
            suitabilities["magnesium"] = suit
        
        # Aluminum saturation (inverse - lower is better)
        if "al_saturation" in soil_req:
            req = soil_req["al_saturation"]
            score, suit = self.score_parameter(
                sample.aluminum_saturation,
                req["optimal"],
                req["suitable"],
                inverse=True
            )
            scores["al_saturation"] = score
            weights["al_saturation"] = req["weight"]
            suitabilities["al_saturation"] = suit
        
        # Micronutrients
        for nutrient in ["zn", "b", "fe", "mn", "cu"]:
            if nutrient in soil_req:
                req = soil_req[nutrient]
                value = getattr(sample, {"zn": "zinc", "b": "boron", "fe": "iron", 
                                         "mn": "manganese", "cu": "copper"}.get(nutrient, nutrient), 0)
                score, suit = self.score_parameter(value, req["optimal"], req["suitable"])
                scores[nutrient] = score
                weights[nutrient] = req["weight"]
                suitabilities[nutrient] = suit
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight
        else:
            weighted_score = 0
        
        # Determine overall suitability (limited by worst factor)
        worst_suitability = SuitabilityClass.S1
        limiting_factors = []
        
        for param, suit in suitabilities.items():
            if suit.severity > worst_suitability.severity:
                worst_suitability = suit
                limiting_factors = [param]
            elif suit == worst_suitability and suit != SuitabilityClass.S1:
                limiting_factors.append(param)
        
        return {
            "overall_score": weighted_score,
            "overall_suitability": worst_suitability,
            "parameter_scores": scores,
            "parameter_suitabilities": suitabilities,
            "limiting_factors": limiting_factors,
            "weights": weights
        }
    
    def score_climate(self, climate: ClimateData) -> Dict[str, Any]:
        """Score climate suitability for the crop."""
        climate_req = self.requirements.get_climate_requirements()
        scores = {}
        weights = {}
        suitabilities = {}
        
        # Annual rainfall
        if "annual_rainfall" in climate_req:
            req = climate_req["annual_rainfall"]
            score, suit = self.score_parameter(
                climate.annual_rainfall,
                req["optimal"],
                req["suitable"]
            )
            scores["rainfall"] = score
            weights["rainfall"] = req["weight"]
            suitabilities["rainfall"] = suit
        
        # Dry months (inverse - fewer is better)
        if "dry_months" in climate_req:
            req = climate_req["dry_months"]
            score, suit = self.score_parameter(
                climate.dry_months,
                req["optimal"],
                req["suitable"],
                inverse=True
            )
            scores["dry_months"] = score
            weights["dry_months"] = req["weight"]
            suitabilities["dry_months"] = suit
        
        # Mean temperature
        if "mean_temp" in climate_req:
            req = climate_req["mean_temp"]
            score, suit = self.score_parameter(
                climate.mean_annual_temp,
                req["optimal"],
                req["suitable"]
            )
            scores["temperature"] = score
            weights["temperature"] = req["weight"]
            suitabilities["temperature"] = suit
        
        # Minimum temperature
        if "min_temp" in climate_req:
            req = climate_req["min_temp"]
            score, suit = self.score_parameter(
                climate.absolute_min_temp,
                req["optimal"],
                req["suitable"]
            )
            scores["min_temp"] = score
            weights["min_temp"] = req["weight"]
            suitabilities["min_temp"] = suit
        
        # Humidity
        if "humidity" in climate_req:
            req = climate_req["humidity"]
            score, suit = self.score_parameter(
                climate.relative_humidity,
                req["optimal"],
                req["suitable"]
            )
            scores["humidity"] = score
            weights["humidity"] = req["weight"]
            suitabilities["humidity"] = suit
        
        # Sunshine hours
        if "sunshine_hours" in climate_req:
            req = climate_req["sunshine_hours"]
            score, suit = self.score_parameter(
                climate.sunshine_hours,
                req["optimal"],
                req["suitable"]
            )
            scores["sunshine"] = score
            weights["sunshine"] = req["weight"]
            suitabilities["sunshine"] = suit
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight
        else:
            weighted_score = 0
        
        # Determine overall suitability
        worst_suitability = SuitabilityClass.S1
        limiting_factors = []
        
        for param, suit in suitabilities.items():
            if suit.severity > worst_suitability.severity:
                worst_suitability = suit
                limiting_factors = [param]
            elif suit == worst_suitability and suit != SuitabilityClass.S1:
                limiting_factors.append(param)
        
        return {
            "overall_score": weighted_score,
            "overall_suitability": worst_suitability,
            "parameter_scores": scores,
            "parameter_suitabilities": suitabilities,
            "limiting_factors": limiting_factors
        }
    
    def score_topography(self, topo: TopographyData) -> Dict[str, Any]:
        """Score topography suitability for the crop."""
        topo_req = self.requirements.get_topography_requirements()
        scores = {}
        weights = {}
        suitabilities = {}
        
        # Slope
        if "slope" in topo_req:
            req = topo_req["slope"]
            score, suit = self.score_parameter(
                topo.slope_percent,
                req["optimal"],
                req["suitable"],
                inverse=True
            )
            scores["slope"] = score
            weights["slope"] = req["weight"]
            suitabilities["slope"] = suit
        
        # Elevation
        if "elevation" in topo_req:
            req = topo_req["elevation"]
            score, suit = self.score_parameter(
                topo.elevation_m,
                req["optimal"],
                req["suitable"]
            )
            scores["elevation"] = score
            weights["elevation"] = req["weight"]
            suitabilities["elevation"] = suit
        
        # Calculate weighted average
        total_weight = sum(weights.values())
        if total_weight > 0:
            weighted_score = sum(scores[k] * weights[k] for k in scores) / total_weight
        else:
            weighted_score = 0
        
        # Determine overall suitability
        worst_suitability = SuitabilityClass.S1
        limiting_factors = []
        
        for param, suit in suitabilities.items():
            if suit.severity > worst_suitability.severity:
                worst_suitability = suit
                limiting_factors = [param]
            elif suit == worst_suitability and suit != SuitabilityClass.S1:
                limiting_factors.append(param)
        
        return {
            "overall_score": weighted_score,
            "overall_suitability": worst_suitability,
            "parameter_scores": scores,
            "parameter_suitabilities": suitabilities,
            "limiting_factors": limiting_factors
        }


class RemediationRecommender:
    """Generate soil remediation recommendations."""
    
    # Lime requirement factors (kg CaCO3/ha per unit pH change per cmol Al)
    LIME_FACTORS = {
        SoilTextureClass.SAND: 500,
        SoilTextureClass.LOAMY_SAND: 750,
        SoilTextureClass.SANDY_LOAM: 1000,
        SoilTextureClass.LOAM: 1500,
        SoilTextureClass.SILT_LOAM: 1750,
        SoilTextureClass.CLAY_LOAM: 2000,
        SoilTextureClass.CLAY: 2500
    }
    
    # Fertilizer recommendations (kg/ha for deficiency correction)
    FERTILIZER_RATES = {
        "nitrogen": {
            "urea": {"n_content": 0.46, "rate_per_kg_n": 2.17},
            "ammonium_sulfate": {"n_content": 0.21, "rate_per_kg_n": 4.76}
        },
        "phosphorus": {
            "tsp": {"p2o5_content": 0.46, "rate_per_kg_p2o5": 2.17},
            "rock_phosphate": {"p2o5_content": 0.30, "rate_per_kg_p2o5": 3.33}
        },
        "potassium": {
            "kcl": {"k2o_content": 0.60, "rate_per_kg_k2o": 1.67},
            "k2so4": {"k2o_content": 0.50, "rate_per_kg_k2o": 2.00}
        }
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.requirements = CropRequirements(crop_type)
    
    def calculate_lime_requirement(
        self,
        sample: SoilSample,
        target_ph: float = 6.0,
        depth_cm: float = 20
    ) -> Dict[str, Any]:
        """
        Calculate lime requirement to raise pH.
        
        Returns lime recommendation in kg/ha.
        """
        if sample.ph_water >= target_ph:
            return {
                "lime_required_kg_ha": 0,
                "recommendation": "No liming required",
                "target_ph": target_ph,
                "current_ph": sample.ph_water
            }
        
        ph_change = target_ph - sample.ph_water
        
        # Get lime factor based on texture
        lime_factor = self.LIME_FACTORS.get(
            sample.texture_class, 
            1500  # Default for loam
        )
        
        # Adjust for Al saturation
        al_factor = 1 + (sample.aluminum_saturation / 100)
        
        # Calculate lime requirement
        lime_kg_ha = lime_factor * ph_change * al_factor * (depth_cm / 20)
        
        # Determine lime type
        if sample.magnesium_exchangeable < 1.0:
            lime_type = "dolomitic_lime"
            recommendation = f"Apply {lime_kg_ha:.0f} kg/ha dolomitic lime (provides Mg)"
        else:
            lime_type = "calcitic_lime"
            recommendation = f"Apply {lime_kg_ha:.0f} kg/ha calcitic lime"
        
        return {
            "lime_required_kg_ha": lime_kg_ha,
            "lime_type": lime_type,
            "recommendation": recommendation,
            "target_ph": target_ph,
            "current_ph": sample.ph_water,
            "ph_change_needed": ph_change
        }
    
    def calculate_gypsum_requirement(
        self,
        sample: SoilSample,
        target_esp: float = 10
    ) -> Dict[str, Any]:
        """
        Calculate gypsum requirement for sodic soil reclamation.
        """
        if sample.esp <= target_esp:
            return {
                "gypsum_required_kg_ha": 0,
                "recommendation": "No gypsum required",
                "current_esp": sample.esp
            }
        
        # Gypsum requirement formula
        esp_reduction = sample.esp - target_esp
        gypsum_kg_ha = esp_reduction * sample.cec * 1.72 * 10  # Simplified formula
        
        return {
            "gypsum_required_kg_ha": gypsum_kg_ha,
            "recommendation": f"Apply {gypsum_kg_ha:.0f} kg/ha gypsum to reduce ESP",
            "current_esp": sample.esp,
            "target_esp": target_esp
        }
    
    def calculate_fertilizer_requirements(
        self,
        sample: SoilSample,
        yield_target: str = "medium"
    ) -> Dict[str, Any]:
        """
        Calculate fertilizer requirements based on soil test and crop needs.
        """
        soil_req = self.requirements.get_soil_requirements()
        recommendations = {}
        
        # Yield target multipliers
        yield_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
        multiplier = yield_multipliers.get(yield_target, 1.0)
        
        # Nitrogen
        if "n_available" in soil_req:
            optimal_n = soil_req["n_available"]["optimal"][0]
            if sample.nitrogen_available < optimal_n:
                n_deficit = optimal_n - sample.nitrogen_available
                n_kg_ha = n_deficit * 2 * multiplier  # Simplified conversion
                recommendations["nitrogen"] = {
                    "deficit_ppm": n_deficit,
                    "n_kg_ha": n_kg_ha,
                    "urea_kg_ha": n_kg_ha / 0.46,
                    "recommendation": f"Apply {n_kg_ha:.0f} kg N/ha ({n_kg_ha/0.46:.0f} kg urea/ha)"
                }
        
        # Phosphorus
        if "p_available" in soil_req:
            optimal_p = soil_req["p_available"]["optimal"][0]
            if sample.phosphorus_available < optimal_p:
                p_deficit = optimal_p - sample.phosphorus_available
                p2o5_kg_ha = p_deficit * 4 * multiplier  # Simplified conversion
                recommendations["phosphorus"] = {
                    "deficit_ppm": p_deficit,
                    "p2o5_kg_ha": p2o5_kg_ha,
                    "tsp_kg_ha": p2o5_kg_ha / 0.46,
                    "recommendation": f"Apply {p2o5_kg_ha:.0f} kg P2O5/ha ({p2o5_kg_ha/0.46:.0f} kg TSP/ha)"
                }
        
        # Potassium
        if "k_available" in soil_req:
            optimal_k = soil_req["k_available"]["optimal"][0]
            if sample.potassium_available < optimal_k:
                k_deficit = optimal_k - sample.potassium_available
                k2o_kg_ha = k_deficit * 1.2 * multiplier  # Simplified conversion
                recommendations["potassium"] = {
                    "deficit_ppm": k_deficit,
                    "k2o_kg_ha": k2o_kg_ha,
                    "kcl_kg_ha": k2o_kg_ha / 0.60,
                    "recommendation": f"Apply {k2o_kg_ha:.0f} kg K2O/ha ({k2o_kg_ha/0.60:.0f} kg KCl/ha)"
                }
        
        # Organic matter
        if sample.organic_matter < 2.0:
            recommendations["organic_matter"] = {
                "current_percent": sample.organic_matter,
                "recommendation": "Apply 5-10 tonnes/ha organic matter (compost, manure) annually"
            }
        
        # Micronutrients
        if sample.zinc < 1.0:
            recommendations["zinc"] = {
                "current_ppm": sample.zinc,
                "recommendation": "Apply 5-10 kg/ha zinc sulfate"
            }
        
        if sample.boron < 0.3:
            recommendations["boron"] = {
                "current_ppm": sample.boron,
                "recommendation": "Apply 1-2 kg/ha borax"
            }
        
        return recommendations
    
    def generate_management_plan(
        self,
        sample: SoilSample,
        climate: Optional[ClimateData] = None,
        topo: Optional[TopographyData] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive soil management plan.
        """
        plan = {
            "crop": self.crop_type.value,
            "sample_id": sample.sample_id,
            "amendments": {},
            "fertilizers": {},
            "management_practices": [],
            "priority_actions": []
        }
        
        # Lime requirement
        lime_rec = self.calculate_lime_requirement(sample)
        if lime_rec["lime_required_kg_ha"] > 0:
            plan["amendments"]["lime"] = lime_rec
            plan["priority_actions"].append(lime_rec["recommendation"])
        
        # Gypsum for sodic soils
        if sample.esp > 10:
            gypsum_rec = self.calculate_gypsum_requirement(sample)
            plan["amendments"]["gypsum"] = gypsum_rec
            plan["priority_actions"].append(gypsum_rec["recommendation"])
        
        # Fertilizer requirements
        fert_rec = self.calculate_fertilizer_requirements(sample)
        plan["fertilizers"] = fert_rec
        for nutrient, rec in fert_rec.items():
            if "recommendation" in rec:
                plan["priority_actions"].append(rec["recommendation"])
        
        # Drainage recommendations
        if sample.drainage_class in [DrainageClass.POORLY_DRAINED, DrainageClass.VERY_POORLY_DRAINED]:
            plan["management_practices"].append("Install drainage system or raised beds")
        
        # Erosion control
        if topo and topo.slope_percent > 15:
            plan["management_practices"].append("Implement contour planting and terracing")
            plan["management_practices"].append("Establish cover crops between rows")
        
        # Organic matter building
        if sample.organic_matter < 2.5:
            plan["management_practices"].append("Apply organic mulch around plants")
            plan["management_practices"].append("Incorporate crop residues")
        
        # Water management
        if climate:
            if climate.dry_months > 3:
                plan["management_practices"].append("Install irrigation system")
                plan["management_practices"].append("Apply mulch to conserve moisture")
            if climate.annual_rainfall > 3000:
                plan["management_practices"].append("Ensure adequate drainage")
        
        return plan


class SoilSuitabilityPipeline:
    """
    Complete soil suitability assessment pipeline.
    """
    
    def __init__(
        self,
        crop_types: List[CropType],
        project_name: str = "soil_assessment"
    ):
        self.crop_types = crop_types
        self.project_name = project_name
        
        # Initialize scorers and recommenders for each crop
        self.scorers = {crop: SoilSuitabilityScorer(crop) for crop in crop_types}
        self.recommenders = {crop: RemediationRecommender(crop) for crop in crop_types}
        
        # Data storage
        self.soil_samples: List[SoilSample] = []
        self.climate_data: List[ClimateData] = []
        self.topography_data: List[TopographyData] = []
        
        # Results
        self.suitability_results: Dict[str, Dict[str, Any]] = {}
    
    def add_soil_sample(self, sample: SoilSample) -> None:
        """Add a soil sample."""
        # Compute derived properties if not set
        if sample.organic_carbon > 0 and sample.organic_matter == 0:
            sample.organic_matter = sample.organic_carbon * 1.724
        
        # Compute texture class if not set
        if sample.texture_class is None and sample.sand_percent + sample.silt_percent + sample.clay_percent > 0:
            sample.texture_class = self._classify_texture(
                sample.sand_percent,
                sample.silt_percent,
                sample.clay_percent
            )
        
        self.soil_samples.append(sample)
    
    def add_climate_data(self, climate: ClimateData) -> None:
        """Add climate data."""
        self.climate_data.append(climate)
    
    def add_topography_data(self, topo: TopographyData) -> None:
        """Add topography data."""
        self.topography_data.append(topo)
    
    def _classify_texture(
        self,
        sand: float,
        silt: float,
        clay: float
    ) -> SoilTextureClass:
        """Classify soil texture using USDA triangle."""
        # Simplified texture classification
        if clay >= 40:
            if sand >= 45:
                return SoilTextureClass.SANDY_CLAY
            elif silt >= 40:
                return SoilTextureClass.SILTY_CLAY
            else:
                return SoilTextureClass.CLAY
        elif clay >= 27:
            if sand >= 20 and sand < 45:
                return SoilTextureClass.CLAY_LOAM
            elif silt >= 40:
                return SoilTextureClass.SILTY_CLAY_LOAM
            else:
                return SoilTextureClass.SANDY_CLAY_LOAM
        elif clay >= 7:
            if silt >= 50:
                if clay < 12:
                    return SoilTextureClass.SILT_LOAM
                else:
                    return SoilTextureClass.SILT_LOAM
            elif sand >= 52:
                return SoilTextureClass.SANDY_LOAM
            else:
                return SoilTextureClass.LOAM
        else:
            if sand >= 85:
                return SoilTextureClass.SAND
            elif sand >= 70:
                return SoilTextureClass.LOAMY_SAND
            else:
                return SoilTextureClass.SANDY_LOAM
    
    def assess_sample(
        self,
        sample: SoilSample,
        climate: Optional[ClimateData] = None,
        topo: Optional[TopographyData] = None
    ) -> Dict[str, Any]:
        """
        Assess a single sample for all crops.
        """
        results = {
            "sample_id": sample.sample_id,
            "location": {"x": sample.x, "y": sample.y},
            "crops": {}
        }
        
        for crop in self.crop_types:
            scorer = self.scorers[crop]
            
            # Score soil
            soil_score = scorer.score_soil_sample(sample)
            
            # Score climate if available
            climate_score = None
            if climate:
                climate_score = scorer.score_climate(climate)
            
            # Score topography if available
            topo_score = None
            if topo:
                topo_score = scorer.score_topography(topo)
            
            # Calculate combined score
            scores = [soil_score["overall_score"]]
            weights = [0.5]
            
            if climate_score:
                scores.append(climate_score["overall_score"])
                weights.append(0.3)
            
            if topo_score:
                scores.append(topo_score["overall_score"])
                weights.append(0.2)
            
            # Normalize weights
            total_weight = sum(weights)
            combined_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            
            # Determine overall suitability
            all_suitabilities = [soil_score["overall_suitability"]]
            if climate_score:
                all_suitabilities.append(climate_score["overall_suitability"])
            if topo_score:
                all_suitabilities.append(topo_score["overall_suitability"])
            
            # Overall suitability is limited by worst factor (highest severity = worst)
            overall_suitability = max(all_suitabilities, key=lambda x: x.severity)
            
            # Collect all limiting factors
            limiting_factors = soil_score["limiting_factors"].copy()
            if climate_score:
                limiting_factors.extend(climate_score["limiting_factors"])
            if topo_score:
                limiting_factors.extend(topo_score["limiting_factors"])
            
            results["crops"][crop.value] = {
                "combined_score": combined_score,
                "overall_suitability": overall_suitability.value,
                "suitability_class": overall_suitability.name,
                "soil_score": soil_score,
                "climate_score": climate_score,
                "topography_score": topo_score,
                "limiting_factors": limiting_factors
            }
        
        return results
    
    def assess_all_samples(self) -> Dict[str, Any]:
        """
        Assess all samples for all crops.
        """
        results = {
            "project_name": self.project_name,
            "sample_count": len(self.soil_samples),
            "crops_assessed": [c.value for c in self.crop_types],
            "samples": []
        }
        
        for sample in self.soil_samples:
            # Find matching climate and topo data
            climate = None
            for c in self.climate_data:
                if abs(c.x - sample.x) < 1000 and abs(c.y - sample.y) < 1000:
                    climate = c
                    break
            
            topo = None
            for t in self.topography_data:
                if abs(t.x - sample.x) < 100 and abs(t.y - sample.y) < 100:
                    topo = t
                    break
            
            sample_result = self.assess_sample(sample, climate, topo)
            results["samples"].append(sample_result)
            self.suitability_results[sample.sample_id] = sample_result
        
        # Summary statistics
        for crop in self.crop_types:
            crop_scores = [
                r["crops"][crop.value]["combined_score"] 
                for r in results["samples"]
            ]
            crop_suitabilities = [
                r["crops"][crop.value]["suitability_class"]
                for r in results["samples"]
            ]
            
            results[f"{crop.value}_summary"] = {
                "mean_score": np.mean(crop_scores),
                "max_score": np.max(crop_scores),
                "min_score": np.min(crop_scores),
                "s1_count": sum(1 for s in crop_suitabilities if s == "S1"),
                "s2_count": sum(1 for s in crop_suitabilities if s == "S2"),
                "s3_count": sum(1 for s in crop_suitabilities if s == "S3"),
                "n1_count": sum(1 for s in crop_suitabilities if s == "N1"),
                "n2_count": sum(1 for s in crop_suitabilities if s == "N2")
            }
        
        return results
    
    def generate_recommendations(
        self,
        sample_id: str
    ) -> Dict[str, Any]:
        """
        Generate remediation recommendations for a sample.
        """
        # Find sample
        sample = None
        for s in self.soil_samples:
            if s.sample_id == sample_id:
                sample = s
                break
        
        if not sample:
            return {"error": f"Sample {sample_id} not found"}
        
        # Find climate and topo
        climate = None
        for c in self.climate_data:
            if abs(c.x - sample.x) < 1000 and abs(c.y - sample.y) < 1000:
                climate = c
                break
        
        topo = None
        for t in self.topography_data:
            if abs(t.x - sample.x) < 100 and abs(t.y - sample.y) < 100:
                topo = t
                break
        
        recommendations = {
            "sample_id": sample_id,
            "crops": {}
        }
        
        for crop in self.crop_types:
            recommender = self.recommenders[crop]
            plan = recommender.generate_management_plan(sample, climate, topo)
            recommendations["crops"][crop.value] = plan
        
        return recommendations
    
    def rank_sites(self, crop: CropType) -> List[Dict[str, Any]]:
        """
        Rank all sites by suitability for a specific crop.
        """
        if not self.suitability_results:
            self.assess_all_samples()
        
        rankings = []
        for sample_id, result in self.suitability_results.items():
            if crop.value in result["crops"]:
                crop_result = result["crops"][crop.value]
                rankings.append({
                    "sample_id": sample_id,
                    "x": result["location"]["x"],
                    "y": result["location"]["y"],
                    "score": crop_result["combined_score"],
                    "suitability": crop_result["suitability_class"],
                    "limiting_factors": crop_result["limiting_factors"]
                })
        
        # Sort by score descending
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
        
        return rankings
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive suitability report.
        """
        if not self.suitability_results:
            self.assess_all_samples()
        
        report = {
            "project_name": self.project_name,
            "assessment_date": datetime.now().isoformat(),
            "sample_count": len(self.soil_samples),
            "crops_assessed": [c.value for c in self.crop_types],
            "crop_rankings": {},
            "best_crop_per_site": {},
            "recommendations_summary": []
        }
        
        # Rankings for each crop
        for crop in self.crop_types:
            report["crop_rankings"][crop.value] = self.rank_sites(crop)
        
        # Best crop for each site
        for sample_id, result in self.suitability_results.items():
            best_crop = None
            best_score = 0
            
            for crop_name, crop_result in result["crops"].items():
                if crop_result["combined_score"] > best_score:
                    best_score = crop_result["combined_score"]
                    best_crop = crop_name
            
            report["best_crop_per_site"][sample_id] = {
                "recommended_crop": best_crop,
                "score": best_score,
                "suitability": result["crops"][best_crop]["suitability_class"] if best_crop else None
            }
        
        # Summary recommendations
        for sample in self.soil_samples:
            recs = self.generate_recommendations(sample.sample_id)
            if "error" not in recs:
                # Get priority actions across all crops
                all_actions = set()
                for crop_plan in recs["crops"].values():
                    all_actions.update(crop_plan.get("priority_actions", []))
                
                if all_actions:
                    report["recommendations_summary"].append({
                        "sample_id": sample.sample_id,
                        "priority_actions": list(all_actions)
                    })
        
        return report


def create_soil_suitability_pipeline(
    crops: List[str],
    project_name: str = "soil_assessment"
) -> SoilSuitabilityPipeline:
    """
    Factory function to create a soil suitability pipeline.
    
    Args:
        crops: List of crop names ('oil_palm', 'cocoa', 'ginger')
        project_name: Project identifier
    
    Returns:
        Configured SoilSuitabilityPipeline
    """
    crop_mapping = {
        "oil_palm": CropType.OIL_PALM,
        "palm": CropType.OIL_PALM,
        "cocoa": CropType.COCOA,
        "cacao": CropType.COCOA,
        "ginger": CropType.GINGER,
        "rubber": CropType.RUBBER,
        "coffee": CropType.COFFEE_ROBUSTA,
        "coffee_arabica": CropType.COFFEE_ARABICA,
        "coffee_robusta": CropType.COFFEE_ROBUSTA,
        "cassava": CropType.CASSAVA,
        "maize": CropType.MAIZE,
        "rice": CropType.RICE_PADDY
    }
    
    crop_types = []
    for crop in crops:
        crop_enum = crop_mapping.get(crop.lower())
        if crop_enum:
            crop_types.append(crop_enum)
    
    if not crop_types:
        crop_types = [CropType.OIL_PALM, CropType.COCOA, CropType.GINGER]
    
    return SoilSuitabilityPipeline(crop_types, project_name)


def create_synthetic_soil_dataset(
    n_samples: int = 50,
    region: str = "tropical",
    seed: int = 42
) -> Tuple[List[SoilSample], List[ClimateData], List[TopographyData]]:
    """
    Create synthetic soil dataset for testing.
    """
    np.random.seed(seed)
    
    soil_samples = []
    climate_data = []
    topo_data = []
    
    # Regional parameters
    if region == "tropical":
        ph_range = (4.5, 7.0)
        om_range = (1.5, 5.0)
        rainfall_range = (1500, 3000)
        temp_range = (22, 30)
    else:
        ph_range = (5.5, 7.5)
        om_range = (2.0, 6.0)
        rainfall_range = (800, 1500)
        temp_range = (15, 25)
    
    for i in range(n_samples):
        x = np.random.uniform(0, 50000)
        y = np.random.uniform(0, 50000)
        
        # Generate soil sample
        sand = np.random.uniform(20, 60)
        clay = np.random.uniform(15, 45)
        silt = 100 - sand - clay
        
        sample = SoilSample(
            sample_id=f"SS_{i:04d}",
            x=x,
            y=y,
            sampling_date=datetime.now(),
            sampling_depth_cm=(0, 30),
            sand_percent=sand,
            silt_percent=silt,
            clay_percent=clay,
            ph_water=np.random.uniform(*ph_range),
            ec=np.random.uniform(0.1, 2.0),
            cec=np.random.uniform(8, 30),
            base_saturation=np.random.uniform(30, 80),
            organic_matter=np.random.uniform(*om_range),
            nitrogen_available=np.random.uniform(10, 80),
            phosphorus_available=np.random.uniform(5, 40),
            potassium_available=np.random.uniform(50, 300),
            calcium_exchangeable=np.random.uniform(2, 15),
            magnesium_exchangeable=np.random.uniform(0.5, 5),
            aluminum_saturation=np.random.uniform(0, 40),
            zinc=np.random.uniform(0.5, 5),
            boron=np.random.uniform(0.1, 1.5),
            drainage_class=np.random.choice(list(DrainageClass))
        )
        soil_samples.append(sample)
        
        # Generate climate data
        climate = ClimateData(
            location_id=f"CL_{i:04d}",
            x=x,
            y=y,
            annual_rainfall=np.random.uniform(*rainfall_range),
            mean_annual_temp=np.random.uniform(*temp_range),
            dry_months=np.random.randint(0, 5),
            relative_humidity=np.random.uniform(60, 90)
        )
        climate_data.append(climate)
        
        # Generate topography data
        topo = TopographyData(
            location_id=f"TP_{i:04d}",
            x=x,
            y=y,
            elevation_m=np.random.uniform(0, 1000),
            slope_percent=np.random.uniform(0, 30)
        )
        topo_data.append(topo)
    
    return soil_samples, climate_data, topo_data
