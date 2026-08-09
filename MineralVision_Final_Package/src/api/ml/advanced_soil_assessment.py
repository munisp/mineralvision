"""
Advanced Soil Assessment Module for MineralVision Platform.

Enhancements to the base soil_suitability module:
1. Toxicity hazard flags with override logic
2. Nutrient budgeting with split application schedules
3. Soil physical constraints (hardpan, compaction, coarse fragments)
4. Uncertainty quantification with confidence bounds
5. Spatial interpolation for continuous suitability maps
6. Water balance and seasonality scoring
7. Economic analysis for remediation ROI
8. Disease/pest risk integration
9. CRS-aware coordinate matching
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from datetime import datetime
import math

from .soil_suitability import (
    CropType, SuitabilityClass, SoilTextureClass, DrainageClass,
    SoilSample, ClimateData, TopographyData, CropRequirements,
    SoilSuitabilityScorer, RemediationRecommender, SoilSuitabilityPipeline
)


class HazardSeverity(Enum):
    """Hazard severity levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ToxicityType(Enum):
    """Types of soil toxicity."""
    ALUMINUM = "aluminum"
    MANGANESE = "manganese"
    IRON = "iron"
    SALINITY = "salinity"
    SODICITY = "sodicity"
    BORON = "boron"
    HEAVY_METALS = "heavy_metals"


class DiseaseRiskType(Enum):
    """Crop disease risk types."""
    RHIZOME_ROT = "rhizome_rot"  # Ginger
    PHYTOPHTHORA = "phytophthora"  # Cocoa, Oil Palm
    GANODERMA = "ganoderma"  # Oil Palm
    FUSARIUM = "fusarium"  # Multiple crops
    ROOT_ROT = "root_rot"  # General
    NEMATODES = "nematodes"  # Multiple crops


class PhosphorusMethod(Enum):
    """Phosphorus extraction methods with critical values."""
    BRAY_1 = "bray_1"
    BRAY_2 = "bray_2"
    OLSEN = "olsen"
    MEHLICH_1 = "mehlich_1"
    MEHLICH_3 = "mehlich_3"
    COLWELL = "colwell"


@dataclass
class SoilPhysicalConstraints:
    """Extended soil physical constraints."""
    sample_id: str
    
    # Depth constraints
    effective_rooting_depth_cm: float = 100.0
    restrictive_layer_depth_cm: Optional[float] = None
    restrictive_layer_type: str = ""  # hardpan, gravel, bedrock, water_table
    
    # Coarse fragments
    coarse_fragments_percent: float = 0.0  # >2mm particles
    gravel_percent: float = 0.0  # 2-75mm
    cobble_percent: float = 0.0  # 75-250mm
    stone_percent: float = 0.0  # >250mm
    
    # Compaction
    bulk_density_surface: float = 1.3  # g/cm3, 0-20cm
    bulk_density_subsurface: float = 1.4  # g/cm3, 20-50cm
    penetration_resistance_kpa: float = 0.0  # Cone penetrometer
    
    # Water dynamics
    infiltration_rate_mm_hr: float = 20.0
    saturated_hydraulic_conductivity: float = 10.0  # mm/hr
    perched_water_table_depth_cm: Optional[float] = None
    seasonal_high_water_table_cm: Optional[float] = None
    
    # Flooding
    flooding_frequency: str = "none"  # none, rare, occasional, frequent
    flooding_duration: str = "none"  # none, brief, long, very_long
    ponding_frequency: str = "none"
    
    # Erosion
    erosion_class: str = "none"  # none, slight, moderate, severe
    k_factor: float = 0.3  # Soil erodibility factor
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToxicityHazard:
    """Toxicity hazard assessment result."""
    toxicity_type: ToxicityType
    severity: HazardSeverity
    current_value: float
    threshold_value: float
    unit: str
    override_suitability: bool  # If True, overrides weighted score
    mitigation_required: bool
    mitigation_strategy: str
    estimated_remediation_cost_per_ha: float = 0.0


@dataclass
class NutrientBudget:
    """Nutrient budget with application schedule."""
    nutrient: str
    annual_requirement_kg_ha: float
    soil_supply_kg_ha: float
    fertilizer_requirement_kg_ha: float
    recovery_efficiency: float
    
    # Split applications
    applications: List[Dict[str, Any]] = field(default_factory=list)
    
    # Product recommendations
    recommended_products: List[Dict[str, Any]] = field(default_factory=list)
    
    # Fixation/loss adjustments
    fixation_factor: float = 1.0  # Multiplier for P fixation
    leaching_risk: str = "low"
    volatilization_risk: str = "low"


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification for assessments."""
    parameter: str
    point_estimate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    confidence_level: float  # e.g., 0.95 for 95% CI
    
    # Sources of uncertainty
    measurement_error: float = 0.0
    spatial_variability: float = 0.0
    temporal_variability: float = 0.0
    model_uncertainty: float = 0.0
    
    # Data quality
    sample_count: int = 1
    data_quality_score: float = 1.0  # 0-1


@dataclass
class WaterBalanceResult:
    """Water balance and seasonality assessment."""
    location_id: str
    
    # Annual totals
    annual_precipitation_mm: float = 0.0
    annual_pet_mm: float = 0.0
    annual_aet_mm: float = 0.0
    annual_water_surplus_mm: float = 0.0
    annual_water_deficit_mm: float = 0.0
    
    # Indices
    aridity_index: float = 1.0  # P/PET
    moisture_index: float = 0.0  # (P-PET)/PET
    seasonality_index: float = 0.0
    
    # Monthly analysis
    monthly_precipitation: List[float] = field(default_factory=list)
    monthly_pet: List[float] = field(default_factory=list)
    monthly_water_balance: List[float] = field(default_factory=list)
    
    # Stress periods
    drought_stress_days: int = 0
    waterlogging_risk_months: int = 0
    dry_season_length_months: int = 0
    
    # Crop-specific
    critical_period_water_available: bool = True
    irrigation_requirement_mm: float = 0.0


@dataclass
class DiseaseRiskAssessment:
    """Disease/pest risk assessment."""
    disease_type: DiseaseRiskType
    risk_level: HazardSeverity
    probability: float  # 0-1
    
    # Contributing factors
    contributing_factors: List[str] = field(default_factory=list)
    
    # Management
    preventive_measures: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)


@dataclass
class EconomicAnalysis:
    """Economic analysis for remediation."""
    sample_id: str
    crop: str
    
    # Costs
    amendment_costs: Dict[str, float] = field(default_factory=dict)
    fertilizer_costs: Dict[str, float] = field(default_factory=dict)
    labor_costs: float = 0.0
    equipment_costs: float = 0.0
    total_remediation_cost_per_ha: float = 0.0
    
    # Benefits
    expected_yield_without_remediation: float = 0.0
    expected_yield_with_remediation: float = 0.0
    yield_increase_percent: float = 0.0
    crop_price_per_kg: float = 0.0
    additional_revenue_per_ha: float = 0.0
    
    # ROI
    net_benefit_per_ha: float = 0.0
    benefit_cost_ratio: float = 0.0
    payback_period_years: float = 0.0
    
    # Priority
    priority_rank: int = 0
    recommendation: str = ""


class ToxicityHazardAssessor:
    """Assess soil toxicity hazards with override logic."""
    
    # Toxicity thresholds by crop type
    THRESHOLDS = {
        CropType.OIL_PALM: {
            ToxicityType.ALUMINUM: {
                "low": 30, "moderate": 50, "high": 70, "critical": 85,
                "unit": "% saturation", "field": "aluminum_saturation"
            },
            ToxicityType.SALINITY: {
                "low": 2, "moderate": 4, "high": 6, "critical": 8,
                "unit": "dS/m", "field": "ec"
            },
            ToxicityType.SODICITY: {
                "low": 6, "moderate": 10, "high": 15, "critical": 25,
                "unit": "% ESP", "field": "esp"
            }
        },
        CropType.COCOA: {
            ToxicityType.ALUMINUM: {
                "low": 20, "moderate": 35, "high": 50, "critical": 70,
                "unit": "% saturation", "field": "aluminum_saturation"
            },
            ToxicityType.SALINITY: {
                "low": 1.5, "moderate": 3, "high": 4, "critical": 6,
                "unit": "dS/m", "field": "ec"
            },
            ToxicityType.SODICITY: {
                "low": 5, "moderate": 8, "high": 12, "critical": 20,
                "unit": "% ESP", "field": "esp"
            },
            ToxicityType.MANGANESE: {
                "low": 100, "moderate": 200, "high": 400, "critical": 600,
                "unit": "ppm", "field": "manganese"
            }
        },
        CropType.GINGER: {
            ToxicityType.ALUMINUM: {
                "low": 25, "moderate": 40, "high": 60, "critical": 80,
                "unit": "% saturation", "field": "aluminum_saturation"
            },
            ToxicityType.SALINITY: {
                "low": 1, "moderate": 2, "high": 3, "critical": 4,
                "unit": "dS/m", "field": "ec"
            },
            ToxicityType.SODICITY: {
                "low": 5, "moderate": 8, "high": 12, "critical": 18,
                "unit": "% ESP", "field": "esp"
            }
        }
    }
    
    # Mitigation strategies
    MITIGATION = {
        ToxicityType.ALUMINUM: {
            "strategy": "Apply agricultural lime (CaCO3) or dolomite to raise pH and reduce Al solubility",
            "cost_per_ha": 150  # Base cost, adjusted by severity
        },
        ToxicityType.MANGANESE: {
            "strategy": "Apply lime to raise pH above 5.5; ensure good drainage",
            "cost_per_ha": 120
        },
        ToxicityType.SALINITY: {
            "strategy": "Leaching with good quality water; improve drainage; salt-tolerant varieties",
            "cost_per_ha": 300
        },
        ToxicityType.SODICITY: {
            "strategy": "Apply gypsum (CaSO4) to displace Na; improve drainage",
            "cost_per_ha": 250
        },
        ToxicityType.BORON: {
            "strategy": "Leaching; avoid high-B irrigation water; tolerant varieties",
            "cost_per_ha": 100
        }
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.thresholds = self.THRESHOLDS.get(crop_type, self.THRESHOLDS[CropType.OIL_PALM])
    
    def assess_toxicity(
        self,
        sample: SoilSample,
        toxicity_type: ToxicityType
    ) -> Optional[ToxicityHazard]:
        """Assess a specific toxicity type."""
        if toxicity_type not in self.thresholds:
            return None
        
        threshold_info = self.thresholds[toxicity_type]
        field_name = threshold_info["field"]
        
        # Get current value
        current_value = getattr(sample, field_name, 0)
        
        # Determine severity
        if current_value >= threshold_info["critical"]:
            severity = HazardSeverity.CRITICAL
            override = True
        elif current_value >= threshold_info["high"]:
            severity = HazardSeverity.HIGH
            override = True
        elif current_value >= threshold_info["moderate"]:
            severity = HazardSeverity.MODERATE
            override = False
        elif current_value >= threshold_info["low"]:
            severity = HazardSeverity.LOW
            override = False
        else:
            severity = HazardSeverity.NONE
            override = False
        
        # Get mitigation info
        mitigation_info = self.MITIGATION.get(toxicity_type, {})
        mitigation_strategy = mitigation_info.get("strategy", "Consult agronomist")
        base_cost = mitigation_info.get("cost_per_ha", 100)
        
        # Adjust cost by severity
        severity_multipliers = {
            HazardSeverity.NONE: 0,
            HazardSeverity.LOW: 0.5,
            HazardSeverity.MODERATE: 1.0,
            HazardSeverity.HIGH: 1.5,
            HazardSeverity.CRITICAL: 2.5
        }
        estimated_cost = base_cost * severity_multipliers[severity]
        
        return ToxicityHazard(
            toxicity_type=toxicity_type,
            severity=severity,
            current_value=current_value,
            threshold_value=threshold_info["moderate"],
            unit=threshold_info["unit"],
            override_suitability=override,
            mitigation_required=severity in [HazardSeverity.HIGH, HazardSeverity.CRITICAL],
            mitigation_strategy=mitigation_strategy,
            estimated_remediation_cost_per_ha=estimated_cost
        )
    
    def assess_all_toxicities(self, sample: SoilSample) -> List[ToxicityHazard]:
        """Assess all relevant toxicities for the crop."""
        hazards = []
        for toxicity_type in self.thresholds.keys():
            hazard = self.assess_toxicity(sample, toxicity_type)
            if hazard and hazard.severity != HazardSeverity.NONE:
                hazards.append(hazard)
        return hazards
    
    def get_override_suitability(
        self,
        hazards: List[ToxicityHazard]
    ) -> Optional[SuitabilityClass]:
        """Determine if toxicity should override suitability class."""
        critical_hazards = [h for h in hazards if h.severity == HazardSeverity.CRITICAL]
        high_hazards = [h for h in hazards if h.severity == HazardSeverity.HIGH]
        
        if critical_hazards:
            return SuitabilityClass.N1  # Currently not suitable
        elif len(high_hazards) >= 2:
            return SuitabilityClass.N1
        elif high_hazards:
            return SuitabilityClass.S3  # Marginally suitable at best
        
        return None  # No override


class NutrientBudgetCalculator:
    """Calculate nutrient budgets with split application schedules."""
    
    # Crop nutrient uptake (kg/tonne of product)
    CROP_UPTAKE = {
        CropType.OIL_PALM: {
            "N": {"uptake_per_tonne": 7.5, "typical_yield": 20},  # FFB
            "P2O5": {"uptake_per_tonne": 1.8, "typical_yield": 20},
            "K2O": {"uptake_per_tonne": 9.5, "typical_yield": 20},
            "MgO": {"uptake_per_tonne": 2.0, "typical_yield": 20}
        },
        CropType.COCOA: {
            "N": {"uptake_per_tonne": 30, "typical_yield": 1.5},  # Dry beans
            "P2O5": {"uptake_per_tonne": 8, "typical_yield": 1.5},
            "K2O": {"uptake_per_tonne": 45, "typical_yield": 1.5},
            "MgO": {"uptake_per_tonne": 6, "typical_yield": 1.5}
        },
        CropType.GINGER: {
            "N": {"uptake_per_tonne": 5.5, "typical_yield": 25},  # Fresh rhizome
            "P2O5": {"uptake_per_tonne": 1.2, "typical_yield": 25},
            "K2O": {"uptake_per_tonne": 8.0, "typical_yield": 25},
            "MgO": {"uptake_per_tonne": 1.5, "typical_yield": 25}
        }
    }
    
    # Recovery efficiencies by soil texture
    RECOVERY_EFFICIENCY = {
        "N": {
            SoilTextureClass.SAND: 0.45,
            SoilTextureClass.LOAMY_SAND: 0.50,
            SoilTextureClass.SANDY_LOAM: 0.55,
            SoilTextureClass.LOAM: 0.60,
            SoilTextureClass.SILT_LOAM: 0.60,
            SoilTextureClass.CLAY_LOAM: 0.55,
            SoilTextureClass.CLAY: 0.50
        },
        "P2O5": {
            SoilTextureClass.SAND: 0.25,
            SoilTextureClass.LOAMY_SAND: 0.22,
            SoilTextureClass.SANDY_LOAM: 0.20,
            SoilTextureClass.LOAM: 0.18,
            SoilTextureClass.SILT_LOAM: 0.18,
            SoilTextureClass.CLAY_LOAM: 0.15,
            SoilTextureClass.CLAY: 0.12
        },
        "K2O": {
            SoilTextureClass.SAND: 0.55,
            SoilTextureClass.LOAMY_SAND: 0.60,
            SoilTextureClass.SANDY_LOAM: 0.65,
            SoilTextureClass.LOAM: 0.70,
            SoilTextureClass.SILT_LOAM: 0.70,
            SoilTextureClass.CLAY_LOAM: 0.75,
            SoilTextureClass.CLAY: 0.80
        }
    }
    
    # P fixation factors by soil properties
    P_FIXATION_FACTORS = {
        "high_fe_al": 1.5,  # High Fe/Al oxides (tropical soils)
        "acidic": 1.3,  # pH < 5.5
        "calcareous": 1.4,  # pH > 7.5, high CaCO3
        "normal": 1.0
    }
    
    # Fertilizer products
    FERTILIZER_PRODUCTS = {
        "N": [
            {"name": "Urea", "content": 0.46, "unit_cost": 0.50},
            {"name": "Ammonium Sulfate", "content": 0.21, "unit_cost": 0.35},
            {"name": "Ammonium Nitrate", "content": 0.34, "unit_cost": 0.55},
            {"name": "CAN", "content": 0.27, "unit_cost": 0.45}
        ],
        "P2O5": [
            {"name": "TSP", "content": 0.46, "unit_cost": 0.60},
            {"name": "DAP", "content": 0.46, "unit_cost": 0.65},
            {"name": "Rock Phosphate", "content": 0.30, "unit_cost": 0.25},
            {"name": "SSP", "content": 0.18, "unit_cost": 0.30}
        ],
        "K2O": [
            {"name": "MOP (KCl)", "content": 0.60, "unit_cost": 0.55},
            {"name": "SOP (K2SO4)", "content": 0.50, "unit_cost": 0.80},
            {"name": "Kieserite", "content": 0.22, "unit_cost": 0.40}
        ]
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.uptake_data = self.CROP_UPTAKE.get(crop_type, self.CROP_UPTAKE[CropType.OIL_PALM])
    
    def estimate_soil_supply(
        self,
        sample: SoilSample,
        nutrient: str
    ) -> float:
        """Estimate nutrient supply from soil (kg/ha)."""
        if nutrient == "N":
            # N supply from organic matter mineralization
            # Assume 2-3% of total N mineralizes annually
            total_n_percent = sample.nitrogen_total if sample.nitrogen_total > 0 else sample.organic_matter * 0.05
            soil_n_kg_ha = total_n_percent * 10000 * 0.15 * 0.025  # 2.5% mineralization
            return soil_n_kg_ha
        
        elif nutrient == "P2O5":
            # Convert available P (ppm) to kg P2O5/ha
            # Assume 20cm depth, bulk density 1.3
            p_kg_ha = sample.phosphorus_available * 2.6  # Simplified conversion
            p2o5_kg_ha = p_kg_ha * 2.29  # P to P2O5
            return min(p2o5_kg_ha * 0.3, 30)  # Cap at 30 kg/ha available
        
        elif nutrient == "K2O":
            # Convert available K (ppm) to kg K2O/ha
            k_kg_ha = sample.potassium_available * 2.6
            k2o_kg_ha = k_kg_ha * 1.2  # K to K2O
            return min(k2o_kg_ha * 0.4, 60)  # Cap at 60 kg/ha available
        
        return 0
    
    def calculate_p_fixation_factor(self, sample: SoilSample) -> float:
        """Calculate P fixation factor based on soil properties."""
        factor = 1.0
        
        # Acidic soils with high Al
        if sample.ph_water < 5.5 and sample.aluminum_saturation > 30:
            factor = max(factor, self.P_FIXATION_FACTORS["high_fe_al"])
        elif sample.ph_water < 5.5:
            factor = max(factor, self.P_FIXATION_FACTORS["acidic"])
        
        # Calcareous soils
        if sample.ph_water > 7.5:
            factor = max(factor, self.P_FIXATION_FACTORS["calcareous"])
        
        # Clay content effect
        if sample.clay_percent > 40:
            factor *= 1.2
        
        return factor
    
    def calculate_nutrient_budget(
        self,
        sample: SoilSample,
        nutrient: str,
        target_yield: Optional[float] = None,
        yield_class: str = "medium"
    ) -> NutrientBudget:
        """Calculate complete nutrient budget."""
        uptake_info = self.uptake_data.get(nutrient, {})
        
        # Determine target yield
        if target_yield is None:
            base_yield = uptake_info.get("typical_yield", 10)
            yield_multipliers = {"low": 0.7, "medium": 1.0, "high": 1.3}
            target_yield = base_yield * yield_multipliers.get(yield_class, 1.0)
        
        # Calculate crop requirement
        uptake_per_tonne = uptake_info.get("uptake_per_tonne", 5)
        annual_requirement = uptake_per_tonne * target_yield
        
        # Estimate soil supply
        soil_supply = self.estimate_soil_supply(sample, nutrient)
        
        # Get recovery efficiency
        texture = sample.texture_class or SoilTextureClass.LOAM
        efficiency_table = self.RECOVERY_EFFICIENCY.get(nutrient, {})
        recovery_efficiency = efficiency_table.get(texture, 0.5)
        
        # Calculate fertilizer requirement
        net_requirement = max(0, annual_requirement - soil_supply)
        
        # Apply fixation factor for P
        fixation_factor = 1.0
        if nutrient == "P2O5":
            fixation_factor = self.calculate_p_fixation_factor(sample)
        
        fertilizer_requirement = (net_requirement / recovery_efficiency) * fixation_factor
        
        # Determine leaching/volatilization risk
        leaching_risk = "low"
        volatilization_risk = "low"
        
        if nutrient == "N":
            if texture in [SoilTextureClass.SAND, SoilTextureClass.LOAMY_SAND]:
                leaching_risk = "high"
            elif texture == SoilTextureClass.SANDY_LOAM:
                leaching_risk = "moderate"
            
            if sample.ph_water > 7.5:
                volatilization_risk = "high"
            elif sample.ph_water > 7.0:
                volatilization_risk = "moderate"
        
        # Generate split applications
        applications = self._generate_split_applications(
            nutrient, fertilizer_requirement, leaching_risk
        )
        
        # Recommend products
        products = self._recommend_products(nutrient, fertilizer_requirement, sample)
        
        return NutrientBudget(
            nutrient=nutrient,
            annual_requirement_kg_ha=annual_requirement,
            soil_supply_kg_ha=soil_supply,
            fertilizer_requirement_kg_ha=fertilizer_requirement,
            recovery_efficiency=recovery_efficiency,
            applications=applications,
            recommended_products=products,
            fixation_factor=fixation_factor,
            leaching_risk=leaching_risk,
            volatilization_risk=volatilization_risk
        )
    
    def _generate_split_applications(
        self,
        nutrient: str,
        total_requirement: float,
        leaching_risk: str
    ) -> List[Dict[str, Any]]:
        """Generate split application schedule."""
        applications = []
        
        if nutrient == "N":
            if leaching_risk == "high":
                # 4 splits for high leaching risk
                splits = [
                    {"timing": "Planting/Start of season", "percent": 20},
                    {"timing": "4-6 weeks after planting", "percent": 30},
                    {"timing": "8-10 weeks after planting", "percent": 30},
                    {"timing": "12-14 weeks after planting", "percent": 20}
                ]
            elif leaching_risk == "moderate":
                # 3 splits
                splits = [
                    {"timing": "Planting/Start of season", "percent": 25},
                    {"timing": "6-8 weeks after planting", "percent": 40},
                    {"timing": "12 weeks after planting", "percent": 35}
                ]
            else:
                # 2 splits
                splits = [
                    {"timing": "Planting/Start of season", "percent": 40},
                    {"timing": "8-10 weeks after planting", "percent": 60}
                ]
        
        elif nutrient == "P2O5":
            # P typically applied at planting
            splits = [
                {"timing": "Pre-planting/Planting", "percent": 100}
            ]
        
        elif nutrient == "K2O":
            if leaching_risk == "high":
                splits = [
                    {"timing": "Planting", "percent": 30},
                    {"timing": "6-8 weeks after planting", "percent": 40},
                    {"timing": "12 weeks after planting", "percent": 30}
                ]
            else:
                splits = [
                    {"timing": "Planting", "percent": 50},
                    {"timing": "8-10 weeks after planting", "percent": 50}
                ]
        else:
            splits = [{"timing": "As needed", "percent": 100}]
        
        for split in splits:
            applications.append({
                "timing": split["timing"],
                "rate_kg_ha": total_requirement * split["percent"] / 100,
                "percent_of_total": split["percent"]
            })
        
        return applications
    
    def _recommend_products(
        self,
        nutrient: str,
        requirement: float,
        sample: SoilSample
    ) -> List[Dict[str, Any]]:
        """Recommend fertilizer products."""
        products = self.FERTILIZER_PRODUCTS.get(nutrient, [])
        recommendations = []
        
        for product in products:
            product_rate = requirement / product["content"]
            cost = product_rate * product["unit_cost"]
            
            # Suitability notes
            notes = []
            if nutrient == "N":
                if product["name"] == "Urea" and sample.ph_water > 7.5:
                    notes.append("High volatilization risk - incorporate immediately")
                if product["name"] == "Ammonium Sulfate" and sample.ph_water < 5.5:
                    notes.append("May increase acidity - monitor pH")
            
            recommendations.append({
                "product": product["name"],
                "nutrient_content": product["content"],
                "rate_kg_ha": product_rate,
                "cost_per_ha": cost,
                "notes": notes
            })
        
        # Sort by cost
        recommendations.sort(key=lambda x: x["cost_per_ha"])
        
        return recommendations


class PhysicalConstraintAssessor:
    """Assess soil physical constraints."""
    
    # Bulk density thresholds by texture (g/cm3)
    COMPACTION_THRESHOLDS = {
        SoilTextureClass.SAND: {"optimal": 1.6, "limiting": 1.8},
        SoilTextureClass.LOAMY_SAND: {"optimal": 1.6, "limiting": 1.75},
        SoilTextureClass.SANDY_LOAM: {"optimal": 1.5, "limiting": 1.7},
        SoilTextureClass.LOAM: {"optimal": 1.4, "limiting": 1.6},
        SoilTextureClass.SILT_LOAM: {"optimal": 1.4, "limiting": 1.55},
        SoilTextureClass.CLAY_LOAM: {"optimal": 1.35, "limiting": 1.5},
        SoilTextureClass.CLAY: {"optimal": 1.25, "limiting": 1.4}
    }
    
    # Minimum rooting depths by crop (cm)
    MIN_ROOTING_DEPTH = {
        CropType.OIL_PALM: {"optimal": 100, "minimum": 75},
        CropType.COCOA: {"optimal": 150, "minimum": 100},
        CropType.GINGER: {"optimal": 45, "minimum": 30}
    }
    
    # Coarse fragment limits
    COARSE_FRAGMENT_LIMITS = {
        "optimal": 15,  # %
        "moderate": 35,
        "severe": 60
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
    
    def assess_compaction(
        self,
        sample: SoilSample,
        constraints: Optional[SoilPhysicalConstraints] = None
    ) -> Dict[str, Any]:
        """Assess soil compaction."""
        texture = sample.texture_class or SoilTextureClass.LOAM
        thresholds = self.COMPACTION_THRESHOLDS.get(texture, self.COMPACTION_THRESHOLDS[SoilTextureClass.LOAM])
        
        bulk_density = sample.bulk_density
        if constraints:
            bulk_density = constraints.bulk_density_surface
        
        if bulk_density <= thresholds["optimal"]:
            severity = HazardSeverity.NONE
            score = 100
        elif bulk_density <= thresholds["limiting"]:
            severity = HazardSeverity.MODERATE
            score = 70 - 30 * (bulk_density - thresholds["optimal"]) / (thresholds["limiting"] - thresholds["optimal"])
        else:
            severity = HazardSeverity.HIGH
            score = max(0, 40 - 40 * (bulk_density - thresholds["limiting"]) / 0.3)
        
        return {
            "parameter": "compaction",
            "bulk_density": bulk_density,
            "threshold_optimal": thresholds["optimal"],
            "threshold_limiting": thresholds["limiting"],
            "severity": severity,
            "score": score,
            "mitigation": "Deep tillage, subsoiling, or cover crops" if severity != HazardSeverity.NONE else None
        }
    
    def assess_rooting_depth(
        self,
        constraints: SoilPhysicalConstraints
    ) -> Dict[str, Any]:
        """Assess effective rooting depth."""
        depth_req = self.MIN_ROOTING_DEPTH.get(self.crop_type, {"optimal": 100, "minimum": 75})
        
        effective_depth = constraints.effective_rooting_depth_cm
        if constraints.restrictive_layer_depth_cm:
            effective_depth = min(effective_depth, constraints.restrictive_layer_depth_cm)
        
        if effective_depth >= depth_req["optimal"]:
            severity = HazardSeverity.NONE
            score = 100
        elif effective_depth >= depth_req["minimum"]:
            severity = HazardSeverity.MODERATE
            score = 70 + 30 * (effective_depth - depth_req["minimum"]) / (depth_req["optimal"] - depth_req["minimum"])
        else:
            severity = HazardSeverity.HIGH
            score = max(0, 70 * effective_depth / depth_req["minimum"])
        
        return {
            "parameter": "rooting_depth",
            "effective_depth_cm": effective_depth,
            "restrictive_layer": constraints.restrictive_layer_type,
            "required_minimum": depth_req["minimum"],
            "required_optimal": depth_req["optimal"],
            "severity": severity,
            "score": score,
            "mitigation": "Deep ripping, raised beds, or alternative crop" if severity != HazardSeverity.NONE else None
        }
    
    def assess_coarse_fragments(
        self,
        constraints: SoilPhysicalConstraints
    ) -> Dict[str, Any]:
        """Assess coarse fragment content."""
        fragments = constraints.coarse_fragments_percent
        
        if fragments <= self.COARSE_FRAGMENT_LIMITS["optimal"]:
            severity = HazardSeverity.NONE
            score = 100
        elif fragments <= self.COARSE_FRAGMENT_LIMITS["moderate"]:
            severity = HazardSeverity.MODERATE
            score = 70
        elif fragments <= self.COARSE_FRAGMENT_LIMITS["severe"]:
            severity = HazardSeverity.HIGH
            score = 40
        else:
            severity = HazardSeverity.CRITICAL
            score = 10
        
        return {
            "parameter": "coarse_fragments",
            "percent": fragments,
            "severity": severity,
            "score": score,
            "mitigation": "Stone removal, raised beds, or alternative land use" if severity != HazardSeverity.NONE else None
        }
    
    def assess_drainage_risk(
        self,
        sample: SoilSample,
        constraints: Optional[SoilPhysicalConstraints] = None
    ) -> Dict[str, Any]:
        """Assess drainage and waterlogging risk."""
        drainage = sample.drainage_class
        
        # Base score from drainage class
        drainage_scores = {
            DrainageClass.EXCESSIVELY_DRAINED: 70,
            DrainageClass.WELL_DRAINED: 100,
            DrainageClass.MODERATELY_WELL_DRAINED: 85,
            DrainageClass.SOMEWHAT_POORLY_DRAINED: 60,
            DrainageClass.POORLY_DRAINED: 30,
            DrainageClass.VERY_POORLY_DRAINED: 10
        }
        
        score = drainage_scores.get(drainage, 70)
        
        # Adjust for water table if available
        if constraints and constraints.seasonal_high_water_table_cm:
            if constraints.seasonal_high_water_table_cm < 50:
                score = min(score, 30)
            elif constraints.seasonal_high_water_table_cm < 100:
                score = min(score, 60)
        
        # Adjust for flooding
        if constraints:
            if constraints.flooding_frequency == "frequent":
                score = min(score, 20)
            elif constraints.flooding_frequency == "occasional":
                score = min(score, 50)
        
        if score >= 80:
            severity = HazardSeverity.NONE
        elif score >= 60:
            severity = HazardSeverity.LOW
        elif score >= 40:
            severity = HazardSeverity.MODERATE
        elif score >= 20:
            severity = HazardSeverity.HIGH
        else:
            severity = HazardSeverity.CRITICAL
        
        return {
            "parameter": "drainage",
            "drainage_class": drainage.value if drainage else "unknown",
            "severity": severity,
            "score": score,
            "mitigation": "Install drainage, raised beds, or mounding" if severity != HazardSeverity.NONE else None
        }


class UncertaintyQuantifier:
    """Quantify uncertainty in soil assessments."""
    
    # Typical lab measurement CVs (coefficient of variation)
    LAB_CV = {
        "ph_water": 0.02,
        "ec": 0.05,
        "organic_matter": 0.08,
        "nitrogen_total": 0.10,
        "phosphorus_available": 0.12,
        "potassium_available": 0.10,
        "cec": 0.08,
        "aluminum_saturation": 0.15
    }
    
    # Spatial variability CVs (typical field-scale)
    SPATIAL_CV = {
        "ph_water": 0.05,
        "ec": 0.30,
        "organic_matter": 0.25,
        "nitrogen_total": 0.30,
        "phosphorus_available": 0.40,
        "potassium_available": 0.35,
        "cec": 0.20,
        "aluminum_saturation": 0.35
    }
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level == 0.95 else 1.645  # 95% or 90%
    
    def estimate_parameter_uncertainty(
        self,
        parameter: str,
        value: float,
        sample_count: int = 1,
        lab_replicates: int = 1
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for a single parameter."""
        lab_cv = self.LAB_CV.get(parameter, 0.10)
        spatial_cv = self.SPATIAL_CV.get(parameter, 0.30)
        
        # Measurement error (reduced by replicates)
        measurement_error = value * lab_cv / math.sqrt(lab_replicates)
        
        # Spatial variability (reduced by sample count)
        spatial_variability = value * spatial_cv / math.sqrt(sample_count)
        
        # Combined uncertainty
        total_uncertainty = math.sqrt(measurement_error**2 + spatial_variability**2)
        
        # Confidence interval
        ci_half_width = self.z_score * total_uncertainty
        
        # Data quality score
        quality_score = min(1.0, sample_count / 10)  # Max quality at 10+ samples
        
        return UncertaintyEstimate(
            parameter=parameter,
            point_estimate=value,
            confidence_interval_lower=max(0, value - ci_half_width),
            confidence_interval_upper=value + ci_half_width,
            confidence_level=self.confidence_level,
            measurement_error=measurement_error,
            spatial_variability=spatial_variability,
            sample_count=sample_count,
            data_quality_score=quality_score
        )
    
    def estimate_score_uncertainty(
        self,
        parameter_uncertainties: List[UncertaintyEstimate],
        weights: Dict[str, float]
    ) -> UncertaintyEstimate:
        """Propagate uncertainty to final score."""
        # Weighted average of uncertainties
        total_weight = sum(weights.values())
        
        weighted_variance = 0
        weighted_mean = 0
        
        for unc in parameter_uncertainties:
            if unc.parameter in weights:
                w = weights[unc.parameter] / total_weight
                weighted_mean += w * unc.point_estimate
                # Propagate variance
                param_variance = ((unc.confidence_interval_upper - unc.confidence_interval_lower) / (2 * self.z_score))**2
                weighted_variance += (w**2) * param_variance
        
        combined_std = math.sqrt(weighted_variance)
        ci_half_width = self.z_score * combined_std
        
        # Average data quality
        avg_quality = np.mean([u.data_quality_score for u in parameter_uncertainties])
        
        return UncertaintyEstimate(
            parameter="overall_score",
            point_estimate=weighted_mean,
            confidence_interval_lower=max(0, weighted_mean - ci_half_width),
            confidence_interval_upper=min(100, weighted_mean + ci_half_width),
            confidence_level=self.confidence_level,
            sample_count=int(np.mean([u.sample_count for u in parameter_uncertainties])),
            data_quality_score=avg_quality
        )
    
    def check_data_sufficiency(
        self,
        sample: SoilSample,
        required_parameters: List[str]
    ) -> Dict[str, Any]:
        """Check if required parameters are present."""
        missing = []
        present = []
        
        parameter_fields = {
            "ph": "ph_water",
            "texture": "texture_class",
            "drainage": "drainage_class",
            "organic_matter": "organic_matter",
            "nitrogen": "nitrogen_available",
            "phosphorus": "phosphorus_available",
            "potassium": "potassium_available",
            "cec": "cec",
            "al_saturation": "aluminum_saturation"
        }
        
        for param in required_parameters:
            field = parameter_fields.get(param, param)
            value = getattr(sample, field, None)
            
            if value is None or (isinstance(value, (int, float)) and value == 0):
                missing.append(param)
            else:
                present.append(param)
        
        sufficiency_score = len(present) / len(required_parameters) if required_parameters else 1.0
        
        return {
            "sufficient": len(missing) == 0,
            "sufficiency_score": sufficiency_score,
            "missing_parameters": missing,
            "present_parameters": present,
            "recommendation": f"Missing critical parameters: {', '.join(missing)}" if missing else "All required parameters present"
        }


class WaterBalanceCalculator:
    """Calculate water balance and seasonality indices."""
    
    # Crop water requirements (mm/day during peak demand)
    CROP_WATER_DEMAND = {
        CropType.OIL_PALM: {"peak_etcrop": 5.5, "kc": 1.0},
        CropType.COCOA: {"peak_etcrop": 4.0, "kc": 0.9},
        CropType.GINGER: {"peak_etcrop": 4.5, "kc": 0.85}
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
        self.water_demand = self.CROP_WATER_DEMAND.get(crop_type, {"peak_etcrop": 4.5, "kc": 0.9})
    
    def calculate_water_balance(
        self,
        climate: ClimateData,
        monthly_rainfall: Optional[List[float]] = None,
        monthly_pet: Optional[List[float]] = None
    ) -> WaterBalanceResult:
        """Calculate comprehensive water balance."""
        result = WaterBalanceResult(location_id=climate.location_id)
        
        # Use provided monthly data or estimate from annual
        if monthly_rainfall is None:
            monthly_rainfall = self._estimate_monthly_rainfall(climate)
        if monthly_pet is None:
            monthly_pet = self._estimate_monthly_pet(climate)
        
        result.monthly_precipitation = monthly_rainfall
        result.monthly_pet = monthly_pet
        
        # Calculate monthly water balance
        result.monthly_water_balance = [p - pet for p, pet in zip(monthly_rainfall, monthly_pet)]
        
        # Annual totals
        result.annual_precipitation_mm = sum(monthly_rainfall)
        result.annual_pet_mm = sum(monthly_pet)
        
        # Calculate surplus and deficit
        surplus = sum(max(0, wb) for wb in result.monthly_water_balance)
        deficit = sum(abs(min(0, wb)) for wb in result.monthly_water_balance)
        
        result.annual_water_surplus_mm = surplus
        result.annual_water_deficit_mm = deficit
        
        # Indices
        if result.annual_pet_mm > 0:
            result.aridity_index = result.annual_precipitation_mm / result.annual_pet_mm
            result.moisture_index = (result.annual_precipitation_mm - result.annual_pet_mm) / result.annual_pet_mm
        
        # Seasonality index (Walsh & Lawler)
        mean_monthly = result.annual_precipitation_mm / 12
        if mean_monthly > 0:
            result.seasonality_index = sum(abs(p - mean_monthly) for p in monthly_rainfall) / result.annual_precipitation_mm
        
        # Stress periods
        result.drought_stress_days = self._calculate_drought_days(result.monthly_water_balance)
        result.waterlogging_risk_months = sum(1 for wb in result.monthly_water_balance if wb > 150)
        result.dry_season_length_months = sum(1 for p in monthly_rainfall if p < 60)
        
        # Irrigation requirement
        kc = self.water_demand["kc"]
        crop_water_need = [pet * kc for pet in monthly_pet]
        result.irrigation_requirement_mm = max(0, sum(crop_water_need) - result.annual_precipitation_mm)
        
        return result
    
    def _estimate_monthly_rainfall(self, climate: ClimateData) -> List[float]:
        """Estimate monthly rainfall distribution."""
        annual = climate.annual_rainfall
        dry_months = climate.dry_months
        
        # Simple bimodal distribution for tropical regions
        if dry_months <= 2:
            # Relatively uniform
            base = annual / 12
            monthly = [base * (0.9 + 0.2 * np.sin(i * np.pi / 6)) for i in range(12)]
        else:
            # More seasonal
            wet_months = 12 - dry_months
            wet_rain = annual * 0.85 / wet_months
            dry_rain = annual * 0.15 / dry_months
            
            monthly = []
            for i in range(12):
                if i < dry_months // 2 or i >= 12 - dry_months // 2:
                    monthly.append(dry_rain)
                else:
                    monthly.append(wet_rain)
        
        # Normalize to match annual total
        total = sum(monthly)
        if total > 0:
            monthly = [m * annual / total for m in monthly]
        
        return monthly
    
    def _estimate_monthly_pet(self, climate: ClimateData) -> List[float]:
        """Estimate monthly PET."""
        annual_pet = climate.potential_evapotranspiration
        if annual_pet == 0:
            # Estimate from temperature (simplified Thornthwaite)
            annual_pet = climate.mean_annual_temp * 60  # Very rough estimate
        
        # Assume relatively uniform PET in tropics with slight seasonal variation
        base = annual_pet / 12
        monthly = [base * (0.95 + 0.1 * np.sin(i * np.pi / 6)) for i in range(12)]
        
        # Normalize
        total = sum(monthly)
        if total > 0:
            monthly = [m * annual_pet / total for m in monthly]
        
        return monthly
    
    def _calculate_drought_days(self, monthly_water_balance: List[float]) -> int:
        """Estimate drought stress days."""
        drought_days = 0
        for wb in monthly_water_balance:
            if wb < -50:  # Significant deficit
                drought_days += 30  # Full month
            elif wb < 0:
                drought_days += int(15 * abs(wb) / 50)  # Partial month
        return drought_days
    
    def score_water_availability(self, result: WaterBalanceResult) -> Dict[str, Any]:
        """Score water availability for crop suitability."""
        scores = {}
        
        # Aridity index score
        ai = result.aridity_index
        if ai >= 1.0:
            scores["aridity"] = 100
        elif ai >= 0.65:
            scores["aridity"] = 70 + 30 * (ai - 0.65) / 0.35
        elif ai >= 0.2:
            scores["aridity"] = 30 + 40 * (ai - 0.2) / 0.45
        else:
            scores["aridity"] = 30 * ai / 0.2
        
        # Drought stress score
        if result.drought_stress_days <= 30:
            scores["drought"] = 100
        elif result.drought_stress_days <= 60:
            scores["drought"] = 70
        elif result.drought_stress_days <= 90:
            scores["drought"] = 50
        else:
            scores["drought"] = max(0, 50 - (result.drought_stress_days - 90))
        
        # Waterlogging score
        if result.waterlogging_risk_months == 0:
            scores["waterlogging"] = 100
        elif result.waterlogging_risk_months <= 2:
            scores["waterlogging"] = 70
        elif result.waterlogging_risk_months <= 4:
            scores["waterlogging"] = 50
        else:
            scores["waterlogging"] = 30
        
        # Combined score
        overall = np.mean(list(scores.values()))
        
        return {
            "scores": scores,
            "overall_score": overall,
            "limiting_factor": min(scores, key=scores.get),
            "irrigation_needed": result.irrigation_requirement_mm > 200
        }


class DiseaseRiskAssessor:
    """Assess disease and pest risks based on soil and climate conditions."""
    
    # Disease risk factors
    DISEASE_FACTORS = {
        DiseaseRiskType.RHIZOME_ROT: {
            "crops": [CropType.GINGER],
            "soil_factors": {
                "drainage": {"risk_classes": [DrainageClass.POORLY_DRAINED, DrainageClass.VERY_POORLY_DRAINED], "weight": 0.4},
                "ph": {"risk_range": (0, 5.0), "weight": 0.2},
                "organic_matter": {"risk_range": (0, 1.5), "weight": 0.1}
            },
            "climate_factors": {
                "rainfall": {"risk_range": (2500, 9999), "weight": 0.3}
            },
            "preventive_measures": [
                "Ensure excellent drainage",
                "Use disease-free seed rhizomes",
                "Practice crop rotation (3-4 years)",
                "Apply Trichoderma-based biocontrol",
                "Avoid waterlogging"
            ]
        },
        DiseaseRiskType.PHYTOPHTHORA: {
            "crops": [CropType.COCOA, CropType.OIL_PALM],
            "soil_factors": {
                "drainage": {"risk_classes": [DrainageClass.POORLY_DRAINED, DrainageClass.SOMEWHAT_POORLY_DRAINED], "weight": 0.35},
                "ph": {"risk_range": (0, 5.5), "weight": 0.15}
            },
            "climate_factors": {
                "rainfall": {"risk_range": (2000, 9999), "weight": 0.25},
                "humidity": {"risk_range": (85, 100), "weight": 0.25}
            },
            "preventive_measures": [
                "Improve drainage",
                "Prune to improve air circulation",
                "Remove infected plant material",
                "Apply copper-based fungicides preventively",
                "Use resistant varieties where available"
            ]
        },
        DiseaseRiskType.GANODERMA: {
            "crops": [CropType.OIL_PALM],
            "soil_factors": {
                "ph": {"risk_range": (0, 5.0), "weight": 0.2},
                "organic_matter": {"risk_range": (0, 2.0), "weight": 0.15}
            },
            "climate_factors": {
                "rainfall": {"risk_range": (2500, 9999), "weight": 0.2}
            },
            "history_factors": {
                "previous_palm": {"weight": 0.45}  # Previous oil palm increases risk
            },
            "preventive_measures": [
                "Remove all old palm debris before replanting",
                "Apply dolomite to raise pH",
                "Use Trichoderma-based biocontrol",
                "Maintain good drainage",
                "Regular monitoring and early removal of infected palms"
            ]
        }
    }
    
    def __init__(self, crop_type: CropType):
        self.crop_type = crop_type
    
    def assess_disease_risk(
        self,
        disease_type: DiseaseRiskType,
        sample: SoilSample,
        climate: Optional[ClimateData] = None,
        previous_crop: Optional[str] = None
    ) -> Optional[DiseaseRiskAssessment]:
        """Assess risk for a specific disease."""
        disease_info = self.DISEASE_FACTORS.get(disease_type)
        if not disease_info:
            return None
        
        if self.crop_type not in disease_info["crops"]:
            return None
        
        risk_score = 0
        total_weight = 0
        contributing_factors = []
        
        # Soil factors
        for factor, config in disease_info.get("soil_factors", {}).items():
            weight = config["weight"]
            total_weight += weight
            
            if factor == "drainage":
                if sample.drainage_class in config["risk_classes"]:
                    risk_score += weight
                    contributing_factors.append(f"Poor drainage ({sample.drainage_class.value})")
            
            elif factor == "ph":
                risk_range = config["risk_range"]
                if risk_range[0] <= sample.ph_water <= risk_range[1]:
                    risk_score += weight
                    contributing_factors.append(f"Low pH ({sample.ph_water:.1f})")
            
            elif factor == "organic_matter":
                risk_range = config["risk_range"]
                if risk_range[0] <= sample.organic_matter <= risk_range[1]:
                    risk_score += weight
                    contributing_factors.append(f"Low organic matter ({sample.organic_matter:.1f}%)")
        
        # Climate factors
        if climate:
            for factor, config in disease_info.get("climate_factors", {}).items():
                weight = config["weight"]
                total_weight += weight
                
                if factor == "rainfall":
                    risk_range = config["risk_range"]
                    if risk_range[0] <= climate.annual_rainfall <= risk_range[1]:
                        risk_score += weight
                        contributing_factors.append(f"High rainfall ({climate.annual_rainfall:.0f} mm)")
                
                elif factor == "humidity":
                    risk_range = config["risk_range"]
                    if risk_range[0] <= climate.relative_humidity <= risk_range[1]:
                        risk_score += weight
                        contributing_factors.append(f"High humidity ({climate.relative_humidity:.0f}%)")
        
        # History factors
        if previous_crop:
            for factor, config in disease_info.get("history_factors", {}).items():
                weight = config["weight"]
                total_weight += weight
                
                if factor == "previous_palm" and "palm" in previous_crop.lower():
                    risk_score += weight
                    contributing_factors.append("Previous oil palm cultivation")
        
        # Calculate probability
        probability = risk_score / total_weight if total_weight > 0 else 0
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = HazardSeverity.HIGH
        elif probability >= 0.5:
            risk_level = HazardSeverity.MODERATE
        elif probability >= 0.3:
            risk_level = HazardSeverity.LOW
        else:
            risk_level = HazardSeverity.NONE
        
        return DiseaseRiskAssessment(
            disease_type=disease_type,
            risk_level=risk_level,
            probability=probability,
            contributing_factors=contributing_factors,
            preventive_measures=disease_info.get("preventive_measures", []),
            monitoring_recommendations=[
                "Regular field inspections",
                "Monitor for early symptoms",
                "Keep records of disease incidence"
            ]
        )
    
    def assess_all_diseases(
        self,
        sample: SoilSample,
        climate: Optional[ClimateData] = None,
        previous_crop: Optional[str] = None
    ) -> List[DiseaseRiskAssessment]:
        """Assess all relevant disease risks."""
        assessments = []
        
        for disease_type in DiseaseRiskType:
            assessment = self.assess_disease_risk(disease_type, sample, climate, previous_crop)
            if assessment and assessment.risk_level != HazardSeverity.NONE:
                assessments.append(assessment)
        
        return assessments


class EconomicAnalyzer:
    """Economic analysis for soil remediation."""
    
    # Default costs (USD/kg or USD/ha)
    DEFAULT_COSTS = {
        "lime_per_kg": 0.08,
        "gypsum_per_kg": 0.10,
        "urea_per_kg": 0.50,
        "tsp_per_kg": 0.60,
        "kcl_per_kg": 0.55,
        "organic_matter_per_tonne": 30,
        "labor_per_ha": 50,
        "equipment_per_ha": 30
    }
    
    # Crop economics
    CROP_ECONOMICS = {
        CropType.OIL_PALM: {
            "price_per_kg": 0.15,  # FFB
            "base_yield_kg_ha": 18000,
            "yield_response_to_ph": 0.05,  # 5% per pH unit
            "yield_response_to_nutrients": 0.10
        },
        CropType.COCOA: {
            "price_per_kg": 2.50,  # Dry beans
            "base_yield_kg_ha": 1000,
            "yield_response_to_ph": 0.08,
            "yield_response_to_nutrients": 0.15
        },
        CropType.GINGER: {
            "price_per_kg": 0.80,  # Fresh rhizome
            "base_yield_kg_ha": 20000,
            "yield_response_to_ph": 0.06,
            "yield_response_to_nutrients": 0.12
        }
    }
    
    def __init__(
        self,
        crop_type: CropType,
        costs: Optional[Dict[str, float]] = None
    ):
        self.crop_type = crop_type
        self.costs = costs or self.DEFAULT_COSTS
        self.crop_economics = self.CROP_ECONOMICS.get(crop_type, self.CROP_ECONOMICS[CropType.OIL_PALM])
    
    def analyze_remediation_economics(
        self,
        sample: SoilSample,
        lime_kg_ha: float = 0,
        gypsum_kg_ha: float = 0,
        fertilizer_costs: Dict[str, float] = None,
        organic_matter_tonnes_ha: float = 0
    ) -> EconomicAnalysis:
        """Analyze economics of soil remediation."""
        analysis = EconomicAnalysis(
            sample_id=sample.sample_id,
            crop=self.crop_type.value
        )
        
        # Calculate amendment costs
        if lime_kg_ha > 0:
            analysis.amendment_costs["lime"] = lime_kg_ha * self.costs["lime_per_kg"]
        
        if gypsum_kg_ha > 0:
            analysis.amendment_costs["gypsum"] = gypsum_kg_ha * self.costs["gypsum_per_kg"]
        
        if organic_matter_tonnes_ha > 0:
            analysis.amendment_costs["organic_matter"] = organic_matter_tonnes_ha * self.costs["organic_matter_per_tonne"]
        
        # Fertilizer costs
        if fertilizer_costs:
            analysis.fertilizer_costs = fertilizer_costs
        
        # Labor and equipment
        analysis.labor_costs = self.costs["labor_per_ha"]
        analysis.equipment_costs = self.costs["equipment_per_ha"]
        
        # Total cost
        analysis.total_remediation_cost_per_ha = (
            sum(analysis.amendment_costs.values()) +
            sum(analysis.fertilizer_costs.values()) +
            analysis.labor_costs +
            analysis.equipment_costs
        )
        
        # Estimate yield impact
        base_yield = self.crop_economics["base_yield_kg_ha"]
        
        # Yield penalty from current conditions
        yield_penalty = 0
        
        # pH penalty
        optimal_ph = 6.0
        if sample.ph_water < optimal_ph:
            ph_gap = optimal_ph - sample.ph_water
            yield_penalty += ph_gap * self.crop_economics["yield_response_to_ph"]
        
        # Nutrient penalty (simplified)
        if sample.nitrogen_available < 30:
            yield_penalty += 0.1
        if sample.phosphorus_available < 10:
            yield_penalty += 0.1
        if sample.potassium_available < 100:
            yield_penalty += 0.1
        
        analysis.expected_yield_without_remediation = base_yield * (1 - min(yield_penalty, 0.5))
        
        # Expected yield with remediation
        remediation_benefit = min(yield_penalty * 0.7, 0.35)  # Can recover 70% of penalty
        analysis.expected_yield_with_remediation = base_yield * (1 - yield_penalty + remediation_benefit)
        
        # Yield increase
        yield_increase = analysis.expected_yield_with_remediation - analysis.expected_yield_without_remediation
        analysis.yield_increase_percent = (yield_increase / analysis.expected_yield_without_remediation) * 100
        
        # Revenue
        analysis.crop_price_per_kg = self.crop_economics["price_per_kg"]
        analysis.additional_revenue_per_ha = yield_increase * analysis.crop_price_per_kg
        
        # ROI
        analysis.net_benefit_per_ha = analysis.additional_revenue_per_ha - analysis.total_remediation_cost_per_ha
        
        if analysis.total_remediation_cost_per_ha > 0:
            analysis.benefit_cost_ratio = analysis.additional_revenue_per_ha / analysis.total_remediation_cost_per_ha
            if analysis.net_benefit_per_ha > 0:
                analysis.payback_period_years = analysis.total_remediation_cost_per_ha / analysis.additional_revenue_per_ha
            else:
                analysis.payback_period_years = float('inf')
        
        # Recommendation
        if analysis.benefit_cost_ratio >= 2.0:
            analysis.recommendation = "Highly recommended - excellent ROI"
            analysis.priority_rank = 1
        elif analysis.benefit_cost_ratio >= 1.5:
            analysis.recommendation = "Recommended - good ROI"
            analysis.priority_rank = 2
        elif analysis.benefit_cost_ratio >= 1.0:
            analysis.recommendation = "Consider - marginal ROI"
            analysis.priority_rank = 3
        else:
            analysis.recommendation = "Not recommended - poor ROI"
            analysis.priority_rank = 4
        
        return analysis


class SpatialInterpolator:
    """Spatial interpolation for continuous suitability surfaces."""
    
    def __init__(self, method: str = "idw"):
        """
        Initialize interpolator.
        
        Args:
            method: Interpolation method ('idw', 'kriging', 'rbf')
        """
        self.method = method
    
    def interpolate_to_grid(
        self,
        points: List[Tuple[float, float]],
        values: List[float],
        grid_resolution: float = 100,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Interpolate point values to a regular grid.
        
        Args:
            points: List of (x, y) coordinates
            values: List of values at each point
            grid_resolution: Grid cell size
            bounds: (xmin, ymin, xmax, ymax) or auto-detect
        
        Returns:
            Dictionary with grid, values, and metadata
        """
        if len(points) < 3:
            return {"error": "Need at least 3 points for interpolation"}
        
        points_array = np.array(points)
        values_array = np.array(values)
        
        # Determine bounds
        if bounds is None:
            xmin, ymin = points_array.min(axis=0)
            xmax, ymax = points_array.max(axis=0)
            # Add 10% buffer
            buffer_x = (xmax - xmin) * 0.1
            buffer_y = (ymax - ymin) * 0.1
            bounds = (xmin - buffer_x, ymin - buffer_y, xmax + buffer_x, ymax + buffer_y)
        
        xmin, ymin, xmax, ymax = bounds
        
        # Create grid
        x_grid = np.arange(xmin, xmax, grid_resolution)
        y_grid = np.arange(ymin, ymax, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Interpolate
        if self.method == "idw":
            grid_values = self._idw_interpolate(points_array, values_array, xx, yy)
        else:
            # Default to IDW
            grid_values = self._idw_interpolate(points_array, values_array, xx, yy)
        
        return {
            "x_grid": x_grid.tolist(),
            "y_grid": y_grid.tolist(),
            "values": grid_values.tolist(),
            "bounds": bounds,
            "resolution": grid_resolution,
            "method": self.method,
            "n_points": len(points)
        }
    
    def _idw_interpolate(
        self,
        points: np.ndarray,
        values: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        power: float = 2
    ) -> np.ndarray:
        """Inverse Distance Weighting interpolation."""
        grid_values = np.zeros_like(xx)
        
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x, y = xx[i, j], yy[i, j]
                
                # Calculate distances
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
                
                # Handle exact point matches
                if np.any(distances < 1e-10):
                    grid_values[i, j] = values[distances < 1e-10][0]
                else:
                    # IDW weights
                    weights = 1 / (distances ** power)
                    weights /= weights.sum()
                    grid_values[i, j] = np.sum(weights * values)
        
        return grid_values
    
    def cross_validate(
        self,
        points: List[Tuple[float, float]],
        values: List[float],
        n_folds: int = 5
    ) -> Dict[str, float]:
        """Leave-one-out cross-validation for interpolation accuracy."""
        points_array = np.array(points)
        values_array = np.array(values)
        n = len(points)
        
        errors = []
        
        for i in range(n):
            # Leave one out
            train_points = np.delete(points_array, i, axis=0)
            train_values = np.delete(values_array, i)
            test_point = points_array[i]
            test_value = values_array[i]
            
            # Predict
            distances = np.sqrt(np.sum((train_points - test_point)**2, axis=1))
            weights = 1 / (distances ** 2 + 1e-10)
            weights /= weights.sum()
            predicted = np.sum(weights * train_values)
            
            errors.append(test_value - predicted)
        
        errors = np.array(errors)
        
        return {
            "mae": np.mean(np.abs(errors)),
            "rmse": np.sqrt(np.mean(errors**2)),
            "bias": np.mean(errors),
            "r_squared": 1 - np.var(errors) / np.var(values_array)
        }


class AdvancedSoilSuitabilityPipeline(SoilSuitabilityPipeline):
    """
    Enhanced soil suitability pipeline with all advanced features.
    """
    
    def __init__(
        self,
        crop_types: List[CropType],
        project_name: str = "advanced_soil_assessment",
        coordinate_system: str = "meters"  # 'meters' or 'degrees'
    ):
        super().__init__(crop_types, project_name)
        
        self.coordinate_system = coordinate_system
        
        # Advanced assessors
        self.toxicity_assessors = {crop: ToxicityHazardAssessor(crop) for crop in crop_types}
        self.nutrient_calculators = {crop: NutrientBudgetCalculator(crop) for crop in crop_types}
        self.physical_assessors = {crop: PhysicalConstraintAssessor(crop) for crop in crop_types}
        self.disease_assessors = {crop: DiseaseRiskAssessor(crop) for crop in crop_types}
        self.economic_analyzers = {crop: EconomicAnalyzer(crop) for crop in crop_types}
        
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.water_balance_calculators = {crop: WaterBalanceCalculator(crop) for crop in crop_types}
        self.spatial_interpolator = SpatialInterpolator()
        
        # Physical constraints storage
        self.physical_constraints: Dict[str, SoilPhysicalConstraints] = {}
        
        # Required parameters per crop
        self.required_parameters = {
            CropType.OIL_PALM: ["ph", "texture", "drainage", "organic_matter", "potassium"],
            CropType.COCOA: ["ph", "texture", "drainage", "organic_matter", "nitrogen", "phosphorus"],
            CropType.GINGER: ["ph", "texture", "drainage", "organic_matter", "nitrogen", "potassium"]
        }
    
    def add_physical_constraints(self, constraints: SoilPhysicalConstraints) -> None:
        """Add physical constraints for a sample."""
        self.physical_constraints[constraints.sample_id] = constraints
    
    def _get_coordinate_tolerance(self, tolerance_meters: float) -> float:
        """Convert tolerance to appropriate units based on coordinate system."""
        if self.coordinate_system == "degrees":
            # Approximate conversion: 1 degree ~ 111km at equator
            return tolerance_meters / 111000
        return tolerance_meters
    
    def _find_matching_data(
        self,
        sample: SoilSample,
        data_list: List[Any],
        tolerance_meters: float
    ) -> Optional[Any]:
        """Find matching data with CRS-aware tolerance."""
        tolerance = self._get_coordinate_tolerance(tolerance_meters)
        
        for data in data_list:
            if abs(data.x - sample.x) < tolerance and abs(data.y - sample.y) < tolerance:
                return data
        return None
    
    def assess_sample_advanced(
        self,
        sample: SoilSample,
        climate: Optional[ClimateData] = None,
        topo: Optional[TopographyData] = None,
        previous_crop: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive sample assessment with all advanced features.
        """
        results = {
            "sample_id": sample.sample_id,
            "location": {"x": sample.x, "y": sample.y},
            "crops": {},
            "data_quality": {},
            "hazards": [],
            "disease_risks": []
        }
        
        # Get physical constraints if available
        constraints = self.physical_constraints.get(sample.sample_id)
        
        for crop in self.crop_types:
            crop_result = {
                "base_assessment": {},
                "toxicity_hazards": [],
                "nutrient_budgets": {},
                "physical_constraints": {},
                "water_balance": None,
                "disease_risks": [],
                "economic_analysis": None,
                "uncertainty": {},
                "final_suitability": None,
                "confidence_score": 0
            }
            
            # 1. Check data sufficiency
            required = self.required_parameters.get(crop, ["ph", "texture", "drainage"])
            data_check = self.uncertainty_quantifier.check_data_sufficiency(sample, required)
            crop_result["data_sufficiency"] = data_check
            
            if not data_check["sufficient"]:
                crop_result["warning"] = f"Missing parameters: {', '.join(data_check['missing_parameters'])}"
            
            # 2. Base suitability scoring
            scorer = self.scorers[crop]
            base_score = scorer.score_soil_sample(sample)
            crop_result["base_assessment"]["soil"] = base_score
            
            if climate:
                climate_score = scorer.score_climate(climate)
                crop_result["base_assessment"]["climate"] = climate_score
            
            if topo:
                topo_score = scorer.score_topography(topo)
                crop_result["base_assessment"]["topography"] = topo_score
            
            # 3. Toxicity hazard assessment
            toxicity_assessor = self.toxicity_assessors[crop]
            hazards = toxicity_assessor.assess_all_toxicities(sample)
            crop_result["toxicity_hazards"] = [
                {
                    "type": h.toxicity_type.value,
                    "severity": h.severity.value,
                    "value": h.current_value,
                    "threshold": h.threshold_value,
                    "unit": h.unit,
                    "override": h.override_suitability,
                    "mitigation": h.mitigation_strategy,
                    "cost": h.estimated_remediation_cost_per_ha
                }
                for h in hazards
            ]
            
            # 4. Nutrient budgets
            nutrient_calc = self.nutrient_calculators[crop]
            for nutrient in ["N", "P2O5", "K2O"]:
                budget = nutrient_calc.calculate_nutrient_budget(sample, nutrient)
                crop_result["nutrient_budgets"][nutrient] = {
                    "annual_requirement": budget.annual_requirement_kg_ha,
                    "soil_supply": budget.soil_supply_kg_ha,
                    "fertilizer_requirement": budget.fertilizer_requirement_kg_ha,
                    "recovery_efficiency": budget.recovery_efficiency,
                    "applications": budget.applications,
                    "leaching_risk": budget.leaching_risk
                }
            
            # 5. Physical constraints
            if constraints:
                physical_assessor = self.physical_assessors[crop]
                crop_result["physical_constraints"] = {
                    "compaction": physical_assessor.assess_compaction(sample, constraints),
                    "rooting_depth": physical_assessor.assess_rooting_depth(constraints),
                    "coarse_fragments": physical_assessor.assess_coarse_fragments(constraints),
                    "drainage": physical_assessor.assess_drainage_risk(sample, constraints)
                }
            
            # 6. Water balance
            if climate:
                water_calc = self.water_balance_calculators[crop]
                water_balance = water_calc.calculate_water_balance(climate)
                water_score = water_calc.score_water_availability(water_balance)
                crop_result["water_balance"] = {
                    "aridity_index": water_balance.aridity_index,
                    "drought_stress_days": water_balance.drought_stress_days,
                    "waterlogging_risk_months": water_balance.waterlogging_risk_months,
                    "irrigation_requirement_mm": water_balance.irrigation_requirement_mm,
                    "score": water_score
                }
            
            # 7. Disease risks
            disease_assessor = self.disease_assessors[crop]
            diseases = disease_assessor.assess_all_diseases(sample, climate, previous_crop)
            crop_result["disease_risks"] = [
                {
                    "disease": d.disease_type.value,
                    "risk_level": d.risk_level.value,
                    "probability": d.probability,
                    "factors": d.contributing_factors,
                    "prevention": d.preventive_measures
                }
                for d in diseases
            ]
            
            # 8. Uncertainty quantification
            param_uncertainties = []
            for param in ["ph_water", "organic_matter", "phosphorus_available", "potassium_available"]:
                value = getattr(sample, param, 0)
                if value > 0:
                    unc = self.uncertainty_quantifier.estimate_parameter_uncertainty(param, value)
                    param_uncertainties.append(unc)
            
            if param_uncertainties:
                score_uncertainty = self.uncertainty_quantifier.estimate_score_uncertainty(
                    param_uncertainties,
                    base_score.get("weights", {})
                )
                crop_result["uncertainty"] = {
                    "score_estimate": score_uncertainty.point_estimate,
                    "confidence_interval": [
                        score_uncertainty.confidence_interval_lower,
                        score_uncertainty.confidence_interval_upper
                    ],
                    "data_quality_score": score_uncertainty.data_quality_score
                }
                crop_result["confidence_score"] = score_uncertainty.data_quality_score * 100
            
            # 9. Determine final suitability with overrides
            base_suitability = base_score["overall_suitability"]
            
            # Check for toxicity overrides
            override_suitability = toxicity_assessor.get_override_suitability(hazards)
            if override_suitability and override_suitability.severity > base_suitability.severity:
                final_suitability = override_suitability
                crop_result["suitability_override_reason"] = "Toxicity hazard"
            else:
                final_suitability = base_suitability
            
            # Penalize for missing data
            if not data_check["sufficient"]:
                if final_suitability == SuitabilityClass.S1:
                    final_suitability = SuitabilityClass.S2
                crop_result["suitability_penalized"] = True
            
            crop_result["final_suitability"] = {
                "class": final_suitability.name,
                "description": final_suitability.value,
                "score": base_score["overall_score"]
            }
            
            # 10. Economic analysis
            recommender = self.recommenders[crop]
            lime_rec = recommender.calculate_lime_requirement(sample)
            fert_rec = recommender.calculate_fertilizer_requirements(sample)
            
            economic_analyzer = self.economic_analyzers[crop]
            fert_costs = {}
            for nutrient, rec in fert_rec.items():
                if "urea_kg_ha" in rec:
                    fert_costs["N"] = rec["urea_kg_ha"] * 0.50
                if "tsp_kg_ha" in rec:
                    fert_costs["P"] = rec["tsp_kg_ha"] * 0.60
                if "kcl_kg_ha" in rec:
                    fert_costs["K"] = rec["kcl_kg_ha"] * 0.55
            
            economic = economic_analyzer.analyze_remediation_economics(
                sample,
                lime_kg_ha=lime_rec.get("lime_required_kg_ha", 0),
                fertilizer_costs=fert_costs
            )
            
            crop_result["economic_analysis"] = {
                "total_cost_per_ha": economic.total_remediation_cost_per_ha,
                "expected_yield_increase_percent": economic.yield_increase_percent,
                "additional_revenue_per_ha": economic.additional_revenue_per_ha,
                "net_benefit_per_ha": economic.net_benefit_per_ha,
                "benefit_cost_ratio": economic.benefit_cost_ratio,
                "payback_years": economic.payback_period_years,
                "recommendation": economic.recommendation
            }
            
            results["crops"][crop.value] = crop_result
        
        return results
    
    def generate_suitability_maps(
        self,
        parameter: str = "overall_score",
        crop: Optional[CropType] = None,
        grid_resolution: float = 100
    ) -> Dict[str, Any]:
        """Generate interpolated suitability maps."""
        if not self.suitability_results:
            self.assess_all_samples()
        
        crop = crop or self.crop_types[0]
        
        points = []
        values = []
        
        for sample_id, result in self.suitability_results.items():
            if crop.value in result.get("crops", {}):
                crop_result = result["crops"][crop.value]
                
                x = result["location"]["x"]
                y = result["location"]["y"]
                
                if parameter == "overall_score":
                    value = crop_result.get("base_assessment", {}).get("soil", {}).get("overall_score", 50)
                else:
                    value = crop_result.get("base_assessment", {}).get("soil", {}).get("parameter_scores", {}).get(parameter, 50)
                
                points.append((x, y))
                values.append(value)
        
        if len(points) < 3:
            return {"error": "Need at least 3 samples for interpolation"}
        
        # Interpolate
        grid_result = self.spatial_interpolator.interpolate_to_grid(
            points, values, grid_resolution
        )
        
        # Cross-validation
        cv_result = self.spatial_interpolator.cross_validate(points, values)
        
        return {
            "parameter": parameter,
            "crop": crop.value,
            "grid": grid_result,
            "cross_validation": cv_result,
            "n_samples": len(points)
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive assessment report with all advanced features."""
        # First run advanced assessments
        advanced_results = {}
        
        for sample in self.soil_samples:
            climate = self._find_matching_data(sample, self.climate_data, 1000)
            topo = self._find_matching_data(sample, self.topography_data, 100)
            
            result = self.assess_sample_advanced(sample, climate, topo)
            advanced_results[sample.sample_id] = result
            self.suitability_results[sample.sample_id] = result
        
        report = {
            "project_name": self.project_name,
            "assessment_date": datetime.now().isoformat(),
            "sample_count": len(self.soil_samples),
            "crops_assessed": [c.value for c in self.crop_types],
            "coordinate_system": self.coordinate_system,
            "samples": list(advanced_results.values()),
            "summaries": {},
            "hazard_summary": {},
            "economic_summary": {},
            "recommendations_priority": []
        }
        
        # Generate summaries per crop
        for crop in self.crop_types:
            crop_samples = []
            total_cost = 0
            total_benefit = 0
            hazard_counts = {}
            
            for sample_id, result in advanced_results.items():
                if crop.value in result["crops"]:
                    crop_result = result["crops"][crop.value]
                    crop_samples.append({
                        "sample_id": sample_id,
                        "suitability": crop_result["final_suitability"]["class"],
                        "score": crop_result["final_suitability"]["score"],
                        "confidence": crop_result.get("confidence_score", 0)
                    })
                    
                    # Economic totals
                    econ = crop_result.get("economic_analysis", {})
                    total_cost += econ.get("total_cost_per_ha", 0)
                    total_benefit += econ.get("net_benefit_per_ha", 0)
                    
                    # Hazard counts
                    for hazard in crop_result.get("toxicity_hazards", []):
                        hazard_type = hazard["type"]
                        hazard_counts[hazard_type] = hazard_counts.get(hazard_type, 0) + 1
            
            # Sort by score
            crop_samples.sort(key=lambda x: x["score"], reverse=True)
            
            report["summaries"][crop.value] = {
                "sample_rankings": crop_samples,
                "suitability_distribution": {
                    "S1": sum(1 for s in crop_samples if s["suitability"] == "S1"),
                    "S2": sum(1 for s in crop_samples if s["suitability"] == "S2"),
                    "S3": sum(1 for s in crop_samples if s["suitability"] == "S3"),
                    "N1": sum(1 for s in crop_samples if s["suitability"] == "N1"),
                    "N2": sum(1 for s in crop_samples if s["suitability"] == "N2")
                },
                "mean_score": np.mean([s["score"] for s in crop_samples]) if crop_samples else 0,
                "mean_confidence": np.mean([s["confidence"] for s in crop_samples]) if crop_samples else 0
            }
            
            report["hazard_summary"][crop.value] = hazard_counts
            report["economic_summary"][crop.value] = {
                "total_remediation_cost": total_cost,
                "total_net_benefit": total_benefit,
                "average_cost_per_ha": total_cost / len(crop_samples) if crop_samples else 0,
                "average_benefit_per_ha": total_benefit / len(crop_samples) if crop_samples else 0
            }
        
        # Priority recommendations
        all_recommendations = []
        for sample_id, result in advanced_results.items():
            for crop_name, crop_result in result["crops"].items():
                econ = crop_result.get("economic_analysis", {})
                if econ.get("benefit_cost_ratio", 0) > 1:
                    all_recommendations.append({
                        "sample_id": sample_id,
                        "crop": crop_name,
                        "bcr": econ["benefit_cost_ratio"],
                        "net_benefit": econ["net_benefit_per_ha"],
                        "recommendation": econ["recommendation"]
                    })
        
        # Sort by BCR
        all_recommendations.sort(key=lambda x: x["bcr"], reverse=True)
        report["recommendations_priority"] = all_recommendations[:20]  # Top 20
        
        return report


def create_advanced_soil_pipeline(
    crops: List[str],
    project_name: str = "advanced_soil_assessment",
    coordinate_system: str = "meters"
) -> AdvancedSoilSuitabilityPipeline:
    """
    Factory function to create an advanced soil suitability pipeline.
    
    Args:
        crops: List of crop names ('oil_palm', 'cocoa', 'ginger')
        project_name: Project identifier
        coordinate_system: 'meters' or 'degrees'
    
    Returns:
        Configured AdvancedSoilSuitabilityPipeline
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
    
    return AdvancedSoilSuitabilityPipeline(crop_types, project_name, coordinate_system)
