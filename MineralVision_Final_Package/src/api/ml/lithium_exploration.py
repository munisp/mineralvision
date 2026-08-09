"""
Lithium Exploration Module for MineralVision Platform.

Comprehensive lithium exploration support for all deposit types:
- Pegmatite (LCT - Li-Cs-Ta, hard rock)
- Clay/Sedimentary (hectorite, jadarite)
- Brine (salars, geothermal)

Includes pathfinder element analysis, brine chemistry, hydrogeology integration,
and deposit-type-specific prospectivity modeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime


class LithiumDepositType(Enum):
    """Lithium deposit classification."""
    PEGMATITE_LCT = "pegmatite_lct"  # Li-Cs-Ta pegmatites (spodumene, petalite, lepidolite)
    PEGMATITE_NYF = "pegmatite_nyf"  # Nb-Y-F pegmatites (minor Li)
    CLAY_HECTORITE = "clay_hectorite"  # Hectorite clay deposits
    CLAY_JADARITE = "clay_jadarite"  # Jadarite deposits (Serbia-type)
    CLAY_SMECTITE = "clay_smectite"  # Li-bearing smectite clays
    BRINE_SALAR = "brine_salar"  # Salar/playa brine deposits
    BRINE_GEOTHERMAL = "brine_geothermal"  # Geothermal brine deposits
    BRINE_OILFIELD = "brine_oilfield"  # Oilfield brine deposits


class LithiumMineral(Enum):
    """Primary lithium-bearing minerals."""
    SPODUMENE = "spodumene"  # LiAlSi2O6, 8% Li2O
    PETALITE = "petalite"  # LiAlSi4O10, 4.5% Li2O
    LEPIDOLITE = "lepidolite"  # K(Li,Al)3(Si,Al)4O10(F,OH)2, 3-4% Li2O
    AMBLYGONITE = "amblygonite"  # LiAlPO4F, 10% Li2O
    EUCRYPTITE = "eucryptite"  # LiAlSiO4, 11.8% Li2O
    HECTORITE = "hectorite"  # Na0.3(Mg,Li)3Si4O10(OH)2
    JADARITE = "jadarite"  # LiNaSiB3O7(OH)
    BRINE = "brine"  # Dissolved Li in brines


class BrineType(Enum):
    """Brine classification for lithium extraction."""
    CONTINENTAL_SALAR = "continental_salar"  # Closed basin salars (Atacama, Uyuni)
    GEOTHERMAL = "geothermal"  # Geothermal fluids (Salton Sea)
    OILFIELD = "oilfield"  # Smackover, Leduc formations
    GROUNDWATER = "groundwater"  # Li-enriched groundwater


@dataclass
class BrineSample:
    """Brine sample data for lithium exploration."""
    sample_id: str
    x: float  # Easting/longitude
    y: float  # Northing/latitude
    z: float  # Depth (m below surface, negative)
    sample_date: datetime
    brine_type: BrineType
    
    # Major ions (mg/L)
    lithium: float = 0.0
    sodium: float = 0.0
    potassium: float = 0.0
    magnesium: float = 0.0
    calcium: float = 0.0
    chloride: float = 0.0
    sulfate: float = 0.0
    bicarbonate: float = 0.0
    boron: float = 0.0
    
    # Physical properties
    tds: float = 0.0  # Total dissolved solids (mg/L)
    density: float = 1.0  # g/cm3
    ph: float = 7.0
    temperature: float = 20.0  # Celsius
    conductivity: float = 0.0  # mS/cm
    
    # Isotopes (optional)
    delta_d: Optional[float] = None  # Deuterium
    delta_o18: Optional[float] = None  # Oxygen-18
    sr87_sr86: Optional[float] = None  # Strontium isotope ratio
    
    # Quality flags
    quality_flags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WellData:
    """Well/borehole data for hydrogeology."""
    well_id: str
    x: float
    y: float
    surface_elevation: float  # m
    total_depth: float  # m
    well_type: str  # production, monitoring, exploration
    
    # Aquifer properties
    screened_intervals: List[Tuple[float, float]] = field(default_factory=list)  # (top, bottom) m
    static_water_level: float = 0.0  # m below surface
    transmissivity: float = 0.0  # m2/day
    storativity: float = 0.0  # dimensionless
    hydraulic_conductivity: float = 0.0  # m/day
    
    # Production data
    pump_rate: float = 0.0  # L/s
    drawdown: float = 0.0  # m
    
    # Lithology log
    lithology_log: List[Dict[str, Any]] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PegmatiteSample:
    """Pegmatite rock/soil sample for lithium exploration."""
    sample_id: str
    x: float
    y: float
    z: float
    sample_type: str  # rock, soil, stream_sediment, drill_core
    
    # Lithium and pathfinders (ppm unless noted)
    li: float = 0.0
    cs: float = 0.0
    rb: float = 0.0
    ta: float = 0.0
    nb: float = 0.0
    sn: float = 0.0
    be: float = 0.0
    b: float = 0.0
    f: float = 0.0  # Fluorine
    p: float = 0.0  # Phosphorus
    
    # Major elements (%)
    k: float = 0.0
    na: float = 0.0
    al: float = 0.0
    si: float = 0.0
    fe: float = 0.0
    
    # Ratios (computed)
    k_rb_ratio: Optional[float] = None
    nb_ta_ratio: Optional[float] = None
    
    # Mineralogy
    minerals_identified: List[str] = field(default_factory=list)
    li2o_percent: float = 0.0  # Li2O grade
    
    quality_flags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class LithiumPathfinderElements:
    """Pathfinder element associations for lithium deposit types."""
    
    # Pathfinder associations by deposit type
    PATHFINDERS: Dict[LithiumDepositType, Dict[str, Dict[str, float]]] = {
        LithiumDepositType.PEGMATITE_LCT: {
            "primary": {"Li": 100, "Cs": 50, "Ta": 20, "Rb": 100, "Sn": 50},
            "secondary": {"Nb": 20, "Be": 10, "B": 50, "F": 500, "P": 500},
            "ratios": {"K_Rb": 50, "Nb_Ta": 5}  # Low K/Rb indicates fractionation
        },
        LithiumDepositType.PEGMATITE_NYF: {
            "primary": {"Nb": 100, "Y": 50, "F": 1000, "Li": 50},
            "secondary": {"REE": 100, "Zr": 100, "Th": 20, "U": 10},
            "ratios": {}
        },
        LithiumDepositType.CLAY_HECTORITE: {
            "primary": {"Li": 500, "Mg": 10000, "F": 500},
            "secondary": {"B": 100, "Na": 5000, "K": 1000},
            "ratios": {"Mg_Li": 20}
        },
        LithiumDepositType.CLAY_SMECTITE: {
            "primary": {"Li": 200, "Mg": 5000},
            "secondary": {"K": 2000, "Na": 3000, "Ca": 2000},
            "ratios": {}
        },
        LithiumDepositType.BRINE_SALAR: {
            "primary": {"Li": 200, "K": 5000, "Mg": 5000, "B": 500},
            "secondary": {"Na": 50000, "Cl": 100000, "SO4": 10000},
            "ratios": {"Mg_Li": 10, "K_Li": 50}  # Low Mg/Li is favorable
        },
        LithiumDepositType.BRINE_GEOTHERMAL: {
            "primary": {"Li": 100, "K": 10000, "B": 200},
            "secondary": {"Na": 30000, "Cl": 50000, "SiO2": 500},
            "ratios": {"K_Li": 100}
        },
        LithiumDepositType.BRINE_OILFIELD: {
            "primary": {"Li": 50, "Br": 500, "Sr": 500},
            "secondary": {"Na": 50000, "Cl": 100000, "Ca": 10000},
            "ratios": {"Br_Cl": 0.003}
        }
    }
    
    # Anomaly thresholds (ppm for solids, mg/L for brines)
    THRESHOLDS: Dict[str, Dict[str, float]] = {
        "pegmatite_rock": {
            "Li": 100, "Cs": 20, "Rb": 500, "Ta": 10, "Nb": 50,
            "Sn": 20, "Be": 5, "B": 20, "F": 1000
        },
        "pegmatite_soil": {
            "Li": 50, "Cs": 10, "Rb": 200, "Ta": 5, "Nb": 20,
            "Sn": 10, "Be": 2, "B": 10
        },
        "clay_rock": {
            "Li": 300, "Mg": 50000, "F": 500, "B": 50
        },
        "brine": {
            "Li": 100, "K": 5000, "Mg": 2000, "B": 100
        }
    }
    
    def __init__(self, deposit_type: LithiumDepositType):
        self.deposit_type = deposit_type
        self.pathfinders = self.PATHFINDERS.get(deposit_type, {})
    
    def compute_pathfinder_score(
        self,
        sample: Union[PegmatiteSample, BrineSample],
        sample_type: str = "rock"
    ) -> float:
        """
        Compute pathfinder score for a sample.
        
        Returns score 0-100 based on pathfinder element enrichment.
        """
        if isinstance(sample, BrineSample):
            thresholds = self.THRESHOLDS["brine"]
            elements = {
                "Li": sample.lithium,
                "K": sample.potassium,
                "Mg": sample.magnesium,
                "B": sample.boron,
                "Na": sample.sodium
            }
        else:
            threshold_key = f"pegmatite_{sample_type}" if "pegmatite" in self.deposit_type.value else "clay_rock"
            thresholds = self.THRESHOLDS.get(threshold_key, self.THRESHOLDS["pegmatite_rock"])
            elements = {
                "Li": sample.li,
                "Cs": sample.cs,
                "Rb": sample.rb,
                "Ta": sample.ta,
                "Nb": sample.nb,
                "Sn": sample.sn,
                "Be": sample.be,
                "B": sample.b
            }
        
        scores = []
        primary = self.pathfinders.get("primary", {})
        
        for element, value in elements.items():
            if element in thresholds and thresholds[element] > 0:
                # Score based on enrichment above threshold
                enrichment = value / thresholds[element]
                if enrichment > 1:
                    # Log scale for enrichment
                    score = min(100, 50 + 50 * np.log10(enrichment))
                else:
                    score = 50 * enrichment
                
                # Weight by pathfinder importance
                weight = primary.get(element, 10) / 100
                scores.append(score * weight)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_fractionation_index(self, sample: PegmatiteSample) -> float:
        """
        Compute pegmatite fractionation index.
        
        Lower K/Rb ratio indicates more evolved (Li-rich) pegmatite.
        """
        if sample.rb > 0 and sample.k > 0:
            k_rb = (sample.k * 10000) / sample.rb  # K in %, Rb in ppm
            # Highly fractionated pegmatites have K/Rb < 50
            if k_rb < 20:
                return 100.0
            elif k_rb < 50:
                return 80.0 - (k_rb - 20) * 2
            elif k_rb < 100:
                return 40.0 - (k_rb - 50) * 0.4
            else:
                return max(0, 20 - (k_rb - 100) * 0.1)
        return 0.0
    
    def compute_mg_li_ratio(self, sample: BrineSample) -> Tuple[float, str]:
        """
        Compute Mg/Li ratio for brine quality assessment.
        
        Lower Mg/Li is more favorable for extraction.
        """
        if sample.lithium > 0:
            mg_li = sample.magnesium / sample.lithium
            
            if mg_li < 3:
                return mg_li, "excellent"
            elif mg_li < 6:
                return mg_li, "good"
            elif mg_li < 10:
                return mg_li, "moderate"
            elif mg_li < 20:
                return mg_li, "challenging"
            else:
                return mg_li, "difficult"
        return float('inf'), "no_lithium"


class BrineChemistry:
    """Brine chemistry analysis for lithium exploration."""
    
    # Reference brine compositions (mg/L)
    REFERENCE_BRINES: Dict[str, Dict[str, float]] = {
        "atacama": {"Li": 1500, "K": 23000, "Mg": 9500, "Na": 91000, "Cl": 160000, "SO4": 16000, "B": 400},
        "uyuni": {"Li": 350, "K": 7200, "Mg": 6500, "Na": 80000, "Cl": 130000, "SO4": 8500, "B": 300},
        "hombre_muerto": {"Li": 650, "K": 6100, "Mg": 1100, "Na": 100000, "Cl": 160000, "SO4": 7500, "B": 1200},
        "salton_sea": {"Li": 200, "K": 16000, "Mg": 50, "Na": 50000, "Cl": 130000, "SO4": 100, "B": 300},
        "smackover": {"Li": 150, "K": 3000, "Mg": 2000, "Na": 80000, "Cl": 150000, "SO4": 500, "B": 50}
    }
    
    def __init__(self):
        pass
    
    def classify_brine_type(self, sample: BrineSample) -> str:
        """Classify brine based on chemistry."""
        # Na-Cl dominant
        if sample.chloride > 50000 and sample.sodium > 30000:
            if sample.sulfate > 5000:
                return "Na-Cl-SO4"
            else:
                return "Na-Cl"
        # Ca-Cl (oilfield type)
        elif sample.calcium > 5000 and sample.chloride > 50000:
            return "Ca-Cl"
        # Mixed
        else:
            return "mixed"
    
    def compute_evaporation_index(self, sample: BrineSample) -> float:
        """
        Compute evaporation concentration index.
        
        Higher values indicate more concentrated brine.
        """
        # Use Cl as conservative tracer
        seawater_cl = 19000  # mg/L
        if sample.chloride > 0:
            return sample.chloride / seawater_cl
        return 0.0
    
    def compute_lithium_enrichment(self, sample: BrineSample) -> float:
        """
        Compute lithium enrichment factor relative to seawater.
        """
        seawater_li = 0.17  # mg/L
        if sample.lithium > 0:
            return sample.lithium / seawater_li
        return 0.0
    
    def estimate_resource_potential(
        self,
        samples: List[BrineSample],
        aquifer_area_km2: float,
        aquifer_thickness_m: float,
        porosity: float = 0.3,
        specific_yield: float = 0.1
    ) -> Dict[str, float]:
        """
        Estimate lithium resource potential from brine samples.
        
        Returns resource estimates in tonnes Li and tonnes LCE.
        """
        if not samples:
            return {"tonnes_li": 0, "tonnes_lce": 0, "avg_li_mg_l": 0}
        
        # Average lithium concentration
        avg_li = np.mean([s.lithium for s in samples])
        avg_density = np.mean([s.density for s in samples])
        
        # Volume calculation
        volume_m3 = aquifer_area_km2 * 1e6 * aquifer_thickness_m * porosity
        brine_volume_l = volume_m3 * 1000 * specific_yield
        
        # Lithium mass
        li_kg = (avg_li / 1000) * brine_volume_l * avg_density
        li_tonnes = li_kg / 1000
        
        # LCE conversion (Li2CO3 equivalent)
        lce_tonnes = li_tonnes * 5.323
        
        return {
            "tonnes_li": li_tonnes,
            "tonnes_lce": lce_tonnes,
            "avg_li_mg_l": avg_li,
            "brine_volume_m3": volume_m3,
            "recoverable_volume_l": brine_volume_l
        }
    
    def assess_extraction_feasibility(self, sample: BrineSample) -> Dict[str, Any]:
        """
        Assess lithium extraction feasibility based on brine chemistry.
        """
        mg_li = sample.magnesium / sample.lithium if sample.lithium > 0 else float('inf')
        ca_li = sample.calcium / sample.lithium if sample.lithium > 0 else float('inf')
        so4_li = sample.sulfate / sample.lithium if sample.lithium > 0 else float('inf')
        
        # Scoring
        scores = {}
        
        # Lithium grade score
        if sample.lithium >= 1000:
            scores["grade"] = 100
        elif sample.lithium >= 500:
            scores["grade"] = 80
        elif sample.lithium >= 200:
            scores["grade"] = 60
        elif sample.lithium >= 100:
            scores["grade"] = 40
        else:
            scores["grade"] = 20
        
        # Mg/Li ratio score (lower is better)
        if mg_li < 3:
            scores["mg_li"] = 100
        elif mg_li < 6:
            scores["mg_li"] = 80
        elif mg_li < 10:
            scores["mg_li"] = 60
        elif mg_li < 20:
            scores["mg_li"] = 40
        else:
            scores["mg_li"] = 20
        
        # Impurity score
        impurity_score = 100
        if ca_li > 50:
            impurity_score -= 20
        if so4_li > 100:
            impurity_score -= 20
        if sample.boron > 500:
            impurity_score -= 10
        scores["impurities"] = max(0, impurity_score)
        
        # Overall feasibility
        overall = np.mean(list(scores.values()))
        
        # Recommended extraction method
        if mg_li < 6 and sample.lithium > 200:
            method = "solar_evaporation"
        elif mg_li < 15:
            method = "direct_lithium_extraction"
        else:
            method = "dle_with_pretreatment"
        
        return {
            "scores": scores,
            "overall_feasibility": overall,
            "recommended_method": method,
            "mg_li_ratio": mg_li,
            "ca_li_ratio": ca_li,
            "challenges": self._identify_challenges(sample, mg_li, ca_li, so4_li)
        }
    
    def _identify_challenges(
        self,
        sample: BrineSample,
        mg_li: float,
        ca_li: float,
        so4_li: float
    ) -> List[str]:
        """Identify extraction challenges."""
        challenges = []
        
        if mg_li > 10:
            challenges.append("High Mg/Li ratio requires selective extraction")
        if ca_li > 50:
            challenges.append("High calcium may cause scaling")
        if so4_li > 100:
            challenges.append("High sulfate complicates evaporation")
        if sample.boron > 500:
            challenges.append("High boron requires removal for battery-grade")
        if sample.lithium < 100:
            challenges.append("Low lithium grade increases processing costs")
        if sample.tds > 300000:
            challenges.append("Very high TDS may require dilution")
        
        return challenges


class HydrogeologyModel:
    """Hydrogeology modeling for brine lithium exploration."""
    
    def __init__(self):
        pass
    
    def estimate_aquifer_properties(
        self,
        wells: List[WellData]
    ) -> Dict[str, float]:
        """
        Estimate regional aquifer properties from well data.
        """
        if not wells:
            return {}
        
        transmissivities = [w.transmissivity for w in wells if w.transmissivity > 0]
        storativities = [w.storativity for w in wells if w.storativity > 0]
        conductivities = [w.hydraulic_conductivity for w in wells if w.hydraulic_conductivity > 0]
        
        return {
            "avg_transmissivity": np.mean(transmissivities) if transmissivities else 0,
            "avg_storativity": np.mean(storativities) if storativities else 0,
            "avg_conductivity": np.mean(conductivities) if conductivities else 0,
            "well_count": len(wells)
        }
    
    def compute_recharge_rate(
        self,
        precipitation_mm_yr: float,
        evaporation_mm_yr: float,
        runoff_coefficient: float = 0.1,
        infiltration_coefficient: float = 0.05
    ) -> float:
        """
        Estimate groundwater recharge rate (mm/yr).
        """
        net_precipitation = precipitation_mm_yr - evaporation_mm_yr
        if net_precipitation > 0:
            return net_precipitation * infiltration_coefficient
        else:
            # Arid climate - minimal recharge
            return precipitation_mm_yr * 0.01
    
    def estimate_brine_residence_time(
        self,
        aquifer_volume_m3: float,
        recharge_rate_m3_yr: float
    ) -> float:
        """
        Estimate brine residence time in years.
        
        Longer residence times allow more lithium concentration.
        """
        if recharge_rate_m3_yr > 0:
            return aquifer_volume_m3 / recharge_rate_m3_yr
        return float('inf')
    
    def model_brine_evolution(
        self,
        initial_li_mg_l: float,
        evaporation_factor: float,
        years: int = 1000
    ) -> List[float]:
        """
        Model brine lithium concentration evolution through evaporation.
        """
        concentrations = [initial_li_mg_l]
        current = initial_li_mg_l
        
        annual_factor = evaporation_factor ** (1 / years)
        
        for _ in range(years):
            current *= annual_factor
            # Cap at saturation (approximately)
            current = min(current, 5000)
            concentrations.append(current)
        
        return concentrations


class ClayLithiumAnalysis:
    """Analysis for clay-hosted lithium deposits."""
    
    # Clay mineral Li content ranges (ppm)
    CLAY_LI_CONTENT: Dict[str, Tuple[float, float]] = {
        "hectorite": (500, 3000),
        "smectite": (100, 1000),
        "illite": (50, 500),
        "kaolinite": (10, 100),
        "montmorillonite": (50, 500)
    }
    
    def __init__(self):
        pass
    
    def estimate_clay_li_grade(
        self,
        clay_mineralogy: Dict[str, float],  # mineral: percentage
        bulk_li_ppm: float
    ) -> Dict[str, Any]:
        """
        Estimate lithium distribution in clay minerals.
        """
        results = {
            "bulk_li_ppm": bulk_li_ppm,
            "mineral_contributions": {},
            "dominant_li_host": None,
            "estimated_li2o_percent": bulk_li_ppm * 2.153 / 10000
        }
        
        max_contribution = 0
        for mineral, percentage in clay_mineralogy.items():
            if mineral.lower() in self.CLAY_LI_CONTENT:
                min_li, max_li = self.CLAY_LI_CONTENT[mineral.lower()]
                avg_li = (min_li + max_li) / 2
                contribution = (percentage / 100) * avg_li
                results["mineral_contributions"][mineral] = contribution
                
                if contribution > max_contribution:
                    max_contribution = contribution
                    results["dominant_li_host"] = mineral
        
        return results
    
    def assess_extraction_amenability(
        self,
        clay_type: str,
        li_grade_ppm: float,
        impurities: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assess clay lithium extraction amenability.
        """
        scores = {}
        
        # Grade score
        if li_grade_ppm >= 2000:
            scores["grade"] = 100
        elif li_grade_ppm >= 1000:
            scores["grade"] = 80
        elif li_grade_ppm >= 500:
            scores["grade"] = 60
        else:
            scores["grade"] = 40
        
        # Clay type score (hectorite most amenable)
        clay_scores = {
            "hectorite": 100,
            "smectite": 70,
            "montmorillonite": 60,
            "illite": 40,
            "kaolinite": 30
        }
        scores["clay_type"] = clay_scores.get(clay_type.lower(), 50)
        
        # Impurity score
        impurity_score = 100
        if impurities.get("Fe", 0) > 5:
            impurity_score -= 20
        if impurities.get("Ca", 0) > 10:
            impurity_score -= 15
        if impurities.get("Mg", 0) > 15:
            impurity_score -= 15
        scores["impurities"] = max(0, impurity_score)
        
        overall = np.mean(list(scores.values()))
        
        # Recommended process
        if clay_type.lower() == "hectorite":
            process = "acid_leach"
        elif clay_type.lower() in ["smectite", "montmorillonite"]:
            process = "roast_leach"
        else:
            process = "sulfation_roast"
        
        return {
            "scores": scores,
            "overall_amenability": overall,
            "recommended_process": process
        }


class LithiumDepositPriors:
    """Geological priors for lithium deposit types."""
    
    PRIORS: Dict[LithiumDepositType, Dict[str, Any]] = {
        LithiumDepositType.PEGMATITE_LCT: {
            "host_rocks": ["granite", "gneiss", "schist", "metasediment"],
            "tectonic_setting": ["orogenic", "post_orogenic", "anorogenic"],
            "age_preference": ["archean", "proterozoic", "paleozoic"],
            "structural_controls": ["shear_zone", "fold_hinge", "contact"],
            "depth_km": (0, 5),
            "favorable_indicators": [
                "fractionated_granite_nearby",
                "pegmatite_swarm",
                "regional_metamorphism",
                "structural_complexity"
            ]
        },
        LithiumDepositType.CLAY_HECTORITE: {
            "host_rocks": ["volcanic_tuff", "lacustrine_sediment", "altered_volcanic"],
            "tectonic_setting": ["extensional", "basin"],
            "age_preference": ["cenozoic", "mesozoic"],
            "structural_controls": ["basin_margin", "caldera"],
            "depth_km": (0, 0.5),
            "favorable_indicators": [
                "volcanic_activity",
                "closed_basin",
                "arid_climate",
                "hot_springs"
            ]
        },
        LithiumDepositType.BRINE_SALAR: {
            "host_rocks": ["evaporite", "alluvium", "volcanic"],
            "tectonic_setting": ["extensional", "foreland", "intramontane"],
            "age_preference": ["quaternary", "neogene"],
            "structural_controls": ["graben", "half_graben", "closed_basin"],
            "depth_km": (0, 0.3),
            "favorable_indicators": [
                "closed_drainage",
                "arid_climate",
                "volcanic_input",
                "geothermal_activity",
                "long_residence_time"
            ]
        },
        LithiumDepositType.BRINE_GEOTHERMAL: {
            "host_rocks": ["volcanic", "sedimentary", "metamorphic"],
            "tectonic_setting": ["extensional", "volcanic_arc", "rift"],
            "age_preference": ["quaternary"],
            "structural_controls": ["fault", "caldera", "volcanic_center"],
            "depth_km": (0.5, 3),
            "favorable_indicators": [
                "active_geothermal",
                "high_heat_flow",
                "volcanic_activity",
                "deep_circulation"
            ]
        }
    }
    
    def __init__(self, deposit_type: LithiumDepositType):
        self.deposit_type = deposit_type
        self.priors = self.PRIORS.get(deposit_type, {})
    
    def compute_prior(
        self,
        host_rock: str,
        tectonic_setting: str,
        age: str,
        structural_features: List[str],
        indicators: List[str]
    ) -> float:
        """
        Compute prior probability based on geological setting.
        
        Returns probability 0-1.
        """
        score = 0.0
        max_score = 5.0
        
        # Host rock match
        if host_rock.lower() in [r.lower() for r in self.priors.get("host_rocks", [])]:
            score += 1.0
        
        # Tectonic setting match
        if tectonic_setting.lower() in [t.lower() for t in self.priors.get("tectonic_setting", [])]:
            score += 1.0
        
        # Age preference match
        if age.lower() in [a.lower() for a in self.priors.get("age_preference", [])]:
            score += 1.0
        
        # Structural controls
        structural_controls = self.priors.get("structural_controls", [])
        structural_match = sum(1 for s in structural_features if s.lower() in [c.lower() for c in structural_controls])
        score += min(1.0, structural_match / max(1, len(structural_controls)))
        
        # Favorable indicators
        favorable = self.priors.get("favorable_indicators", [])
        indicator_match = sum(1 for i in indicators if i.lower() in [f.lower() for f in favorable])
        score += min(1.0, indicator_match / max(1, len(favorable)))
        
        return score / max_score


class LithiumExplorationPipeline:
    """
    Complete lithium exploration pipeline integrating all modules.
    """
    
    def __init__(
        self,
        deposit_type: LithiumDepositType,
        project_name: str = "lithium_exploration"
    ):
        self.deposit_type = deposit_type
        self.project_name = project_name
        
        # Initialize components
        self.pathfinders = LithiumPathfinderElements(deposit_type)
        self.priors = LithiumDepositPriors(deposit_type)
        
        # Brine-specific components
        if "brine" in deposit_type.value:
            self.brine_chemistry = BrineChemistry()
            self.hydrogeology = HydrogeologyModel()
        else:
            self.brine_chemistry = None
            self.hydrogeology = None
        
        # Clay-specific components
        if "clay" in deposit_type.value:
            self.clay_analysis = ClayLithiumAnalysis()
        else:
            self.clay_analysis = None
        
        # Data storage
        self.pegmatite_samples: List[PegmatiteSample] = []
        self.brine_samples: List[BrineSample] = []
        self.wells: List[WellData] = []
        
        # Results
        self.pathfinder_scores: Dict[str, float] = {}
        self.targets: List[Dict[str, Any]] = []
    
    def add_pegmatite_sample(self, sample: PegmatiteSample) -> None:
        """Add a pegmatite/rock sample."""
        # Compute K/Rb ratio if not set
        if sample.k_rb_ratio is None and sample.rb > 0:
            sample.k_rb_ratio = (sample.k * 10000) / sample.rb
        
        # Compute Nb/Ta ratio if not set
        if sample.nb_ta_ratio is None and sample.ta > 0:
            sample.nb_ta_ratio = sample.nb / sample.ta
        
        self.pegmatite_samples.append(sample)
    
    def add_brine_sample(self, sample: BrineSample) -> None:
        """Add a brine sample."""
        self.brine_samples.append(sample)
    
    def add_well(self, well: WellData) -> None:
        """Add well data."""
        self.wells.append(well)
    
    def process_samples(self) -> Dict[str, Any]:
        """
        Process all samples and compute pathfinder scores.
        """
        results = {
            "sample_count": 0,
            "pathfinder_scores": {},
            "anomalies": [],
            "statistics": {}
        }
        
        if "brine" in self.deposit_type.value and self.brine_samples:
            results["sample_count"] = len(self.brine_samples)
            
            for sample in self.brine_samples:
                score = self.pathfinders.compute_pathfinder_score(sample)
                self.pathfinder_scores[sample.sample_id] = score
                results["pathfinder_scores"][sample.sample_id] = score
                
                if score > 70:
                    results["anomalies"].append({
                        "sample_id": sample.sample_id,
                        "x": sample.x,
                        "y": sample.y,
                        "score": score,
                        "li_mg_l": sample.lithium
                    })
            
            # Statistics
            li_values = [s.lithium for s in self.brine_samples]
            results["statistics"] = {
                "li_mean": np.mean(li_values),
                "li_max": np.max(li_values),
                "li_min": np.min(li_values),
                "li_std": np.std(li_values)
            }
        
        elif self.pegmatite_samples:
            results["sample_count"] = len(self.pegmatite_samples)
            
            for sample in self.pegmatite_samples:
                score = self.pathfinders.compute_pathfinder_score(sample, sample.sample_type)
                frac_index = self.pathfinders.compute_fractionation_index(sample)
                
                combined_score = 0.7 * score + 0.3 * frac_index
                self.pathfinder_scores[sample.sample_id] = combined_score
                results["pathfinder_scores"][sample.sample_id] = combined_score
                
                if combined_score > 60:
                    results["anomalies"].append({
                        "sample_id": sample.sample_id,
                        "x": sample.x,
                        "y": sample.y,
                        "score": combined_score,
                        "li_ppm": sample.li,
                        "k_rb_ratio": sample.k_rb_ratio
                    })
            
            # Statistics
            li_values = [s.li for s in self.pegmatite_samples]
            results["statistics"] = {
                "li_mean": np.mean(li_values),
                "li_max": np.max(li_values),
                "li_min": np.min(li_values),
                "li_std": np.std(li_values)
            }
        
        return results
    
    def assess_brine_resources(
        self,
        aquifer_area_km2: float,
        aquifer_thickness_m: float,
        porosity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Assess brine lithium resources.
        """
        if not self.brine_chemistry or not self.brine_samples:
            return {"error": "No brine data available"}
        
        # Resource estimate
        resource = self.brine_chemistry.estimate_resource_potential(
            self.brine_samples,
            aquifer_area_km2,
            aquifer_thickness_m,
            porosity
        )
        
        # Extraction feasibility for each sample
        feasibility_results = []
        for sample in self.brine_samples:
            feasibility = self.brine_chemistry.assess_extraction_feasibility(sample)
            feasibility_results.append({
                "sample_id": sample.sample_id,
                **feasibility
            })
        
        # Average feasibility
        avg_feasibility = np.mean([f["overall_feasibility"] for f in feasibility_results])
        
        return {
            "resource_estimate": resource,
            "feasibility_assessments": feasibility_results,
            "average_feasibility": avg_feasibility,
            "recommended_method": max(
                set(f["recommended_method"] for f in feasibility_results),
                key=lambda x: sum(1 for f in feasibility_results if f["recommended_method"] == x)
            )
        }
    
    def generate_targets(
        self,
        min_score: float = 50.0,
        max_targets: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked exploration targets.
        """
        targets = []
        
        # From pathfinder scores
        for sample_id, score in self.pathfinder_scores.items():
            if score >= min_score:
                # Find sample location
                sample = None
                for s in self.pegmatite_samples:
                    if s.sample_id == sample_id:
                        sample = s
                        break
                for s in self.brine_samples:
                    if s.sample_id == sample_id:
                        sample = s
                        break
                
                if sample:
                    target = {
                        "target_id": f"T_{sample_id}",
                        "x": sample.x,
                        "y": sample.y,
                        "score": score,
                        "source": "pathfinder_analysis",
                        "deposit_type": self.deposit_type.value
                    }
                    
                    if isinstance(sample, BrineSample):
                        target["li_mg_l"] = sample.lithium
                        target["mg_li_ratio"] = sample.magnesium / sample.lithium if sample.lithium > 0 else None
                    else:
                        target["li_ppm"] = sample.li
                        target["k_rb_ratio"] = sample.k_rb_ratio
                    
                    targets.append(target)
        
        # Sort by score
        targets.sort(key=lambda x: x["score"], reverse=True)
        self.targets = targets[:max_targets]
        
        return self.targets
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive exploration report.
        """
        report = {
            "project_name": self.project_name,
            "deposit_type": self.deposit_type.value,
            "sample_summary": {
                "pegmatite_samples": len(self.pegmatite_samples),
                "brine_samples": len(self.brine_samples),
                "wells": len(self.wells)
            },
            "pathfinder_analysis": self.process_samples(),
            "targets": self.targets if self.targets else self.generate_targets()
        }
        
        # Add brine-specific analysis
        if self.brine_chemistry and self.brine_samples:
            # Use default aquifer parameters if not specified
            report["brine_analysis"] = {
                "chemistry_classification": [
                    {
                        "sample_id": s.sample_id,
                        "type": self.brine_chemistry.classify_brine_type(s),
                        "evaporation_index": self.brine_chemistry.compute_evaporation_index(s),
                        "li_enrichment": self.brine_chemistry.compute_lithium_enrichment(s)
                    }
                    for s in self.brine_samples
                ]
            }
        
        # Add hydrogeology if wells available
        if self.hydrogeology and self.wells:
            report["hydrogeology"] = self.hydrogeology.estimate_aquifer_properties(self.wells)
        
        return report


def create_lithium_exploration_pipeline(
    deposit_type: str,
    project_name: str = "lithium_exploration"
) -> LithiumExplorationPipeline:
    """
    Factory function to create a lithium exploration pipeline.
    
    Args:
        deposit_type: One of 'pegmatite', 'clay', 'brine_salar', 'brine_geothermal'
        project_name: Project identifier
    
    Returns:
        Configured LithiumExplorationPipeline
    """
    type_mapping = {
        "pegmatite": LithiumDepositType.PEGMATITE_LCT,
        "pegmatite_lct": LithiumDepositType.PEGMATITE_LCT,
        "pegmatite_nyf": LithiumDepositType.PEGMATITE_NYF,
        "clay": LithiumDepositType.CLAY_HECTORITE,
        "clay_hectorite": LithiumDepositType.CLAY_HECTORITE,
        "clay_smectite": LithiumDepositType.CLAY_SMECTITE,
        "brine": LithiumDepositType.BRINE_SALAR,
        "brine_salar": LithiumDepositType.BRINE_SALAR,
        "brine_geothermal": LithiumDepositType.BRINE_GEOTHERMAL,
        "brine_oilfield": LithiumDepositType.BRINE_OILFIELD
    }
    
    deposit_enum = type_mapping.get(deposit_type.lower(), LithiumDepositType.PEGMATITE_LCT)
    return LithiumExplorationPipeline(deposit_enum, project_name)


def create_synthetic_lithium_dataset(
    deposit_type: LithiumDepositType,
    n_samples: int = 100,
    seed: int = 42
) -> Tuple[List[Union[PegmatiteSample, BrineSample]], List[WellData]]:
    """
    Create synthetic lithium exploration dataset for testing.
    """
    np.random.seed(seed)
    samples = []
    wells = []
    
    if "brine" in deposit_type.value:
        # Generate brine samples
        for i in range(n_samples):
            # Random location in a salar-like basin
            x = np.random.uniform(0, 50000)
            y = np.random.uniform(0, 30000)
            z = -np.random.uniform(10, 200)
            
            # Li concentration varies with location (higher in center)
            dist_from_center = np.sqrt((x - 25000)**2 + (y - 15000)**2)
            base_li = 500 * np.exp(-dist_from_center / 20000)
            li = base_li * np.random.uniform(0.5, 1.5)
            
            sample = BrineSample(
                sample_id=f"BR_{i:04d}",
                x=x,
                y=y,
                z=z,
                sample_date=datetime.now(),
                brine_type=BrineType.CONTINENTAL_SALAR,
                lithium=li,
                sodium=np.random.uniform(50000, 100000),
                potassium=np.random.uniform(5000, 20000),
                magnesium=li * np.random.uniform(3, 15),  # Mg/Li ratio
                calcium=np.random.uniform(500, 5000),
                chloride=np.random.uniform(100000, 180000),
                sulfate=np.random.uniform(5000, 20000),
                boron=np.random.uniform(100, 500),
                tds=np.random.uniform(200000, 350000),
                density=np.random.uniform(1.1, 1.25),
                ph=np.random.uniform(6.5, 8.5)
            )
            samples.append(sample)
        
        # Generate wells
        for i in range(n_samples // 10):
            well = WellData(
                well_id=f"W_{i:03d}",
                x=np.random.uniform(0, 50000),
                y=np.random.uniform(0, 30000),
                surface_elevation=np.random.uniform(3500, 4000),
                total_depth=np.random.uniform(100, 300),
                well_type="exploration",
                static_water_level=np.random.uniform(5, 30),
                transmissivity=np.random.uniform(100, 1000),
                storativity=np.random.uniform(0.01, 0.1),
                hydraulic_conductivity=np.random.uniform(1, 50)
            )
            wells.append(well)
    
    else:
        # Generate pegmatite samples
        for i in range(n_samples):
            x = np.random.uniform(0, 10000)
            y = np.random.uniform(0, 10000)
            z = np.random.uniform(0, 500)
            
            # Li concentration varies
            is_anomalous = np.random.random() < 0.2
            if is_anomalous:
                li = np.random.uniform(500, 5000)
                cs = np.random.uniform(50, 500)
                rb = np.random.uniform(500, 2000)
                k_rb = np.random.uniform(20, 60)
            else:
                li = np.random.uniform(10, 200)
                cs = np.random.uniform(1, 20)
                rb = np.random.uniform(50, 300)
                k_rb = np.random.uniform(100, 300)
            
            sample = PegmatiteSample(
                sample_id=f"PG_{i:04d}",
                x=x,
                y=y,
                z=z,
                sample_type="rock" if np.random.random() > 0.3 else "soil",
                li=li,
                cs=cs,
                rb=rb,
                ta=np.random.uniform(5, 100),
                nb=np.random.uniform(10, 200),
                sn=np.random.uniform(5, 100),
                be=np.random.uniform(1, 20),
                b=np.random.uniform(5, 50),
                k=np.random.uniform(1, 5),
                k_rb_ratio=k_rb
            )
            samples.append(sample)
    
    return samples, wells
