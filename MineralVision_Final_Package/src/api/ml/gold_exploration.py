"""
Gold Exploration Module for MineralVision.

This module provides specialized capabilities for gold deposit discovery including:
- Deposit-type-specific feature templates (orogenic, epithermal, intrusion-related, IOCG)
- Geochemistry/soil sampling integration
- Alteration indices and regolith models
- Gold-specific pathfinder element analysis
- Structural complexity metrics
- Deposit-type priors and training set generation

Supports comprehensive gold exploration workflows from regional targeting to prospect-scale analysis.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage, interpolate, stats
from scipy.spatial import cKDTree, Voronoi
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import re

logger = logging.getLogger(__name__)


class GoldDepositType(Enum):
    """Gold deposit classification types."""
    OROGENIC = "orogenic"
    EPITHERMAL_HS = "epithermal_high_sulfidation"
    EPITHERMAL_LS = "epithermal_low_sulfidation"
    INTRUSION_RELATED = "intrusion_related"
    IOCG = "iron_oxide_copper_gold"
    CARLIN = "carlin"
    PLACER = "placer"
    VMS_GOLD = "vms_gold_rich"
    PORPHYRY_GOLD = "porphyry_gold"
    SKARN_GOLD = "skarn_gold"


class AlterationType(Enum):
    """Hydrothermal alteration types."""
    POTASSIC = "potassic"
    PHYLLIC = "phyllic"
    ARGILLIC = "argillic"
    ADVANCED_ARGILLIC = "advanced_argillic"
    PROPYLITIC = "propylitic"
    SILICIC = "silicic"
    CARBONATE = "carbonate"
    CHLORITE_SERICITE = "chlorite_sericite"
    ALBITE = "albite"
    SKARN = "skarn"


class RegolithType(Enum):
    """Regolith classification types."""
    RESIDUAL = "residual"
    TRANSPORTED = "transported"
    LATERITE = "laterite"
    SAPROLITE = "saprolite"
    FERRICRETE = "ferricrete"
    CALCRETE = "calcrete"
    SILCRETE = "silcrete"
    ALLUVIUM = "alluvium"
    COLLUVIUM = "colluvium"
    OUTCROP = "outcrop"


@dataclass
class GeochemSample:
    """Geochemistry sample data."""
    sample_id: str
    x: float
    y: float
    z: Optional[float] = None
    sample_type: str = "soil"  # soil, rock, stream, drill
    elements: Dict[str, float] = field(default_factory=dict)  # element: concentration
    units: Dict[str, str] = field(default_factory=dict)  # element: unit (ppm, ppb, %)
    detection_limits: Dict[str, float] = field(default_factory=dict)
    quality_flags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldOccurrence:
    """Known gold occurrence/deposit."""
    name: str
    x: float
    y: float
    deposit_type: GoldDepositType
    commodity: str = "Au"
    size_class: str = "occurrence"  # occurrence, prospect, deposit, mine
    tonnage_mt: Optional[float] = None
    grade_gpt: Optional[float] = None
    contained_oz: Optional[float] = None
    host_rock: Optional[str] = None
    alteration: Optional[List[AlterationType]] = None
    age_ma: Optional[float] = None
    structural_setting: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralFeature:
    """Structural geology feature."""
    feature_id: str
    feature_type: str  # fault, shear_zone, fold_axis, contact, lineament
    geometry: List[Tuple[float, float]]  # List of (x, y) coordinates
    strike: Optional[float] = None
    dip: Optional[float] = None
    movement_sense: Optional[str] = None
    generation: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GoldPathfinderElements:
    """
    Gold pathfinder element analysis.
    
    Different gold deposit types have characteristic pathfinder element associations.
    """
    
    # Pathfinder associations by deposit type
    PATHFINDERS = {
        GoldDepositType.OROGENIC: {
            'primary': ['Au', 'As', 'Sb', 'W', 'Bi', 'Te'],
            'secondary': ['Ag', 'Cu', 'Pb', 'Zn', 'Mo', 'S'],
            'ratios': [('Au', 'As'), ('Au', 'Sb'), ('As', 'Sb')],
            'thresholds': {'Au': 0.01, 'As': 50, 'Sb': 5, 'W': 5}  # ppm
        },
        GoldDepositType.EPITHERMAL_HS: {
            'primary': ['Au', 'Ag', 'Cu', 'As', 'Sb', 'Hg', 'Te', 'Se'],
            'secondary': ['Bi', 'Sn', 'Mo', 'Pb', 'Zn'],
            'ratios': [('Au', 'Ag'), ('Cu', 'Au'), ('As', 'Sb')],
            'thresholds': {'Au': 0.05, 'Ag': 5, 'Cu': 100, 'As': 100}
        },
        GoldDepositType.EPITHERMAL_LS: {
            'primary': ['Au', 'Ag', 'Hg', 'Sb', 'As', 'Se', 'Te'],
            'secondary': ['Pb', 'Zn', 'Cu', 'Mn'],
            'ratios': [('Au', 'Ag'), ('Ag', 'Au'), ('Hg', 'As')],
            'thresholds': {'Au': 0.02, 'Ag': 10, 'Hg': 0.5, 'Sb': 10}
        },
        GoldDepositType.INTRUSION_RELATED: {
            'primary': ['Au', 'Bi', 'W', 'Te', 'As', 'Mo', 'Sn'],
            'secondary': ['Ag', 'Cu', 'Pb', 'Zn', 'Sb'],
            'ratios': [('Au', 'Bi'), ('W', 'Mo'), ('Au', 'As')],
            'thresholds': {'Au': 0.01, 'Bi': 1, 'W': 10, 'As': 30}
        },
        GoldDepositType.IOCG: {
            'primary': ['Au', 'Cu', 'Fe', 'U', 'REE', 'Co', 'Ag'],
            'secondary': ['Bi', 'Mo', 'Ni', 'Se', 'Te'],
            'ratios': [('Cu', 'Au'), ('Fe', 'Cu'), ('U', 'Au')],
            'thresholds': {'Au': 0.1, 'Cu': 500, 'Fe': 20000, 'U': 10}
        },
        GoldDepositType.CARLIN: {
            'primary': ['Au', 'As', 'Sb', 'Hg', 'Tl', 'Te'],
            'secondary': ['Ag', 'Cu', 'Pb', 'Zn', 'Ba'],
            'ratios': [('Au', 'As'), ('As', 'Sb'), ('Hg', 'Tl')],
            'thresholds': {'Au': 0.1, 'As': 200, 'Sb': 20, 'Hg': 1}
        },
        GoldDepositType.PORPHYRY_GOLD: {
            'primary': ['Au', 'Cu', 'Mo', 'Ag', 'As'],
            'secondary': ['Bi', 'Te', 'Se', 'Pb', 'Zn'],
            'ratios': [('Cu', 'Au'), ('Cu', 'Mo'), ('Au', 'Ag')],
            'thresholds': {'Au': 0.1, 'Cu': 200, 'Mo': 5, 'As': 50}
        }
    }
    
    def __init__(self, deposit_type: GoldDepositType = None):
        self.deposit_type = deposit_type
        
    def get_pathfinders(self, deposit_type: GoldDepositType = None) -> Dict[str, Any]:
        """Get pathfinder elements for deposit type."""
        dt = deposit_type or self.deposit_type
        if dt is None:
            # Return union of all pathfinders
            all_primary = set()
            all_secondary = set()
            for config in self.PATHFINDERS.values():
                all_primary.update(config['primary'])
                all_secondary.update(config['secondary'])
            return {
                'primary': list(all_primary),
                'secondary': list(all_secondary),
                'ratios': [],
                'thresholds': {}
            }
        return self.PATHFINDERS.get(dt, self.PATHFINDERS[GoldDepositType.OROGENIC])
        
    def compute_pathfinder_score(self, sample: GeochemSample,
                                deposit_type: GoldDepositType = None) -> float:
        """
        Compute pathfinder element score for a sample.
        
        Args:
            sample: Geochemistry sample
            deposit_type: Target deposit type
            
        Returns:
            Score between 0 and 1
        """
        config = self.get_pathfinders(deposit_type)
        thresholds = config['thresholds']
        
        score = 0.0
        count = 0
        
        for element in config['primary']:
            if element in sample.elements and element in thresholds:
                value = sample.elements[element]
                threshold = thresholds[element]
                
                if value > threshold:
                    # Log-scale scoring above threshold
                    score += min(1.0, np.log10(value / threshold) / 2)
                count += 1
                
        return score / max(count, 1)
        
    def compute_element_ratios(self, sample: GeochemSample,
                              deposit_type: GoldDepositType = None) -> Dict[str, float]:
        """Compute diagnostic element ratios."""
        config = self.get_pathfinders(deposit_type)
        ratios = {}
        
        for elem1, elem2 in config['ratios']:
            if elem1 in sample.elements and elem2 in sample.elements:
                val1 = sample.elements[elem1]
                val2 = sample.elements[elem2]
                
                if val2 > 0:
                    ratios[f'{elem1}/{elem2}'] = val1 / val2
                    
        return ratios


class AlterationIndices:
    """
    Alteration indices for gold exploration.
    
    Computes spectral and geochemical alteration indices.
    """
    
    # Spectral band indices (wavelength ranges in nm)
    SPECTRAL_INDICES = {
        'ferric_iron': {'bands': [(850, 900), (630, 690)], 'formula': 'ratio'},
        'ferrous_iron': {'bands': [(900, 1000), (1000, 1100)], 'formula': 'ratio'},
        'clay_content': {'bands': [(2200, 2250), (2100, 2150)], 'formula': 'ratio'},
        'alunite': {'bands': [(2160, 2180), (2200, 2220)], 'formula': 'ratio'},
        'kaolinite': {'bands': [(2160, 2180), (2200, 2220)], 'formula': 'depth'},
        'sericite': {'bands': [(2190, 2210), (2250, 2270)], 'formula': 'ratio'},
        'chlorite': {'bands': [(2320, 2360), (2250, 2280)], 'formula': 'ratio'},
        'carbonate': {'bands': [(2300, 2350), (2250, 2280)], 'formula': 'ratio'},
        'silica': {'bands': [(8500, 9500), (10000, 11000)], 'formula': 'ratio'},
        'gossan': {'bands': [(850, 900), (450, 520)], 'formula': 'ratio'}
    }
    
    # Geochemical alteration indices
    GEOCHEM_INDICES = {
        'ishikawa': {
            'formula': '100 * (K2O + MgO) / (K2O + MgO + Na2O + CaO)',
            'elements': ['K2O', 'MgO', 'Na2O', 'CaO'],
            'interpretation': 'Chlorite-sericite alteration'
        },
        'ccpi': {
            'formula': '100 * (MgO + FeO) / (MgO + FeO + Na2O + K2O)',
            'elements': ['MgO', 'FeO', 'Na2O', 'K2O'],
            'interpretation': 'Chlorite-carbonate-pyrite index'
        },
        'sericite_index': {
            'formula': 'K2O / (K2O + Na2O)',
            'elements': ['K2O', 'Na2O'],
            'interpretation': 'Sericite alteration intensity'
        },
        'potassic_index': {
            'formula': 'K2O / (K2O + Na2O + CaO)',
            'elements': ['K2O', 'Na2O', 'CaO'],
            'interpretation': 'Potassic alteration'
        },
        'silicification': {
            'formula': 'SiO2 / (SiO2 + Al2O3 + Fe2O3)',
            'elements': ['SiO2', 'Al2O3', 'Fe2O3'],
            'interpretation': 'Silicification intensity'
        },
        'mafic_index': {
            'formula': '(FeO + MgO) / (FeO + MgO + SiO2)',
            'elements': ['FeO', 'MgO', 'SiO2'],
            'interpretation': 'Mafic mineral content'
        }
    }
    
    def __init__(self):
        pass
        
    def compute_spectral_index(self, hyperspectral_data: np.ndarray,
                              wavelengths: np.ndarray,
                              index_name: str) -> np.ndarray:
        """
        Compute spectral alteration index.
        
        Args:
            hyperspectral_data: 3D array (rows, cols, bands)
            wavelengths: Array of wavelengths for each band
            index_name: Name of index to compute
            
        Returns:
            2D array of index values
        """
        if index_name not in self.SPECTRAL_INDICES:
            raise ValueError(f"Unknown spectral index: {index_name}")
            
        config = self.SPECTRAL_INDICES[index_name]
        bands = config['bands']
        formula = config['formula']
        
        # Find band indices for each wavelength range
        band_values = []
        for wl_min, wl_max in bands:
            mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
            if mask.any():
                band_idx = np.where(mask)[0]
                band_values.append(np.mean(hyperspectral_data[:, :, band_idx], axis=2))
            else:
                # Find nearest band
                nearest = np.argmin(np.abs(wavelengths - (wl_min + wl_max) / 2))
                band_values.append(hyperspectral_data[:, :, nearest])
                
        if len(band_values) < 2:
            return np.zeros(hyperspectral_data.shape[:2])
            
        # Compute index
        if formula == 'ratio':
            with np.errstate(divide='ignore', invalid='ignore'):
                result = band_values[0] / band_values[1]
                result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
        elif formula == 'depth':
            # Absorption depth
            continuum = (band_values[0] + band_values[1]) / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                result = 1 - band_values[0] / continuum
                result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
        elif formula == 'ndvi_style':
            with np.errstate(divide='ignore', invalid='ignore'):
                result = (band_values[0] - band_values[1]) / (band_values[0] + band_values[1])
                result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
        else:
            result = band_values[0] - band_values[1]
            
        return result
        
    def compute_geochem_index(self, samples: List[GeochemSample],
                             index_name: str) -> List[float]:
        """
        Compute geochemical alteration index for samples.
        
        Args:
            samples: List of geochemistry samples
            index_name: Name of index to compute
            
        Returns:
            List of index values
        """
        if index_name not in self.GEOCHEM_INDICES:
            raise ValueError(f"Unknown geochemical index: {index_name}")
            
        config = self.GEOCHEM_INDICES[index_name]
        required_elements = config['elements']
        
        results = []
        
        for sample in samples:
            # Check if all required elements are present
            values = {}
            for elem in required_elements:
                if elem in sample.elements:
                    values[elem] = sample.elements[elem]
                else:
                    values[elem] = 0
                    
            # Compute index based on formula
            if index_name == 'ishikawa':
                numerator = values['K2O'] + values['MgO']
                denominator = values['K2O'] + values['MgO'] + values['Na2O'] + values['CaO']
            elif index_name == 'ccpi':
                numerator = values['MgO'] + values.get('FeO', 0)
                denominator = values['MgO'] + values.get('FeO', 0) + values['Na2O'] + values['K2O']
            elif index_name == 'sericite_index':
                numerator = values['K2O']
                denominator = values['K2O'] + values['Na2O']
            elif index_name == 'potassic_index':
                numerator = values['K2O']
                denominator = values['K2O'] + values['Na2O'] + values['CaO']
            elif index_name == 'silicification':
                numerator = values.get('SiO2', 0)
                denominator = values.get('SiO2', 0) + values.get('Al2O3', 0) + values.get('Fe2O3', 0)
            elif index_name == 'mafic_index':
                numerator = values.get('FeO', 0) + values['MgO']
                denominator = values.get('FeO', 0) + values['MgO'] + values.get('SiO2', 0)
            else:
                numerator = 0
                denominator = 1
                
            if denominator > 0:
                results.append(100 * numerator / denominator)
            else:
                results.append(np.nan)
                
        return results
        
    def classify_alteration(self, indices: Dict[str, float]) -> List[AlterationType]:
        """
        Classify alteration type based on index values.
        
        Args:
            indices: Dictionary of index name to value
            
        Returns:
            List of likely alteration types
        """
        alterations = []
        
        # Potassic alteration
        if indices.get('potassic_index', 0) > 60:
            alterations.append(AlterationType.POTASSIC)
            
        # Phyllic/sericite alteration
        if indices.get('sericite_index', 0) > 70:
            alterations.append(AlterationType.PHYLLIC)
            
        # Silicic alteration
        if indices.get('silicification', 0) > 80:
            alterations.append(AlterationType.SILICIC)
            
        # Propylitic alteration (high CCPI, low sericite)
        if indices.get('ccpi', 0) > 60 and indices.get('sericite_index', 0) < 50:
            alterations.append(AlterationType.PROPYLITIC)
            
        # Argillic alteration (moderate Ishikawa)
        if 40 < indices.get('ishikawa', 0) < 70:
            alterations.append(AlterationType.ARGILLIC)
            
        return alterations if alterations else [AlterationType.PROPYLITIC]


class RegolithModel:
    """
    Regolith modeling for gold exploration.
    
    Models regolith thickness, type, and geochemical dispersion.
    """
    
    # Regolith geochemical dispersion factors
    DISPERSION_FACTORS = {
        RegolithType.RESIDUAL: {'lateral': 50, 'vertical': 1.0, 'preservation': 0.9},
        RegolithType.TRANSPORTED: {'lateral': 500, 'vertical': 0.3, 'preservation': 0.5},
        RegolithType.LATERITE: {'lateral': 100, 'vertical': 0.7, 'preservation': 0.8},
        RegolithType.SAPROLITE: {'lateral': 30, 'vertical': 0.9, 'preservation': 0.85},
        RegolithType.FERRICRETE: {'lateral': 200, 'vertical': 0.5, 'preservation': 0.7},
        RegolithType.CALCRETE: {'lateral': 300, 'vertical': 0.4, 'preservation': 0.6},
        RegolithType.ALLUVIUM: {'lateral': 1000, 'vertical': 0.2, 'preservation': 0.4},
        RegolithType.COLLUVIUM: {'lateral': 200, 'vertical': 0.5, 'preservation': 0.6},
        RegolithType.OUTCROP: {'lateral': 10, 'vertical': 1.0, 'preservation': 1.0}
    }
    
    def __init__(self):
        pass
        
    def estimate_regolith_thickness(self, dem: np.ndarray,
                                   slope: np.ndarray,
                                   geology: np.ndarray = None,
                                   rainfall: float = 500) -> np.ndarray:
        """
        Estimate regolith thickness from terrain and climate.
        
        Args:
            dem: Digital elevation model
            slope: Slope in degrees
            geology: Lithology classification (optional)
            rainfall: Mean annual rainfall in mm
            
        Returns:
            Estimated regolith thickness in meters
        """
        # Base thickness from rainfall (weathering intensity)
        base_thickness = rainfall / 50  # 10m at 500mm rainfall
        
        # Slope factor (thinner on steep slopes)
        slope_factor = np.exp(-slope / 30)
        
        # Elevation factor (thinner at high elevation)
        dem_normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-10)
        elevation_factor = 1 - 0.5 * dem_normalized
        
        # Combine factors
        thickness = base_thickness * slope_factor * elevation_factor
        
        # Apply geology modifier if available
        if geology is not None:
            # Assume geology is a categorical array
            # Different rock types weather at different rates
            geology_factor = np.ones_like(geology, dtype=float)
            # This would be customized based on actual lithology codes
            thickness = thickness * geology_factor
            
        return np.clip(thickness, 0, 100)
        
    def classify_regolith(self, dem: np.ndarray,
                         slope: np.ndarray,
                         curvature: np.ndarray,
                         drainage_distance: np.ndarray = None) -> np.ndarray:
        """
        Classify regolith type from terrain attributes.
        
        Args:
            dem: Digital elevation model
            slope: Slope in degrees
            curvature: Terrain curvature
            drainage_distance: Distance to drainage (optional)
            
        Returns:
            Regolith type classification
        """
        # Initialize with residual
        regolith = np.full(dem.shape, RegolithType.RESIDUAL.value)
        
        # Outcrop on steep slopes
        regolith = np.where(slope > 30, RegolithType.OUTCROP.value, regolith)
        
        # Colluvium on moderate slopes with negative curvature (concave)
        regolith = np.where(
            (slope > 10) & (slope <= 30) & (curvature < -0.01),
            RegolithType.COLLUVIUM.value,
            regolith
        )
        
        # Alluvium in flat areas near drainage
        if drainage_distance is not None:
            regolith = np.where(
                (slope < 5) & (drainage_distance < 100),
                RegolithType.ALLUVIUM.value,
                regolith
            )
            
        # Saprolite on gentle slopes
        regolith = np.where(
            (slope > 2) & (slope <= 10),
            RegolithType.SAPROLITE.value,
            regolith
        )
        
        return regolith
        
    def model_dispersion(self, source_locations: List[Tuple[float, float]],
                        regolith_type: np.ndarray,
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        source_strength: float = 1.0) -> np.ndarray:
        """
        Model geochemical dispersion through regolith.
        
        Args:
            source_locations: List of (x, y) source locations
            regolith_type: Regolith classification grid
            grid_x: X coordinates of grid
            grid_y: Y coordinates of grid
            source_strength: Relative source strength
            
        Returns:
            Dispersion model grid
        """
        dispersion = np.zeros_like(regolith_type, dtype=float)
        
        for src_x, src_y in source_locations:
            # Find nearest grid cell
            distances = np.sqrt((grid_x - src_x)**2 + (grid_y - src_y)**2)
            
            # Get dispersion factor for each cell based on regolith type
            for reg_type in RegolithType:
                mask = regolith_type == reg_type.value
                if not mask.any():
                    continue
                    
                factors = self.DISPERSION_FACTORS.get(
                    reg_type, 
                    self.DISPERSION_FACTORS[RegolithType.RESIDUAL]
                )
                
                lateral = factors['lateral']
                preservation = factors['preservation']
                
                # Gaussian dispersion
                contribution = source_strength * preservation * np.exp(
                    -distances[mask]**2 / (2 * lateral**2)
                )
                dispersion[mask] += contribution
                
        return dispersion


class StructuralComplexity:
    """
    Structural complexity analysis for gold targeting.
    
    Gold deposits often occur at structural intersections and
    areas of high structural complexity.
    """
    
    def __init__(self):
        pass
        
    def compute_fault_density(self, faults: List[StructuralFeature],
                             grid_x: np.ndarray,
                             grid_y: np.ndarray,
                             search_radius: float = 1000) -> np.ndarray:
        """
        Compute fault/structure density.
        
        Args:
            faults: List of structural features
            grid_x: X coordinates of grid
            grid_y: Y coordinates of grid
            search_radius: Search radius in meters
            
        Returns:
            Fault density grid (km/km2)
        """
        density = np.zeros_like(grid_x)
        
        for fault in faults:
            if fault.feature_type not in ['fault', 'shear_zone', 'lineament']:
                continue
                
            # Compute distance from each grid cell to fault
            for i in range(len(fault.geometry) - 1):
                x1, y1 = fault.geometry[i]
                x2, y2 = fault.geometry[i + 1]
                
                # Segment length
                seg_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Distance from each grid point to segment
                # Using point-to-line-segment distance
                dx = x2 - x1
                dy = y2 - y1
                
                t = np.clip(
                    ((grid_x - x1) * dx + (grid_y - y1) * dy) / (dx**2 + dy**2 + 1e-10),
                    0, 1
                )
                
                nearest_x = x1 + t * dx
                nearest_y = y1 + t * dy
                
                dist = np.sqrt((grid_x - nearest_x)**2 + (grid_y - nearest_y)**2)
                
                # Add contribution within search radius
                mask = dist < search_radius
                density[mask] += seg_length / 1000  # Convert to km
                
        # Normalize by area
        cell_area = abs(grid_x[0, 1] - grid_x[0, 0]) * abs(grid_y[1, 0] - grid_y[0, 0]) / 1e6  # km2
        density = density / (np.pi * (search_radius / 1000)**2)
        
        return density
        
    def compute_intersection_density(self, faults: List[StructuralFeature],
                                    grid_x: np.ndarray,
                                    grid_y: np.ndarray,
                                    search_radius: float = 500) -> np.ndarray:
        """
        Compute fault intersection density.
        
        Args:
            faults: List of structural features
            grid_x: X coordinates of grid
            grid_y: Y coordinates of grid
            search_radius: Search radius in meters
            
        Returns:
            Intersection density grid
        """
        # Find all intersections
        intersections = []
        
        for i, fault1 in enumerate(faults):
            for j, fault2 in enumerate(faults[i+1:], i+1):
                # Check each segment pair
                for k in range(len(fault1.geometry) - 1):
                    x1, y1 = fault1.geometry[k]
                    x2, y2 = fault1.geometry[k + 1]
                    
                    for l in range(len(fault2.geometry) - 1):
                        x3, y3 = fault2.geometry[l]
                        x4, y4 = fault2.geometry[l + 1]
                        
                        # Line intersection
                        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        
                        if abs(denom) < 1e-10:
                            continue
                            
                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                        
                        if 0 <= t <= 1 and 0 <= u <= 1:
                            ix = x1 + t * (x2 - x1)
                            iy = y1 + t * (y2 - y1)
                            intersections.append((ix, iy))
                            
        # Compute density
        density = np.zeros_like(grid_x)
        
        for ix, iy in intersections:
            dist = np.sqrt((grid_x - ix)**2 + (grid_y - iy)**2)
            mask = dist < search_radius
            density[mask] += 1
            
        return density
        
    def compute_structural_complexity_index(self, 
                                           fault_density: np.ndarray,
                                           intersection_density: np.ndarray,
                                           magnetic_gradient: np.ndarray = None) -> np.ndarray:
        """
        Compute overall structural complexity index.
        
        Args:
            fault_density: Fault density grid
            intersection_density: Intersection density grid
            magnetic_gradient: Magnetic gradient (optional)
            
        Returns:
            Structural complexity index (0-1)
        """
        # Normalize each component
        def normalize(arr):
            arr_min = np.nanmin(arr)
            arr_max = np.nanmax(arr)
            if arr_max - arr_min > 0:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.zeros_like(arr)
            
        fd_norm = normalize(fault_density)
        id_norm = normalize(intersection_density)
        
        # Weighted combination
        complexity = 0.4 * fd_norm + 0.4 * id_norm
        
        if magnetic_gradient is not None:
            mg_norm = normalize(magnetic_gradient)
            complexity = 0.3 * fd_norm + 0.3 * id_norm + 0.4 * mg_norm
            
        return complexity


class GoldDepositPriors:
    """
    Prior probability models for different gold deposit types.
    
    Encodes geological knowledge about favorable settings for each deposit type.
    """
    
    # Favorable settings by deposit type
    FAVORABLE_SETTINGS = {
        GoldDepositType.OROGENIC: {
            'host_rocks': ['greenstone', 'bif', 'turbidite', 'metasediment'],
            'structures': ['shear_zone', 'fault', 'fold_hinge'],
            'metamorphic_grade': ['greenschist', 'amphibolite'],
            'depth_km': (3, 15),
            'age_preference': 'archean_proterozoic'
        },
        GoldDepositType.EPITHERMAL_HS: {
            'host_rocks': ['andesite', 'dacite', 'volcanic'],
            'structures': ['fault', 'breccia', 'dome'],
            'alteration': ['advanced_argillic', 'silicic'],
            'depth_km': (0, 1),
            'age_preference': 'cenozoic'
        },
        GoldDepositType.EPITHERMAL_LS: {
            'host_rocks': ['rhyolite', 'andesite', 'volcanic', 'sediment'],
            'structures': ['fault', 'vein', 'stockwork'],
            'alteration': ['adularia', 'sericite', 'silicic'],
            'depth_km': (0, 1.5),
            'age_preference': 'cenozoic'
        },
        GoldDepositType.INTRUSION_RELATED: {
            'host_rocks': ['granite', 'granodiorite', 'sediment'],
            'structures': ['contact', 'fault', 'sheeted_vein'],
            'alteration': ['potassic', 'sericite'],
            'depth_km': (1, 5),
            'age_preference': 'any'
        },
        GoldDepositType.IOCG: {
            'host_rocks': ['granite', 'volcanic', 'metasediment'],
            'structures': ['fault', 'breccia', 'shear_zone'],
            'alteration': ['potassic', 'sodic', 'iron_oxide'],
            'depth_km': (1, 10),
            'age_preference': 'proterozoic'
        },
        GoldDepositType.CARLIN: {
            'host_rocks': ['limestone', 'dolostone', 'siltstone'],
            'structures': ['fault', 'fold', 'window'],
            'alteration': ['silicic', 'carbonate', 'argillic'],
            'depth_km': (0.5, 3),
            'age_preference': 'paleozoic_host'
        }
    }
    
    def __init__(self):
        pass
        
    def compute_prior(self, deposit_type: GoldDepositType,
                     features: Dict[str, Any]) -> float:
        """
        Compute prior probability for deposit type given features.
        
        Args:
            deposit_type: Target deposit type
            features: Dictionary of geological features
            
        Returns:
            Prior probability (0-1)
        """
        settings = self.FAVORABLE_SETTINGS.get(deposit_type)
        if settings is None:
            return 0.5
            
        score = 0.0
        weights = 0.0
        
        # Host rock match
        if 'host_rock' in features:
            host = features['host_rock'].lower()
            if any(h in host for h in settings['host_rocks']):
                score += 1.0
            weights += 1.0
            
        # Structure match
        if 'structures' in features:
            structs = [s.lower() for s in features['structures']]
            matches = sum(1 for s in settings['structures'] if any(s in st for st in structs))
            score += matches / len(settings['structures'])
            weights += 1.0
            
        # Alteration match
        if 'alteration' in features and 'alteration' in settings:
            alts = [a.lower() for a in features['alteration']]
            matches = sum(1 for a in settings['alteration'] if any(a in al for al in alts))
            score += matches / len(settings['alteration'])
            weights += 1.0
            
        # Depth constraint
        if 'depth_km' in features:
            depth = features['depth_km']
            min_d, max_d = settings['depth_km']
            if min_d <= depth <= max_d:
                score += 1.0
            weights += 1.0
            
        return score / max(weights, 1)
        
    def get_favorable_features(self, deposit_type: GoldDepositType) -> Dict[str, Any]:
        """Get favorable features for deposit type."""
        return self.FAVORABLE_SETTINGS.get(deposit_type, {})


class GoldExplorationPipeline:
    """
    Complete gold exploration pipeline.
    
    Integrates all gold-specific modules for comprehensive targeting.
    """
    
    def __init__(self):
        self.pathfinders = GoldPathfinderElements()
        self.alteration = AlterationIndices()
        self.regolith = RegolithModel()
        self.structural = StructuralComplexity()
        self.priors = GoldDepositPriors()
        
        self.processing_history: List[Dict] = []
        
    def process_geochemistry(self, samples: List[GeochemSample],
                            deposit_types: List[GoldDepositType] = None) -> pd.DataFrame:
        """
        Process geochemistry samples for gold exploration.
        
        Args:
            samples: List of geochemistry samples
            deposit_types: Target deposit types (all if None)
            
        Returns:
            DataFrame with processed geochemistry features
        """
        if deposit_types is None:
            deposit_types = list(GoldDepositType)
            
        results = []
        
        for sample in samples:
            row = {
                'sample_id': sample.sample_id,
                'x': sample.x,
                'y': sample.y,
                'sample_type': sample.sample_type
            }
            
            # Add raw element values
            for elem, value in sample.elements.items():
                row[f'{elem}_raw'] = value
                
            # Compute pathfinder scores for each deposit type
            for dt in deposit_types:
                score = self.pathfinders.compute_pathfinder_score(sample, dt)
                row[f'pathfinder_{dt.value}'] = score
                
                # Compute element ratios
                ratios = self.pathfinders.compute_element_ratios(sample, dt)
                for ratio_name, ratio_value in ratios.items():
                    row[f'ratio_{ratio_name}_{dt.value}'] = ratio_value
                    
            # Compute alteration indices
            for index_name in self.alteration.GEOCHEM_INDICES.keys():
                values = self.alteration.compute_geochem_index([sample], index_name)
                row[f'alteration_{index_name}'] = values[0]
                
            results.append(row)
            
        df = pd.DataFrame(results)
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'process_geochemistry',
            'n_samples': len(samples),
            'deposit_types': [dt.value for dt in deposit_types]
        })
        
        return df
        
    def generate_gold_features(self, 
                              magnetic_grid: np.ndarray,
                              radiometric_k: np.ndarray,
                              radiometric_th: np.ndarray,
                              radiometric_u: np.ndarray,
                              dem: np.ndarray,
                              faults: List[StructuralFeature] = None,
                              grid_x: np.ndarray = None,
                              grid_y: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Generate gold-specific features from geophysical data.
        
        Args:
            magnetic_grid: Total magnetic intensity grid
            radiometric_k: Potassium grid (%)
            radiometric_th: Thorium grid (ppm)
            radiometric_u: Uranium grid (ppm)
            dem: Digital elevation model
            faults: List of structural features
            grid_x: X coordinates
            grid_y: Y coordinates
            
        Returns:
            Dictionary of feature grids
        """
        features = {}
        
        # Magnetic features
        features['mag_tmi'] = magnetic_grid
        
        # Magnetic derivatives
        dx = ndimage.sobel(magnetic_grid, axis=1)
        dy = ndimage.sobel(magnetic_grid, axis=0)
        features['mag_gradient'] = np.sqrt(dx**2 + dy**2)
        features['mag_tilt'] = np.degrees(np.arctan2(
            ndimage.laplace(magnetic_grid),
            features['mag_gradient'] + 1e-10
        ))
        
        # Radiometric features
        features['rad_k'] = radiometric_k
        features['rad_th'] = radiometric_th
        features['rad_u'] = radiometric_u
        
        # Radiometric ratios (gold-relevant)
        with np.errstate(divide='ignore', invalid='ignore'):
            features['rad_k_th'] = np.nan_to_num(radiometric_k / (radiometric_th + 0.1))
            features['rad_u_th'] = np.nan_to_num(radiometric_u / (radiometric_th + 0.1))
            features['rad_k_u'] = np.nan_to_num(radiometric_k / (radiometric_u + 0.1))
            
        # Potassic alteration indicator (high K, low Th)
        k_norm = (radiometric_k - np.nanmean(radiometric_k)) / (np.nanstd(radiometric_k) + 1e-10)
        th_norm = (radiometric_th - np.nanmean(radiometric_th)) / (np.nanstd(radiometric_th) + 1e-10)
        features['potassic_indicator'] = k_norm - th_norm
        
        # Terrain features
        features['elevation'] = dem
        slope = np.degrees(np.arctan(np.sqrt(
            ndimage.sobel(dem, axis=0)**2 + ndimage.sobel(dem, axis=1)**2
        )))
        features['slope'] = slope
        features['curvature'] = ndimage.laplace(dem)
        
        # Regolith thickness estimate
        features['regolith_thickness'] = self.regolith.estimate_regolith_thickness(
            dem, slope
        )
        
        # Structural features
        if faults is not None and grid_x is not None and grid_y is not None:
            features['fault_density'] = self.structural.compute_fault_density(
                faults, grid_x, grid_y
            )
            features['intersection_density'] = self.structural.compute_intersection_density(
                faults, grid_x, grid_y
            )
            features['structural_complexity'] = self.structural.compute_structural_complexity_index(
                features['fault_density'],
                features['intersection_density'],
                features['mag_gradient']
            )
            
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'generate_gold_features',
            'n_features': len(features)
        })
        
        return features
        
    def score_targets(self, features: Dict[str, np.ndarray],
                     deposit_type: GoldDepositType = None,
                     weights: Dict[str, float] = None) -> np.ndarray:
        """
        Score targets based on gold-specific features.
        
        Args:
            features: Dictionary of feature grids
            deposit_type: Target deposit type (generic if None)
            weights: Custom feature weights
            
        Returns:
            Target score grid (0-1)
        """
        # Default weights by deposit type
        default_weights = {
            GoldDepositType.OROGENIC: {
                'structural_complexity': 0.3,
                'mag_gradient': 0.2,
                'potassic_indicator': 0.15,
                'fault_density': 0.2,
                'rad_k_th': 0.15
            },
            GoldDepositType.EPITHERMAL_HS: {
                'potassic_indicator': 0.25,
                'rad_k': 0.2,
                'structural_complexity': 0.2,
                'elevation': -0.1,  # Negative = prefer lower
                'slope': 0.15,
                'mag_tilt': 0.1
            },
            GoldDepositType.INTRUSION_RELATED: {
                'mag_tmi': 0.2,
                'potassic_indicator': 0.25,
                'structural_complexity': 0.2,
                'rad_k_th': 0.2,
                'fault_density': 0.15
            },
            GoldDepositType.IOCG: {
                'mag_tmi': 0.3,
                'rad_u': 0.2,
                'structural_complexity': 0.2,
                'fault_density': 0.15,
                'rad_k': 0.15
            }
        }
        
        if weights is None:
            if deposit_type is not None:
                weights = default_weights.get(deposit_type, {})
            else:
                # Generic weights
                weights = {
                    'structural_complexity': 0.25,
                    'mag_gradient': 0.2,
                    'potassic_indicator': 0.2,
                    'fault_density': 0.2,
                    'rad_k_th': 0.15
                }
                
        # Normalize features and compute weighted score
        score = np.zeros_like(list(features.values())[0], dtype=float)
        total_weight = 0
        
        for feature_name, weight in weights.items():
            if feature_name not in features:
                continue
                
            feature = features[feature_name]
            
            # Normalize to 0-1
            f_min = np.nanmin(feature)
            f_max = np.nanmax(feature)
            
            if f_max - f_min > 0:
                normalized = (feature - f_min) / (f_max - f_min)
            else:
                normalized = np.zeros_like(feature)
                
            # Handle negative weights (inverse relationship)
            if weight < 0:
                normalized = 1 - normalized
                weight = abs(weight)
                
            score += weight * normalized
            total_weight += weight
            
        if total_weight > 0:
            score = score / total_weight
            
        return score
        
    def train_gold_model(self, features: Dict[str, np.ndarray],
                        occurrences: List[GoldOccurrence],
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        deposit_type: GoldDepositType = None,
                        n_negative: int = None) -> Dict[str, Any]:
        """
        Train ML model for gold prospectivity.
        
        Args:
            features: Dictionary of feature grids
            occurrences: Known gold occurrences
            grid_x: X coordinates
            grid_y: Y coordinates
            deposit_type: Filter occurrences by deposit type
            n_negative: Number of negative samples
            
        Returns:
            Training results
        """
        # Filter occurrences by deposit type if specified
        if deposit_type is not None:
            occurrences = [o for o in occurrences if o.deposit_type == deposit_type]
            
        if len(occurrences) < 5:
            raise ValueError(f"Need at least 5 occurrences, got {len(occurrences)}")
            
        # Extract features at occurrence locations
        feature_names = list(features.keys())
        n_features = len(feature_names)
        
        # Positive samples
        positive_features = []
        for occ in occurrences:
            # Find nearest grid cell
            dist = np.sqrt((grid_x - occ.x)**2 + (grid_y - occ.y)**2)
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            sample = [features[fn][idx] for fn in feature_names]
            positive_features.append(sample)
            
        positive_features = np.array(positive_features)
        
        # Negative samples (random locations away from occurrences)
        if n_negative is None:
            n_negative = len(occurrences) * 3
            
        occ_coords = np.array([[o.x, o.y] for o in occurrences])
        occ_tree = cKDTree(occ_coords)
        
        negative_features = []
        attempts = 0
        max_attempts = n_negative * 10
        
        while len(negative_features) < n_negative and attempts < max_attempts:
            # Random grid cell
            i = np.random.randint(0, grid_x.shape[0])
            j = np.random.randint(0, grid_x.shape[1])
            
            x, y = grid_x[i, j], grid_y[i, j]
            
            # Check distance to occurrences
            dist, _ = occ_tree.query([x, y])
            
            if dist > 2000:  # At least 2km from known occurrences
                sample = [features[fn][i, j] for fn in feature_names]
                if not any(np.isnan(sample)):
                    negative_features.append(sample)
                    
            attempts += 1
            
        negative_features = np.array(negative_features)
        
        # Combine and train
        X = np.vstack([positive_features, negative_features])
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        # Handle NaN values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_scaled, y)
        
        # Feature importance
        importance = dict(zip(feature_names, model.feature_importances_))
        
        # Cross-validation score (simple)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
        
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'train_gold_model',
            'n_positive': len(positive_features),
            'n_negative': len(negative_features),
            'deposit_type': deposit_type.value if deposit_type else 'all',
            'cv_auc': float(np.mean(cv_scores))
        })
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'feature_importance': importance,
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'n_positive': len(positive_features),
            'n_negative': len(negative_features)
        }
        
    def predict_prospectivity(self, model_result: Dict[str, Any],
                             features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict gold prospectivity using trained model.
        
        Args:
            model_result: Result from train_gold_model
            features: Dictionary of feature grids
            
        Returns:
            Prospectivity probability grid
        """
        model = model_result['model']
        scaler = model_result['scaler']
        feature_names = model_result['feature_names']
        
        # Stack features
        shape = features[feature_names[0]].shape
        X = np.stack([features[fn].flatten() for fn in feature_names], axis=1)
        
        # Handle NaN
        valid_mask = ~np.any(np.isnan(X), axis=1)
        
        # Predict
        predictions = np.zeros(X.shape[0])
        if valid_mask.any():
            X_valid = scaler.transform(X[valid_mask])
            predictions[valid_mask] = model.predict_proba(X_valid)[:, 1]
            
        return predictions.reshape(shape)
        
    def generate_targets(self, prospectivity: np.ndarray,
                        grid_x: np.ndarray,
                        grid_y: np.ndarray,
                        threshold: float = 0.7,
                        min_area_km2: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate exploration targets from prospectivity map.
        
        Args:
            prospectivity: Prospectivity probability grid
            grid_x: X coordinates
            grid_y: Y coordinates
            threshold: Probability threshold
            min_area_km2: Minimum target area
            
        Returns:
            List of target dictionaries
        """
        # Threshold and label connected regions
        binary = prospectivity > threshold
        labeled, n_labels = ndimage.label(binary)
        
        # Cell area
        dx = abs(grid_x[0, 1] - grid_x[0, 0])
        dy = abs(grid_y[1, 0] - grid_y[0, 0])
        cell_area_km2 = dx * dy / 1e6
        
        targets = []
        
        for label_id in range(1, n_labels + 1):
            mask = labeled == label_id
            area_km2 = mask.sum() * cell_area_km2
            
            if area_km2 < min_area_km2:
                continue
                
            # Target properties
            target_prospectivity = prospectivity[mask]
            target_x = grid_x[mask]
            target_y = grid_y[mask]
            
            # Centroid (weighted by prospectivity)
            weights = target_prospectivity / target_prospectivity.sum()
            centroid_x = np.sum(target_x * weights)
            centroid_y = np.sum(target_y * weights)
            
            # Peak location
            peak_idx = np.argmax(target_prospectivity)
            peak_x = target_x[peak_idx]
            peak_y = target_y[peak_idx]
            
            targets.append({
                'target_id': f'T{label_id:03d}',
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'peak_x': float(peak_x),
                'peak_y': float(peak_y),
                'area_km2': float(area_km2),
                'mean_prospectivity': float(np.mean(target_prospectivity)),
                'max_prospectivity': float(np.max(target_prospectivity)),
                'rank': 0  # Will be set after sorting
            })
            
        # Rank by max prospectivity
        targets.sort(key=lambda t: t['max_prospectivity'], reverse=True)
        for i, target in enumerate(targets):
            target['rank'] = i + 1
            
        return targets


def create_gold_exploration_pipeline() -> GoldExplorationPipeline:
    """Factory function to create gold exploration pipeline."""
    return GoldExplorationPipeline()


def create_synthetic_gold_dataset(deposit_type: GoldDepositType,
                                 n_occurrences: int = 20,
                                 grid_size: int = 100) -> Dict[str, Any]:
    """
    Create synthetic gold exploration dataset for testing.
    
    Args:
        deposit_type: Type of gold deposit to simulate
        n_occurrences: Number of occurrences to generate
        grid_size: Grid dimension
        
    Returns:
        Dictionary with features, occurrences, and grids
    """
    # Create coordinate grids
    x = np.linspace(0, 100000, grid_size)
    y = np.linspace(0, 100000, grid_size)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Generate synthetic features
    np.random.seed(42)
    
    # Magnetic grid with anomalies
    magnetic = np.random.randn(grid_size, grid_size) * 50 + 50000
    
    # Add magnetic anomalies at random locations
    for _ in range(5):
        cx, cy = np.random.randint(10, grid_size - 10, 2)
        anomaly = 500 * np.exp(-((np.arange(grid_size)[:, None] - cy)**2 + 
                                 (np.arange(grid_size)[None, :] - cx)**2) / 200)
        magnetic += anomaly
        
    # Radiometric grids
    rad_k = np.random.rand(grid_size, grid_size) * 3 + 1  # 1-4%
    rad_th = np.random.rand(grid_size, grid_size) * 20 + 5  # 5-25 ppm
    rad_u = np.random.rand(grid_size, grid_size) * 5 + 1  # 1-6 ppm
    
    # DEM
    dem = np.random.randn(grid_size, grid_size) * 100 + 500
    dem = ndimage.gaussian_filter(dem, sigma=5)
    
    # Generate occurrences at favorable locations
    occurrences = []
    for i in range(n_occurrences):
        # Place near magnetic anomalies and high K
        idx = np.random.randint(0, grid_size, 2)
        occ = GoldOccurrence(
            name=f'Occurrence_{i+1}',
            x=float(grid_x[idx[0], idx[1]]),
            y=float(grid_y[idx[0], idx[1]]),
            deposit_type=deposit_type,
            size_class='occurrence',
            grade_gpt=np.random.uniform(0.5, 5.0)
        )
        occurrences.append(occ)
        
    return {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'magnetic': magnetic,
        'rad_k': rad_k,
        'rad_th': rad_th,
        'rad_u': rad_u,
        'dem': dem,
        'occurrences': occurrences
    }
