"""
Variography Module for MineralVision Platform.

Comprehensive variogram analysis including:
1. Experimental variogram calculation (omnidirectional and directional)
2. Variogram model fitting (spherical, exponential, gaussian, etc.)
3. Anisotropy analysis and search ellipse optimization
4. Cross-variograms for multivariate analysis
5. Variogram maps and rose diagrams
6. Automatic model fitting with optimization
7. Nested structure support
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import numpy as np
from collections import defaultdict


class VariogramModel(Enum):
    """Variogram model types."""
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"
    CUBIC = "cubic"
    PENTASPHERICAL = "pentaspherical"
    HOLE_EFFECT = "hole_effect"
    NUGGET = "nugget"


class AnisotropyType(Enum):
    """Types of anisotropy."""
    ISOTROPIC = "isotropic"
    GEOMETRIC = "geometric"
    ZONAL = "zonal"
    COMBINED = "combined"


@dataclass
class Point3D:
    """3D point with optional value."""
    x: float
    y: float
    z: float
    value: float = 0.0
    weight: float = 1.0
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def direction_to(self, other: 'Point3D') -> Tuple[float, float]:
        """Calculate azimuth and dip to another point."""
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        
        horiz_dist = math.sqrt(dx**2 + dy**2)
        
        azimuth = math.degrees(math.atan2(dx, dy)) % 360
        dip = math.degrees(math.atan2(dz, horiz_dist)) if horiz_dist > 0 else (90 if dz > 0 else -90)
        
        return (azimuth, dip)


@dataclass
class LagBin:
    """Variogram lag bin."""
    lag_distance: float
    lag_tolerance: float
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    squared_differences: List[float] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.pairs)
    
    @property
    def semivariance(self) -> float:
        if not self.squared_differences:
            return 0.0
        return sum(self.squared_differences) / (2 * len(self.squared_differences))
    
    @property
    def mean_distance(self) -> float:
        if not self.distances:
            return self.lag_distance
        return sum(self.distances) / len(self.distances)
    
    @property
    def variance(self) -> float:
        if len(self.squared_differences) < 2:
            return 0.0
        mean = self.semivariance * 2
        return sum((sd - mean)**2 for sd in self.squared_differences) / (len(self.squared_differences) - 1)


@dataclass
class ExperimentalVariogram:
    """Experimental variogram data."""
    name: str
    element: str
    azimuth: float
    dip: float
    azimuth_tolerance: float
    dip_tolerance: float
    bandwidth: float
    lag_bins: List[LagBin]
    data_variance: float
    data_mean: float
    n_points: int
    calculation_date: datetime = field(default_factory=datetime.now)
    
    @property
    def lags(self) -> List[float]:
        return [bin.mean_distance for bin in self.lag_bins]
    
    @property
    def semivariances(self) -> List[float]:
        return [bin.semivariance for bin in self.lag_bins]
    
    @property
    def pair_counts(self) -> List[int]:
        return [bin.count for bin in self.lag_bins]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "element": self.element,
            "direction": {"azimuth": self.azimuth, "dip": self.dip},
            "tolerances": {
                "azimuth": self.azimuth_tolerance,
                "dip": self.dip_tolerance,
                "bandwidth": self.bandwidth
            },
            "lags": self.lags,
            "semivariances": self.semivariances,
            "pair_counts": self.pair_counts,
            "data_variance": self.data_variance,
            "data_mean": self.data_mean,
            "n_points": self.n_points
        }


@dataclass
class VariogramStructure:
    """Single variogram structure (nested model component)."""
    model_type: VariogramModel
    contribution: float
    range_major: float
    range_minor: float = 0.0
    range_vertical: float = 0.0
    azimuth: float = 0.0
    dip: float = 0.0
    plunge: float = 0.0
    power: float = 1.0
    
    def __post_init__(self):
        if self.range_minor == 0:
            self.range_minor = self.range_major
        if self.range_vertical == 0:
            self.range_vertical = self.range_major
    
    def evaluate(self, h: float) -> float:
        """Evaluate structure contribution at lag h."""
        if h <= 0:
            return 0.0
        
        a = self.range_major
        c = self.contribution
        
        if self.model_type == VariogramModel.NUGGET:
            return c if h > 0 else 0.0
        
        elif self.model_type == VariogramModel.SPHERICAL:
            if h >= a:
                return c
            hr = h / a
            return c * (1.5 * hr - 0.5 * hr**3)
        
        elif self.model_type == VariogramModel.EXPONENTIAL:
            return c * (1 - math.exp(-3 * h / a))
        
        elif self.model_type == VariogramModel.GAUSSIAN:
            return c * (1 - math.exp(-3 * (h / a)**2))
        
        elif self.model_type == VariogramModel.LINEAR:
            if h >= a:
                return c
            return c * h / a
        
        elif self.model_type == VariogramModel.POWER:
            return c * h**self.power
        
        elif self.model_type == VariogramModel.CUBIC:
            if h >= a:
                return c
            hr = h / a
            return c * (7 * hr**2 - 8.75 * hr**3 + 3.5 * hr**5 - 0.75 * hr**7)
        
        elif self.model_type == VariogramModel.PENTASPHERICAL:
            if h >= a:
                return c
            hr = h / a
            return c * (1.875 * hr - 1.25 * hr**3 + 0.375 * hr**5)
        
        elif self.model_type == VariogramModel.HOLE_EFFECT:
            if h == 0:
                return 0.0
            return c * (1 - math.sin(math.pi * h / a) / (math.pi * h / a))
        
        return 0.0
    
    def evaluate_anisotropic(self, dx: float, dy: float, dz: float) -> float:
        """Evaluate structure with anisotropy."""
        az_rad = math.radians(self.azimuth)
        dip_rad = math.radians(self.dip)
        
        cos_az = math.cos(az_rad)
        sin_az = math.sin(az_rad)
        cos_dip = math.cos(dip_rad)
        sin_dip = math.sin(dip_rad)
        
        dx_rot = dx * cos_az + dy * sin_az
        dy_rot = -dx * sin_az + dy * cos_az
        dz_rot = dz
        
        dx_rot2 = dx_rot * cos_dip + dz_rot * sin_dip
        dz_rot2 = -dx_rot * sin_dip + dz_rot * cos_dip
        
        h_scaled = math.sqrt(
            (dx_rot2 / self.range_major)**2 +
            (dy_rot / self.range_minor)**2 +
            (dz_rot2 / self.range_vertical)**2
        ) * self.range_major
        
        return self.evaluate(h_scaled)


@dataclass
class FittedVariogramModel:
    """Complete fitted variogram model with nested structures."""
    name: str
    element: str
    structures: List[VariogramStructure]
    nugget: float = 0.0
    sill: float = 0.0
    fit_quality: float = 0.0
    rmse: float = 0.0
    experimental: Optional[ExperimentalVariogram] = None
    
    def __post_init__(self):
        self.sill = self.nugget + sum(s.contribution for s in self.structures)
    
    def evaluate(self, h: float) -> float:
        """Evaluate model at lag h."""
        gamma = self.nugget if h > 0 else 0.0
        for structure in self.structures:
            gamma += structure.evaluate(h)
        return gamma
    
    def evaluate_anisotropic(self, dx: float, dy: float, dz: float) -> float:
        """Evaluate model with anisotropy."""
        h = math.sqrt(dx**2 + dy**2 + dz**2)
        gamma = self.nugget if h > 0 else 0.0
        for structure in self.structures:
            gamma += structure.evaluate_anisotropic(dx, dy, dz)
        return gamma
    
    def covariance(self, h: float) -> float:
        """Calculate covariance at lag h."""
        return self.sill - self.evaluate(h)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "element": self.element,
            "nugget": self.nugget,
            "sill": self.sill,
            "structures": [{
                "model": s.model_type.value,
                "contribution": s.contribution,
                "range_major": s.range_major,
                "range_minor": s.range_minor,
                "range_vertical": s.range_vertical,
                "azimuth": s.azimuth,
                "dip": s.dip
            } for s in self.structures],
            "fit_quality": self.fit_quality,
            "rmse": self.rmse
        }


class ExperimentalVariogramCalculator:
    """Calculate experimental variograms from point data."""
    
    def __init__(self):
        self.points: List[Point3D] = []
        self.kdtree = None
    
    def set_data(self, points: List[Point3D]):
        """Set point data for variogram calculation."""
        self.points = points
    
    def add_point(self, x: float, y: float, z: float, value: float, weight: float = 1.0):
        """Add a single point."""
        self.points.append(Point3D(x, y, z, value, weight))
    
    def calculate_omnidirectional(self, n_lags: int = 15, 
                                  lag_distance: Optional[float] = None,
                                  lag_tolerance: Optional[float] = None,
                                  max_distance: Optional[float] = None,
                                  element: str = "value") -> ExperimentalVariogram:
        """Calculate omnidirectional variogram."""
        return self.calculate_directional(
            azimuth=0, dip=0,
            azimuth_tolerance=180, dip_tolerance=90,
            bandwidth=float('inf'),
            n_lags=n_lags,
            lag_distance=lag_distance,
            lag_tolerance=lag_tolerance,
            max_distance=max_distance,
            element=element,
            name="Omnidirectional"
        )
    
    def calculate_directional(self, azimuth: float, dip: float,
                             azimuth_tolerance: float = 22.5,
                             dip_tolerance: float = 22.5,
                             bandwidth: float = float('inf'),
                             n_lags: int = 15,
                             lag_distance: Optional[float] = None,
                             lag_tolerance: Optional[float] = None,
                             max_distance: Optional[float] = None,
                             element: str = "value",
                             name: Optional[str] = None) -> ExperimentalVariogram:
        """Calculate directional variogram."""
        
        if not self.points:
            raise ValueError("No data points set")
        
        values = [p.value for p in self.points]
        data_mean = np.mean(values)
        data_variance = np.var(values, ddof=1)
        
        if max_distance is None:
            xs = [p.x for p in self.points]
            ys = [p.y for p in self.points]
            zs = [p.z for p in self.points]
            max_distance = math.sqrt(
                (max(xs) - min(xs))**2 + 
                (max(ys) - min(ys))**2 + 
                (max(zs) - min(zs))**2
            ) / 2
        
        if lag_distance is None:
            lag_distance = max_distance / n_lags
        
        if lag_tolerance is None:
            lag_tolerance = lag_distance / 2
        
        lag_bins = []
        for i in range(n_lags):
            lag_center = (i + 0.5) * lag_distance
            lag_bins.append(LagBin(
                lag_distance=lag_center,
                lag_tolerance=lag_tolerance
            ))
        
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.points[i]
                p2 = self.points[j]
                
                dist = p1.distance_to(p2)
                
                if dist > max_distance or dist == 0:
                    continue
                
                pair_az, pair_dip = p1.direction_to(p2)
                
                if not self._within_direction_tolerance(
                    pair_az, pair_dip, azimuth, dip, 
                    azimuth_tolerance, dip_tolerance
                ):
                    if not self._within_direction_tolerance(
                        (pair_az + 180) % 360, -pair_dip, azimuth, dip,
                        azimuth_tolerance, dip_tolerance
                    ):
                        continue
                
                if bandwidth < float('inf'):
                    perp_dist = self._perpendicular_distance(p1, p2, azimuth, dip)
                    if perp_dist > bandwidth:
                        continue
                
                sq_diff = (p1.value - p2.value)**2
                
                for bin in lag_bins:
                    if abs(dist - bin.lag_distance) <= bin.lag_tolerance:
                        bin.pairs.append((i, j))
                        bin.squared_differences.append(sq_diff)
                        bin.distances.append(dist)
                        break
        
        if name is None:
            name = f"Az{azimuth:.0f}_Dip{dip:.0f}"
        
        return ExperimentalVariogram(
            name=name,
            element=element,
            azimuth=azimuth,
            dip=dip,
            azimuth_tolerance=azimuth_tolerance,
            dip_tolerance=dip_tolerance,
            bandwidth=bandwidth,
            lag_bins=lag_bins,
            data_variance=data_variance,
            data_mean=data_mean,
            n_points=len(self.points)
        )
    
    def _within_direction_tolerance(self, pair_az: float, pair_dip: float,
                                   target_az: float, target_dip: float,
                                   az_tol: float, dip_tol: float) -> bool:
        """Check if pair direction is within tolerance of target direction."""
        az_diff = abs(pair_az - target_az)
        if az_diff > 180:
            az_diff = 360 - az_diff
        
        dip_diff = abs(pair_dip - target_dip)
        
        return az_diff <= az_tol and dip_diff <= dip_tol
    
    def _perpendicular_distance(self, p1: Point3D, p2: Point3D,
                               azimuth: float, dip: float) -> float:
        """Calculate perpendicular distance from pair midpoint to search direction."""
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        mid_z = (p1.z + p2.z) / 2
        
        az_rad = math.radians(azimuth)
        dip_rad = math.radians(dip)
        
        dir_x = math.sin(az_rad) * math.cos(dip_rad)
        dir_y = math.cos(az_rad) * math.cos(dip_rad)
        dir_z = math.sin(dip_rad)
        
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dz = p2.z - p1.z
        
        proj = dx * dir_x + dy * dir_y + dz * dir_z
        
        perp_x = dx - proj * dir_x
        perp_y = dy - proj * dir_y
        perp_z = dz - proj * dir_z
        
        return math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
    
    def calculate_variogram_map(self, max_distance: float,
                               cell_size: float = 10.0,
                               element: str = "value") -> Dict[str, Any]:
        """Calculate variogram map for anisotropy detection."""
        
        n_cells = int(max_distance / cell_size) * 2 + 1
        center = n_cells // 2
        
        map_counts = np.zeros((n_cells, n_cells))
        map_semivar = np.zeros((n_cells, n_cells))
        
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.points[i]
                p2 = self.points[j]
                
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                
                if abs(dx) > max_distance or abs(dy) > max_distance:
                    continue
                
                cell_x = int(dx / cell_size) + center
                cell_y = int(dy / cell_size) + center
                
                if 0 <= cell_x < n_cells and 0 <= cell_y < n_cells:
                    sq_diff = (p1.value - p2.value)**2
                    map_semivar[cell_y, cell_x] += sq_diff
                    map_counts[cell_y, cell_x] += 1
                    
                    cell_x_neg = int(-dx / cell_size) + center
                    cell_y_neg = int(-dy / cell_size) + center
                    if 0 <= cell_x_neg < n_cells and 0 <= cell_y_neg < n_cells:
                        map_semivar[cell_y_neg, cell_x_neg] += sq_diff
                        map_counts[cell_y_neg, cell_x_neg] += 1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            map_result = np.where(map_counts > 0, map_semivar / (2 * map_counts), np.nan)
        
        return {
            "semivariance_map": map_result.tolist(),
            "count_map": map_counts.tolist(),
            "cell_size": cell_size,
            "max_distance": max_distance,
            "n_cells": n_cells,
            "center": center
        }
    
    def detect_anisotropy(self, n_directions: int = 36,
                         n_lags: int = 15,
                         max_distance: Optional[float] = None) -> Dict[str, Any]:
        """Detect anisotropy by calculating variograms in multiple directions."""
        
        directions = []
        variograms = []
        
        for i in range(n_directions):
            azimuth = i * 180 / n_directions
            
            vario = self.calculate_directional(
                azimuth=azimuth, dip=0,
                azimuth_tolerance=180 / n_directions / 2,
                dip_tolerance=90,
                n_lags=n_lags,
                max_distance=max_distance
            )
            
            directions.append(azimuth)
            variograms.append(vario)
        
        ranges = []
        for vario in variograms:
            semivar = vario.semivariances
            lags = vario.lags
            
            if semivar and max(semivar) > 0:
                sill_estimate = vario.data_variance
                range_estimate = None
                
                for i, sv in enumerate(semivar):
                    if sv >= sill_estimate * 0.95:
                        range_estimate = lags[i]
                        break
                
                if range_estimate is None and lags:
                    range_estimate = lags[-1]
                
                ranges.append(range_estimate or 0)
            else:
                ranges.append(0)
        
        if ranges:
            max_range_idx = np.argmax(ranges)
            min_range_idx = np.argmin([r for r in ranges if r > 0] or [0])
            
            major_azimuth = directions[max_range_idx]
            minor_azimuth = (major_azimuth + 90) % 180
            
            anisotropy_ratio = max(ranges) / min([r for r in ranges if r > 0] or [1])
        else:
            major_azimuth = 0
            minor_azimuth = 90
            anisotropy_ratio = 1.0
        
        return {
            "directions": directions,
            "ranges": ranges,
            "major_azimuth": major_azimuth,
            "minor_azimuth": minor_azimuth,
            "anisotropy_ratio": anisotropy_ratio,
            "is_anisotropic": anisotropy_ratio > 1.5,
            "variograms": [v.to_dict() for v in variograms]
        }


class VariogramModelFitter:
    """Fit variogram models to experimental data."""
    
    def __init__(self):
        self.available_models = list(VariogramModel)
    
    def fit_single_structure(self, experimental: ExperimentalVariogram,
                            model_type: VariogramModel = VariogramModel.SPHERICAL,
                            nugget: Optional[float] = None,
                            sill: Optional[float] = None,
                            range_val: Optional[float] = None) -> FittedVariogramModel:
        """Fit a single structure variogram model."""
        
        lags = np.array(experimental.lags)
        semivar = np.array(experimental.semivariances)
        counts = np.array(experimental.pair_counts)
        
        valid = (counts > 0) & ~np.isnan(semivar)
        lags = lags[valid]
        semivar = semivar[valid]
        counts = counts[valid]
        
        if len(lags) < 3:
            raise ValueError("Not enough valid lag bins for fitting")
        
        if nugget is None:
            if len(semivar) > 1:
                nugget = max(0, semivar[0] - (semivar[1] - semivar[0]))
            else:
                nugget = 0
        
        if sill is None:
            sill = experimental.data_variance
        
        if range_val is None:
            for i, sv in enumerate(semivar):
                if sv >= sill * 0.95:
                    range_val = lags[i]
                    break
            if range_val is None:
                range_val = lags[-1] * 0.7
        
        contribution = sill - nugget
        
        best_params = self._optimize_fit(
            lags, semivar, counts,
            model_type, nugget, contribution, range_val
        )
        
        structure = VariogramStructure(
            model_type=model_type,
            contribution=best_params["contribution"],
            range_major=best_params["range"],
            azimuth=experimental.azimuth,
            dip=experimental.dip
        )
        
        fitted = FittedVariogramModel(
            name=f"{experimental.name}_fitted",
            element=experimental.element,
            structures=[structure],
            nugget=best_params["nugget"],
            experimental=experimental
        )
        
        fitted.rmse = self._calculate_rmse(fitted, lags, semivar, counts)
        fitted.fit_quality = self._calculate_fit_quality(fitted, lags, semivar, counts)
        
        return fitted
    
    def fit_nested_structures(self, experimental: ExperimentalVariogram,
                             n_structures: int = 2,
                             model_types: Optional[List[VariogramModel]] = None
                             ) -> FittedVariogramModel:
        """Fit nested variogram model with multiple structures."""
        
        if model_types is None:
            model_types = [VariogramModel.SPHERICAL] * n_structures
        
        lags = np.array(experimental.lags)
        semivar = np.array(experimental.semivariances)
        counts = np.array(experimental.pair_counts)
        
        valid = (counts > 0) & ~np.isnan(semivar)
        lags = lags[valid]
        semivar = semivar[valid]
        counts = counts[valid]
        
        total_sill = experimental.data_variance
        nugget = max(0, semivar[0] * 0.5) if len(semivar) > 0 else 0
        
        structures = []
        remaining_sill = total_sill - nugget
        
        for i, model_type in enumerate(model_types):
            contribution = remaining_sill / (n_structures - i)
            
            if i == 0:
                range_val = lags[len(lags) // 3] if len(lags) > 3 else lags[-1] * 0.3
            else:
                range_val = lags[-1] * (i + 1) / n_structures
            
            structures.append(VariogramStructure(
                model_type=model_type,
                contribution=contribution,
                range_major=range_val,
                azimuth=experimental.azimuth,
                dip=experimental.dip
            ))
            
            remaining_sill -= contribution
        
        fitted = FittedVariogramModel(
            name=f"{experimental.name}_nested",
            element=experimental.element,
            structures=structures,
            nugget=nugget,
            experimental=experimental
        )
        
        fitted.rmse = self._calculate_rmse(fitted, lags, semivar, counts)
        fitted.fit_quality = self._calculate_fit_quality(fitted, lags, semivar, counts)
        
        return fitted
    
    def auto_fit(self, experimental: ExperimentalVariogram,
                max_structures: int = 3) -> FittedVariogramModel:
        """Automatically fit best variogram model."""
        
        best_model = None
        best_rmse = float('inf')
        
        for model_type in [VariogramModel.SPHERICAL, VariogramModel.EXPONENTIAL, 
                          VariogramModel.GAUSSIAN]:
            try:
                fitted = self.fit_single_structure(experimental, model_type)
                if fitted.rmse < best_rmse:
                    best_rmse = fitted.rmse
                    best_model = fitted
            except Exception:
                continue
        
        for n_struct in range(2, max_structures + 1):
            try:
                fitted = self.fit_nested_structures(experimental, n_struct)
                if fitted.rmse < best_rmse * 0.9:
                    best_rmse = fitted.rmse
                    best_model = fitted
            except Exception:
                continue
        
        return best_model
    
    def _optimize_fit(self, lags: np.ndarray, semivar: np.ndarray, 
                     counts: np.ndarray, model_type: VariogramModel,
                     nugget: float, contribution: float, range_val: float
                     ) -> Dict[str, float]:
        """Optimize model parameters using weighted least squares."""
        
        weights = counts / np.sum(counts)
        
        best_params = {
            "nugget": nugget,
            "contribution": contribution,
            "range": range_val
        }
        best_error = float('inf')
        
        for nugget_mult in [0.5, 0.75, 1.0, 1.25]:
            for contrib_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for range_mult in [0.6, 0.8, 1.0, 1.2, 1.4]:
                    test_nugget = nugget * nugget_mult
                    test_contrib = contribution * contrib_mult
                    test_range = range_val * range_mult
                    
                    structure = VariogramStructure(
                        model_type=model_type,
                        contribution=test_contrib,
                        range_major=test_range
                    )
                    
                    predicted = np.array([
                        test_nugget + structure.evaluate(h) for h in lags
                    ])
                    
                    error = np.sum(weights * (semivar - predicted)**2)
                    
                    if error < best_error:
                        best_error = error
                        best_params = {
                            "nugget": test_nugget,
                            "contribution": test_contrib,
                            "range": test_range
                        }
        
        return best_params
    
    def _calculate_rmse(self, model: FittedVariogramModel,
                       lags: np.ndarray, semivar: np.ndarray,
                       counts: np.ndarray) -> float:
        """Calculate root mean square error of fit."""
        predicted = np.array([model.evaluate(h) for h in lags])
        weights = counts / np.sum(counts)
        mse = np.sum(weights * (semivar - predicted)**2)
        return math.sqrt(mse)
    
    def _calculate_fit_quality(self, model: FittedVariogramModel,
                              lags: np.ndarray, semivar: np.ndarray,
                              counts: np.ndarray) -> float:
        """Calculate R-squared fit quality."""
        predicted = np.array([model.evaluate(h) for h in lags])
        
        ss_res = np.sum((semivar - predicted)**2)
        ss_tot = np.sum((semivar - np.mean(semivar))**2)
        
        if ss_tot == 0:
            return 1.0
        
        r_squared = 1 - ss_res / ss_tot
        return max(0, r_squared)


class CrossVariogramCalculator:
    """Calculate cross-variograms for multivariate analysis."""
    
    def __init__(self):
        self.points: List[Dict[str, Any]] = []
    
    def set_data(self, points: List[Dict[str, Any]]):
        """Set multivariate point data."""
        self.points = points
    
    def calculate_cross_variogram(self, element1: str, element2: str,
                                 n_lags: int = 15,
                                 lag_distance: Optional[float] = None,
                                 max_distance: Optional[float] = None
                                 ) -> Dict[str, Any]:
        """Calculate cross-variogram between two variables."""
        
        if not self.points:
            raise ValueError("No data points set")
        
        values1 = [p.get(element1, 0) for p in self.points]
        values2 = [p.get(element2, 0) for p in self.points]
        xs = [p.get('x', 0) for p in self.points]
        ys = [p.get('y', 0) for p in self.points]
        zs = [p.get('z', 0) for p in self.points]
        
        if max_distance is None:
            max_distance = math.sqrt(
                (max(xs) - min(xs))**2 + 
                (max(ys) - min(ys))**2 + 
                (max(zs) - min(zs))**2
            ) / 2
        
        if lag_distance is None:
            lag_distance = max_distance / n_lags
        
        lag_centers = [(i + 0.5) * lag_distance for i in range(n_lags)]
        lag_tolerance = lag_distance / 2
        
        cross_semivar = [0.0] * n_lags
        counts = [0] * n_lags
        
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt(
                    (xs[i] - xs[j])**2 + 
                    (ys[i] - ys[j])**2 + 
                    (zs[i] - zs[j])**2
                )
                
                if dist > max_distance or dist == 0:
                    continue
                
                cross_diff = (values1[i] - values1[j]) * (values2[i] - values2[j])
                
                for k, lag_center in enumerate(lag_centers):
                    if abs(dist - lag_center) <= lag_tolerance:
                        cross_semivar[k] += cross_diff
                        counts[k] += 1
                        break
        
        for k in range(n_lags):
            if counts[k] > 0:
                cross_semivar[k] /= (2 * counts[k])
        
        correlation = np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0
        
        return {
            "element1": element1,
            "element2": element2,
            "lags": lag_centers,
            "cross_semivariances": cross_semivar,
            "pair_counts": counts,
            "correlation": correlation,
            "coregionalization": "positive" if correlation > 0 else "negative"
        }


class VariographyWorkflow:
    """
    Complete variography workflow manager.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.calculator = ExperimentalVariogramCalculator()
        self.fitter = VariogramModelFitter()
        self.cross_calculator = CrossVariogramCalculator()
        
        self.experimental_variograms: Dict[str, ExperimentalVariogram] = {}
        self.fitted_models: Dict[str, FittedVariogramModel] = {}
    
    def load_data(self, points: List[Dict[str, Any]], value_field: str = "value"):
        """Load point data for variography."""
        point_objects = []
        for p in points:
            point_objects.append(Point3D(
                x=p.get('x', p.get('easting', 0)),
                y=p.get('y', p.get('northing', 0)),
                z=p.get('z', p.get('elevation', 0)),
                value=p.get(value_field, 0),
                weight=p.get('weight', 1.0)
            ))
        
        self.calculator.set_data(point_objects)
        self.cross_calculator.set_data(points)
    
    def calculate_experimental(self, name: str,
                              azimuth: float = 0, dip: float = 0,
                              azimuth_tolerance: float = 22.5,
                              dip_tolerance: float = 22.5,
                              n_lags: int = 15,
                              element: str = "value") -> ExperimentalVariogram:
        """Calculate and store experimental variogram."""
        
        if azimuth_tolerance >= 180 and dip_tolerance >= 90:
            vario = self.calculator.calculate_omnidirectional(
                n_lags=n_lags, element=element
            )
        else:
            vario = self.calculator.calculate_directional(
                azimuth=azimuth, dip=dip,
                azimuth_tolerance=azimuth_tolerance,
                dip_tolerance=dip_tolerance,
                n_lags=n_lags,
                element=element,
                name=name
            )
        
        self.experimental_variograms[name] = vario
        return vario
    
    def fit_model(self, experimental_name: str,
                 model_type: str = "spherical",
                 auto: bool = False) -> FittedVariogramModel:
        """Fit variogram model to experimental data."""
        
        if experimental_name not in self.experimental_variograms:
            raise ValueError(f"Experimental variogram '{experimental_name}' not found")
        
        experimental = self.experimental_variograms[experimental_name]
        
        if auto:
            fitted = self.fitter.auto_fit(experimental)
        else:
            model_enum = VariogramModel(model_type)
            fitted = self.fitter.fit_single_structure(experimental, model_enum)
        
        self.fitted_models[experimental_name] = fitted
        return fitted
    
    def detect_anisotropy(self, n_directions: int = 18) -> Dict[str, Any]:
        """Detect anisotropy in the data."""
        return self.calculator.detect_anisotropy(n_directions=n_directions)
    
    def get_variogram_map(self, max_distance: float, 
                         cell_size: float = 10.0) -> Dict[str, Any]:
        """Generate variogram map for anisotropy visualization."""
        return self.calculator.calculate_variogram_map(max_distance, cell_size)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of variography analysis."""
        return {
            "project": self.project_name,
            "n_points": len(self.calculator.points),
            "experimental_variograms": list(self.experimental_variograms.keys()),
            "fitted_models": {
                name: model.to_dict() 
                for name, model in self.fitted_models.items()
            }
        }


def create_variography_workflow(project_name: str = "default") -> VariographyWorkflow:
    """Factory function to create a variography workflow."""
    return VariographyWorkflow(project_name)
