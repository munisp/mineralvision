"""
Kriging Module for MineralVision Platform.

Comprehensive kriging implementation including:
1. Ordinary Kriging (OK)
2. Simple Kriging (SK)
3. Universal Kriging (UK)
4. Indicator Kriging (IK)
5. Cokriging for multivariate estimation
6. Search neighborhood optimization
7. Cross-validation and jackknife analysis
8. Kriging variance and efficiency metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import numpy as np
from collections import defaultdict


class KrigingType(Enum):
    """Types of kriging methods."""
    ORDINARY = "ordinary"
    SIMPLE = "simple"
    UNIVERSAL = "universal"
    INDICATOR = "indicator"
    LOGNORMAL = "lognormal"
    MULTIGAUSSIAN = "multigaussian"


class SearchType(Enum):
    """Search neighborhood types."""
    OCTANT = "octant"
    QUADRANT = "quadrant"
    ELLIPSOID = "ellipsoid"
    NEAREST = "nearest"


class DriftType(Enum):
    """Drift functions for Universal Kriging."""
    CONSTANT = "constant"
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    CUSTOM = "custom"


@dataclass
class Point3D:
    """3D point with value and weight."""
    x: float
    y: float
    z: float
    value: float = 0.0
    weight: float = 1.0
    sample_id: str = ""
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class SearchEllipsoid:
    """Search ellipsoid parameters."""
    radius_major: float
    radius_minor: float
    radius_vertical: float
    azimuth: float = 0.0
    dip: float = 0.0
    plunge: float = 0.0
    
    def contains(self, dx: float, dy: float, dz: float) -> bool:
        """Check if point offset is within ellipsoid."""
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
        
        scaled_dist = math.sqrt(
            (dx_rot2 / self.radius_major)**2 +
            (dy_rot / self.radius_minor)**2 +
            (dz_rot2 / self.radius_vertical)**2
        )
        
        return scaled_dist <= 1.0
    
    def scaled_distance(self, dx: float, dy: float, dz: float) -> float:
        """Calculate anisotropic scaled distance."""
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
        
        return math.sqrt(
            (dx_rot2 / self.radius_major)**2 +
            (dy_rot / self.radius_minor)**2 +
            (dz_rot2 / self.radius_vertical)**2
        ) * self.radius_major


@dataclass
class SearchParameters:
    """Search neighborhood parameters."""
    ellipsoid: SearchEllipsoid
    search_type: SearchType = SearchType.OCTANT
    min_samples: int = 4
    max_samples: int = 16
    max_per_octant: int = 4
    max_per_quadrant: int = 8
    use_octant_search: bool = True
    discretization: int = 1


@dataclass
class VariogramModel:
    """Variogram model for kriging."""
    nugget: float
    structures: List[Dict[str, Any]]
    sill: float = 0.0
    
    def __post_init__(self):
        self.sill = self.nugget + sum(s.get('contribution', 0) for s in self.structures)
    
    def covariance(self, h: float) -> float:
        """Calculate covariance at lag h."""
        return self.sill - self.semivariance(h)
    
    def semivariance(self, h: float) -> float:
        """Calculate semivariance at lag h."""
        if h <= 0:
            return 0.0
        
        gamma = self.nugget
        
        for structure in self.structures:
            model_type = structure.get('model', 'spherical')
            contribution = structure.get('contribution', 0)
            range_val = structure.get('range', 1)
            
            if model_type == 'spherical':
                if h >= range_val:
                    gamma += contribution
                else:
                    hr = h / range_val
                    gamma += contribution * (1.5 * hr - 0.5 * hr**3)
            
            elif model_type == 'exponential':
                gamma += contribution * (1 - math.exp(-3 * h / range_val))
            
            elif model_type == 'gaussian':
                gamma += contribution * (1 - math.exp(-3 * (h / range_val)**2))
            
            elif model_type == 'linear':
                if h >= range_val:
                    gamma += contribution
                else:
                    gamma += contribution * h / range_val
        
        return gamma
    
    def covariance_anisotropic(self, dx: float, dy: float, dz: float,
                               ellipsoid: SearchEllipsoid) -> float:
        """Calculate covariance with anisotropy."""
        h = ellipsoid.scaled_distance(dx, dy, dz)
        return self.covariance(h)


@dataclass
class KrigingResult:
    """Result of kriging estimation at a single point."""
    x: float
    y: float
    z: float
    estimate: float
    variance: float
    std_error: float
    n_samples: int
    kriging_efficiency: float
    slope_of_regression: float
    weights: List[float] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    lagrange_multiplier: float = 0.0


@dataclass
class CrossValidationResult:
    """Cross-validation result for a single sample."""
    sample_id: str
    actual_value: float
    estimated_value: float
    kriging_variance: float
    error: float
    standardized_error: float
    n_neighbors: int


@dataclass
class KrigingStatistics:
    """Summary statistics for kriging estimation."""
    n_estimated: int
    n_failed: int
    mean_estimate: float
    variance_estimate: float
    mean_kriging_variance: float
    mean_n_samples: float
    mean_kriging_efficiency: float
    min_estimate: float
    max_estimate: float
    negative_weights_count: int


class OrdinaryKriging:
    """Ordinary Kriging implementation."""
    
    def __init__(self, variogram: VariogramModel, search_params: SearchParameters):
        self.variogram = variogram
        self.search = search_params
        self.data_points: List[Point3D] = []
        self.data_mean: float = 0.0
        self.data_variance: float = 0.0
    
    def set_data(self, points: List[Point3D]):
        """Set sample data for kriging."""
        self.data_points = points
        if points:
            values = [p.value for p in points]
            self.data_mean = np.mean(values)
            self.data_variance = np.var(values, ddof=1)
    
    def find_neighbors(self, target: Point3D) -> List[Tuple[Point3D, float]]:
        """Find neighboring samples within search ellipsoid."""
        neighbors = []
        
        for point in self.data_points:
            dx = point.x - target.x
            dy = point.y - target.y
            dz = point.z - target.z
            
            if self.search.ellipsoid.contains(dx, dy, dz):
                dist = self.search.ellipsoid.scaled_distance(dx, dy, dz)
                neighbors.append((point, dist))
        
        neighbors.sort(key=lambda x: x[1])
        
        if self.search.use_octant_search and self.search.search_type == SearchType.OCTANT:
            neighbors = self._octant_search(target, neighbors)
        elif self.search.search_type == SearchType.QUADRANT:
            neighbors = self._quadrant_search(target, neighbors)
        else:
            neighbors = neighbors[:self.search.max_samples]
        
        return neighbors
    
    def _octant_search(self, target: Point3D, 
                      neighbors: List[Tuple[Point3D, float]]) -> List[Tuple[Point3D, float]]:
        """Apply octant search constraints."""
        octants: Dict[int, List[Tuple[Point3D, float]]] = defaultdict(list)
        
        for point, dist in neighbors:
            dx = point.x - target.x
            dy = point.y - target.y
            dz = point.z - target.z
            
            octant = 0
            if dx >= 0:
                octant += 1
            if dy >= 0:
                octant += 2
            if dz >= 0:
                octant += 4
            
            if len(octants[octant]) < self.search.max_per_octant:
                octants[octant].append((point, dist))
        
        result = []
        for oct_neighbors in octants.values():
            result.extend(oct_neighbors)
        
        result.sort(key=lambda x: x[1])
        return result[:self.search.max_samples]
    
    def _quadrant_search(self, target: Point3D,
                        neighbors: List[Tuple[Point3D, float]]) -> List[Tuple[Point3D, float]]:
        """Apply quadrant search constraints."""
        quadrants: Dict[int, List[Tuple[Point3D, float]]] = defaultdict(list)
        
        for point, dist in neighbors:
            dx = point.x - target.x
            dy = point.y - target.y
            
            quadrant = 0
            if dx >= 0:
                quadrant += 1
            if dy >= 0:
                quadrant += 2
            
            if len(quadrants[quadrant]) < self.search.max_per_quadrant:
                quadrants[quadrant].append((point, dist))
        
        result = []
        for quad_neighbors in quadrants.values():
            result.extend(quad_neighbors)
        
        result.sort(key=lambda x: x[1])
        return result[:self.search.max_samples]
    
    def estimate(self, target: Point3D) -> Optional[KrigingResult]:
        """Estimate value at target location using Ordinary Kriging."""
        neighbors = self.find_neighbors(target)
        
        if len(neighbors) < self.search.min_samples:
            return None
        
        n = len(neighbors)
        
        K = np.zeros((n + 1, n + 1))
        k = np.zeros(n + 1)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    K[i, j] = self.variogram.sill
                else:
                    pi = neighbors[i][0]
                    pj = neighbors[j][0]
                    dx = pi.x - pj.x
                    dy = pi.y - pj.y
                    dz = pi.z - pj.z
                    h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
                    K[i, j] = self.variogram.covariance(h)
        
        K[:n, n] = 1.0
        K[n, :n] = 1.0
        K[n, n] = 0.0
        
        for i in range(n):
            pi = neighbors[i][0]
            dx = target.x - pi.x
            dy = target.y - pi.y
            dz = target.z - pi.z
            h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
            k[i] = self.variogram.covariance(h)
        
        k[n] = 1.0
        
        try:
            weights = np.linalg.solve(K, k)
        except np.linalg.LinAlgError:
            try:
                weights = np.linalg.lstsq(K, k, rcond=None)[0]
            except:
                return None
        
        lambda_weights = weights[:n]
        lagrange = weights[n]
        
        estimate = sum(lambda_weights[i] * neighbors[i][0].value for i in range(n))
        
        kriging_variance = self.variogram.sill - sum(lambda_weights[i] * k[i] for i in range(n)) - lagrange
        kriging_variance = max(0, kriging_variance)
        
        kriging_efficiency = 1 - kriging_variance / self.data_variance if self.data_variance > 0 else 0
        
        slope = sum(lambda_weights)
        
        return KrigingResult(
            x=target.x,
            y=target.y,
            z=target.z,
            estimate=estimate,
            variance=kriging_variance,
            std_error=math.sqrt(kriging_variance),
            n_samples=n,
            kriging_efficiency=kriging_efficiency,
            slope_of_regression=slope,
            weights=lambda_weights.tolist(),
            sample_ids=[neighbors[i][0].sample_id for i in range(n)],
            lagrange_multiplier=lagrange
        )
    
    def estimate_grid(self, x_coords: List[float], y_coords: List[float], 
                     z_coords: List[float]) -> List[KrigingResult]:
        """Estimate values on a regular grid."""
        results = []
        
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    target = Point3D(x, y, z)
                    result = self.estimate(target)
                    if result:
                        results.append(result)
        
        return results
    
    def cross_validate(self, leave_out: int = 1) -> List[CrossValidationResult]:
        """Perform leave-one-out cross-validation."""
        results = []
        
        original_points = self.data_points.copy()
        
        for i, point in enumerate(original_points):
            self.data_points = original_points[:i] + original_points[i+1:]
            
            target = Point3D(point.x, point.y, point.z)
            estimate_result = self.estimate(target)
            
            if estimate_result:
                error = point.value - estimate_result.estimate
                std_error = error / estimate_result.std_error if estimate_result.std_error > 0 else 0
                
                results.append(CrossValidationResult(
                    sample_id=point.sample_id,
                    actual_value=point.value,
                    estimated_value=estimate_result.estimate,
                    kriging_variance=estimate_result.variance,
                    error=error,
                    standardized_error=std_error,
                    n_neighbors=estimate_result.n_samples
                ))
        
        self.data_points = original_points
        
        return results
    
    def get_statistics(self, results: List[KrigingResult]) -> KrigingStatistics:
        """Calculate summary statistics from kriging results."""
        if not results:
            return KrigingStatistics(
                n_estimated=0, n_failed=0, mean_estimate=0, variance_estimate=0,
                mean_kriging_variance=0, mean_n_samples=0, mean_kriging_efficiency=0,
                min_estimate=0, max_estimate=0, negative_weights_count=0
            )
        
        estimates = [r.estimate for r in results]
        variances = [r.variance for r in results]
        n_samples = [r.n_samples for r in results]
        efficiencies = [r.kriging_efficiency for r in results]
        
        neg_weights = sum(1 for r in results if any(w < 0 for w in r.weights))
        
        return KrigingStatistics(
            n_estimated=len(results),
            n_failed=0,
            mean_estimate=np.mean(estimates),
            variance_estimate=np.var(estimates, ddof=1),
            mean_kriging_variance=np.mean(variances),
            mean_n_samples=np.mean(n_samples),
            mean_kriging_efficiency=np.mean(efficiencies),
            min_estimate=min(estimates),
            max_estimate=max(estimates),
            negative_weights_count=neg_weights
        )


class SimpleKriging(OrdinaryKriging):
    """Simple Kriging implementation with known mean."""
    
    def __init__(self, variogram: VariogramModel, search_params: SearchParameters,
                 known_mean: Optional[float] = None):
        super().__init__(variogram, search_params)
        self.known_mean = known_mean
    
    def estimate(self, target: Point3D) -> Optional[KrigingResult]:
        """Estimate value at target location using Simple Kriging."""
        neighbors = self.find_neighbors(target)
        
        if len(neighbors) < self.search.min_samples:
            return None
        
        n = len(neighbors)
        
        mean = self.known_mean if self.known_mean is not None else self.data_mean
        
        K = np.zeros((n, n))
        k = np.zeros(n)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    K[i, j] = self.variogram.sill
                else:
                    pi = neighbors[i][0]
                    pj = neighbors[j][0]
                    dx = pi.x - pj.x
                    dy = pi.y - pj.y
                    dz = pi.z - pj.z
                    h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
                    K[i, j] = self.variogram.covariance(h)
        
        for i in range(n):
            pi = neighbors[i][0]
            dx = target.x - pi.x
            dy = target.y - pi.y
            dz = target.z - pi.z
            h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
            k[i] = self.variogram.covariance(h)
        
        try:
            weights = np.linalg.solve(K, k)
        except np.linalg.LinAlgError:
            try:
                weights = np.linalg.lstsq(K, k, rcond=None)[0]
            except:
                return None
        
        residuals = [neighbors[i][0].value - mean for i in range(n)]
        estimate = mean + sum(weights[i] * residuals[i] for i in range(n))
        
        kriging_variance = self.variogram.sill - sum(weights[i] * k[i] for i in range(n))
        kriging_variance = max(0, kriging_variance)
        
        kriging_efficiency = 1 - kriging_variance / self.data_variance if self.data_variance > 0 else 0
        
        return KrigingResult(
            x=target.x,
            y=target.y,
            z=target.z,
            estimate=estimate,
            variance=kriging_variance,
            std_error=math.sqrt(kriging_variance),
            n_samples=n,
            kriging_efficiency=kriging_efficiency,
            slope_of_regression=sum(weights),
            weights=weights.tolist(),
            sample_ids=[neighbors[i][0].sample_id for i in range(n)],
            lagrange_multiplier=0.0
        )


class UniversalKriging(OrdinaryKriging):
    """Universal Kriging with drift functions."""
    
    def __init__(self, variogram: VariogramModel, search_params: SearchParameters,
                 drift_type: DriftType = DriftType.LINEAR):
        super().__init__(variogram, search_params)
        self.drift_type = drift_type
    
    def _drift_functions(self, x: float, y: float, z: float) -> List[float]:
        """Calculate drift function values at a point."""
        if self.drift_type == DriftType.CONSTANT:
            return [1.0]
        elif self.drift_type == DriftType.LINEAR:
            return [1.0, x, y, z]
        elif self.drift_type == DriftType.QUADRATIC:
            return [1.0, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z]
        else:
            return [1.0]
    
    def estimate(self, target: Point3D) -> Optional[KrigingResult]:
        """Estimate value at target location using Universal Kriging."""
        neighbors = self.find_neighbors(target)
        
        if len(neighbors) < self.search.min_samples:
            return None
        
        n = len(neighbors)
        n_drift = len(self._drift_functions(0, 0, 0))
        
        K = np.zeros((n + n_drift, n + n_drift))
        k = np.zeros(n + n_drift)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    K[i, j] = self.variogram.sill
                else:
                    pi = neighbors[i][0]
                    pj = neighbors[j][0]
                    dx = pi.x - pj.x
                    dy = pi.y - pj.y
                    dz = pi.z - pj.z
                    h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
                    K[i, j] = self.variogram.covariance(h)
        
        for i in range(n):
            pi = neighbors[i][0]
            drift_vals = self._drift_functions(pi.x, pi.y, pi.z)
            for j, dv in enumerate(drift_vals):
                K[i, n + j] = dv
                K[n + j, i] = dv
        
        for i in range(n):
            pi = neighbors[i][0]
            dx = target.x - pi.x
            dy = target.y - pi.y
            dz = target.z - pi.z
            h = self.search.ellipsoid.scaled_distance(dx, dy, dz)
            k[i] = self.variogram.covariance(h)
        
        target_drift = self._drift_functions(target.x, target.y, target.z)
        for j, dv in enumerate(target_drift):
            k[n + j] = dv
        
        try:
            weights = np.linalg.solve(K, k)
        except np.linalg.LinAlgError:
            try:
                weights = np.linalg.lstsq(K, k, rcond=None)[0]
            except:
                return None
        
        lambda_weights = weights[:n]
        lagrange = weights[n:]
        
        estimate = sum(lambda_weights[i] * neighbors[i][0].value for i in range(n))
        
        kriging_variance = self.variogram.sill - sum(lambda_weights[i] * k[i] for i in range(n))
        for j in range(n_drift):
            kriging_variance -= lagrange[j] * target_drift[j]
        kriging_variance = max(0, kriging_variance)
        
        kriging_efficiency = 1 - kriging_variance / self.data_variance if self.data_variance > 0 else 0
        
        return KrigingResult(
            x=target.x,
            y=target.y,
            z=target.z,
            estimate=estimate,
            variance=kriging_variance,
            std_error=math.sqrt(kriging_variance),
            n_samples=n,
            kriging_efficiency=kriging_efficiency,
            slope_of_regression=sum(lambda_weights),
            weights=lambda_weights.tolist(),
            sample_ids=[neighbors[i][0].sample_id for i in range(n)],
            lagrange_multiplier=lagrange[0] if lagrange.size > 0 else 0.0
        )


class IndicatorKriging:
    """Indicator Kriging for categorical or threshold estimation."""
    
    def __init__(self, variogram: VariogramModel, search_params: SearchParameters,
                 thresholds: List[float]):
        self.variogram = variogram
        self.search = search_params
        self.thresholds = sorted(thresholds)
        self.indicator_variograms: Dict[float, VariogramModel] = {}
        self.ok_krigers: Dict[float, OrdinaryKriging] = {}
    
    def set_data(self, points: List[Point3D]):
        """Set sample data and create indicator transforms."""
        self.data_points = points
        
        for threshold in self.thresholds:
            indicator_points = []
            for p in points:
                ind_value = 1.0 if p.value <= threshold else 0.0
                indicator_points.append(Point3D(
                    p.x, p.y, p.z, ind_value, p.weight, p.sample_id
                ))
            
            if threshold not in self.indicator_variograms:
                self.indicator_variograms[threshold] = self.variogram
            
            kriging = OrdinaryKriging(
                self.indicator_variograms[threshold],
                self.search
            )
            kriging.set_data(indicator_points)
            self.ok_krigers[threshold] = kriging
    
    def set_indicator_variogram(self, threshold: float, variogram: VariogramModel):
        """Set variogram for a specific threshold."""
        self.indicator_variograms[threshold] = variogram
        if threshold in self.ok_krigers:
            self.ok_krigers[threshold].variogram = variogram
    
    def estimate(self, target: Point3D) -> Dict[str, Any]:
        """Estimate indicator probabilities at target location."""
        probabilities = {}
        variances = {}
        
        for threshold in self.thresholds:
            if threshold in self.ok_krigers:
                result = self.ok_krigers[threshold].estimate(target)
                if result:
                    prob = max(0, min(1, result.estimate))
                    probabilities[threshold] = prob
                    variances[threshold] = result.variance
        
        probabilities = self._order_correct(probabilities)
        
        ccdf = {}
        prev_prob = 0
        for threshold in self.thresholds:
            if threshold in probabilities:
                ccdf[threshold] = probabilities[threshold] - prev_prob
                prev_prob = probabilities[threshold]
        
        e_type = self._calculate_e_type(probabilities)
        
        return {
            "x": target.x,
            "y": target.y,
            "z": target.z,
            "probabilities": probabilities,
            "variances": variances,
            "ccdf": ccdf,
            "e_type_estimate": e_type,
            "thresholds": self.thresholds
        }
    
    def _order_correct(self, probabilities: Dict[float, float]) -> Dict[float, float]:
        """Apply order correction to ensure monotonic CDF."""
        corrected = {}
        prev_prob = 0
        
        for threshold in self.thresholds:
            if threshold in probabilities:
                prob = max(prev_prob, probabilities[threshold])
                corrected[threshold] = prob
                prev_prob = prob
        
        return corrected
    
    def _calculate_e_type(self, probabilities: Dict[float, float]) -> float:
        """Calculate E-type estimate from indicator probabilities."""
        if not probabilities:
            return 0.0
        
        e_type = 0.0
        prev_prob = 0.0
        prev_threshold = self.thresholds[0] if self.thresholds else 0
        
        for threshold in self.thresholds:
            if threshold in probabilities:
                prob = probabilities[threshold]
                mid_value = (prev_threshold + threshold) / 2
                e_type += mid_value * (prob - prev_prob)
                prev_prob = prob
                prev_threshold = threshold
        
        return e_type


class Cokriging:
    """Cokriging for multivariate estimation."""
    
    def __init__(self, primary_variogram: VariogramModel,
                 secondary_variogram: VariogramModel,
                 cross_variogram: VariogramModel,
                 search_params: SearchParameters):
        self.primary_variogram = primary_variogram
        self.secondary_variogram = secondary_variogram
        self.cross_variogram = cross_variogram
        self.search = search_params
        self.primary_points: List[Point3D] = []
        self.secondary_points: List[Point3D] = []
    
    def set_data(self, primary_points: List[Point3D], secondary_points: List[Point3D]):
        """Set primary and secondary variable data."""
        self.primary_points = primary_points
        self.secondary_points = secondary_points
    
    def estimate(self, target: Point3D) -> Optional[Dict[str, Any]]:
        """Estimate primary variable using cokriging."""
        primary_neighbors = self._find_neighbors(target, self.primary_points)
        secondary_neighbors = self._find_neighbors(target, self.secondary_points)
        
        if len(primary_neighbors) < self.search.min_samples:
            return None
        
        n1 = len(primary_neighbors)
        n2 = len(secondary_neighbors)
        n = n1 + n2
        
        K = np.zeros((n + 2, n + 2))
        k = np.zeros(n + 2)
        
        for i in range(n1):
            for j in range(n1):
                if i == j:
                    K[i, j] = self.primary_variogram.sill
                else:
                    pi = primary_neighbors[i][0]
                    pj = primary_neighbors[j][0]
                    h = self._scaled_distance(pi, pj)
                    K[i, j] = self.primary_variogram.covariance(h)
        
        for i in range(n2):
            for j in range(n2):
                if i == j:
                    K[n1 + i, n1 + j] = self.secondary_variogram.sill
                else:
                    pi = secondary_neighbors[i][0]
                    pj = secondary_neighbors[j][0]
                    h = self._scaled_distance(pi, pj)
                    K[n1 + i, n1 + j] = self.secondary_variogram.covariance(h)
        
        for i in range(n1):
            for j in range(n2):
                pi = primary_neighbors[i][0]
                pj = secondary_neighbors[j][0]
                h = self._scaled_distance(pi, pj)
                cov = self.cross_variogram.covariance(h)
                K[i, n1 + j] = cov
                K[n1 + j, i] = cov
        
        K[:n1, n] = 1.0
        K[n, :n1] = 1.0
        K[n1:n, n + 1] = 1.0
        K[n + 1, n1:n] = 1.0
        
        for i in range(n1):
            pi = primary_neighbors[i][0]
            h = self._scaled_distance_to_target(target, pi)
            k[i] = self.primary_variogram.covariance(h)
        
        for i in range(n2):
            pi = secondary_neighbors[i][0]
            h = self._scaled_distance_to_target(target, pi)
            k[n1 + i] = self.cross_variogram.covariance(h)
        
        k[n] = 1.0
        k[n + 1] = 0.0
        
        try:
            weights = np.linalg.solve(K, k)
        except np.linalg.LinAlgError:
            return None
        
        primary_weights = weights[:n1]
        secondary_weights = weights[n1:n]
        
        estimate = (sum(primary_weights[i] * primary_neighbors[i][0].value for i in range(n1)) +
                   sum(secondary_weights[i] * secondary_neighbors[i][0].value for i in range(n2)))
        
        variance = self.primary_variogram.sill - sum(weights[i] * k[i] for i in range(n + 2))
        variance = max(0, variance)
        
        return {
            "x": target.x,
            "y": target.y,
            "z": target.z,
            "estimate": estimate,
            "variance": variance,
            "std_error": math.sqrt(variance),
            "n_primary": n1,
            "n_secondary": n2,
            "primary_weights": primary_weights.tolist(),
            "secondary_weights": secondary_weights.tolist()
        }
    
    def _find_neighbors(self, target: Point3D, 
                       points: List[Point3D]) -> List[Tuple[Point3D, float]]:
        """Find neighboring samples."""
        neighbors = []
        
        for point in points:
            dx = point.x - target.x
            dy = point.y - target.y
            dz = point.z - target.z
            
            if self.search.ellipsoid.contains(dx, dy, dz):
                dist = self.search.ellipsoid.scaled_distance(dx, dy, dz)
                neighbors.append((point, dist))
        
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:self.search.max_samples]
    
    def _scaled_distance(self, p1: Point3D, p2: Point3D) -> float:
        """Calculate scaled distance between two points."""
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return self.search.ellipsoid.scaled_distance(dx, dy, dz)
    
    def _scaled_distance_to_target(self, target: Point3D, point: Point3D) -> float:
        """Calculate scaled distance from target to point."""
        dx = target.x - point.x
        dy = target.y - point.y
        dz = target.z - point.z
        return self.search.ellipsoid.scaled_distance(dx, dy, dz)


class KrigingWorkflow:
    """
    Complete kriging workflow manager.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.data_points: List[Point3D] = []
        self.variogram: Optional[VariogramModel] = None
        self.search_params: Optional[SearchParameters] = None
        self.kriging_type: KrigingType = KrigingType.ORDINARY
        self.kriging_engine: Optional[OrdinaryKriging] = None
        self.results: List[KrigingResult] = []
    
    def load_data(self, points: List[Dict[str, Any]], value_field: str = "value"):
        """Load point data for kriging."""
        self.data_points = []
        for i, p in enumerate(points):
            self.data_points.append(Point3D(
                x=p.get('x', p.get('easting', 0)),
                y=p.get('y', p.get('northing', 0)),
                z=p.get('z', p.get('elevation', 0)),
                value=p.get(value_field, 0),
                weight=p.get('weight', 1.0),
                sample_id=p.get('sample_id', str(i))
            ))
    
    def set_variogram(self, nugget: float, structures: List[Dict[str, Any]]):
        """Set variogram model."""
        self.variogram = VariogramModel(nugget=nugget, structures=structures)
    
    def set_search_parameters(self, radius_major: float, radius_minor: float,
                             radius_vertical: float, azimuth: float = 0,
                             dip: float = 0, min_samples: int = 4,
                             max_samples: int = 16, search_type: str = "octant"):
        """Set search neighborhood parameters."""
        ellipsoid = SearchEllipsoid(
            radius_major=radius_major,
            radius_minor=radius_minor,
            radius_vertical=radius_vertical,
            azimuth=azimuth,
            dip=dip
        )
        
        search_type_enum = SearchType(search_type) if search_type in [e.value for e in SearchType] else SearchType.OCTANT
        
        self.search_params = SearchParameters(
            ellipsoid=ellipsoid,
            search_type=search_type_enum,
            min_samples=min_samples,
            max_samples=max_samples
        )
    
    def initialize_kriging(self, kriging_type: str = "ordinary", **kwargs):
        """Initialize kriging engine."""
        if not self.variogram or not self.search_params:
            raise ValueError("Variogram and search parameters must be set first")
        
        self.kriging_type = KrigingType(kriging_type) if kriging_type in [e.value for e in KrigingType] else KrigingType.ORDINARY
        
        if self.kriging_type == KrigingType.ORDINARY:
            self.kriging_engine = OrdinaryKriging(self.variogram, self.search_params)
        elif self.kriging_type == KrigingType.SIMPLE:
            known_mean = kwargs.get('known_mean')
            self.kriging_engine = SimpleKriging(self.variogram, self.search_params, known_mean)
        elif self.kriging_type == KrigingType.UNIVERSAL:
            drift_type = DriftType(kwargs.get('drift_type', 'linear'))
            self.kriging_engine = UniversalKriging(self.variogram, self.search_params, drift_type)
        else:
            self.kriging_engine = OrdinaryKriging(self.variogram, self.search_params)
        
        self.kriging_engine.set_data(self.data_points)
    
    def estimate_point(self, x: float, y: float, z: float) -> Optional[KrigingResult]:
        """Estimate value at a single point."""
        if not self.kriging_engine:
            raise ValueError("Kriging engine not initialized")
        
        target = Point3D(x, y, z)
        return self.kriging_engine.estimate(target)
    
    def estimate_grid(self, x_min: float, x_max: float, x_step: float,
                     y_min: float, y_max: float, y_step: float,
                     z_min: float, z_max: float, z_step: float) -> List[KrigingResult]:
        """Estimate values on a regular grid."""
        if not self.kriging_engine:
            raise ValueError("Kriging engine not initialized")
        
        x_coords = np.arange(x_min, x_max + x_step, x_step).tolist()
        y_coords = np.arange(y_min, y_max + y_step, y_step).tolist()
        z_coords = np.arange(z_min, z_max + z_step, z_step).tolist()
        
        self.results = self.kriging_engine.estimate_grid(x_coords, y_coords, z_coords)
        return self.results
    
    def cross_validate(self) -> List[CrossValidationResult]:
        """Perform cross-validation."""
        if not self.kriging_engine:
            raise ValueError("Kriging engine not initialized")
        
        return self.kriging_engine.cross_validate()
    
    def get_statistics(self) -> KrigingStatistics:
        """Get kriging statistics."""
        if not self.kriging_engine:
            raise ValueError("Kriging engine not initialized")
        
        return self.kriging_engine.get_statistics(self.results)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        stats = self.get_statistics() if self.results else None
        
        return {
            "project": self.project_name,
            "kriging_type": self.kriging_type.value,
            "n_data_points": len(self.data_points),
            "n_estimated": len(self.results),
            "variogram": {
                "nugget": self.variogram.nugget if self.variogram else 0,
                "sill": self.variogram.sill if self.variogram else 0,
                "structures": self.variogram.structures if self.variogram else []
            },
            "search": {
                "radius_major": self.search_params.ellipsoid.radius_major if self.search_params else 0,
                "radius_minor": self.search_params.ellipsoid.radius_minor if self.search_params else 0,
                "radius_vertical": self.search_params.ellipsoid.radius_vertical if self.search_params else 0,
                "min_samples": self.search_params.min_samples if self.search_params else 0,
                "max_samples": self.search_params.max_samples if self.search_params else 0
            },
            "statistics": {
                "mean_estimate": stats.mean_estimate if stats else 0,
                "mean_kriging_variance": stats.mean_kriging_variance if stats else 0,
                "mean_kriging_efficiency": stats.mean_kriging_efficiency if stats else 0
            } if stats else None
        }


def create_kriging_workflow(project_name: str = "default") -> KrigingWorkflow:
    """Factory function to create a kriging workflow."""
    return KrigingWorkflow(project_name)


def create_variogram_model(nugget: float, model_type: str, contribution: float,
                          range_val: float) -> VariogramModel:
    """Factory function to create a simple variogram model."""
    return VariogramModel(
        nugget=nugget,
        structures=[{
            "model": model_type,
            "contribution": contribution,
            "range": range_val
        }]
    )


def create_search_ellipsoid(radius_major: float, radius_minor: float,
                           radius_vertical: float, azimuth: float = 0,
                           dip: float = 0) -> SearchEllipsoid:
    """Factory function to create a search ellipsoid."""
    return SearchEllipsoid(
        radius_major=radius_major,
        radius_minor=radius_minor,
        radius_vertical=radius_vertical,
        azimuth=azimuth,
        dip=dip
    )
