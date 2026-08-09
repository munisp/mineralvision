"""
Geophysical Inversion Module for MineralVision Platform.

Comprehensive geophysical inversion including:
1. Magnetic inversion (susceptibility)
2. Gravity inversion (density)
3. Electromagnetic inversion (conductivity)
4. IP inversion (chargeability)
5. Forward modeling
6. Regularization and constraints
7. Depth weighting and reference models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import numpy as np
from collections import defaultdict


class InversionType(Enum):
    """Types of geophysical inversion."""
    MAGNETIC = "magnetic"
    GRAVITY = "gravity"
    ELECTROMAGNETIC = "electromagnetic"
    IP = "induced_polarization"
    RESISTIVITY = "resistivity"
    SEISMIC = "seismic"


class RegularizationType(Enum):
    """Regularization types."""
    TIKHONOV = "tikhonov"
    TOTAL_VARIATION = "total_variation"
    MINIMUM_SUPPORT = "minimum_support"
    COMPACT = "compact"
    SMOOTH = "smooth"


class DepthWeightingType(Enum):
    """Depth weighting schemes."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SENSITIVITY = "sensitivity"
    CUSTOM = "custom"


@dataclass
class Point3D:
    """3D point."""
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass
class MeshCell:
    """Single cell in the inversion mesh."""
    ix: int
    iy: int
    iz: int
    center: Point3D
    size: Tuple[float, float, float]
    property_value: float = 0.0
    reference_value: float = 0.0
    bounds: Tuple[float, float] = (-float('inf'), float('inf'))
    is_active: bool = True
    sensitivity: float = 0.0
    
    @property
    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]


@dataclass
class InversionMesh:
    """3D mesh for inversion."""
    origin: Point3D
    cell_sizes: Tuple[float, float, float]
    n_cells: Tuple[int, int, int]
    cells: List[MeshCell] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.cells:
            self._create_cells()
    
    def _create_cells(self):
        """Create mesh cells."""
        nx, ny, nz = self.n_cells
        dx, dy, dz = self.cell_sizes
        
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    center = Point3D(
                        self.origin.x + (ix + 0.5) * dx,
                        self.origin.y + (iy + 0.5) * dy,
                        self.origin.z - (iz + 0.5) * dz
                    )
                    
                    self.cells.append(MeshCell(
                        ix=ix, iy=iy, iz=iz,
                        center=center,
                        size=(dx, dy, dz)
                    ))
    
    @property
    def n_active_cells(self) -> int:
        return sum(1 for c in self.cells if c.is_active)
    
    def get_cell(self, ix: int, iy: int, iz: int) -> Optional[MeshCell]:
        """Get cell by indices."""
        nx, ny, nz = self.n_cells
        if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
            idx = iz * nx * ny + iy * nx + ix
            return self.cells[idx]
        return None
    
    def get_property_array(self) -> np.ndarray:
        """Get property values as array."""
        return np.array([c.property_value for c in self.cells if c.is_active])
    
    def set_property_array(self, values: np.ndarray):
        """Set property values from array."""
        active_cells = [c for c in self.cells if c.is_active]
        for i, cell in enumerate(active_cells):
            if i < len(values):
                cell.property_value = values[i]


@dataclass
class ObservationPoint:
    """Single observation point."""
    location: Point3D
    observed_value: float
    uncertainty: float = 1.0
    predicted_value: float = 0.0
    residual: float = 0.0
    weight: float = 1.0


@dataclass
class SurveyData:
    """Survey data for inversion."""
    name: str
    survey_type: InversionType
    observations: List[ObservationPoint]
    inclination: float = 0.0
    declination: float = 0.0
    field_strength: float = 50000.0
    
    @property
    def n_observations(self) -> int:
        return len(self.observations)
    
    def get_observed_data(self) -> np.ndarray:
        return np.array([o.observed_value for o in self.observations])
    
    def get_uncertainties(self) -> np.ndarray:
        return np.array([o.uncertainty for o in self.observations])
    
    def get_locations(self) -> np.ndarray:
        return np.array([[o.location.x, o.location.y, o.location.z] 
                        for o in self.observations])


@dataclass
class InversionParameters:
    """Inversion parameters."""
    max_iterations: int = 50
    target_misfit: float = 1.0
    beta_initial: float = 1.0
    beta_cooling: float = 0.5
    beta_min: float = 1e-8
    regularization_type: RegularizationType = RegularizationType.TIKHONOV
    depth_weighting_type: DepthWeightingType = DepthWeightingType.SENSITIVITY
    depth_weighting_exponent: float = 2.0
    alpha_s: float = 1.0
    alpha_x: float = 1.0
    alpha_y: float = 1.0
    alpha_z: float = 1.0
    use_reference_model: bool = False
    use_bounds: bool = True
    solver: str = "cg"
    tolerance: float = 1e-6


@dataclass
class InversionResult:
    """Result of inversion."""
    mesh: InversionMesh
    survey: SurveyData
    parameters: InversionParameters
    final_model: np.ndarray
    predicted_data: np.ndarray
    data_misfit: float
    model_norm: float
    objective_function: float
    n_iterations: int
    convergence_history: List[Dict[str, float]]
    computation_time: float = 0.0
    success: bool = True
    message: str = ""


class ForwardModeler:
    """Forward modeling for geophysical methods."""
    
    def __init__(self, mesh: InversionMesh, survey: SurveyData):
        self.mesh = mesh
        self.survey = survey
        self.sensitivity_matrix: Optional[np.ndarray] = None
    
    def compute_sensitivity_matrix(self) -> np.ndarray:
        """Compute sensitivity (Jacobian) matrix."""
        n_obs = self.survey.n_observations
        n_cells = self.mesh.n_active_cells
        
        G = np.zeros((n_obs, n_cells))
        
        active_cells = [c for c in self.mesh.cells if c.is_active]
        
        for i, obs in enumerate(self.survey.observations):
            for j, cell in enumerate(active_cells):
                G[i, j] = self._compute_kernel(obs.location, cell)
        
        self.sensitivity_matrix = G
        return G
    
    def _compute_kernel(self, obs_loc: Point3D, cell: MeshCell) -> float:
        """Compute kernel function for a single observation-cell pair."""
        if self.survey.survey_type == InversionType.MAGNETIC:
            return self._magnetic_kernel(obs_loc, cell)
        elif self.survey.survey_type == InversionType.GRAVITY:
            return self._gravity_kernel(obs_loc, cell)
        elif self.survey.survey_type == InversionType.ELECTROMAGNETIC:
            return self._em_kernel(obs_loc, cell)
        else:
            return self._generic_kernel(obs_loc, cell)
    
    def _magnetic_kernel(self, obs_loc: Point3D, cell: MeshCell) -> float:
        """Compute magnetic kernel (TMI response of a prism)."""
        dx = obs_loc.x - cell.center.x
        dy = obs_loc.y - cell.center.y
        dz = obs_loc.z - cell.center.z
        
        r = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if r < 0.001:
            return 0.0
        
        inc_rad = math.radians(self.survey.inclination)
        dec_rad = math.radians(self.survey.declination)
        
        fx = math.cos(inc_rad) * math.cos(dec_rad)
        fy = math.cos(inc_rad) * math.sin(dec_rad)
        fz = math.sin(inc_rad)
        
        dot_product = (dx * fx + dy * fy + dz * fz) / r
        
        kernel = cell.volume * (3 * dot_product**2 - 1) / (4 * math.pi * r**3)
        
        return kernel * 1e9
    
    def _gravity_kernel(self, obs_loc: Point3D, cell: MeshCell) -> float:
        """Compute gravity kernel (vertical gravity response of a prism)."""
        dx = obs_loc.x - cell.center.x
        dy = obs_loc.y - cell.center.y
        dz = obs_loc.z - cell.center.z
        
        r = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if r < 0.001:
            return 0.0
        
        G = 6.674e-11
        
        kernel = G * cell.volume * dz / (r**3)
        
        return kernel * 1e5
    
    def _em_kernel(self, obs_loc: Point3D, cell: MeshCell) -> float:
        """Compute EM kernel (simplified)."""
        dx = obs_loc.x - cell.center.x
        dy = obs_loc.y - cell.center.y
        dz = obs_loc.z - cell.center.z
        
        r = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if r < 0.001:
            return 0.0
        
        kernel = cell.volume / (r**2)
        
        return kernel
    
    def _generic_kernel(self, obs_loc: Point3D, cell: MeshCell) -> float:
        """Generic kernel based on distance."""
        r = obs_loc.distance_to(cell.center)
        
        if r < 0.001:
            return 0.0
        
        return cell.volume / (r**2)
    
    def forward(self, model: np.ndarray) -> np.ndarray:
        """Compute forward response."""
        if self.sensitivity_matrix is None:
            self.compute_sensitivity_matrix()
        
        return self.sensitivity_matrix @ model
    
    def compute_residuals(self, model: np.ndarray) -> np.ndarray:
        """Compute data residuals."""
        predicted = self.forward(model)
        observed = self.survey.get_observed_data()
        
        return observed - predicted


class RegularizationOperator:
    """Regularization operators for inversion."""
    
    def __init__(self, mesh: InversionMesh, params: InversionParameters):
        self.mesh = mesh
        self.params = params
        self.Ws: Optional[np.ndarray] = None
        self.Wx: Optional[np.ndarray] = None
        self.Wy: Optional[np.ndarray] = None
        self.Wz: Optional[np.ndarray] = None
    
    def build_operators(self):
        """Build regularization operators."""
        n = self.mesh.n_active_cells
        
        self.Ws = np.eye(n) * self.params.alpha_s
        
        self.Wx = self._build_gradient_operator('x') * self.params.alpha_x
        self.Wy = self._build_gradient_operator('y') * self.params.alpha_y
        self.Wz = self._build_gradient_operator('z') * self.params.alpha_z
    
    def _build_gradient_operator(self, direction: str) -> np.ndarray:
        """Build gradient operator in specified direction."""
        nx, ny, nz = self.mesh.n_cells
        n = self.mesh.n_active_cells
        
        active_indices = {}
        idx = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cell = self.mesh.get_cell(ix, iy, iz)
                    if cell and cell.is_active:
                        active_indices[(ix, iy, iz)] = idx
                        idx += 1
        
        rows = []
        cols = []
        vals = []
        
        for (ix, iy, iz), idx in active_indices.items():
            if direction == 'x' and ix < nx - 1:
                neighbor_idx = active_indices.get((ix + 1, iy, iz))
                if neighbor_idx is not None:
                    rows.extend([len(rows) // 2, len(rows) // 2])
                    cols.extend([idx, neighbor_idx])
                    vals.extend([-1, 1])
            elif direction == 'y' and iy < ny - 1:
                neighbor_idx = active_indices.get((ix, iy + 1, iz))
                if neighbor_idx is not None:
                    rows.extend([len(rows) // 2, len(rows) // 2])
                    cols.extend([idx, neighbor_idx])
                    vals.extend([-1, 1])
            elif direction == 'z' and iz < nz - 1:
                neighbor_idx = active_indices.get((ix, iy, iz + 1))
                if neighbor_idx is not None:
                    rows.extend([len(rows) // 2, len(rows) // 2])
                    cols.extend([idx, neighbor_idx])
                    vals.extend([-1, 1])
        
        if not rows:
            return np.zeros((1, n))
        
        n_rows = max(rows) + 1
        W = np.zeros((n_rows, n))
        for r, c, v in zip(rows, cols, vals):
            W[r, c] = v
        
        return W
    
    def compute_model_norm(self, model: np.ndarray, 
                          reference: Optional[np.ndarray] = None) -> float:
        """Compute model norm (regularization term)."""
        if reference is None:
            reference = np.zeros_like(model)
        
        m_diff = model - reference
        
        norm = 0.0
        
        if self.Ws is not None:
            norm += np.sum((self.Ws @ m_diff)**2)
        
        if self.Wx is not None:
            norm += np.sum((self.Wx @ m_diff)**2)
        
        if self.Wy is not None:
            norm += np.sum((self.Wy @ m_diff)**2)
        
        if self.Wz is not None:
            norm += np.sum((self.Wz @ m_diff)**2)
        
        return norm
    
    def get_regularization_matrix(self) -> np.ndarray:
        """Get combined regularization matrix."""
        matrices = []
        
        if self.Ws is not None:
            matrices.append(self.Ws)
        if self.Wx is not None:
            matrices.append(self.Wx)
        if self.Wy is not None:
            matrices.append(self.Wy)
        if self.Wz is not None:
            matrices.append(self.Wz)
        
        if matrices:
            return np.vstack(matrices)
        else:
            return np.eye(self.mesh.n_active_cells)


class DepthWeighting:
    """Depth weighting for inversion."""
    
    def __init__(self, mesh: InversionMesh, params: InversionParameters,
                 survey: SurveyData):
        self.mesh = mesh
        self.params = params
        self.survey = survey
        self.weights: Optional[np.ndarray] = None
    
    def compute_weights(self) -> np.ndarray:
        """Compute depth weights."""
        active_cells = [c for c in self.mesh.cells if c.is_active]
        n = len(active_cells)
        
        if self.params.depth_weighting_type == DepthWeightingType.NONE:
            self.weights = np.ones(n)
        
        elif self.params.depth_weighting_type == DepthWeightingType.LINEAR:
            obs_z = np.mean([o.location.z for o in self.survey.observations])
            depths = np.array([obs_z - c.center.z for c in active_cells])
            depths = np.maximum(depths, 1.0)
            self.weights = depths ** self.params.depth_weighting_exponent
        
        elif self.params.depth_weighting_type == DepthWeightingType.EXPONENTIAL:
            obs_z = np.mean([o.location.z for o in self.survey.observations])
            depths = np.array([obs_z - c.center.z for c in active_cells])
            depths = np.maximum(depths, 1.0)
            self.weights = np.exp(self.params.depth_weighting_exponent * depths / np.max(depths))
        
        elif self.params.depth_weighting_type == DepthWeightingType.SENSITIVITY:
            sensitivities = np.array([c.sensitivity for c in active_cells])
            if np.max(sensitivities) > 0:
                self.weights = 1.0 / (sensitivities + 1e-10)
            else:
                self.weights = np.ones(n)
        
        else:
            self.weights = np.ones(n)
        
        self.weights = self.weights / np.max(self.weights)
        
        return self.weights
    
    def get_weight_matrix(self) -> np.ndarray:
        """Get diagonal weight matrix."""
        if self.weights is None:
            self.compute_weights()
        
        return np.diag(self.weights)


class GaussNewtonSolver:
    """Gauss-Newton solver for inversion."""
    
    def __init__(self, forward_modeler: ForwardModeler,
                 regularization: RegularizationOperator,
                 depth_weighting: DepthWeighting,
                 params: InversionParameters):
        self.forward = forward_modeler
        self.regularization = regularization
        self.depth_weighting = depth_weighting
        self.params = params
    
    def solve(self, initial_model: np.ndarray,
             reference_model: Optional[np.ndarray] = None) -> InversionResult:
        """Run inversion."""
        import time
        start_time = time.time()
        
        if self.forward.sensitivity_matrix is None:
            self.forward.compute_sensitivity_matrix()
        
        self.regularization.build_operators()
        self.depth_weighting.compute_weights()
        
        G = self.forward.sensitivity_matrix
        d_obs = self.forward.survey.get_observed_data()
        uncertainties = self.forward.survey.get_uncertainties()
        
        Wd = np.diag(1.0 / uncertainties)
        
        Wm = self.regularization.get_regularization_matrix()
        Wz = self.depth_weighting.get_weight_matrix()
        
        if reference_model is None:
            reference_model = np.zeros_like(initial_model)
        
        model = initial_model.copy()
        beta = self.params.beta_initial
        
        convergence_history = []
        
        for iteration in range(self.params.max_iterations):
            d_pred = self.forward.forward(model)
            residuals = d_obs - d_pred
            
            data_misfit = np.sum((Wd @ residuals)**2) / len(d_obs)
            model_norm = self.regularization.compute_model_norm(model, reference_model)
            objective = data_misfit + beta * model_norm
            
            convergence_history.append({
                "iteration": iteration,
                "data_misfit": data_misfit,
                "model_norm": model_norm,
                "objective": objective,
                "beta": beta
            })
            
            if data_misfit <= self.params.target_misfit:
                break
            
            GtWd = G.T @ (Wd.T @ Wd)
            WmtWm = Wm.T @ Wm
            
            A = GtWd @ G + beta * WmtWm
            b = GtWd @ residuals - beta * WmtWm @ (model - reference_model)
            
            try:
                delta_m = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                delta_m = np.linalg.lstsq(A, b, rcond=None)[0]
            
            model = model + delta_m
            
            if self.params.use_bounds:
                active_cells = [c for c in self.forward.mesh.cells if c.is_active]
                for i, cell in enumerate(active_cells):
                    model[i] = np.clip(model[i], cell.bounds[0], cell.bounds[1])
            
            if data_misfit > self.params.target_misfit:
                beta = max(beta * self.params.beta_cooling, self.params.beta_min)
        
        d_pred = self.forward.forward(model)
        final_misfit = np.sum((Wd @ (d_obs - d_pred))**2) / len(d_obs)
        final_model_norm = self.regularization.compute_model_norm(model, reference_model)
        
        self.forward.mesh.set_property_array(model)
        
        for i, obs in enumerate(self.forward.survey.observations):
            obs.predicted_value = d_pred[i]
            obs.residual = d_obs[i] - d_pred[i]
        
        computation_time = time.time() - start_time
        
        return InversionResult(
            mesh=self.forward.mesh,
            survey=self.forward.survey,
            parameters=self.params,
            final_model=model,
            predicted_data=d_pred,
            data_misfit=final_misfit,
            model_norm=final_model_norm,
            objective_function=final_misfit + beta * final_model_norm,
            n_iterations=len(convergence_history),
            convergence_history=convergence_history,
            computation_time=computation_time,
            success=final_misfit <= self.params.target_misfit * 2,
            message="Inversion completed"
        )


class MagneticInversion:
    """Magnetic susceptibility inversion."""
    
    def __init__(self, mesh: InversionMesh, survey: SurveyData,
                 params: Optional[InversionParameters] = None):
        self.mesh = mesh
        self.survey = survey
        self.params = params or InversionParameters()
        
        self.forward_modeler = ForwardModeler(mesh, survey)
        self.regularization = RegularizationOperator(mesh, self.params)
        self.depth_weighting = DepthWeighting(mesh, self.params, survey)
        self.solver = GaussNewtonSolver(
            self.forward_modeler, self.regularization,
            self.depth_weighting, self.params
        )
    
    def set_bounds(self, lower: float = 0.0, upper: float = 1.0):
        """Set susceptibility bounds."""
        for cell in self.mesh.cells:
            cell.bounds = (lower, upper)
    
    def set_reference_model(self, values: np.ndarray):
        """Set reference model."""
        active_cells = [c for c in self.mesh.cells if c.is_active]
        for i, cell in enumerate(active_cells):
            if i < len(values):
                cell.reference_value = values[i]
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """Run magnetic inversion."""
        if initial_model is None:
            initial_model = np.ones(self.mesh.n_active_cells) * 0.001
        
        reference = np.array([c.reference_value for c in self.mesh.cells if c.is_active])
        
        return self.solver.solve(initial_model, reference)


class GravityInversion:
    """Gravity density inversion."""
    
    def __init__(self, mesh: InversionMesh, survey: SurveyData,
                 params: Optional[InversionParameters] = None):
        self.mesh = mesh
        self.survey = survey
        self.params = params or InversionParameters()
        
        self.forward_modeler = ForwardModeler(mesh, survey)
        self.regularization = RegularizationOperator(mesh, self.params)
        self.depth_weighting = DepthWeighting(mesh, self.params, survey)
        self.solver = GaussNewtonSolver(
            self.forward_modeler, self.regularization,
            self.depth_weighting, self.params
        )
    
    def set_bounds(self, lower: float = -1.0, upper: float = 1.0):
        """Set density contrast bounds."""
        for cell in self.mesh.cells:
            cell.bounds = (lower, upper)
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """Run gravity inversion."""
        if initial_model is None:
            initial_model = np.zeros(self.mesh.n_active_cells)
        
        reference = np.array([c.reference_value for c in self.mesh.cells if c.is_active])
        
        return self.solver.solve(initial_model, reference)


class EMInversion:
    """Electromagnetic conductivity inversion."""
    
    def __init__(self, mesh: InversionMesh, survey: SurveyData,
                 params: Optional[InversionParameters] = None):
        self.mesh = mesh
        self.survey = survey
        self.params = params or InversionParameters()
        
        self.forward_modeler = ForwardModeler(mesh, survey)
        self.regularization = RegularizationOperator(mesh, self.params)
        self.depth_weighting = DepthWeighting(mesh, self.params, survey)
        self.solver = GaussNewtonSolver(
            self.forward_modeler, self.regularization,
            self.depth_weighting, self.params
        )
    
    def set_bounds(self, lower: float = 1e-5, upper: float = 10.0):
        """Set conductivity bounds (S/m)."""
        for cell in self.mesh.cells:
            cell.bounds = (lower, upper)
    
    def run(self, initial_model: Optional[np.ndarray] = None) -> InversionResult:
        """Run EM inversion."""
        if initial_model is None:
            initial_model = np.ones(self.mesh.n_active_cells) * 0.01
        
        reference = np.array([c.reference_value for c in self.mesh.cells if c.is_active])
        
        return self.solver.solve(initial_model, reference)


class InversionWorkflow:
    """
    Complete geophysical inversion workflow.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.mesh: Optional[InversionMesh] = None
        self.survey: Optional[SurveyData] = None
        self.params = InversionParameters()
        self.result: Optional[InversionResult] = None
    
    def create_mesh(self, x_min: float, y_min: float, z_max: float,
                   dx: float, dy: float, dz: float,
                   nx: int, ny: int, nz: int) -> InversionMesh:
        """Create inversion mesh."""
        self.mesh = InversionMesh(
            origin=Point3D(x_min, y_min, z_max),
            cell_sizes=(dx, dy, dz),
            n_cells=(nx, ny, nz)
        )
        return self.mesh
    
    def load_survey(self, survey_type: str, observations: List[Dict[str, Any]],
                   name: str = "Survey",
                   inclination: float = 0.0, declination: float = 0.0,
                   field_strength: float = 50000.0) -> SurveyData:
        """Load survey data."""
        obs_points = []
        for obs in observations:
            obs_points.append(ObservationPoint(
                location=Point3D(
                    obs.get('x', obs.get('easting', 0)),
                    obs.get('y', obs.get('northing', 0)),
                    obs.get('z', obs.get('elevation', 0))
                ),
                observed_value=obs.get('value', obs.get('tmi', obs.get('gz', 0))),
                uncertainty=obs.get('uncertainty', obs.get('error', 1.0))
            ))
        
        survey_type_enum = InversionType(survey_type) if survey_type in [t.value for t in InversionType] else InversionType.MAGNETIC
        
        self.survey = SurveyData(
            name=name,
            survey_type=survey_type_enum,
            observations=obs_points,
            inclination=inclination,
            declination=declination,
            field_strength=field_strength
        )
        
        return self.survey
    
    def set_parameters(self, **kwargs):
        """Set inversion parameters."""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def run_inversion(self, inversion_type: Optional[str] = None) -> InversionResult:
        """Run inversion."""
        if self.mesh is None:
            raise ValueError("Mesh not created")
        if self.survey is None:
            raise ValueError("Survey not loaded")
        
        inv_type = inversion_type or self.survey.survey_type.value
        
        if inv_type == "magnetic":
            inversion = MagneticInversion(self.mesh, self.survey, self.params)
            inversion.set_bounds(0.0, 1.0)
        elif inv_type == "gravity":
            inversion = GravityInversion(self.mesh, self.survey, self.params)
            inversion.set_bounds(-1.0, 1.0)
        elif inv_type == "electromagnetic":
            inversion = EMInversion(self.mesh, self.survey, self.params)
            inversion.set_bounds(1e-5, 10.0)
        else:
            inversion = MagneticInversion(self.mesh, self.survey, self.params)
        
        self.result = inversion.run()
        return self.result
    
    def export_model(self, filepath: str, format: str = "csv"):
        """Export inversion result."""
        if self.result is None:
            raise ValueError("No inversion result")
        
        if format == "csv":
            self._export_csv(filepath)
        elif format == "json":
            self._export_json(filepath)
    
    def _export_csv(self, filepath: str):
        """Export to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z', 'value', 'ix', 'iy', 'iz'])
            
            for cell in self.result.mesh.cells:
                if cell.is_active:
                    writer.writerow([
                        cell.center.x, cell.center.y, cell.center.z,
                        cell.property_value,
                        cell.ix, cell.iy, cell.iz
                    ])
    
    def _export_json(self, filepath: str):
        """Export to JSON."""
        import json
        
        data = {
            "project": self.project_name,
            "mesh": {
                "origin": self.result.mesh.origin.to_array().tolist(),
                "cell_sizes": self.result.mesh.cell_sizes,
                "n_cells": self.result.mesh.n_cells
            },
            "result": {
                "data_misfit": self.result.data_misfit,
                "model_norm": self.result.model_norm,
                "n_iterations": self.result.n_iterations,
                "computation_time": self.result.computation_time
            },
            "model": self.result.final_model.tolist(),
            "predicted_data": self.result.predicted_data.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        summary = {
            "project": self.project_name,
            "mesh": {
                "n_cells": self.mesh.n_cells if self.mesh else None,
                "n_active": self.mesh.n_active_cells if self.mesh else None
            },
            "survey": {
                "type": self.survey.survey_type.value if self.survey else None,
                "n_observations": self.survey.n_observations if self.survey else None
            },
            "parameters": {
                "max_iterations": self.params.max_iterations,
                "target_misfit": self.params.target_misfit,
                "regularization": self.params.regularization_type.value
            }
        }
        
        if self.result:
            summary["result"] = {
                "success": self.result.success,
                "data_misfit": self.result.data_misfit,
                "model_norm": self.result.model_norm,
                "n_iterations": self.result.n_iterations,
                "computation_time": self.result.computation_time
            }
        
        return summary


def create_inversion_workflow(project_name: str = "default") -> InversionWorkflow:
    """Factory function to create an inversion workflow."""
    return InversionWorkflow(project_name)


def create_magnetic_inversion(mesh: InversionMesh, survey: SurveyData,
                             params: Optional[InversionParameters] = None) -> MagneticInversion:
    """Factory function to create magnetic inversion."""
    return MagneticInversion(mesh, survey, params)


def create_gravity_inversion(mesh: InversionMesh, survey: SurveyData,
                            params: Optional[InversionParameters] = None) -> GravityInversion:
    """Factory function to create gravity inversion."""
    return GravityInversion(mesh, survey, params)
