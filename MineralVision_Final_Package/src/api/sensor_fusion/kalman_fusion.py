"""
Kalman Filter-based Sensor Fusion for MineralVision.

This module provides advanced temporal fusion capabilities using various
Kalman filter implementations for combining time-series sensor data.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from scipy import linalg

from .core import SensorData, SensorFusionAlgorithm, SensorType, DataDimension

logger = logging.getLogger(__name__)


class KalmanFilterType(Enum):
    """Types of Kalman filters available."""
    STANDARD = "standard"
    EXTENDED = "extended"
    UNSCENTED = "unscented"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


@dataclass
class KalmanState:
    """State representation for Kalman filter."""
    mean: np.ndarray
    covariance: np.ndarray
    timestamp: datetime
    
    def copy(self) -> 'KalmanState':
        """Create a deep copy of the state."""
        return KalmanState(
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            timestamp=self.timestamp
        )


@dataclass
class KalmanConfig:
    """Configuration for Kalman filter."""
    state_dim: int
    measurement_dim: int
    process_noise: float = 0.01
    measurement_noise: float = 0.1
    initial_covariance: float = 1.0
    filter_type: KalmanFilterType = KalmanFilterType.STANDARD
    adaptive_window: int = 10
    ensemble_size: int = 100
    ukf_alpha: float = 0.001
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0


class StandardKalmanFilter:
    """
    Standard Linear Kalman Filter implementation.
    
    Optimal for linear systems with Gaussian noise.
    """
    
    def __init__(self, config: KalmanConfig):
        """
        Initialize the Kalman filter.
        
        Args:
            config: Kalman filter configuration
        """
        self.config = config
        self.state_dim = config.state_dim
        self.measurement_dim = config.measurement_dim
        
        # State transition matrix (identity by default - random walk model)
        self.F = np.eye(self.state_dim)
        
        # Observation matrix (identity by default - direct observation)
        self.H = np.eye(self.measurement_dim, self.state_dim)
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * config.process_noise
        
        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * config.measurement_noise
        
        # Control input matrix (no control by default)
        self.B = np.zeros((self.state_dim, 1))
        
        # Current state
        self.state: Optional[KalmanState] = None
        
        # History for analysis
        self.state_history: List[KalmanState] = []
        self.innovation_history: List[np.ndarray] = []
        
    def initialize(self, initial_mean: np.ndarray, 
                   initial_covariance: Optional[np.ndarray] = None,
                   timestamp: Optional[datetime] = None) -> None:
        """
        Initialize the filter state.
        
        Args:
            initial_mean: Initial state mean
            initial_covariance: Initial state covariance (optional)
            timestamp: Initial timestamp
        """
        if initial_covariance is None:
            initial_covariance = np.eye(self.state_dim) * self.config.initial_covariance
            
        self.state = KalmanState(
            mean=initial_mean.copy(),
            covariance=initial_covariance.copy(),
            timestamp=timestamp or datetime.now()
        )
        self.state_history = [self.state.copy()]
        
    def set_transition_matrix(self, F: np.ndarray) -> None:
        """Set the state transition matrix."""
        if F.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Transition matrix must be {self.state_dim}x{self.state_dim}")
        self.F = F.copy()
        
    def set_observation_matrix(self, H: np.ndarray) -> None:
        """Set the observation matrix."""
        if H.shape != (self.measurement_dim, self.state_dim):
            raise ValueError(f"Observation matrix must be {self.measurement_dim}x{self.state_dim}")
        self.H = H.copy()
        
    def set_process_noise(self, Q: np.ndarray) -> None:
        """Set the process noise covariance."""
        if Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Process noise must be {self.state_dim}x{self.state_dim}")
        self.Q = Q.copy()
        
    def set_measurement_noise(self, R: np.ndarray) -> None:
        """Set the measurement noise covariance."""
        if R.shape != (self.measurement_dim, self.measurement_dim):
            raise ValueError(f"Measurement noise must be {self.measurement_dim}x{self.measurement_dim}")
        self.R = R.copy()
        
    def predict(self, control_input: Optional[np.ndarray] = None,
                dt: float = 1.0) -> KalmanState:
        """
        Predict the next state.
        
        Args:
            control_input: Optional control input
            dt: Time step for time-varying models
            
        Returns:
            Predicted state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        # State prediction: x_pred = F * x + B * u
        x_pred = self.F @ self.state.mean
        if control_input is not None:
            x_pred += self.B @ control_input
            
        # Covariance prediction: P_pred = F * P * F^T + Q
        P_pred = self.F @ self.state.covariance @ self.F.T + self.Q
        
        # Update state
        self.state = KalmanState(
            mean=x_pred,
            covariance=P_pred,
            timestamp=datetime.now()
        )
        
        return self.state.copy()
        
    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update the state with a measurement.
        
        Args:
            measurement: Measurement vector
            measurement_noise: Optional measurement-specific noise covariance
            
        Returns:
            Updated state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        R = measurement_noise if measurement_noise is not None else self.R
        
        # Innovation (measurement residual): y = z - H * x
        y = measurement - self.H @ self.state.mean
        self.innovation_history.append(y.copy())
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.state.covariance @ self.H.T + R
        
        # Kalman gain: K = P * H^T * S^(-1)
        K = self.state.covariance @ self.H.T @ np.linalg.inv(S)
        
        # State update: x = x + K * y
        x_updated = self.state.mean + K @ y
        
        # Covariance update: P = (I - K * H) * P
        # Using Joseph form for numerical stability
        I_KH = np.eye(self.state_dim) - K @ self.H
        P_updated = I_KH @ self.state.covariance @ I_KH.T + K @ R @ K.T
        
        # Update state
        self.state = KalmanState(
            mean=x_updated,
            covariance=P_updated,
            timestamp=datetime.now()
        )
        self.state_history.append(self.state.copy())
        
        return self.state.copy()
        
    def filter_sequence(self, measurements: np.ndarray,
                       timestamps: Optional[List[datetime]] = None) -> List[KalmanState]:
        """
        Filter a sequence of measurements.
        
        Args:
            measurements: Array of measurements (n_measurements x measurement_dim)
            timestamps: Optional list of timestamps
            
        Returns:
            List of filtered states
        """
        if self.state is None:
            # Initialize with first measurement
            initial_mean = np.zeros(self.state_dim)
            initial_mean[:self.measurement_dim] = measurements[0]
            self.initialize(initial_mean)
            
        results = []
        for i, measurement in enumerate(measurements):
            self.predict()
            state = self.update(measurement)
            if timestamps is not None:
                state.timestamp = timestamps[i]
            results.append(state)
            
        return results
        
    def smooth(self, states: Optional[List[KalmanState]] = None) -> List[KalmanState]:
        """
        Apply Rauch-Tung-Striebel (RTS) smoother.
        
        Args:
            states: Optional list of states to smooth (uses history if None)
            
        Returns:
            List of smoothed states
        """
        if states is None:
            states = self.state_history
            
        if len(states) < 2:
            return states
            
        # Forward pass already done, now backward pass
        n = len(states)
        smoothed = [None] * n
        smoothed[-1] = states[-1].copy()
        
        for k in range(n - 2, -1, -1):
            # Predicted state at k+1 given k
            x_pred = self.F @ states[k].mean
            P_pred = self.F @ states[k].covariance @ self.F.T + self.Q
            
            # Smoother gain
            G = states[k].covariance @ self.F.T @ np.linalg.inv(P_pred)
            
            # Smoothed state
            x_smooth = states[k].mean + G @ (smoothed[k + 1].mean - x_pred)
            P_smooth = states[k].covariance + G @ (smoothed[k + 1].covariance - P_pred) @ G.T
            
            smoothed[k] = KalmanState(
                mean=x_smooth,
                covariance=P_smooth,
                timestamp=states[k].timestamp
            )
            
        return smoothed
        
    def get_state(self) -> Optional[KalmanState]:
        """Get the current state."""
        return self.state.copy() if self.state else None
        
    def get_innovation_statistics(self) -> Dict[str, float]:
        """Get innovation sequence statistics for filter health monitoring."""
        if not self.innovation_history:
            return {}
            
        innovations = np.array(self.innovation_history)
        return {
            'mean': float(np.mean(innovations)),
            'std': float(np.std(innovations)),
            'normalized_mean': float(np.mean(innovations) / (np.std(innovations) + 1e-10)),
            'count': len(innovations)
        }


class ExtendedKalmanFilter(StandardKalmanFilter):
    """
    Extended Kalman Filter for nonlinear systems.
    
    Uses first-order Taylor expansion to linearize the system.
    """
    
    def __init__(self, config: KalmanConfig,
                 state_transition_fn: Optional[callable] = None,
                 observation_fn: Optional[callable] = None,
                 state_jacobian_fn: Optional[callable] = None,
                 observation_jacobian_fn: Optional[callable] = None):
        """
        Initialize the Extended Kalman Filter.
        
        Args:
            config: Kalman filter configuration
            state_transition_fn: Nonlinear state transition function f(x, u)
            observation_fn: Nonlinear observation function h(x)
            state_jacobian_fn: Jacobian of state transition df/dx
            observation_jacobian_fn: Jacobian of observation dh/dx
        """
        super().__init__(config)
        
        self.f = state_transition_fn or (lambda x, u: self.F @ x)
        self.h = observation_fn or (lambda x: self.H @ x)
        self.F_jacobian = state_jacobian_fn or (lambda x, u: self.F)
        self.H_jacobian = observation_jacobian_fn or (lambda x: self.H)
        
    def predict(self, control_input: Optional[np.ndarray] = None,
                dt: float = 1.0) -> KalmanState:
        """
        Predict the next state using nonlinear transition.
        
        Args:
            control_input: Optional control input
            dt: Time step
            
        Returns:
            Predicted state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        u = control_input if control_input is not None else np.zeros(1)
        
        # Nonlinear state prediction
        x_pred = self.f(self.state.mean, u)
        
        # Linearize around current state
        F = self.F_jacobian(self.state.mean, u)
        
        # Covariance prediction
        P_pred = F @ self.state.covariance @ F.T + self.Q
        
        self.state = KalmanState(
            mean=x_pred,
            covariance=P_pred,
            timestamp=datetime.now()
        )
        
        return self.state.copy()
        
    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update the state with a measurement using nonlinear observation.
        
        Args:
            measurement: Measurement vector
            measurement_noise: Optional measurement-specific noise covariance
            
        Returns:
            Updated state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        R = measurement_noise if measurement_noise is not None else self.R
        
        # Predicted measurement
        z_pred = self.h(self.state.mean)
        
        # Innovation
        y = measurement - z_pred
        self.innovation_history.append(y.copy())
        
        # Linearize observation around predicted state
        H = self.H_jacobian(self.state.mean)
        
        # Innovation covariance
        S = H @ self.state.covariance @ H.T + R
        
        # Kalman gain
        K = self.state.covariance @ H.T @ np.linalg.inv(S)
        
        # State update
        x_updated = self.state.mean + K @ y
        
        # Covariance update (Joseph form)
        I_KH = np.eye(self.state_dim) - K @ H
        P_updated = I_KH @ self.state.covariance @ I_KH.T + K @ R @ K.T
        
        self.state = KalmanState(
            mean=x_updated,
            covariance=P_updated,
            timestamp=datetime.now()
        )
        self.state_history.append(self.state.copy())
        
        return self.state.copy()


class UnscentedKalmanFilter(StandardKalmanFilter):
    """
    Unscented Kalman Filter for highly nonlinear systems.
    
    Uses sigma points to capture the mean and covariance of the state distribution.
    """
    
    def __init__(self, config: KalmanConfig,
                 state_transition_fn: Optional[callable] = None,
                 observation_fn: Optional[callable] = None):
        """
        Initialize the Unscented Kalman Filter.
        
        Args:
            config: Kalman filter configuration
            state_transition_fn: Nonlinear state transition function f(x, u)
            observation_fn: Nonlinear observation function h(x)
        """
        super().__init__(config)
        
        self.f = state_transition_fn or (lambda x, u: self.F @ x)
        self.h = observation_fn or (lambda x: self.H @ x)
        
        # UKF parameters
        self.alpha = config.ukf_alpha
        self.beta = config.ukf_beta
        self.kappa = config.ukf_kappa
        
        # Compute weights
        self._compute_weights()
        
    def _compute_weights(self) -> None:
        """Compute sigma point weights."""
        n = self.state_dim
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n
        
        # Mean weights
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = lambda_ / (n + lambda_)
        self.Wm[1:] = 1 / (2 * (n + lambda_))
        
        # Covariance weights
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + lambda_))
        
        self.lambda_ = lambda_
        
    def _generate_sigma_points(self, mean: np.ndarray, 
                               covariance: np.ndarray) -> np.ndarray:
        """
        Generate sigma points.
        
        Args:
            mean: State mean
            covariance: State covariance
            
        Returns:
            Sigma points array (2n+1 x n)
        """
        n = self.state_dim
        sigma_points = np.zeros((2 * n + 1, n))
        
        # First sigma point is the mean
        sigma_points[0] = mean
        
        # Square root of scaled covariance
        try:
            sqrt_cov = linalg.cholesky((n + self.lambda_) * covariance, lower=True)
        except linalg.LinAlgError:
            # Add small regularization if not positive definite
            sqrt_cov = linalg.cholesky(
                (n + self.lambda_) * (covariance + 1e-6 * np.eye(n)), 
                lower=True
            )
        
        # Generate sigma points
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[:, i]
            sigma_points[n + i + 1] = mean - sqrt_cov[:, i]
            
        return sigma_points
        
    def predict(self, control_input: Optional[np.ndarray] = None,
                dt: float = 1.0) -> KalmanState:
        """
        Predict the next state using unscented transform.
        
        Args:
            control_input: Optional control input
            dt: Time step
            
        Returns:
            Predicted state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        u = control_input if control_input is not None else np.zeros(1)
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(
            self.state.mean, self.state.covariance
        )
        
        # Transform sigma points through state transition
        transformed_points = np.array([self.f(sp, u) for sp in sigma_points])
        
        # Compute predicted mean
        x_pred = np.sum(self.Wm[:, np.newaxis] * transformed_points, axis=0)
        
        # Compute predicted covariance
        P_pred = self.Q.copy()
        for i, sp in enumerate(transformed_points):
            diff = sp - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
            
        self.state = KalmanState(
            mean=x_pred,
            covariance=P_pred,
            timestamp=datetime.now()
        )
        
        return self.state.copy()
        
    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update the state with a measurement using unscented transform.
        
        Args:
            measurement: Measurement vector
            measurement_noise: Optional measurement-specific noise covariance
            
        Returns:
            Updated state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        R = measurement_noise if measurement_noise is not None else self.R
        
        # Generate sigma points
        sigma_points = self._generate_sigma_points(
            self.state.mean, self.state.covariance
        )
        
        # Transform sigma points through observation function
        z_points = np.array([self.h(sp) for sp in sigma_points])
        
        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * z_points, axis=0)
        
        # Innovation covariance
        S = R.copy()
        for i, zp in enumerate(z_points):
            diff = zp - z_pred
            S += self.Wc[i] * np.outer(diff, diff)
            
        # Cross covariance
        Pxz = np.zeros((self.state_dim, self.measurement_dim))
        for i, (sp, zp) in enumerate(zip(sigma_points, z_points)):
            x_diff = sp - self.state.mean
            z_diff = zp - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
            
        # Kalman gain
        K = Pxz @ np.linalg.inv(S)
        
        # Innovation
        y = measurement - z_pred
        self.innovation_history.append(y.copy())
        
        # State update
        x_updated = self.state.mean + K @ y
        P_updated = self.state.covariance - K @ S @ K.T
        
        self.state = KalmanState(
            mean=x_updated,
            covariance=P_updated,
            timestamp=datetime.now()
        )
        self.state_history.append(self.state.copy())
        
        return self.state.copy()


class EnsembleKalmanFilter(StandardKalmanFilter):
    """
    Ensemble Kalman Filter for high-dimensional systems.
    
    Uses Monte Carlo sampling to approximate the state distribution.
    """
    
    def __init__(self, config: KalmanConfig,
                 state_transition_fn: Optional[callable] = None,
                 observation_fn: Optional[callable] = None):
        """
        Initialize the Ensemble Kalman Filter.
        
        Args:
            config: Kalman filter configuration
            state_transition_fn: Nonlinear state transition function f(x, u)
            observation_fn: Nonlinear observation function h(x)
        """
        super().__init__(config)
        
        self.f = state_transition_fn or (lambda x, u: self.F @ x)
        self.h = observation_fn or (lambda x: self.H @ x)
        self.ensemble_size = config.ensemble_size
        self.ensemble: Optional[np.ndarray] = None
        
    def initialize(self, initial_mean: np.ndarray,
                   initial_covariance: Optional[np.ndarray] = None,
                   timestamp: Optional[datetime] = None) -> None:
        """
        Initialize the filter with an ensemble.
        
        Args:
            initial_mean: Initial state mean
            initial_covariance: Initial state covariance
            timestamp: Initial timestamp
        """
        super().initialize(initial_mean, initial_covariance, timestamp)
        
        # Generate initial ensemble
        if initial_covariance is None:
            initial_covariance = np.eye(self.state_dim) * self.config.initial_covariance
            
        self.ensemble = np.random.multivariate_normal(
            initial_mean, initial_covariance, self.ensemble_size
        )
        
    def predict(self, control_input: Optional[np.ndarray] = None,
                dt: float = 1.0) -> KalmanState:
        """
        Predict the next state by propagating the ensemble.
        
        Args:
            control_input: Optional control input
            dt: Time step
            
        Returns:
            Predicted state
        """
        if self.ensemble is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        u = control_input if control_input is not None else np.zeros(1)
        
        # Propagate each ensemble member
        for i in range(self.ensemble_size):
            self.ensemble[i] = self.f(self.ensemble[i], u)
            # Add process noise
            self.ensemble[i] += np.random.multivariate_normal(
                np.zeros(self.state_dim), self.Q
            )
            
        # Update state estimate
        x_pred = np.mean(self.ensemble, axis=0)
        P_pred = np.cov(self.ensemble.T)
        
        self.state = KalmanState(
            mean=x_pred,
            covariance=P_pred,
            timestamp=datetime.now()
        )
        
        return self.state.copy()
        
    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update the state using ensemble Kalman update.
        
        Args:
            measurement: Measurement vector
            measurement_noise: Optional measurement-specific noise covariance
            
        Returns:
            Updated state
        """
        if self.ensemble is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        R = measurement_noise if measurement_noise is not None else self.R
        
        # Compute predicted measurements for each ensemble member
        z_ensemble = np.array([self.h(member) for member in self.ensemble])
        z_mean = np.mean(z_ensemble, axis=0)
        
        # Ensemble anomalies
        X_anomaly = self.ensemble - np.mean(self.ensemble, axis=0)
        Z_anomaly = z_ensemble - z_mean
        
        # Kalman gain (ensemble formulation)
        Pxz = (X_anomaly.T @ Z_anomaly) / (self.ensemble_size - 1)
        Pzz = (Z_anomaly.T @ Z_anomaly) / (self.ensemble_size - 1) + R
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update each ensemble member with perturbed observations
        for i in range(self.ensemble_size):
            perturbed_obs = measurement + np.random.multivariate_normal(
                np.zeros(self.measurement_dim), R
            )
            innovation = perturbed_obs - z_ensemble[i]
            self.ensemble[i] += K @ innovation
            
        # Update state estimate
        x_updated = np.mean(self.ensemble, axis=0)
        P_updated = np.cov(self.ensemble.T)
        
        # Store innovation
        y = measurement - z_mean
        self.innovation_history.append(y.copy())
        
        self.state = KalmanState(
            mean=x_updated,
            covariance=P_updated,
            timestamp=datetime.now()
        )
        self.state_history.append(self.state.copy())
        
        return self.state.copy()


class AdaptiveKalmanFilter(StandardKalmanFilter):
    """
    Adaptive Kalman Filter with online noise estimation.
    
    Automatically adjusts process and measurement noise based on innovation sequence.
    """
    
    def __init__(self, config: KalmanConfig):
        """
        Initialize the Adaptive Kalman Filter.
        
        Args:
            config: Kalman filter configuration
        """
        super().__init__(config)
        self.window_size = config.adaptive_window
        self.innovation_window: List[np.ndarray] = []
        self.residual_window: List[np.ndarray] = []
        
    def update(self, measurement: np.ndarray,
               measurement_noise: Optional[np.ndarray] = None) -> KalmanState:
        """
        Update the state with adaptive noise estimation.
        
        Args:
            measurement: Measurement vector
            measurement_noise: Optional measurement-specific noise covariance
            
        Returns:
            Updated state
        """
        if self.state is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
            
        # Compute innovation
        y = measurement - self.H @ self.state.mean
        self.innovation_window.append(y.copy())
        
        # Keep window size
        if len(self.innovation_window) > self.window_size:
            self.innovation_window.pop(0)
            
        # Adapt measurement noise if enough samples
        if len(self.innovation_window) >= self.window_size // 2:
            self._adapt_measurement_noise()
            
        # Standard Kalman update
        R = measurement_noise if measurement_noise is not None else self.R
        
        S = self.H @ self.state.covariance @ self.H.T + R
        K = self.state.covariance @ self.H.T @ np.linalg.inv(S)
        
        x_updated = self.state.mean + K @ y
        I_KH = np.eye(self.state_dim) - K @ self.H
        P_updated = I_KH @ self.state.covariance @ I_KH.T + K @ R @ K.T
        
        # Compute residual for process noise adaptation
        residual = measurement - self.H @ x_updated
        self.residual_window.append(residual.copy())
        
        if len(self.residual_window) > self.window_size:
            self.residual_window.pop(0)
            
        # Adapt process noise
        if len(self.residual_window) >= self.window_size // 2:
            self._adapt_process_noise()
            
        self.state = KalmanState(
            mean=x_updated,
            covariance=P_updated,
            timestamp=datetime.now()
        )
        self.state_history.append(self.state.copy())
        self.innovation_history.append(y.copy())
        
        return self.state.copy()
        
    def _adapt_measurement_noise(self) -> None:
        """Adapt measurement noise based on innovation sequence."""
        innovations = np.array(self.innovation_window)
        
        # Estimate innovation covariance
        innovation_cov = np.cov(innovations.T)
        if innovation_cov.ndim == 0:
            innovation_cov = np.array([[innovation_cov]])
            
        # Expected innovation covariance: S = H * P * H^T + R
        expected_S = self.H @ self.state.covariance @ self.H.T + self.R
        
        # Adapt R to match observed innovation covariance
        R_new = innovation_cov - self.H @ self.state.covariance @ self.H.T
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(R_new)
        if np.min(eigvals) < 0:
            R_new += (abs(np.min(eigvals)) + 1e-6) * np.eye(self.measurement_dim)
            
        # Smooth update
        alpha = 0.1
        self.R = (1 - alpha) * self.R + alpha * R_new
        
    def _adapt_process_noise(self) -> None:
        """Adapt process noise based on residual sequence."""
        residuals = np.array(self.residual_window)
        
        # Estimate residual covariance
        residual_cov = np.cov(residuals.T)
        if residual_cov.ndim == 0:
            residual_cov = np.array([[residual_cov]])
            
        # Adapt Q based on residuals
        # This is a simplified adaptation - full adaptation would use
        # the relationship between Q and the state estimation error
        scale = np.trace(residual_cov) / np.trace(self.R)
        
        if scale > 1.5:
            # Increase process noise
            self.Q *= 1.1
        elif scale < 0.5:
            # Decrease process noise
            self.Q *= 0.9


class KalmanFusionAlgorithm(SensorFusionAlgorithm):
    """
    Kalman filter-based sensor fusion algorithm.
    
    Combines multiple sensor measurements using Kalman filtering
    for optimal state estimation with uncertainty quantification.
    """
    
    def __init__(self, filter_type: KalmanFilterType = KalmanFilterType.STANDARD):
        """
        Initialize the Kalman fusion algorithm.
        
        Args:
            filter_type: Type of Kalman filter to use
        """
        self.filter_type = filter_type
        self.filters: Dict[str, StandardKalmanFilter] = {}
        self.fusion_filter: Optional[StandardKalmanFilter] = None
        
        # Compatibility matrix
        self._compatibility_matrix = {
            (SensorType.HYPERSPECTRAL, SensorType.LIDAR): 0.9,
            (SensorType.HYPERSPECTRAL, SensorType.MAGNETOMETRY): 0.7,
            (SensorType.LIDAR, SensorType.MAGNETOMETRY): 0.6,
            (SensorType.MAGNETOMETRY, SensorType.GRAVITY): 0.9,
        }
        
        # Add self-compatibility
        for sensor_type in SensorType:
            self._compatibility_matrix[(sensor_type, sensor_type)] = 1.0
            
        # Add reverse pairs
        for (type1, type2), score in list(self._compatibility_matrix.items()):
            self._compatibility_matrix[(type2, type1)] = score
            
    def _create_filter(self, config: KalmanConfig) -> StandardKalmanFilter:
        """Create a Kalman filter based on the configured type."""
        if self.filter_type == KalmanFilterType.STANDARD:
            return StandardKalmanFilter(config)
        elif self.filter_type == KalmanFilterType.EXTENDED:
            return ExtendedKalmanFilter(config)
        elif self.filter_type == KalmanFilterType.UNSCENTED:
            return UnscentedKalmanFilter(config)
        elif self.filter_type == KalmanFilterType.ENSEMBLE:
            return EnsembleKalmanFilter(config)
        elif self.filter_type == KalmanFilterType.ADAPTIVE:
            return AdaptiveKalmanFilter(config)
        else:
            return StandardKalmanFilter(config)
            
    def fuse(self, sensor_data_list: List[SensorData], **kwargs) -> SensorData:
        """
        Fuse multiple sensor data using Kalman filtering.
        
        Args:
            sensor_data_list: List of SensorData objects to fuse
            **kwargs: Additional parameters
                state_dim: State dimension (default: auto)
                process_noise: Process noise (default: 0.01)
                measurement_noise: Measurement noise per sensor (default: 0.1)
                
        Returns:
            Fused SensorData object
        """
        if len(sensor_data_list) < 2:
            raise ValueError("At least two sensor data objects required for fusion")
            
        # Determine state dimension
        state_dim = kwargs.get('state_dim', None)
        if state_dim is None:
            # Use maximum measurement dimension
            state_dim = max(
                np.prod(data.data.shape) if hasattr(data.data, 'shape') else 1
                for data in sensor_data_list
            )
            state_dim = min(state_dim, 1000)  # Cap for computational reasons
            
        # Create fusion filter
        config = KalmanConfig(
            state_dim=state_dim,
            measurement_dim=state_dim,
            process_noise=kwargs.get('process_noise', 0.01),
            measurement_noise=kwargs.get('measurement_noise', 0.1),
            filter_type=self.filter_type
        )
        
        self.fusion_filter = self._create_filter(config)
        
        # Initialize with first sensor data
        first_data = sensor_data_list[0].data
        if hasattr(first_data, 'values'):
            first_data = first_data.values
        initial_mean = first_data.flatten()[:state_dim]
        
        self.fusion_filter.initialize(initial_mean)
        
        # Process each sensor as a measurement
        for sensor_data in sensor_data_list[1:]:
            data = sensor_data.data
            if hasattr(data, 'values'):
                data = data.values
            measurement = data.flatten()[:state_dim]
            
            # Adjust measurement noise based on sensor quality
            quality = sensor_data.quality_metrics.get('accuracy', 0.8)
            R = np.eye(state_dim) * (config.measurement_noise / quality)
            
            self.fusion_filter.predict()
            self.fusion_filter.update(measurement, R)
            
        # Get fused state
        fused_state = self.fusion_filter.get_state()
        
        # Reshape to original data shape
        original_shape = sensor_data_list[0].data.shape if hasattr(sensor_data_list[0].data, 'shape') else (state_dim,)
        fused_data = fused_state.mean[:np.prod(original_shape)].reshape(original_shape)
        
        # Create fused SensorData
        return SensorData(
            data=fused_data,
            sensor_type=SensorType.CUSTOM,
            dimensions=sensor_data_list[0].dimensions,
            metadata={
                'fusion_method': 'kalman',
                'filter_type': self.filter_type.value,
                'state_covariance': fused_state.covariance.tolist(),
                'source_sensors': [d.sensor_type.value for d in sensor_data_list]
            },
            crs=sensor_data_list[0].crs,
            quality_metrics={
                'uncertainty': float(np.trace(fused_state.covariance)),
                'innovation_stats': self.fusion_filter.get_innovation_statistics()
            }
        )
        
    def get_compatibility_matrix(self) -> Dict[Tuple[SensorType, SensorType], float]:
        """Get the sensor compatibility matrix."""
        return self._compatibility_matrix.copy()


def create_kalman_filter(
    filter_type: KalmanFilterType,
    state_dim: int,
    measurement_dim: int,
    **kwargs
) -> StandardKalmanFilter:
    """
    Factory function to create a Kalman filter.
    
    Args:
        filter_type: Type of Kalman filter
        state_dim: State dimension
        measurement_dim: Measurement dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured Kalman filter instance
    """
    config = KalmanConfig(
        state_dim=state_dim,
        measurement_dim=measurement_dim,
        process_noise=kwargs.get('process_noise', 0.01),
        measurement_noise=kwargs.get('measurement_noise', 0.1),
        initial_covariance=kwargs.get('initial_covariance', 1.0),
        filter_type=filter_type,
        adaptive_window=kwargs.get('adaptive_window', 10),
        ensemble_size=kwargs.get('ensemble_size', 100),
        ukf_alpha=kwargs.get('ukf_alpha', 0.001),
        ukf_beta=kwargs.get('ukf_beta', 2.0),
        ukf_kappa=kwargs.get('ukf_kappa', 0.0)
    )
    
    if filter_type == KalmanFilterType.STANDARD:
        return StandardKalmanFilter(config)
    elif filter_type == KalmanFilterType.EXTENDED:
        return ExtendedKalmanFilter(
            config,
            state_transition_fn=kwargs.get('state_transition_fn'),
            observation_fn=kwargs.get('observation_fn'),
            state_jacobian_fn=kwargs.get('state_jacobian_fn'),
            observation_jacobian_fn=kwargs.get('observation_jacobian_fn')
        )
    elif filter_type == KalmanFilterType.UNSCENTED:
        return UnscentedKalmanFilter(
            config,
            state_transition_fn=kwargs.get('state_transition_fn'),
            observation_fn=kwargs.get('observation_fn')
        )
    elif filter_type == KalmanFilterType.ENSEMBLE:
        return EnsembleKalmanFilter(
            config,
            state_transition_fn=kwargs.get('state_transition_fn'),
            observation_fn=kwargs.get('observation_fn')
        )
    elif filter_type == KalmanFilterType.ADAPTIVE:
        return AdaptiveKalmanFilter(config)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def fuse_time_series(
    measurements: List[np.ndarray],
    timestamps: List[datetime],
    sensor_noises: Optional[List[float]] = None,
    filter_type: KalmanFilterType = KalmanFilterType.ADAPTIVE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fuse time-series measurements from multiple sensors.
    
    Args:
        measurements: List of measurement arrays (one per sensor)
        timestamps: List of timestamps
        sensor_noises: Optional noise levels per sensor
        filter_type: Type of Kalman filter to use
        
    Returns:
        Tuple of (fused_values, uncertainties)
    """
    if not measurements:
        raise ValueError("No measurements provided")
        
    n_sensors = len(measurements)
    n_samples = len(measurements[0])
    state_dim = measurements[0].shape[1] if len(measurements[0].shape) > 1 else 1
    
    # Default sensor noises
    if sensor_noises is None:
        sensor_noises = [0.1] * n_sensors
        
    # Create filter
    config = KalmanConfig(
        state_dim=state_dim,
        measurement_dim=state_dim,
        process_noise=0.01,
        measurement_noise=np.mean(sensor_noises),
        filter_type=filter_type
    )
    
    if filter_type == KalmanFilterType.ADAPTIVE:
        kf = AdaptiveKalmanFilter(config)
    else:
        kf = StandardKalmanFilter(config)
        
    # Initialize
    initial_mean = np.mean([m[0] for m in measurements], axis=0)
    if initial_mean.ndim == 0:
        initial_mean = np.array([initial_mean])
    kf.initialize(initial_mean, timestamp=timestamps[0])
    
    # Process measurements
    fused_values = []
    uncertainties = []
    
    for t in range(n_samples):
        # Predict
        kf.predict()
        
        # Update with each sensor measurement
        for s in range(n_sensors):
            measurement = measurements[s][t]
            if measurement.ndim == 0:
                measurement = np.array([measurement])
            R = np.eye(state_dim) * sensor_noises[s]
            kf.update(measurement, R)
            
        state = kf.get_state()
        fused_values.append(state.mean.copy())
        uncertainties.append(np.diag(state.covariance).copy())
        
    return np.array(fused_values), np.array(uncertainties)
