"""
Kalman Filter for Object Tracking

Implements a Kalman filter for tracking bounding box targets
in image space using (center_x, center_y, aspect_ratio, height) representation.
"""

import numpy as np
import scipy.linalg
from typing import Tuple


class KalmanFilter:
    """
    Kalman filter for target tracking in image space.
    
    Uses an 8-dimensional state space:
        (x, y, a, h, vx, vy, va, vh)
    where:
        - (x, y) is the center position
        - a is the aspect ratio (width/height)
        - h is the height
        - (vx, vy, va, vh) are the respective velocities
    
    Motion model: constant velocity model
    Observation model: (x, y, a, h)
    """
    
    # Motion covariance weights
    _std_weight_position = 1. / 20
    _std_weight_velocity = 1. / 160
    
    def __init__(self):
        """Initialize Kalman filter parameters."""
        ndim = 4  # observation dimension
        dt = 1.0  # time step
        
        # State transition matrix (motion model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        # Observation matrix (maps state to observation)
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Standard deviations for initializing covariance
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track state from first measurement.
        
        Args:
            measurement: Bounding box in format (x, y, a, h) where
                (x, y) is center, a is aspect ratio, h is height
                
        Returns:
            Tuple of (mean, covariance) of initial state
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        
        Args:
            mean: Current state mean (8,)
            covariance: Current state covariance (8, 8)
            
        Returns:
            Tuple of (predicted_mean, predicted_covariance)
        """
        # Motion noise covariance
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # Predict new state
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov
        
        return mean, covariance
    
    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        
        Args:
            mean: State mean (8,)
            covariance: State covariance (8, 8)
            
        Returns:
            Tuple of (projected_mean, projected_covariance)
        """
        # Measurement noise
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        
        # Project to measurement space
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        
        return mean, covariance + innovation_cov
    
    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter update step.
        
        Args:
            mean: Predicted state mean (8,)
            covariance: Predicted state covariance (8, 8)
            measurement: Observed measurement (x, y, a, h)
            
        Returns:
            Tuple of (updated_mean, updated_covariance)
        """
        # Project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Cholesky factorization for numerical stability
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        
        # Kalman gain
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        
        # Innovation (measurement residual)
        innovation = measurement - projected_mean
        
        # Update state
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        
        return new_mean, new_covariance
    
    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        Compute gating distance (Mahalanobis distance).
        
        Args:
            mean: State mean (8,)
            covariance: State covariance (8, 8)
            measurements: Array of measurements (N, 4)
            only_position: Use position components only
            
        Returns:
            Squared Mahalanobis distances (N,)
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        chol_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            chol_factor, d.T,
            lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        
        return squared_maha
