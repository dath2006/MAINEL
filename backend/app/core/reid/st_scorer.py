"""
Spatial-Temporal Scorer

Computes spatial-temporal probabilities for cross-camera matching
using Transition Time Distributions (TTD).
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from loguru import logger

from app.config import settings


@dataclass
class TransitionStats:
    """Statistics for camera pair transitions."""
    from_camera: int
    to_camera: int
    observed_times: List[float] = field(default_factory=list)
    mean_time: float = 0.0
    std_time: float = 10.0
    count: int = 0


class ParzenEstimator:
    """
    Histogram-Parzen window estimator for transition time distributions.
    
    Non-parametric density estimation that can capture multi-modal
    distributions (e.g., people taking different routes).
    """
    
    def __init__(self, bandwidth: float = 5.0):
        """
        Args:
            bandwidth: Kernel bandwidth in seconds
        """
        self.bandwidth = bandwidth
        self.observations: List[float] = []
    
    def add_observation(self, time_delta: float):
        """Add observed transition time."""
        if time_delta > 0:
            self.observations.append(time_delta)
    
    def pdf(self, time_delta: float) -> float:
        """
        Estimate probability density at given time delta.
        
        Uses Gaussian kernel with specified bandwidth.
        """
        if len(self.observations) == 0:
            return 0.0
        
        # Kernel density estimation
        n = len(self.observations)
        density = 0.0
        
        for obs in self.observations:
            # Gaussian kernel
            z = (time_delta - obs) / self.bandwidth
            density += np.exp(-0.5 * z * z)
        
        density /= (n * self.bandwidth * np.sqrt(2 * np.pi))
        return density
    
    @property
    def mean(self) -> float:
        """Mean transition time."""
        return np.mean(self.observations) if self.observations else 0.0
    
    @property
    def std(self) -> float:
        """Standard deviation of transition time."""
        return np.std(self.observations) if len(self.observations) > 1 else 10.0


class SpatioTemporalScorer:
    """
    Spatial-temporal scorer for cross-camera matching.
    
    Computes probability that a person could transition between
    two cameras in a given time interval based on:
    - Physical distance between cameras
    - Observed transition times (learned from data)
    - Typical pedestrian walking speeds
    """
    
    # Typical pedestrian speeds (m/s) - Configurable via settings
    
    @property
    def MIN_SPEED(self) -> float:
        return settings.st_min_speed

    @property
    def AVG_SPEED(self) -> float:
        return settings.st_avg_speed

    @property
    def MAX_SPEED(self) -> float:
        return settings.st_max_speed
    
    def __init__(
        self,
        bandwidth: float = None,
        max_transition_time: float = 300.0,
        use_parzen: bool = True,
    ):
        """
        Args:
            bandwidth: Parzen window bandwidth (from config if None)
            max_transition_time: Max allowed transition time (seconds)
            use_parzen: Use Parzen estimation vs Gaussian
        """
        self.bandwidth = bandwidth if bandwidth is not None else settings.st_bandwidth
        self.max_transition_time = max_transition_time
        self.use_parzen = use_parzen
        
        # Transition Time Distributions: (from_cam, to_cam) -> estimator/stats
        self.ttd: Dict[Tuple[int, int], ParzenEstimator] = {}
        self.transition_stats: Dict[Tuple[int, int], TransitionStats] = {}
        
        # Camera positions: camera_id -> (lat, lon)
        self.camera_positions: Dict[int, Tuple[float, float]] = {}
        
        logger.info(f"ST Scorer initialized (max_time={max_transition_time}s)")
    
    def set_camera_position(self, camera_id: int, lat: float, lon: float):
        """Set camera GPS position."""
        self.camera_positions[camera_id] = (lat, lon)
    
    def get_distance(self, cam1: int, cam2: int) -> Optional[float]:
        """
        Get distance between two cameras in meters.
        
        Uses Haversine formula for GPS coordinates.
        """
        if cam1 not in self.camera_positions or cam2 not in self.camera_positions:
            return None
        
        lat1, lon1 = self.camera_positions[cam1]
        lat2, lon2 = self.camera_positions[cam2]
        
        # Haversine formula
        R = 6371000  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = np.sin(delta_phi / 2) ** 2 + \
            np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c
    
    def update_ttd(
        self,
        from_camera: int,
        to_camera: int,
        time_delta: float,
    ):
        """
        Update transition time distribution with new observation.
        
        Args:
            from_camera: Source camera ID
            to_camera: Destination camera ID
            time_delta: Observed transition time in seconds
        """
        key = (from_camera, to_camera)
        
        if self.use_parzen:
            if key not in self.ttd:
                self.ttd[key] = ParzenEstimator(bandwidth=self.bandwidth)
            self.ttd[key].add_observation(time_delta)
        else:
            # Simple Gaussian stats
            if key not in self.transition_stats:
                self.transition_stats[key] = TransitionStats(
                    from_camera=from_camera,
                    to_camera=to_camera,
                )
            
            stats = self.transition_stats[key]
            stats.observed_times.append(time_delta)
            stats.count += 1
            stats.mean_time = np.mean(stats.observed_times)
            stats.std_time = np.std(stats.observed_times) if len(stats.observed_times) > 1 else 10.0
        
        logger.debug(f"TTD updated: {from_camera} -> {to_camera}, delta={time_delta:.1f}s")
    
    def calculate_score(
        self,
        from_camera: int,
        to_camera: int,
        time_delta: float,
    ) -> float:
        """
        Calculate spatial-temporal probability score.
        
        Args:
            from_camera: Source camera ID
            to_camera: Destination camera ID
            time_delta: Observed transition time in seconds
            
        Returns:
            Probability score in [0, 1]
        """
        if time_delta <= 0 or time_delta > self.max_transition_time:
            return 0.0
        
        key = (from_camera, to_camera)
        
        # Check if we have learned distribution
        if self.use_parzen and key in self.ttd and len(self.ttd[key].observations) >= 5:
            # Use Parzen window estimation
            score = self.ttd[key].pdf(time_delta)
            # Normalize to [0, 1] range
            max_pdf = self.ttd[key].pdf(self.ttd[key].mean)
            score = min(score / max_pdf if max_pdf > 0 else 0, 1.0)
        elif not self.use_parzen and key in self.transition_stats:
            # Use Gaussian distribution
            stats_entry = self.transition_stats[key]
            z = (time_delta - stats_entry.mean_time) / max(stats_entry.std_time, 1.0)
            score = np.exp(-0.5 * z * z)
        else:
            # Fallback: use distance-based estimation
            score = self._distance_based_score(from_camera, to_camera, time_delta)
        
        return float(score)
    
    def _distance_based_score(
        self,
        from_camera: int,
        to_camera: int,
        time_delta: float,
    ) -> float:
        """
        Fallback scoring based on physical distance.
        
        Assumes normal walking speed distribution.
        """
        distance = self.get_distance(from_camera, to_camera)
        
        if distance is None:
            # No position data, use uniform prior
            return 1.0 / self.max_transition_time
        
        # Check if transition is physically possible
        min_time = distance / self.MAX_SPEED
        max_time = distance / self.MIN_SPEED
        
        if time_delta < min_time * 0.5:  # Too fast (allow some margin)
            return 0.0
        if time_delta > max_time * 2.0:  # Too slow
            return 0.1  # Low but non-zero (could have stopped)
        
        # Expected time at average walking speed
        expected_time = distance / self.AVG_SPEED
        
        # Gaussian likelihood centered at expected time
        std = expected_time * 0.3  # 30% relative std
        z = (time_delta - expected_time) / max(std, 1.0)
        score = np.exp(-0.5 * z * z)
        
        return float(score)
    
    def joint_score(
        self,
        visual_score: float,
        st_score: float,
        alpha: float = 0.5,
        use_logistic: bool = True,
    ) -> float:
        """
        Compute joint matching score.
        
        Args:
            visual_score: Visual similarity in [0, 1]
            st_score: Spatial-temporal probability in [0, 1]
            alpha: Weight for ST score
            use_logistic: Apply logistic smoothing
            
        Returns:
            Joint score in [0, 1]
        """
        if use_logistic:
            # Logistic smoothing to prevent zero scores
            st_factor = 1 / (1 + np.exp(-10 * (st_score - 0.3)))
            joint = visual_score * (1 - alpha) + visual_score * alpha * st_factor
        else:
            # Simple weighted average
            joint = (1 - alpha) * visual_score + alpha * st_score
        
        return float(np.clip(joint, 0, 1))
    
    def is_transition_plausible(
        self,
        from_camera: int,
        to_camera: int,
        time_delta: float,
        threshold: float = None,
    ) -> bool:
        """
        Check if transition is physically plausible.
        
        Args:
            from_camera: Source camera
            to_camera: Destination camera
            time_delta: Transition time in seconds
            threshold: Minimum probability threshold (from config if None)
            
        Returns:
            True if transition is plausible
        """
        threshold = threshold if threshold is not None else settings.st_plausibility_threshold
        score = self.calculate_score(from_camera, to_camera, time_delta)
        return score >= threshold
