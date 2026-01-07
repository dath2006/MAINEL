"""
DeepSORT Multi-Object Tracker

Implements the DeepSORT tracking algorithm which combines:
- Kalman filtering for motion prediction
- Deep appearance features for identity matching
- Hungarian algorithm for optimal assignment
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger

from app.core.tracking.kalman import KalmanFilter
from app.core.detection.yolo_detector import Detection


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1    # New track, not yet confirmed
    CONFIRMED = 2    # Confirmed track with enough observations
    DELETED = 3      # Track marked for deletion


@dataclass
class Track:
    """
    Single track object.
    
    Represents a tracked identity within a single camera.
    Maintains state, features, and track history.
    """
    track_id: int
    mean: np.ndarray  # Kalman state mean (8,)
    covariance: np.ndarray  # Kalman state covariance (8, 8)
    n_init: int  # Frames to confirm track
    max_age: int  # Max frames before deletion
    
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1  # Number of successful updates
    age: int = 1  # Total number of frames since creation
    time_since_update: int = 0  # Frames since last successful update
    features: List[np.ndarray] = field(default_factory=list)
    
    # Metadata
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0
    global_id: Optional[str] = None  # Global Re-ID across cameras
    face_bbox: Optional[List[int]] = None  # Face bounding box [x1, y1, x2, y2] in frame coords
    
    def to_xyah(self) -> np.ndarray:
        """Get current position in (x, y, a, h) format."""
        return self.mean[:4].copy()
    
    def to_tlwh(self) -> np.ndarray:
        """Get current position in (x, y, w, h) format (top-left)."""
        ret = self.to_xyah().copy()
        ret[2] *= ret[3]  # aspect ratio * height = width
        ret[:2] -= ret[2:] / 2  # center to top-left
        return ret
    
    def to_tlbr(self) -> np.ndarray:
        """Get current position in (x1, y1, x2, y2) format."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]  # width/height to bottom-right
        return ret
    
    def predict(self, kf: KalmanFilter):
        """Propagate state with Kalman prediction."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf: KalmanFilter, detection: Detection, feature: Optional[np.ndarray] = None):
        """Update track with new detection."""
        # Convert detection to (x, y, a, h) format
        measurement = np.array(detection.to_xyah())
        
        # Kalman update
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, measurement
        )
        
        # Update metadata from latest detection
        self.class_id = detection.class_id
        if hasattr(detection, 'class_name'):
             self.class_name = detection.class_name
        self.confidence = detection.confidence

        # Update feature history
        if feature is not None:
            self.features.append(feature)
            # Keep only recent features
            if len(self.features) > 100:
                self.features = self.features[-100:]
        
        self.hits += 1
        self.time_since_update = 0
        
        # Transition to confirmed if enough hits
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    
    def mark_missed(self):
        """Mark track as missed (no detection assigned)."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED
    
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE
    
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


class NearestNeighborDistanceMetric:
    """
    Distance metric for matching detections to tracks.
    
    Uses appearance features with either cosine or euclidean distance,
    and maintains a gallery of features for each track.
    """
    
    def __init__(
        self,
        metric: str = "cosine",
        matching_threshold: float = 0.4,
        budget: Optional[int] = 100,
    ):
        """
        Args:
            metric: Distance metric ('cosine' or 'euclidean')
            matching_threshold: Max distance for valid matches
            budget: Max features to store per identity
        """
        self.metric = metric
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, List[np.ndarray]] = {}
    
    def partial_fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        active_targets: List[int],
    ):
        """Update feature gallery with new observations."""
        for feature, target in zip(features, targets):
            if target not in self.samples:
                self.samples[target] = []
            self.samples[target].append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        # Remove inactive targets
        self.samples = {k: v for k, v in self.samples.items() if k in active_targets}
    
    def distance(self, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute distance matrix.
        
        Args:
            features: New detection features (N, D)
            targets: Track IDs to compare against (M,)
            
        Returns:
            Distance matrix (N, M)
        """
        cost_matrix = np.zeros((len(features), len(targets)))
        
        for i, target in enumerate(targets):
            if target not in self.samples or len(self.samples[target]) == 0:
                cost_matrix[:, i] = 1e10  # No features, max cost
                continue
            
            gallery = np.array(self.samples[target])
            
            if self.metric == "cosine":
                # Cosine distance = 1 - cosine similarity
                distances = 1 - np.dot(features, gallery.T)
                cost_matrix[:, i] = np.min(distances, axis=1)
            else:
                # Euclidean distance
                distances = np.linalg.norm(features[:, np.newaxis] - gallery, axis=2)
                cost_matrix[:, i] = np.min(distances, axis=1)
        
        return cost_matrix


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker.
    
    Combines Kalman filtering with deep appearance features
    for robust multi-object tracking.
    """
    
    # Chi-squared distribution threshold for gating (95% confidence at 4 DOF)
    CHI2_THRESHOLD = 9.4877
    
    def __init__(
        self,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
        metric: str = "cosine",
        matching_threshold: float = 0.4,
    ):
        """
        Initialize tracker.
        
        Args:
            max_iou_distance: Max IOU distance for unconfirmed track matching
            max_age: Max frames before track deletion
            n_init: Frames to confirm a track
            metric: Distance metric for appearance matching
            matching_threshold: Max distance for valid matches
        """
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        
        self.kf = KalmanFilter()
        self.metric = NearestNeighborDistanceMetric(
            metric=metric,
            matching_threshold=matching_threshold,
        )
        
        self.tracks: List[Track] = []
        self._next_id = 1
    
    def predict(self):
        """Propagate all tracks one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)
    
    def update(
        self,
        detections: List[Detection],
        features: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: New detections for current frame
            features: Optional feature embeddings for detections (N, D)
            
        Returns:
            List of active tracks
        """
        # Run matching cascade
        matches, unmatched_tracks, unmatched_detections = self._match(
            detections, features
        )
        
        # Update matched tracks
        for track_idx, detection_idx in matches:
            feature = features[detection_idx] if features is not None else None
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx], feature
            )
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Initialize new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(
                detections[detection_idx],
                features[detection_idx] if features is not None else None
            )
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Update distance metric with confirmed track features
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        all_features, all_targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            all_features.extend(track.features)
            all_targets.extend([track.track_id] * len(track.features))
            # Keep only the most recent feature for ReID access, clear older ones
            if len(track.features) > 0:
                track.features = [track.features[-1]]  # Keep last feature
            else:
                track.features = []
        
        if len(all_features) > 0:
            self.metric.partial_fit(
                np.array(all_features),
                np.array(all_targets),
                active_targets
            )
        
        return self.tracks
    
    def _match(
        self,
        detections: List[Detection],
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks.
        
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        # Split tracks by confirmation status
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # Associate confirmed tracks using appearance + motion
        matches_a, unmatched_tracks_a, unmatched_detections = self._matching_cascade(
            confirmed_tracks, detections, features
        )
        
        # Associate remaining tracks with IOU matching
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        
        matches_b, unmatched_tracks_b, unmatched_detections = self._iou_matching(
            iou_track_candidates, detections, unmatched_detections
        )
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _matching_cascade(
        self,
        track_indices: List[int],
        detections: List[Detection],
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Cascade matching by track age."""
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        
        if features is None:
            # Fall back to IOU matching if no features
            return [], track_indices, list(range(len(detections)))
        
        unmatched_detections = list(range(len(detections)))
        matches = []
        
        for level in range(self.max_age):
            if len(unmatched_detections) == 0:
                break
            
            track_indices_at_level = [
                k for k in track_indices
                if self.tracks[k].time_since_update == 1 + level
            ]
            if len(track_indices_at_level) == 0:
                continue
            
            # Build cost matrix using appearance features
            feat = features[unmatched_detections]
            targets = np.array([self.tracks[k].track_id for k in track_indices_at_level])
            cost_matrix = self.metric.distance(feat, targets)
            
            # Apply gating based on Mahalanobis distance
            for i, track_idx in enumerate(track_indices_at_level):
                measurements = np.array([
                    detections[d].to_xyah() for d in unmatched_detections
                ])
                gating_distance = self.kf.gating_distance(
                    self.tracks[track_idx].mean,
                    self.tracks[track_idx].covariance,
                    measurements
                )
                cost_matrix[gating_distance > self.CHI2_THRESHOLD, i] = 1e10
            
            # Apply matching threshold
            cost_matrix[cost_matrix > self.metric.matching_threshold] = 1e10
            
            # Hungarian assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] >= 1e10:
                    continue
                matches.append((track_indices_at_level[col], unmatched_detections[row]))
            
            matched_detections = set([m[1] for m in matches if m[0] in track_indices_at_level])
            unmatched_detections = [d for d in unmatched_detections if d not in matched_detections]
        
        unmatched_tracks = [t for t in track_indices if t not in [m[0] for m in matches]]
        return matches, unmatched_tracks, unmatched_detections
    
    def _iou_matching(
        self,
        track_indices: List[int],
        detections: List[Detection],
        detection_indices: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IOU-based matching for unconfirmed tracks."""
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], track_indices, detection_indices
        
        # Build IOU cost matrix
        cost_matrix = np.zeros((len(detection_indices), len(track_indices)))
        for i, det_idx in enumerate(detection_indices):
            det_bbox = detections[det_idx].bbox  # x1, y1, x2, y2
            for j, track_idx in enumerate(track_indices):
                track_bbox = self.tracks[track_idx].to_tlbr()  # x1, y1, x2, y2
                cost_matrix[i, j] = 1 - self._iou(det_bbox, track_bbox)
        
        # Apply threshold
        cost_matrix[cost_matrix > self.max_iou_distance] = 1e10
        
        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] >= 1e10:
                continue
            matches.append((track_indices[col], detection_indices[row]))
        
        unmatched_tracks = [t for t in track_indices if t not in [m[0] for m in matches]]
        unmatched_detections = [d for d in detection_indices if d not in [m[1] for m in matches]]
        
        return matches, unmatched_tracks, unmatched_detections
    
    @staticmethod
    def _iou(bbox1: tuple, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes (x1, y1, x2, y2)."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def _initiate_track(self, detection: Detection, feature: Optional[np.ndarray]):
        """Create new track from detection."""
        measurement = np.array(detection.to_xyah())
        mean, covariance = self.kf.initiate(measurement)
        
        track = Track(
            track_id=self._next_id,
            mean=mean,
            covariance=covariance,
            n_init=self.n_init,
            max_age=self.max_age,
        )
        
        if feature is not None:
            track.features.append(feature)
        
        self.tracks.append(track)
        self._next_id += 1
        
        logger.debug(f"Initiated new track {track.track_id}")
