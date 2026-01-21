"""
ByteTrack Multi-Object Tracker

Pure Python implementation of ByteTrack algorithm which uses:
- Two-stage association: high-confidence then low-confidence detections
- Kalman filtering for motion prediction
- Optional deep appearance features for identity matching

Reference: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
https://arxiv.org/abs/2110.06864
"""

from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger

from app.core.tracking.kalman import KalmanFilter
from app.schemas.track import Detection


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1    # New track, not yet confirmed
    CONFIRMED = 2    # Confirmed track with enough observations
    DELETED = 3      # Track marked for deletion


class CrossCameraTrackState(Enum):
    """Cross-camera lifecycle states for occlusion awareness."""
    TENTATIVE = 1      # New, needs confirmation
    CONFIRMED = 2      # Strong multi-camera evidence
    OCCLUDED = 3       # Currently occluded by another person
    MISSING = 4        # Lost, searching across cameras
    DELETED = 5        # Permanently removed


@dataclass
class OcclusionInfo:
    """Tracks occlusion events for ID correction."""
    occluding_track_ids: List[int] = field(default_factory=list)
    occlusion_start_time: Optional[float] = None  # Frame number
    pre_occlusion_embedding: Optional[np.ndarray] = None
    occlusion_count: int = 0


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
    
    # Detection score for ByteTrack's scoring system
    score: float = 0.0
    
    # Metadata
    class_id: int = -1
    class_name: str = "unknown"
    confidence: float = 0.0
    global_id: Optional[str] = None  # Global Re-ID across cameras
    
    # Cross-camera state tracking
    cross_camera_state: CrossCameraTrackState = CrossCameraTrackState.TENTATIVE
    occlusion_info: OcclusionInfo = field(default_factory=OcclusionInfo)
    last_high_quality_embedding: Optional[np.ndarray] = None
    quality_history: List[float] = field(default_factory=list)
    views_confirmed: set = field(default_factory=set)  # Camera IDs where confirmed
    
    # Face bbox (attached during processing)
    face_bbox: Optional[List[float]] = None
    
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
        self.score = detection.confidence

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
    
    def update_occlusion_state(
        self,
        is_occluded: bool,
        occluding_ids: List[int],
        current_frame: int,
    ):
        """
        Update occlusion state and store pre-occlusion embedding.
        
        Args:
            is_occluded: Whether this track is currently occluded
            occluding_ids: List of track IDs that are occluding this track
            current_frame: Current frame number
        """
        if is_occluded and self.cross_camera_state != CrossCameraTrackState.OCCLUDED:
            # Entering occlusion
            self.cross_camera_state = CrossCameraTrackState.OCCLUDED
            self.occlusion_info.occlusion_start_time = current_frame
            self.occlusion_info.occluding_track_ids = occluding_ids
            self.occlusion_info.occlusion_count += 1
            
            # Store last high-quality embedding before occlusion
            if self.last_high_quality_embedding is not None:
                self.occlusion_info.pre_occlusion_embedding = self.last_high_quality_embedding.copy()
                
        elif not is_occluded and self.cross_camera_state == CrossCameraTrackState.OCCLUDED:
            # Exiting occlusion - flag for verification
            self.cross_camera_state = CrossCameraTrackState.CONFIRMED
            # Keep occlusion_info for post-verification (don't clear yet)
    
    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE
    
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED
    
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED


class ByteTrackTracker:
    """
    ByteTrack multi-object tracker.
    
    Key innovation: Two-stage association that uses both high-confidence
    and low-confidence detections to maintain tracks during occlusion.
    
    Stage 1: Match high-confidence detections with existing tracks
    Stage 2: Match remaining tracks with low-confidence detections
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        max_age: int = 30,
        n_init: int = 3,
        use_appearance: bool = True,
        appearance_thresh: float = 0.4,
    ):
        """
        Initialize ByteTrack tracker.
        
        Args:
            track_thresh: Threshold for high-confidence detections
            low_thresh: Threshold for low-confidence detections
            match_thresh: IoU threshold for matching
            max_age: Max frames before track deletion (track_buffer)
            n_init: Frames to confirm a track
            use_appearance: Whether to use appearance features
            appearance_thresh: Threshold for appearance matching
        """
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.n_init = n_init
        self.use_appearance = use_appearance
        self.appearance_thresh = appearance_thresh
        
        self.kf = KalmanFilter()
        
        self.tracks: List[Track] = []
        self._next_id = 1
        
        # Feature gallery for appearance matching
        self._feature_gallery: Dict[int, List[np.ndarray]] = {}
        self._feature_budget = 100
    
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
        Update tracks with new detections using ByteTrack's two-stage association.
        
        Args:
            detections: All detections for current frame
            features: Optional feature embeddings for detections (N, D)
            
        Returns:
            List of active tracks
        """
        if len(detections) == 0:
            # No detections - mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
            return self.tracks
        
        # Get detection scores
        scores = np.array([d.confidence for d in detections])
        
        # Split detections by confidence
        high_conf_mask = scores >= self.track_thresh
        low_conf_mask = (scores < self.track_thresh) & (scores >= self.low_thresh)
        
        high_conf_dets = [d for i, d in enumerate(detections) if high_conf_mask[i]]
        low_conf_dets = [d for i, d in enumerate(detections) if low_conf_mask[i]]
        
        high_conf_indices = np.where(high_conf_mask)[0].tolist()
        low_conf_indices = np.where(low_conf_mask)[0].tolist()
        
        high_conf_features = features[high_conf_indices] if features is not None and len(high_conf_indices) > 0 else None
        low_conf_features = features[low_conf_indices] if features is not None and len(low_conf_indices) > 0 else None
        
        # Get confirmed and unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        
        # ===== STAGE 1: Match high-confidence detections with confirmed tracks =====
        matches_high, unmatched_tracks_stage1, unmatched_dets_high = self._match_with_tracks(
            confirmed_tracks, high_conf_dets, high_conf_features
        )
        
        # Convert detection indices back to original
        matches_high = [(t, high_conf_indices[d]) for t, d in matches_high]
        unmatched_high_original = [high_conf_indices[d] for d in unmatched_dets_high]
        
        # ===== STAGE 2: Match remaining tracks with low-confidence detections =====
        # Only tracks that were unmatched in stage 1
        matches_low, unmatched_tracks_stage2, unmatched_dets_low = self._match_with_tracks(
            unmatched_tracks_stage1, low_conf_dets, low_conf_features
        )
        
        # Convert detection indices back to original
        matches_low = [(t, low_conf_indices[d]) for t, d in matches_low]
        
        # ===== STAGE 3: Match unconfirmed tracks with remaining high-confidence detections =====
        remaining_high_dets = [detections[i] for i in unmatched_high_original]
        remaining_high_features = features[unmatched_high_original] if features is not None and len(unmatched_high_original) > 0 else None
        
        matches_unconf, unmatched_unconf, unmatched_remaining = self._match_with_tracks(
            unconfirmed_tracks, remaining_high_dets, remaining_high_features
        )
        
        # Convert indices
        matches_unconf = [(t, unmatched_high_original[d]) for t, d in matches_unconf]
        final_unmatched_dets = [unmatched_high_original[d] for d in unmatched_remaining]
        
        # Combine all matches
        all_matches = matches_high + matches_low + matches_unconf
        
        # Update matched tracks
        for track_idx, det_idx in all_matches:
            feature = features[det_idx] if features is not None else None
            self.tracks[track_idx].update(
                self.kf, detections[det_idx], feature
            )
            # Update feature gallery
            if feature is not None:
                self._update_feature_gallery(self.tracks[track_idx].track_id, feature)
        
        # Mark unmatched tracks as missed
        all_unmatched_tracks = set(unmatched_tracks_stage2 + unmatched_unconf)
        for track_idx in all_unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Initialize new tracks from unmatched high-confidence detections
        for det_idx in final_unmatched_dets:
            self._initiate_track(
                detections[det_idx],
                features[det_idx] if features is not None else None
            )
        
        # Remove deleted tracks
        deleted_ids = [t.track_id for t in self.tracks if t.is_deleted()]
        for track_id in deleted_ids:
            self._feature_gallery.pop(track_id, None)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        return self.tracks
    
    def _match_with_tracks(
        self,
        track_indices: List[int],
        detections: List[Detection],
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU (and optionally appearance).
        
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if len(track_indices) == 0 or len(detections) == 0:
            return [], track_indices, list(range(len(detections)))
        
        # Compute IoU cost matrix
        iou_matrix = self._compute_iou_matrix(track_indices, detections)
        
        # Optionally combine with appearance features
        if self.use_appearance and features is not None and len(features) > 0:
            appearance_matrix = self._compute_appearance_matrix(track_indices, features)
            # Combine: IoU dominates (0.7) with appearance boost (0.3)
            cost_matrix = 0.7 * (1 - iou_matrix) + 0.3 * appearance_matrix
        else:
            cost_matrix = 1 - iou_matrix
        
        # Apply IoU threshold
        cost_matrix[iou_matrix < (1 - self.match_thresh)] = 1e10
        
        # Hungarian assignment
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = [], []
        
        matches = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] >= 1e10:
                continue
            matches.append((track_indices[col], row))
        
        matched_tracks = set([m[0] for m in matches])
        matched_dets = set([m[1] for m in matches])
        
        unmatched_tracks = [t for t in track_indices if t not in matched_tracks]
        unmatched_dets = [d for d in range(len(detections)) if d not in matched_dets]
        
        return matches, unmatched_tracks, unmatched_dets
    
    def _compute_iou_matrix(
        self,
        track_indices: List[int],
        detections: List[Detection],
    ) -> np.ndarray:
        """Compute IoU matrix between tracks and detections."""
        iou_matrix = np.zeros((len(detections), len(track_indices)))
        
        for i, det in enumerate(detections):
            det_bbox = (det.x1, det.y1, det.x2, det.y2)
            for j, track_idx in enumerate(track_indices):
                track_bbox = self.tracks[track_idx].to_tlbr()
                iou_matrix[i, j] = self._iou(det_bbox, track_bbox)
        
        return iou_matrix
    
    def _compute_appearance_matrix(
        self,
        track_indices: List[int],
        features: np.ndarray,
    ) -> np.ndarray:
        """Compute appearance distance matrix."""
        cost_matrix = np.ones((len(features), len(track_indices)))
        
        for j, track_idx in enumerate(track_indices):
            track_id = self.tracks[track_idx].track_id
            if track_id not in self._feature_gallery or len(self._feature_gallery[track_id]) == 0:
                continue
            
            gallery = np.array(self._feature_gallery[track_id])
            # Cosine distance = 1 - cosine similarity
            similarities = np.dot(features, gallery.T)
            # Take min distance (max similarity)
            cost_matrix[:, j] = 1 - np.max(similarities, axis=1)
        
        return cost_matrix
    
    def _update_feature_gallery(self, track_id: int, feature: np.ndarray):
        """Update feature gallery for a track."""
        if track_id not in self._feature_gallery:
            self._feature_gallery[track_id] = []
        self._feature_gallery[track_id].append(feature)
        if len(self._feature_gallery[track_id]) > self._feature_budget:
            self._feature_gallery[track_id] = self._feature_gallery[track_id][-self._feature_budget:]
    
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
            score=detection.confidence,
        )
        
        if feature is not None:
            track.features.append(feature)
            self._update_feature_gallery(self._next_id, feature)
        
        self.tracks.append(track)
        self._next_id += 1
        
        logger.debug(f"ByteTrack: Initiated new track {track.track_id}")


# Alias for backward compatibility
DeepSORTTracker = ByteTrackTracker
