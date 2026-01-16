"""
Simple IOU-based Person Tracker.

Tracks persons across video frames using bounding box overlap.
No deep features required - designed for preprocessing before ReID.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Track:
    """Represents a tracked person."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    age: int = 0  # Frames since last update
    hits: int = 1  # Total frames with detections
    frame_indices: List[int] = field(default_factory=list)  # Frame numbers with detections
    
    velocity: Tuple[float, float] = (0.0, 0.0)  # dx, dy (per frame)
    
    def update(self, bbox: Tuple[float, float, float, float], frame_idx: int):
        """Update track with new detection."""
        # Calculate velocity
        prev_center_x = (self.bbox[0] + self.bbox[2]) / 2
        prev_center_y = (self.bbox[1] + self.bbox[3]) / 2
        curr_center_x = (bbox[0] + bbox[2]) / 2
        curr_center_y = (bbox[1] + bbox[3]) / 2
        
        # Simple exponential moving average for smoothing
        alpha = 0.7
        dx = curr_center_x - prev_center_x
        dy = curr_center_y - prev_center_y
        
        # If moving significantly, update velocity
        if abs(dx) > 1 or abs(dy) > 1:
            self.velocity = (
                self.velocity[0] * (1 - alpha) + dx * alpha,
                self.velocity[1] * (1 - alpha) + dy * alpha
            )
            
        self.bbox = bbox
        self.age = 0
        self.hits += 1
        self.frame_indices.append(frame_idx)
    
    def increment_age(self):
        """Increment age when not matched."""
        self.age += 1


class PersonTracker:
    """
    IOU-based multi-object tracker for persons.
    
    Simple tracking suitable for preprocessing:
    - Assigns track IDs based on bbox overlap
    - Handles track creation/deletion with configurable thresholds
    - No appearance features (preprocessing happens before ReID)
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,  # Frames before track deletion
        min_hits: int = 3,  # Minimum hits to confirm track
        min_iou_for_new: float = 0.05  # Don't create new track if high IOU with existing
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IOU to match detection to track
            max_age: Frames without detection before track is deleted
            min_hits: Minimum detections before track is confirmed
            min_iou_for_new: Threshold to prevent duplicate tracks
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_iou_for_new = min_iou_for_new
        
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0
    
    def update(
        self,
        detections: List[Dict],
        frame_idx: int = None
    ) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with 'x1', 'y1', 'x2', 'y2'
            frame_idx: Optional frame index
            
        Returns:
            List of detections with added 'track_id' field
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        self.frame_count += 1
        
        if not detections:
            # Increment age for all tracks
            self._increment_all_ages()
            self._remove_dead_tracks()
            return []
        
        # Extract bboxes from detections
        det_bboxes = np.array([
            [d['x1'], d['y1'], d['x2'], d['y2']]
            for d in detections
        ])
        
        # Get existing track bboxes
        track_ids = list(self.tracks.keys())
        if track_ids:
            track_bboxes = np.array([
                list(self.tracks[tid].bbox)
                for tid in track_ids
            ])
            
            # Compute IOU matrix
            iou_matrix = self._compute_iou_matrix(det_bboxes, track_bboxes)
            
            # Match detections to tracks (greedy assignment)
            matched_dets, matched_tracks, unmatched_dets = self._match(
                iou_matrix, track_ids
            )
        else:
            matched_dets = []
            matched_tracks = []
            unmatched_dets = list(range(len(detections)))
        
        # Update matched tracks
        for det_idx, track_id in zip(matched_dets, matched_tracks):
            bbox = tuple(det_bboxes[det_idx])
            self.tracks[track_id].update(bbox, frame_idx)
            detections[det_idx]['track_id'] = track_id
            detections[det_idx]['velocity'] = self.tracks[track_id].velocity
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            bbox = tuple(det_bboxes[det_idx])
            # Check if too close to existing track (avoid duplicates)
            if not self._is_duplicate(bbox):
                track_id = self._create_track(bbox, frame_idx)
                detections[det_idx]['track_id'] = track_id
                detections[det_idx]['velocity'] = (0.0, 0.0)
            else:
                detections[det_idx]['track_id'] = -1  # Invalid
        
        # Increment age for unmatched tracks
        matched_track_ids = set(matched_tracks)
        for tid in self.tracks:
            if tid not in matched_track_ids:
                self.tracks[tid].increment_age()
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        # Filter out invalid track IDs
        return [d for d in detections if d.get('track_id', -1) > 0]
    
    def _compute_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IOU matrix between two sets of boxes."""
        n1, n2 = len(boxes1), len(boxes2)
        iou = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                iou[i, j] = self._compute_iou(boxes1[i], boxes2[j])
        
        return iou
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IOU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _match(
        self,
        iou_matrix: np.ndarray,
        track_ids: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Greedy matching based on IOU matrix.
        
        Returns:
            matched_det_indices, matched_track_ids, unmatched_det_indices
        """
        n_dets, n_tracks = iou_matrix.shape
        matched_dets = []
        matched_tracks = []
        unmatched_dets = list(range(n_dets))
        
        # Get all matches above threshold
        matches = []
        for i in range(n_dets):
            for j in range(n_tracks):
                if iou_matrix[i, j] >= self.iou_threshold:
                    matches.append((iou_matrix[i, j], i, j))
        
        # Sort by IOU descending (greedy)
        matches.sort(reverse=True, key=lambda x: x[0])
        
        # Assign matches (each det and track can only match once)
        used_dets = set()
        used_tracks = set()
        
        for iou, det_idx, track_idx in matches:
            if det_idx not in used_dets and track_idx not in used_tracks:
                matched_dets.append(det_idx)
                matched_tracks.append(track_ids[track_idx])
                used_dets.add(det_idx)
                used_tracks.add(track_idx)
        
        unmatched_dets = [i for i in range(n_dets) if i not in used_dets]
        
        return matched_dets, matched_tracks, unmatched_dets
    
    def _is_duplicate(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if bbox overlaps too much with existing tracks."""
        for track in self.tracks.values():
            iou = self._compute_iou(np.array(bbox), np.array(track.bbox))
            if iou > self.min_iou_for_new:
                return True
        return False
    
    def _create_track(
        self,
        bbox: Tuple[float, float, float, float],
        frame_idx: int
    ) -> int:
        """Create a new track."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        self.tracks[track_id] = Track(
            track_id=track_id,
            bbox=bbox,
            frame_indices=[frame_idx]
        )
        
        return track_id
    
    def _increment_all_ages(self):
        """Increment age for all tracks."""
        for track in self.tracks.values():
            track.increment_age()
    
    def _remove_dead_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        dead_ids = [
            tid for tid, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]
    
    def get_active_tracks(self) -> List[Track]:
        """Get list of active (confirmed) tracks."""
        return [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0


if __name__ == '__main__':
    # Simple test
    tracker = PersonTracker()
    
    # Simulated detections
    frame1 = [{'x1': 100, 'y1': 100, 'x2': 200, 'y2': 300}]
    frame2 = [{'x1': 110, 'y1': 105, 'x2': 210, 'y2': 305}]  # Same person, moved
    frame3 = [
        {'x1': 115, 'y1': 110, 'x2': 215, 'y2': 310},  # Person 1
        {'x1': 400, 'y1': 100, 'x2': 500, 'y2': 300}   # New person
    ]
    
    print("Frame 1:", tracker.update(frame1, 0))
    print("Frame 2:", tracker.update(frame2, 1))
    print("Frame 3:", tracker.update(frame3, 2))
    print("Active tracks:", [t.track_id for t in tracker.get_active_tracks()])
