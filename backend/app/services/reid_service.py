"""
ReID Service

Handles cross-camera person re-identification.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass
import numpy as np
from loguru import logger

from app.core.reid import VisualMatcher, SpatioTemporalScorer, CameraTopology
from app.config import settings


@dataclass
class MatchResult:
    """ReID match result."""
    global_track_id: UUID
    visual_similarity: float
    st_probability: float
    joint_score: float
    is_new: bool = False


@dataclass
class TrackletInfo:
    """Information about a tracklet for matching."""
    tracklet_id: UUID
    camera_id: int
    start_time: datetime
    end_time: Optional[datetime]
    embedding: np.ndarray
    global_track_id: Optional[UUID] = None


class ReIDService:
    """
    Service for cross-camera person re-identification.
    
    Combines visual similarity with spatial-temporal constraints
    for robust identity matching.
    """
    
    def __init__(
        self,
        match_threshold: float = None,
        st_weight: float = None,
        max_transition_time: float = None,
    ):
        self.match_threshold = match_threshold or settings.reid_match_threshold
        self.st_weight = st_weight or settings.st_weight
        self.max_transition_time = max_transition_time or settings.max_transition_time
        
        # Core components
        self.visual_matcher = VisualMatcher(
            match_threshold=self.match_threshold,
        )
        self.st_scorer = SpatioTemporalScorer(
            max_transition_time=self.max_transition_time,
        )
        self.topology = CameraTopology(
            auto_connect_radius=500.0,  # Auto-connect cameras within 500m
        )
        
        # Recent tracklets for candidate matching
        self._recent_tracklets: Dict[UUID, TrackletInfo] = {}
        self._tracklet_expiry = timedelta(seconds=self.max_transition_time * 2)
        
        # Global track counter
        self._next_global_id = 1
        
        # Person thumbnails for gallery display (global_id -> base64 image)
        self.person_thumbnails: Dict[str, str] = {}
        # Quality scores for thumbnails (to update only if better quality)
        self.thumbnail_quality: Dict[str, float] = {}
        
        logger.info(f"ReIDService initialized (threshold={self.match_threshold})")
    
    def register_camera(
        self,
        camera_id: int,
        lat: float,
        lon: float,
    ):
        """Register camera position for ST scoring."""
        self.topology.add_camera(camera_id, lat, lon)
        self.st_scorer.set_camera_position(camera_id, lat, lon)
        logger.info(f"Registered camera {camera_id} at ({lat:.4f}, {lon:.4f})")
    
    def add_tracklet(
        self,
        tracklet_id: UUID,
        camera_id: int,
        start_time: datetime,
        embedding: np.ndarray,
        global_track_id: Optional[UUID] = None,
    ):
        """Add tracklet to cache for matching."""
        self._recent_tracklets[tracklet_id] = TrackletInfo(
            tracklet_id=tracklet_id,
            camera_id=camera_id,
            start_time=start_time,
            end_time=None,
            embedding=embedding,
            global_track_id=global_track_id,
        )
        
        # Also add to visual matcher gallery if has global ID
        if global_track_id:
            self.visual_matcher.add_to_gallery(
                str(global_track_id),
                embedding,
                camera_id,
                start_time,
            )
    
    def end_tracklet(
        self,
        tracklet_id: UUID,
        end_time: datetime,
        final_embedding: Optional[np.ndarray] = None,
    ):
        """Mark tracklet as ended."""
        if tracklet_id in self._recent_tracklets:
            info = self._recent_tracklets[tracklet_id]
            info.end_time = end_time
            if final_embedding is not None:
                info.embedding = final_embedding
    
    async def match_identity(
        self,
        camera_id: int,
        embedding: np.ndarray,
        timestamp: datetime,
        top_k: int = 5,
    ) -> MatchResult:
        """
        Match detection to existing global identity.
        
        Args:
            camera_id: Camera where detection occurred
            embedding: Feature embedding
            timestamp: Detection timestamp
            top_k: Number of candidates to consider
            
        Returns:
            MatchResult with best match or new identity
        """
        # Clean expired tracklets
        self._clean_expired(timestamp)
        
        # Get candidate matches from visual matcher
        visual_matches = self.visual_matcher.match(embedding, top_k=top_k)
        
        if not visual_matches:
            # No matches, create new identity
            return self._create_new_identity(camera_id, embedding, timestamp)
        
        # Score candidates with spatial-temporal constraints
        best_match = None
        best_score = -1
        
        for global_id, visual_sim, entry in visual_matches:
            # Calculate ST probability
            from_camera = entry.last_camera_id
            time_delta = (timestamp - entry.last_seen).total_seconds()
            
            if time_delta < 0:
                continue  # Skip if timestamp is before last seen
            
            # For same camera, ST probability should be high (no travel time needed)
            if from_camera == camera_id:
                st_prob = 1.0  # Same camera = immediate match is valid
            else:
                st_prob = self.st_scorer.calculate_score(
                    from_camera, camera_id, time_delta
                )
            
            # Joint score: weight visual more heavily since ST may not be configured
            # If cameras not registered, st_prob defaults to low - don't penalize
            joint = visual_sim * 0.8 + st_prob * 0.2
            
            logger.debug(
                f"ReID candidate: {global_id[:8]} visual={visual_sim:.3f} "
                f"st={st_prob:.3f} joint={joint:.3f}"
            )
            
            if joint > best_score:
                best_score = joint
                best_match = (global_id, visual_sim, st_prob, joint)
        
        # Use visual threshold for matching (not joint)
        if best_match and best_match[1] >= self.match_threshold:
            global_id, visual_sim, st_prob, joint = best_match
            
            # Update gallery with new observation
            self.visual_matcher.add_to_gallery(
                global_id, embedding, camera_id, timestamp
            )
            
            # Update ST scorer TTD for cross-camera transitions
            entry = self.visual_matcher.gallery[global_id]
            if entry.last_camera_id != camera_id:
                time_delta = (timestamp - entry.last_seen).total_seconds()
                self.st_scorer.update_ttd(
                    entry.last_camera_id, camera_id, time_delta
                )
                self.topology.update_transition(
                    entry.last_camera_id, camera_id, time_delta
                )
            
            logger.info(
                f"ReID match: {global_id[:8]} (visual={visual_sim:.3f}, "
                f"st={st_prob:.3f}, joint={joint:.3f})"
            )
            
            return MatchResult(
                global_track_id=UUID(global_id),
                visual_similarity=visual_sim,
                st_probability=st_prob,
                joint_score=joint,
                is_new=False,
            )
        
        # No confident match, create new identity
        return self._create_new_identity(camera_id, embedding, timestamp)
    
    def _create_new_identity(
        self,
        camera_id: int,
        embedding: np.ndarray,
        timestamp: datetime,
    ) -> MatchResult:
        """Create a new global identity."""
        from uuid import uuid4
        
        global_id = uuid4()
        
        self.visual_matcher.add_to_gallery(
            str(global_id), embedding, camera_id, timestamp
        )
        
        logger.info(f"Created new identity: {global_id}")
        
        return MatchResult(
            global_track_id=global_id,
            visual_similarity=1.0,
            st_probability=1.0,
            joint_score=1.0,
            is_new=True,
        )
    
    def _clean_expired(self, current_time: datetime):
        """Remove expired tracklets from cache."""
        expired = [
            tid for tid, info in self._recent_tracklets.items()
            if (current_time - info.start_time) > self._tracklet_expiry
        ]
        for tid in expired:
            del self._recent_tracklets[tid]
    
    def search_by_image(
        self,
        image_bytes: bytes,
        top_k: int = 5,
        threshold: float = None,  # Will use self.match_threshold if None
    ) -> List[Dict]:
        """
        Search for a person in the gallery using an uploaded image.
        
        Uses FastReID body-only embedding for matching.
        Supports:
        - Full body crops
        - Full scene images (runs person detection first)
        """
        import cv2
        import numpy as np
        
        if threshold is None:
            threshold = self.match_threshold
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Check if this is a full scene image (larger than typical crop)
        # If so, run YOLO to detect persons
        h, w = img.shape[:2]
        is_full_scene = (h > 300 and w > 300) and (h / w < 0.6 or h / w > 1.7 or min(h, w) > 400)
        
        person_crops = []
        if is_full_scene:
            try:
                from app.services.tracking_service import get_tracking_service
                tracking_service = get_tracking_service()
                detector = tracking_service._get_detector()
                detections = detector.detect(img)
                
                if len(detections) > 0:
                    logger.info(f"Detected {len(detections)} person(s) in uploaded image")
                    # Crop each detected person
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det.bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            crop = img[y1:y2, x1:x2]
                            person_crops.append(crop)
            except Exception as det_err:
                logger.warning(f"Person detection failed, treating as single crop: {det_err}")
        
        # If no detections (or single person image), use the whole image
        if not person_crops:
            person_crops = [img]
        
        # Extract body embedding using FastReID
        from app.services.tracking_service import get_tracking_service
        tracking_service = get_tracking_service()
        embedding = tracking_service.extract_from_image(person_crops[0])
        
        if embedding is None:
            raise ValueError("Failed to extract features from image")
        
        logger.info("Using FastReID body embedding for search")
        
        # Search gallery
        all_matches = self.visual_matcher.match(embedding, top_k=top_k)
        
        results = []
        for item in all_matches:
            global_id, score = item[0], item[1]
            entry = item[2] if len(item) > 2 else None
            logger.info(f"Search candidate: {global_id} score={score:.4f} threshold={threshold}")
            if score >= threshold:
                results.append({
                    "global_track_id": global_id,
                    "score": score,
                    "last_seen": entry.last_seen,
                    "camera_sequence": entry.camera_history if entry.camera_history else [entry.last_camera_id],
                })
        
        return results
    
    def get_plausible_cameras(
        self,
        camera_id: int,
        max_hops: int = 2,
    ) -> List[int]:
        """Get cameras reachable from current camera."""
        return list(self.topology.get_reachable(camera_id, max_hops))
    
    def get_gallery_size(self) -> int:
        """Get number of identities in gallery."""
        return self.visual_matcher.gallery_size
    
    def set_thumbnail(self, global_id: str, thumbnail_base64: str, quality: float = 0.0):
        """
        Store a thumbnail for a person.
        
        Only updates if the new thumbnail has better quality than the existing one.
        
        Args:
            global_id: Global track ID
            thumbnail_base64: Base64 encoded thumbnail image
            quality: Quality score (0.0 - 1.0), higher is better
        """
        current_quality = self.thumbnail_quality.get(global_id, -1.0)
        
        if quality > current_quality:
            self.person_thumbnails[global_id] = thumbnail_base64
            self.thumbnail_quality[global_id] = quality
            logger.debug(f"Updated thumbnail for {global_id[:8]} (quality: {current_quality:.2f} -> {quality:.2f})")
    
    def get_gallery(self) -> List[Dict]:
        """Get all gallery entries with thumbnails."""
        entries = []
        for global_id, entry in self.visual_matcher.gallery.items():
            entries.append({
                "global_id": global_id,
                "last_camera_id": entry.last_camera_id,
                "last_seen": entry.last_seen.isoformat(),
                "appearance_count": entry.appearance_count,
                "thumbnail": self.person_thumbnails.get(global_id),
            })
        return entries
    
    def clear_gallery(self):
        """Clear all identities."""
        self.visual_matcher.clear_gallery()
        self._recent_tracklets.clear()
        logger.info("ReID gallery cleared")


# Service singleton
_reid_service: Optional[ReIDService] = None


def get_reid_service() -> ReIDService:
    """Get or create ReID service singleton."""
    global _reid_service
    if _reid_service is None:
        _reid_service = ReIDService()
    return _reid_service
