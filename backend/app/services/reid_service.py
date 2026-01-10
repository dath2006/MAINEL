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
    
    # Two-threshold system for confident identity matching
    # Values are loaded from config (settings.reid_confirm_threshold, settings.reid_new_identity_threshold)
    # - confirm_threshold: Must exceed this for confident match to existing ID
    # - new_identity_threshold: Below this, definitely create new ID
    
    @property
    def CONFIRM_THRESHOLD(self):
        """High bar for 'same person' confidence. Configurable via REID_CONFIRM_THRESHOLD."""
        return settings.reid_confirm_threshold
    
    @property
    def NEW_IDENTITY_THRESHOLD(self):
        """Below this = definitely new person. Configurable via REID_NEW_IDENTITY_THRESHOLD."""
        return settings.reid_new_identity_threshold
    
    async def match_identity(
        self,
        camera_id: int,
        embedding: np.ndarray,
        timestamp: datetime,
        top_k: int = 5,
    ) -> MatchResult:
        """
        Match detection to existing global identity.
        
        Uses two-threshold system to prevent false merges:
        - similarity >= CONFIRM_THRESHOLD: Confident match, update gallery
        - similarity < NEW_IDENTITY_THRESHOLD: Create new identity
        - Between: Create new identity (conservative approach to prevent pollution)
        
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
        # Use lower threshold for initial candidates, we'll filter ourselves
        visual_matches = self.visual_matcher.match(
            embedding, 
            top_k=top_k, 
            threshold=settings.reid_candidate_threshold  # Get more candidates, filter with CONFIRM_THRESHOLD
        )
        
        if not visual_matches:
            # No matches at all, create new identity
            logger.debug("ReID: No gallery matches found, creating new identity")
            return self._create_new_identity(camera_id, embedding, timestamp)
        
        # Score candidates with spatial-temporal constraints
        best_match = None
        best_score = -1
        best_raw_sim = -1  # Track raw cosine similarity separately
        
        for global_id, visual_sim, entry in visual_matches:
            # Calculate ST probability
            from_camera = entry.last_camera_id
            time_delta = (timestamp - entry.last_seen).total_seconds()
            
            if time_delta < 0:
                continue  # Skip if timestamp is before last seen
            
            # For same camera, ST probability should be high
            if from_camera == camera_id:
                st_prob = 1.0
            else:
                st_prob = self.st_scorer.calculate_score(
                    from_camera, camera_id, time_delta
                )
            
            # Joint score for ranking
            joint = visual_sim * 0.8 + st_prob * 0.2
            
            logger.debug(
                f"ReID candidate: {global_id[:8]} visual={visual_sim:.3f} "
                f"st={st_prob:.3f} joint={joint:.3f}"
            )
            
            if joint > best_score:
                best_score = joint
                best_raw_sim = visual_sim
                best_match = (global_id, visual_sim, st_prob, joint)
        
        # === TWO-THRESHOLD DECISION LOGIC ===
        # Use raw visual similarity for threshold decisions (not reranked joint score)
        
        if best_match is None:
            logger.debug("ReID: No valid candidates after ST filtering")
            return self._create_new_identity(camera_id, embedding, timestamp)
        
        global_id, visual_sim, st_prob, joint = best_match
        
        # Case 1: CONFIDENT MATCH (visual_sim >= CONFIRM_THRESHOLD)
        if visual_sim >= self.CONFIRM_THRESHOLD:
            # High confidence - this IS the same person
            # Update gallery with new observation
            self.visual_matcher.add_to_gallery(
                global_id, 
                embedding, 
                camera_id, 
                timestamp,
                quality_score=visual_sim
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
                f"ReID CONFIRMED match: {global_id[:8]} (visual={visual_sim:.3f}, "
                f"st={st_prob:.3f}, joint={joint:.3f})"
            )
            
            return MatchResult(
                global_track_id=UUID(global_id),
                visual_similarity=visual_sim,
                st_probability=st_prob,
                joint_score=joint,
                is_new=False,
            )
        
        # Case 2: LOW SIMILARITY (visual_sim < NEW_IDENTITY_THRESHOLD)
        # Definitely a new person
        if visual_sim < self.NEW_IDENTITY_THRESHOLD:
            logger.debug(
                f"ReID: Best match {global_id[:8]} too weak ({visual_sim:.3f} < {self.NEW_IDENTITY_THRESHOLD}), "
                f"creating new identity"
            )
            return self._create_new_identity(camera_id, embedding, timestamp)
        
        # Case 3: UNCERTAIN (between thresholds)
        # Conservative approach: Create new identity to avoid polluting gallery
        # Better to have duplicate IDs than merged incorrect IDs
        logger.info(
            f"ReID UNCERTAIN: Best match {global_id[:8]} similarity {visual_sim:.3f} "
            f"is between thresholds [{self.NEW_IDENTITY_THRESHOLD}, {self.CONFIRM_THRESHOLD}]. "
            f"Creating new identity to prevent potential gallery pollution."
        )
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
            str(global_id), embedding, camera_id, timestamp,
            quality_score=0.5  # Default quality for new identity
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
        mode: str = "auto",
    ) -> List[Dict]:
        """
        Search for a person in the gallery using an uploaded image.
        
        Supports:
        - Full body crops (uses body + face embeddings)
        - Face-only images (uses face embeddings)
        - Full scene images (runs person detection first)
        """
        import cv2
        import numpy as np
        
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
        
        # Try to extract face embedding first
        face_embedding = None
        try:
            from app.core.features.face_extractor import get_face_extractor
            face_extractor = get_face_extractor()
            face_embedding, _, face_conf = face_extractor.extract_from_person_crop(person_crops[0])
            if face_embedding is not None:
                logger.info(f"Face detected in search image (confidence={face_conf:.2f})")
        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
        
        # Extract body embedding
        from app.services.tracking_service import get_tracking_service
        tracking_service = get_tracking_service()
        body_embedding = tracking_service.extract_from_image(person_crops[0])
        
        # Determine which embedding(s) to use and search method
        all_matches = []
        
        # Log gallery state before search
        gallery_size = self.visual_matcher.gallery_size
        face_gallery_size = len(self.visual_matcher.face_gallery)
        logger.info(f"=== IMAGE SEARCH: Gallery has {gallery_size} identities, {face_gallery_size} face embeddings ===")
        
        if gallery_size == 0:
            logger.warning("Gallery is EMPTY - no identities to search against")
            return []
        
        # Mode-based Search Logic
        if mode == "face":
            if face_embedding is None:
                logger.warning("Search Mode=FACE but no face detected in query image.")
                return []
            
            logger.info("Search Mode: FACE ONLY")
            # Search face gallery directly
            all_matches = self.visual_matcher.match_face(face_embedding, top_k=top_k, threshold=0.0)
            
        elif mode == "body":
            if body_embedding is None:
                 logger.warning("Search Mode=BODY but no person body detected.")
                 return []
            
            logger.info("Search Mode: BODY ONLY")
            # Search main gallery using body embedding (strict body-to-body/fused match)
            all_matches = self.visual_matcher.match(body_embedding, top_k=top_k, threshold=0.0)
            
        else:
            # Mode = "auto" (Default Legacy Logic)
            if face_embedding is not None and body_embedding is not None:
                # Gallery is now body-only, so use body embedding for main search
                # Face is used as a bonus boost via face_gallery
                logger.info("Search: AUTO (Body Search + Face Boost)")
                
                # Primary search: body-only against main gallery
                all_matches = self.visual_matcher.match(body_embedding, top_k=top_k * 2, threshold=0.0)
                
                # Bonus: Check face gallery for face-to-face matches
                if len(self.visual_matcher.face_gallery) > 0:
                    logger.info(f"Checking face gallery ({len(self.visual_matcher.face_gallery)} faces)...")
                    face_matches = self.visual_matcher.match_face(face_embedding, top_k=top_k, threshold=0.3)
                    
                    # Boost scores for identities that also match by face
                    face_match_ids = {gid: score for gid, score, _ in face_matches}
                    boosted_matches = []
                    for gid, score, entry in all_matches:
                        if gid in face_match_ids:
                            # Boost by combining body + face scores
                            face_score = face_match_ids[gid]
                            boosted = score * 0.7 + face_score * 0.3  # Weight body more
                            logger.info(f"  Face boost: {gid[:8]}... body={score:.3f} + face={face_score:.3f} = {boosted:.3f}")
                            boosted_matches.append((gid, boosted, entry))
                        else:
                            boosted_matches.append((gid, score, entry))
                    
                    # Re-sort by boosted score
                    boosted_matches.sort(key=lambda x: x[1], reverse=True)
                    all_matches = boosted_matches[:top_k]
                else:
                    all_matches = all_matches[:top_k]
                
            elif face_embedding is not None:
                # Face-only image - search face gallery primarily
                logger.info("Search: AUTO (Face Only)")
                
                # Search face gallery
                face_matches = self.visual_matcher.match_face(face_embedding, top_k=top_k, threshold=0.0)
                for gid, score, entry in face_matches:
                    all_matches.append((gid, score, entry))
                
                # If not enough matches, also try main gallery (body)
                if len(all_matches) < top_k:
                    main_matches = self.visual_matcher.match(face_embedding, top_k=top_k, threshold=0.0)
                    for gid, score, entry in main_matches:
                        if not any(m[0] == gid for m in all_matches):
                            all_matches.append((gid, score * 0.5, entry))  # Heavy discount
                    
                    all_matches.sort(key=lambda x: x[1], reverse=True)
                    all_matches = all_matches[:top_k]
                
            elif body_embedding is not None:
                # Body-only - use body embedding
                embedding = body_embedding
                logger.info("Search: AUTO (Body Only)")
                # Search with threshold=0.0 to get ALL candidates for logging
                all_matches = self.visual_matcher.match(embedding, top_k=top_k, threshold=0.0)
            else:
                raise ValueError("Failed to extract any features from image")
        
        # Apply threshold and build results
        search_threshold = threshold if threshold is not None else self.match_threshold
        results = []
        
        logger.info(f"=== FINAL RESULTS (search_threshold={search_threshold}) ===")
        for i, item in enumerate(all_matches):
            global_id, score = item[0], item[1]
            entry = item[2] if len(item) > 2 else None
            
            if score >= search_threshold:
                logger.info(f"  ✓ MATCH #{len(results)+1}: {global_id[:8]}... score={score:.4f} >= {search_threshold}")
                results.append({
                    "global_track_id": global_id,
                    "score": score,
                    "last_seen": entry.last_seen,
                    "camera_sequence": entry.camera_history if entry.camera_history else [entry.last_camera_id],
                })
            else:
                logger.info(f"  ✗ REJECTED: {global_id[:8]}... score={score:.4f} < {search_threshold}")
        
        logger.info(f"=== Search complete: {len(results)} matches returned ===")
        
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
