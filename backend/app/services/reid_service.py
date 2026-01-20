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
        new_threshold: float = None,
        st_weight: float = None,
        max_transition_time: float = None,
    ):
        self.match_threshold = match_threshold or settings.reid_match_threshold
        self.new_threshold = new_threshold or getattr(settings, 'reid_new_threshold', 0.50)
        self.st_weight = st_weight or settings.st_weight
        self.max_transition_time = max_transition_time or settings.max_transition_time
        
        # Core components - use new_threshold as candidate filtering threshold
        # so match() returns candidates between new_threshold and match_threshold
        self.visual_matcher = VisualMatcher(
            match_threshold=self.match_threshold,
            candidate_threshold=self.new_threshold * 0.8,  # Slightly lower for safety margin
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
        
        logger.info(f"ReIDService initialized (match_thresh={self.match_threshold}, new_thresh={self.new_threshold})")
    
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
        
        # Two-threshold matching decision:
        # 1. If visual_sim >= match_threshold: Confident match - accept
        # 2. If visual_sim >= new_threshold but < match_threshold: Tentative match
        #    (likely same person with domain shift, accept to avoid duplicate IDs)
        # 3. If visual_sim < new_threshold: Low confidence - create new identity
        
        if best_match:
            global_id, visual_sim, st_prob, joint = best_match
            
            # Accept match if above EITHER threshold (lenient for cross-camera)
            # The key insight: it's better to merge potentially-same people than
            # to create duplicate IDs that can never be merged later.
            
            # STRATEGY CHANGE: Always prefer existing ID if there is ANY plausible candidate.
            # Only create new ID if similarity is very low.
            
            # Primary check: High confidence visual match
            should_match = visual_sim >= self.match_threshold
            
            # Secondary check: Tentative match (Moderate visual similarity)
            # Accept if > new_threshold/2 (very lenient)
            if not should_match and visual_sim >= (self.new_threshold * 0.8):
                 should_match = True
                 logger.debug(f"Tentative match accepted (lenient): {global_id[:8]} visual={visual_sim:.3f}")
            
            # Tertiary check: Spatio-temporal support
            if not should_match and st_prob > 0.3 and visual_sim > 0.3:
                 should_match = True
                 logger.debug(f"ST-supported match: {global_id[:8]} st={st_prob:.3f} visual={visual_sim:.3f}")

            
            if should_match:
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
        threshold: float = None,
    ) -> List[Dict]:
        """
        Search for a person in the gallery using an uploaded image.
        
        Uses GalleryStore with cached embeddings for fast, accurate matching.
        For each identity, computes MAX similarity across all their high-quality
        captures (up to 5 per person with diverse poses).
        
        Supports:
        - Full body crops
        - Full scene images (runs person detection first)
        """
        import cv2
        import numpy as np
        
        if threshold is None:
            threshold = settings.search_threshold
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Check if this is a full scene image (larger than typical crop)
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
                    for det in detections:
                        # Handle both PeopleNet (dict) and YOLO (Detection object) formats
                        if isinstance(det, dict):
                            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                        else:
                            x1, y1, x2, y2 = map(int, det.bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            person_crops.append(img[y1:y2, x1:x2])
            except Exception as det_err:
                logger.warning(f"Person detection failed, treating as single crop: {det_err}")
        
        if not person_crops:
            person_crops = [img]
        
        # Extract query embedding
        from app.services.tracking_service import get_tracking_service
        tracking_service = get_tracking_service()
        query_embedding = tracking_service.extract_from_image(person_crops[0])
        
        if query_embedding is None:
            raise ValueError("Failed to extract features from image")
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        logger.info("Searching GalleryStore with cached embeddings...")
        
        # Search GalleryStore - compare against all cached embeddings
        from app.services.gallery_store import get_gallery_store
        gallery_store = get_gallery_store()
        
        match_results = []  # List of (global_id, max_score, best_capture)
        
        for gallery in gallery_store._galleries.values():
            global_id = gallery.global_id
            max_score = 0.0
            best_capture = None
            
            for capture in gallery.captures:
                if capture.embedding is None:
                    continue  # Skip captures without cached embeddings
                
                # Normalize stored embedding
                cap_emb = capture.embedding
                cap_norm = np.linalg.norm(cap_emb)
                if cap_norm > 0:
                    cap_emb = cap_emb / cap_norm
                
                # Cosine similarity
                similarity = float(np.dot(query_embedding, cap_emb))
                
                if similarity > max_score:
                    max_score = similarity
                    best_capture = capture
            
            if max_score >= threshold:
                match_results.append((global_id, max_score, best_capture))
        
        # Sort by score descending and take more than needed for potential merging
        match_results.sort(key=lambda x: -x[1])
        match_results = match_results[:top_k * 2]  # Get extra candidates for merging
        
        # =========================================
        # Retrospective Identity Merging
        # =========================================
        # Detect if multiple matches are likely the same person (fragmented IDs)
        # and merge them into a single unified result.
        
        if len(match_results) > 1:
            # Get embeddings for each matched identity
            identity_embeddings = {}
            for global_id, score, capture in match_results:
                entry = self.visual_matcher.gallery.get(global_id)
                if entry and entry.embedding is not None:
                    identity_embeddings[global_id] = entry.embedding
            
            # Pairwise similarity check to detect fragments
            merge_threshold = settings.reid_merge_threshold  # Configurable threshold
            merge_groups = {}  # Maps each ID to its canonical (highest scoring) ID
            used = set()
            
            for i, (gid1, score1, cap1) in enumerate(match_results):
                if gid1 in used:
                    continue
                    
                merge_groups[gid1] = {
                    'ids': [gid1],
                    'primary_score': score1,
                    'primary_capture': cap1,
                    'cameras': set()
                }
                
                # Get cameras for this identity
                entry1 = self.visual_matcher.gallery.get(gid1)
                if entry1 and entry1.camera_history:
                    merge_groups[gid1]['cameras'].update(entry1.camera_history)
                
                emb1 = identity_embeddings.get(gid1)
                if emb1 is None:
                    continue
                
                # Check similarity with remaining candidates
                for j, (gid2, score2, cap2) in enumerate(match_results[i+1:], start=i+1):
                    if gid2 in used:
                        continue
                    
                    emb2 = identity_embeddings.get(gid2)
                    if emb2 is None:
                        continue
                    
                    # Normalize and compute similarity
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
                    mutual_sim = float(np.dot(emb1_norm, emb2_norm))
                    
                    if mutual_sim >= merge_threshold:
                        logger.info(f"MERGE: {gid1[:8]} + {gid2[:8]} (mutual_sim={mutual_sim:.3f})")
                        merge_groups[gid1]['ids'].append(gid2)
                        used.add(gid2)
                        
                        # Merge camera sequences
                        entry2 = self.visual_matcher.gallery.get(gid2)
                        if entry2 and entry2.camera_history:
                            merge_groups[gid1]['cameras'].update(entry2.camera_history)
            
            # Rebuild match_results with merged identities
            merged_results = []
            for gid, group in merge_groups.items():
                merged_results.append((
                    gid,
                    group['primary_score'],
                    group['primary_capture'],
                    list(group['cameras']),  # Merged camera sequence
                    group['ids']  # All merged IDs for reference
                ))
            
            # Sort by score and take top_k
            merged_results.sort(key=lambda x: -x[1])
            merged_results = merged_results[:top_k]
        else:
            merged_results = [
                (gid, score, cap, 
                 self.visual_matcher.gallery.get(gid).camera_history if self.visual_matcher.gallery.get(gid) else [],
                 [gid])
                for gid, score, cap in match_results[:top_k]
            ]
        
        # Build response with merged metadata
        results = []
        for global_id, score, capture, merged_cameras, merged_ids in merged_results:
            entry = self.visual_matcher.gallery.get(global_id)
            
            if len(merged_ids) > 1:
                logger.info(f"Search match (MERGED): {global_id[:8]} score={score:.4f} "
                           f"from {len(merged_ids)} IDs, cameras={merged_cameras}")
            else:
                logger.info(f"Search match: {global_id[:8]} score={score:.4f}")
            
            results.append({
                "global_track_id": global_id,
                "score": score,
                "last_seen": entry.last_seen if entry else (capture.timestamp if capture else None),
                "camera_sequence": merged_cameras if merged_cameras else (entry.camera_history if entry else []),
                "best_capture_pose": capture.pose if capture else "unknown",
                "best_capture_quality": capture.quality_score if capture else 0.0,
                "merged_ids": merged_ids if len(merged_ids) > 1 else None,  # Include for debugging
            })
        
        logger.info(f"Search complete: {len(results)} matches found above threshold {threshold}")
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
    
    # NOTE: set_thumbnail() removed - thumbnails now managed by GalleryStore only
    
    def get_gallery(self) -> List[Dict]:
        """Get all gallery entries with thumbnails from GalleryStore."""
        from app.services.gallery_store import get_gallery_store
        gallery_store = get_gallery_store()
        
        entries = []
        for global_id, entry in self.visual_matcher.gallery.items():
            # Get thumbnail from GalleryStore (single source of truth)
            thumbnail = gallery_store.get_thumbnail(global_id)
            
            # Build camera sequence with timestamps
            camera_sequence = []
            for cam_id in entry.camera_history:
                seq_entry = {"camera_id": cam_id}
                if cam_id in entry.camera_timestamps:
                    seq_entry["first_seen"] = entry.camera_timestamps[cam_id].isoformat()
                camera_sequence.append(seq_entry)
            
            # Build transitions list
            transitions = []
            for t in entry.transitions:
                transitions.append({
                    "from_camera": t.from_camera,
                    "to_camera": t.to_camera,
                    "transition_time": t.transition_time.isoformat(),
                    "time_at_from": t.time_at_from_camera,
                })
            
            entries.append({
                "global_id": global_id,
                "last_camera_id": entry.last_camera_id,
                "first_seen": entry.first_seen.isoformat() if entry.first_seen else None,
                "last_seen": entry.last_seen.isoformat(),
                "appearance_count": entry.appearance_count,
                "thumbnail": thumbnail,
                "camera_sequence": camera_sequence,
                "transitions": transitions,
            })
        return entries
    
    def clear_gallery(self):
        """Clear all identities from both VisualMatcher and GalleryStore with complete memory cleanup."""
        # Explicitly delete embeddings from visual matcher gallery before clearing
        for entry in self.visual_matcher.gallery.values():
            if hasattr(entry, 'embedding') and entry.embedding is not None:
                del entry.embedding
            # Also clear multi-embedding arrays if they exist
            if hasattr(entry, 'all_embeddings') and entry.all_embeddings:
                for emb in entry.all_embeddings:
                    del emb
                entry.all_embeddings.clear()
        
        # Clear visual matcher gallery
        self.visual_matcher.clear_gallery()
        self._recent_tracklets.clear()
        
        # Clear GalleryStore (single source of truth for captures) with proper memory cleanup
        from app.services.gallery_store import get_gallery_store
        gallery_store = get_gallery_store()
        gallery_store.clear()
        
        # Additional garbage collection pass to ensure memory is freed
        import gc
        gc.collect()
        
        logger.info("ReID gallery cleared with complete memory cleanup (VisualMatcher + GalleryStore)")


# Service singleton
_reid_service: Optional[ReIDService] = None


def get_reid_service() -> ReIDService:
    """Get or create ReID service singleton."""
    global _reid_service
    if _reid_service is None:
        _reid_service = ReIDService()
    return _reid_service
