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
from app.db.repositories.global_track_repo import GlobalTrackRepository
from app.db.session import get_db_context
from app.db.models import TrackStatus


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
        
        # Person thumbnails for gallery display (global_id -> base64 image)
        self.person_thumbnails: Dict[str, str] = {}
        
        logger.info(f"ReIDService initialized (threshold={self.match_threshold})")
    
    async def load_initial_state(self):
        """Load known identities from database."""
        logger.info("Loading ReID identities from database...")
        async with get_db_context() as session:
            repo = GlobalTrackRepository(session)
            
            # Load active tracks to populate gallery
            # We load ALL active tracks to ensure we can match returning people
            active_tracks = await repo.get_active(limit=1000)
            
            count = 0
            for track in active_tracks:
                if track.avg_embedding and track.camera_sequence:
                    self.visual_matcher.add_to_gallery(
                        str(track.id),
                        np.array(track.avg_embedding, dtype=np.float32),
                        track.camera_sequence[-1],
                        track.last_seen
                    )
                    # Restore thumbnail from database if available
                    if track.thumbnail_base64:
                        self.person_thumbnails[str(track.id)] = track.thumbnail_base64
                    count += 1
            
            logger.info(f"Loaded {count} identities into ReID gallery (with {len(self.person_thumbnails)} thumbnails)")

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
        """
        # Clean expired tracklets
        self._clean_expired(timestamp)
        
        # Get candidate matches from visual matcher
        visual_matches = self.visual_matcher.match(embedding, top_k=top_k)
        
        if not visual_matches:
            # No matches, create new identity
            return await self._create_new_identity(camera_id, embedding, timestamp)
        
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
            joint = visual_sim * 0.8 + st_prob * 0.2
            
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
            
            # Persist update to DB
            async with get_db_context() as session:
                repo = GlobalTrackRepository(session)
                # Update camera sequence
                await repo.add_camera_to_sequence(
                    UUID(global_id), camera_id, timestamp
                )
                # Update embedding (VisualMatcher keeps history, here we save AVG or latest)
                # Since we don't calculate avg here easily without history, we can save this embedding
                # OR better: let repo handle it if we passed history.
                # For now, simplistic approach: update avg with moving average or just replace?
                # GlobalTrackRepository.update_embedding expects List[float].
                # We can skip updating avg_embedding on every frame for performance,
                # or do it periodically. For now, let's NOT update avg_embedding on every match 
                # to avoid DB spam, unless it's a significant change?
                # Actually, strictly required for Search to improve over time.
                # Let's verify if VisualMatcher exposes the new average.
                # It doesn't. 
                # We'll skip updating avg_embedding in DB for every frame for now, 
                # relying on the initial visual match.
                # BUT we DO update camera_sequence and last_seen.
            
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
            
            return MatchResult(
                global_track_id=UUID(global_id),
                visual_similarity=visual_sim,
                st_probability=st_prob,
                joint_score=joint,
                is_new=False,
            )
        
        # No confident match, create new identity
        return await self._create_new_identity(camera_id, embedding, timestamp)
    
    async def _create_new_identity(
        self,
        camera_id: int,
        embedding: np.ndarray,
        timestamp: datetime,
    ) -> MatchResult:
        """Create a new global identity."""
        
        # Persist to DB first to get ID
        async with get_db_context() as session:
            repo = GlobalTrackRepository(session)
            track = await repo.create(
                first_seen=timestamp,
                camera_id=camera_id,
                avg_embedding=embedding.tolist()
            )
            global_id = track.id
        
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
        threshold: float = 0.6,
    ) -> List[Dict]:
        """
        Search for a person in the gallery using an uploaded image.
        """
        import cv2
        import numpy as np
        
        logger.info(f"search_by_image: Received {len(image_bytes)} bytes, top_k={top_k}, threshold={threshold}")
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Failed to decode image")
            raise ValueError("Invalid image data")
        
        logger.info(f"Decoded image: shape={img.shape}")
            
        # Get embedding using TrackingService
        from app.services.tracking_service import get_tracking_service
        tracking_service = get_tracking_service()
        
        embedding = tracking_service.extract_from_image(img)
        
        if embedding is None:
            logger.error("Failed to extract features from image")
            raise ValueError("Failed to extract features from image")
        
        logger.info(f"Extracted embedding: shape={embedding.shape}")
        logger.info(f"Gallery size: {self.visual_matcher.gallery_size}")
        
        # Search gallery
        matches = self.visual_matcher.match(embedding, top_k=top_k)
        
        logger.info(f"Visual matcher returned {len(matches)} matches")
        
        results = []
        for idx, (global_id, score, entry) in enumerate(matches):
            logger.info(f"Match {idx+1}: global_id={global_id}, score={score:.4f}, threshold={threshold}")
            if score >= threshold:
                results.append({
                    "global_track_id": global_id,
                    "score": score,
                    "last_seen": entry.last_seen,
                    "camera_sequence": [entry.last_camera_id],
                })
            else:
                logger.info(f"  -> FILTERED OUT (below threshold)")
        
        logger.info(f"Returning {len(results)} results after threshold filter (from {len(matches)} total matches)")
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
    
    def set_thumbnail(self, global_id: str, thumbnail_base64: str):
        """Store a thumbnail for a person in memory and database."""
        # Store in memory for immediate access
        self.person_thumbnails[global_id] = thumbnail_base64
        
        # Persist to database asynchronously
        import asyncio
        try:
            # Use background task to avoid blocking
            asyncio.create_task(self._save_thumbnail_to_db(global_id, thumbnail_base64))
        except RuntimeError:
            # If no event loop, log warning - thumbnail will still be in memory
            logger.warning(f"Could not schedule thumbnail save for {global_id} - no event loop")
    
    async def _save_thumbnail_to_db(self, global_id: str, thumbnail_base64: str):
        """Save thumbnail to database."""
        try:
            from uuid import UUID
            async with get_db_context() as session:
                repo = GlobalTrackRepository(session)
                track = await repo.get_by_id(UUID(global_id))
                if track:
                    track.thumbnail_base64 = thumbnail_base64
                    await session.commit()
                    logger.debug(f"Saved thumbnail to DB for {global_id}")
        except Exception as e:
            logger.error(f"Failed to save thumbnail to DB for {global_id}: {e}")
    
    def get_gallery(self) -> List[Dict]:
        """Get all gallery entries with thumbnails."""
        entries = []
        for global_id, entry in self.visual_matcher.gallery.items():
            entries.append({
                "global_id": global_id,
                "last_camera_id": entry.last_camera_id,
                "last_seen": entry.last_seen.isoformat(),
                "appearance_count": entry.appearance_count,
                # Fetch from in-memory cache (populated from DB on startup or newly captured)
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
