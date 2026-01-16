"""
Gallery Store - In-memory multi-capture gallery for ReID.

Mirrors the logic from preprocessor/gallery_manager.py:
- Top-K images per global_id (default 5)
- Diversity bonus for different poses
- Temporal spacing to avoid redundant captures
- Quality threshold filtering

This is used for:
1. Storing high-quality captures for each identity
2. Powering the gallery popup in frontend
3. Multi-image matching for search queries
"""

import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from numpy import ndarray

# Import settings for configuration
from app.config import settings


@dataclass
class CaptureEntry:
    """Single capture in a person's gallery."""
    image_b64: str
    quality_score: float
    pose: str
    timestamp: datetime
    sharpness: float = 0.0
    frame_number: int = 0
    embedding: Optional[np.ndarray] = None  # Cached feature embedding


@dataclass
class PersonGallery:
    """Gallery of captures for one identity."""
    global_id: str
    captures: List[CaptureEntry] = field(default_factory=list)
    best_score: float = 0.0
    pose_counts: Dict[str, int] = field(default_factory=lambda: {'front': 0, 'side': 0, 'back': 0, 'unknown': 0})
    last_capture_frame: int = -999
    
    def get_best(self) -> Optional[CaptureEntry]:
        """Get highest quality capture."""
        if not self.captures:
            return None
        return max(self.captures, key=lambda c: c.quality_score)
    
    def get_thumbnail_b64(self) -> Optional[str]:
        """Get base64 image of best capture."""
        best = self.get_best()
        return best.image_b64 if best else None


class GalleryStore:
    """
    In-memory gallery store for ReID captures.
    
    Implements top-K selection with diversity bonus.
    """
    
    def __init__(
        self,
        max_captures_per_person: int = 5,
        min_frame_gap: int = 10,
        min_quality_for_save: float = None,  # Uses settings.gallery_quality_threshold if None
        diversity_bonus: float = 10.0
    ):
        """
        Initialize gallery store.
        
        Args:
            max_captures_per_person: Max images to keep per identity
            min_frame_gap: Minimum frames between captures
            min_quality_for_save: Minimum quality score to consider (default: from config)
            diversity_bonus: Score bonus for underrepresented poses
        """
        self.max_captures = max_captures_per_person
        self.min_frame_gap = min_frame_gap
        # Use config threshold if not explicitly provided
        self.min_quality = min_quality_for_save if min_quality_for_save is not None else settings.gallery_quality_threshold
        self.diversity_bonus = diversity_bonus
        
        # Galleries by global_id
        self._galleries: Dict[str, PersonGallery] = {}
        
        # Stats
        self.stats = {
            'total_attempts': 0,
            'saved': 0,
            'rejected_quality': 0,
            'rejected_timing': 0,
            'evicted': 0
        }
    
    def should_capture(
        self,
        global_id: str,
        frame_number: int,
        quality_score: float,
        pose: str = 'unknown'
    ) -> Tuple[bool, str]:
        """
        Check if this frame should be captured.
        
        Returns:
            Tuple of (should_capture, reason)
        """
        self.stats['total_attempts'] += 1
        
        # Initialize gallery if new
        if global_id not in self._galleries:
            self._galleries[global_id] = PersonGallery(global_id=global_id)
        
        gallery = self._galleries[global_id]
        
        # Check timing
        if frame_number - gallery.last_capture_frame < self.min_frame_gap:
            self.stats['rejected_timing'] += 1
            return False, 'timing'
        
        # Calculate effective score with diversity bonus
        effective_score = quality_score
        if len(gallery.captures) > 0:
            total_poses = sum(gallery.pose_counts.values())
            if total_poses > 0:
                pose_ratio = gallery.pose_counts.get(pose, 0) / total_poses
                if pose_ratio < 0.3:  # Underrepresented
                    effective_score += self.diversity_bonus
        
        # Check if would make the cut
        if len(gallery.captures) < self.max_captures:
            if effective_score >= self.min_quality:
                return True, 'gallery_not_full'
            self.stats['rejected_quality'] += 1
            return False, 'low_quality'
        else:
            worst_score = min(c.quality_score for c in gallery.captures)
            if effective_score > worst_score:
                return True, 'better_than_worst'
            return False, 'not_better'
    
    def add_capture(
        self,
        global_id: str,
        image_b64: str,
        quality_score: float,
        pose: str = 'unknown',
        sharpness: float = 0.0,
        frame_number: int = 0,
        timestamp: Optional[datetime] = None,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Add a capture to the gallery.
        
        Args:
            global_id: Global track ID
            image_b64: Base64 encoded image
            quality_score: Quality score of the capture
            pose: Detected pose ('front', 'side', 'back', 'unknown')
            sharpness: Sharpness score
            frame_number: Frame number in video
            timestamp: Capture timestamp
            embedding: Pre-computed feature embedding (cached for fast search)
        
        Returns:
            True if capture was added/replaced, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check eligibility
        should, reason = self.should_capture(global_id, frame_number, quality_score, pose)
        if not should:
            return False
        
        gallery = self._galleries[global_id]
        
        # Create entry with cached embedding
        entry = CaptureEntry(
            image_b64=image_b64,
            quality_score=quality_score,
            pose=pose,
            sharpness=sharpness,
            frame_number=frame_number,
            timestamp=timestamp,
            embedding=embedding
        )
        
        # Evict worst if full
        if len(gallery.captures) >= self.max_captures:
            worst_idx = min(range(len(gallery.captures)),
                           key=lambda i: gallery.captures[i].quality_score)
            removed = gallery.captures.pop(worst_idx)
            gallery.pose_counts[removed.pose] = max(0, gallery.pose_counts.get(removed.pose, 1) - 1)
            self.stats['evicted'] += 1
        
        # Add new capture
        gallery.captures.append(entry)
        gallery.pose_counts[pose] = gallery.pose_counts.get(pose, 0) + 1
        gallery.last_capture_frame = frame_number
        
        if quality_score > gallery.best_score:
            gallery.best_score = quality_score
        
        self.stats['saved'] += 1
        logger.debug(f"GalleryStore: Added capture for {global_id[:8]} Q={quality_score:.1f} pose={pose}")
        
        return True
    
    def get_gallery(self, global_id: str) -> Optional[PersonGallery]:
        """Get gallery for a specific identity."""
        return self._galleries.get(global_id)
    
    def get_captures(self, global_id: str) -> List[dict]:
        """Get all captures as serializable dicts."""
        gallery = self._galleries.get(global_id)
        if not gallery:
            return []
        
        return [
            {
                'image_b64': c.image_b64,
                'quality_score': c.quality_score,
                'pose': c.pose,
                'sharpness': c.sharpness,
                'timestamp': c.timestamp.isoformat() if c.timestamp else None
            }
            for c in sorted(gallery.captures, key=lambda x: -x.quality_score)
        ]
    
    def get_thumbnail(self, global_id: str) -> Optional[str]:
        """Get best thumbnail for identity."""
        gallery = self._galleries.get(global_id)
        if gallery:
            return gallery.get_thumbnail_b64()
        return None
    
    def get_all_galleries(self) -> List[dict]:
        """Get summary of all galleries."""
        return [
            {
                'global_id': gid,
                'capture_count': len(g.captures),
                'best_score': g.best_score,
                'thumbnail': g.get_thumbnail_b64()
            }
            for gid, g in self._galleries.items()
        ]
    
    def get_all_embeddings_for_matching(self) -> Dict[str, List[np.ndarray]]:
        """
        Get all embeddings from all galleries for multi-embedding matching.
        
        Returns:
            Dict of global_id -> list of embeddings
        """
        result = {}
        for gid, gallery in self._galleries.items():
            embeddings = [c.embedding for c in gallery.captures if c.embedding is not None]
            if embeddings:
                result[gid] = embeddings
        return result
    
    def compute_max_similarity(
        self,
        query_embedding: np.ndarray,
        global_id: str
    ) -> float:
        """
        Compute MAX similarity between query and all embeddings for an identity.
        
        This is more robust than comparing against a single averaged embedding,
        as it accounts for pose/lighting variations in the gallery.
        
        Args:
            query_embedding: Query embedding (normalized)
            global_id: Identity to compare against
            
        Returns:
            Maximum similarity across all gallery embeddings (0.0 if no embeddings)
        """
        gallery = self._galleries.get(global_id)
        if not gallery:
            return 0.0
        
        embeddings = [c.embedding for c in gallery.captures if c.embedding is not None]
        if not embeddings:
            return 0.0
        
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        max_sim = 0.0
        for emb in embeddings:
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            sim = float(np.dot(query_embedding, emb))
            if sim > max_sim:
                max_sim = sim
        
        return max_sim
    
    def clear(self):
        """Clear all galleries."""
        self._galleries.clear()
        logger.info("GalleryStore cleared")


# Singleton
_gallery_store: Optional[GalleryStore] = None


def get_gallery_store() -> GalleryStore:
    """Get or create singleton GalleryStore."""
    global _gallery_store
    if _gallery_store is None:
        _gallery_store = GalleryStore()
        logger.info(f"GalleryStore initialized (min_quality={_gallery_store.min_quality})")
    return _gallery_store
