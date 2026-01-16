"""
Identity Merger - Merges fragmented identities across cameras.

Runs periodically to detect and merge identities that are likely the same
person but were assigned different global IDs due to:
- Cross-camera appearance with embed variance
- Time gaps between observations
- Threshold edge cases

Based on best practices:
"Maintain per-camera identity entries in the gallery to account for view,
lighting, and occlusion variations. Merge into a single global identity 
only after robust matching."
"""

from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime
import numpy as np
from loguru import logger


class IdentityMerger:
    """
    Merges fragmented identities in the gallery.
    
    Uses multi-embedding comparison to detect same-person cases and
    merges them into a single unified identity.
    
    IMPORTANT: Uses AVERAGE similarity (not MAX) to avoid false merges
    caused by spurious high-similarity pairs from low-quality captures.
    """
    
    def __init__(
        self,
        merge_threshold: float = 0.70,  # RAISED from 0.55 - be conservative
        min_captures_for_merge: int = 2,  # RAISED from 1 - require more evidence
    ):
        """
        Initialize identity merger.
        
        Args:
            merge_threshold: Minimum AVERAGE similarity to consider merging (0.0-1.0)
            min_captures_for_merge: Minimum captures needed per identity to consider
        """
        self.merge_threshold = merge_threshold
        self.min_captures = min_captures_for_merge
        
        # Stats
        self.stats = {
            'checks_performed': 0,
            'merges_performed': 0,
            'last_merge_time': None,
        }
        
        logger.info(f"IdentityMerger initialized (threshold={merge_threshold}, min_captures={min_captures_for_merge})")
    
    def compute_cross_similarity(
        self,
        embeddings1: List[np.ndarray],
        embeddings2: List[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Compute similarity metrics between two sets of embeddings.
        
        Returns:
            Tuple of (max_similarity, avg_similarity, min_similarity)
        """
        if not embeddings1 or not embeddings2:
            return 0.0, 0.0, 0.0
        
        similarities = []
        for e1 in embeddings1:
            e1_norm = e1 / (np.linalg.norm(e1) + 1e-8)
            for e2 in embeddings2:
                e2_norm = e2 / (np.linalg.norm(e2) + 1e-8)
                sim = float(np.dot(e1_norm, e2_norm))
                similarities.append(sim)
        
        if not similarities:
            return 0.0, 0.0, 0.0
        
        return max(similarities), sum(similarities) / len(similarities), min(similarities)
    
    def find_merge_candidates(self) -> List[Tuple[str, str, float]]:
        """
        Find pairs of identities that should be merged.
        
        Uses AVERAGE similarity (not MAX) to be more conservative and
        avoid false merges from spurious high-similarity pairs.
        
        Returns:
            List of (id1, id2, avg_similarity) tuples, sorted by similarity descending
        """
        from app.services.gallery_store import get_gallery_store
        
        gallery_store = get_gallery_store()
        galleries = gallery_store._galleries
        
        if len(galleries) < 2:
            return []
        
        candidates = []
        global_ids = list(galleries.keys())
        
        for i, id1 in enumerate(global_ids):
            gallery1 = galleries[id1]
            if len(gallery1.captures) < self.min_captures:
                continue
            
            embs1 = [c.embedding for c in gallery1.captures if c.embedding is not None]
            if not embs1:
                continue
            
            for id2 in global_ids[i+1:]:
                gallery2 = galleries[id2]
                if len(gallery2.captures) < self.min_captures:
                    continue
                
                embs2 = [c.embedding for c in gallery2.captures if c.embedding is not None]
                if not embs2:
                    continue
                
                max_sim, avg_sim, min_sim = self.compute_cross_similarity(embs1, embs2)
                
                # USE AVERAGE SIMILARITY for merge decision (more conservative)
                # Only consider if BOTH max and avg are high enough
                if avg_sim >= self.merge_threshold and max_sim >= self.merge_threshold:
                    candidates.append((id1, id2, avg_sim))
                    logger.info(
                        f"Merge candidate: {id1[:8]} + {id2[:8]} "
                        f"(max={max_sim:.3f}, avg={avg_sim:.3f}, min={min_sim:.3f})"
                    )
        
        # Sort by average similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        self.stats['checks_performed'] += 1
        
        return candidates
    
    def merge_identities(self, keep_id: str, remove_id: str) -> bool:
        """
        Merge remove_id into keep_id.
        
        Args:
            keep_id: Identity to keep (absorbs the other)
            remove_id: Identity to remove (gets absorbed)
            
        Returns:
            True if merge was successful
        """
        from app.services.gallery_store import get_gallery_store
        from app.services.reid_service import get_reid_service
        
        gallery_store = get_gallery_store()
        reid_service = get_reid_service()
        
        if keep_id not in gallery_store._galleries:
            logger.warning(f"Cannot merge: {keep_id} not in gallery")
            return False
        
        if remove_id not in gallery_store._galleries:
            logger.warning(f"Cannot merge: {remove_id} not in gallery")
            return False
        
        keep_gallery = gallery_store._galleries[keep_id]
        remove_gallery = gallery_store._galleries[remove_id]
        
        logger.info(
            f"MERGING: {remove_id[:8]} -> {keep_id[:8]} "
            f"(captures: {len(remove_gallery.captures)} -> {len(keep_gallery.captures)})"
        )
        
        # 1. Merge captures from remove_id into keep_id
        keep_gallery.captures.extend(remove_gallery.captures)
        
        # 2. Re-select top captures by quality
        keep_gallery.captures = sorted(
            keep_gallery.captures,
            key=lambda c: c.quality_score,
            reverse=True
        )[:gallery_store.max_captures]
        
        # 3. Update best score
        if keep_gallery.captures:
            keep_gallery.best_score = max(c.quality_score for c in keep_gallery.captures)
        
        # 4. Merge pose counts
        for pose, count in remove_gallery.pose_counts.items():
            keep_gallery.pose_counts[pose] = keep_gallery.pose_counts.get(pose, 0) + count
        
        # 5. Remove the absorbed identity from GalleryStore
        del gallery_store._galleries[remove_id]
        
        # 6. Update VisualMatcher gallery
        # Merge embeddings and camera history
        visual_matcher = reid_service.visual_matcher
        
        if remove_id in visual_matcher.gallery:
            remove_entry = visual_matcher.gallery[remove_id]
            
            if keep_id in visual_matcher.gallery:
                keep_entry = visual_matcher.gallery[keep_id]
                
                # Merge embedding history
                keep_entry.embeddings_history.extend(remove_entry.embeddings_history)
                keep_entry.embeddings_history = keep_entry.embeddings_history[-visual_matcher.embedding_history_size:]
                
                # Recompute average embedding
                if keep_entry.embeddings_history:
                    keep_entry.embedding = np.mean(keep_entry.embeddings_history, axis=0)
                    norm = np.linalg.norm(keep_entry.embedding)
                    if norm > 0:
                        keep_entry.embedding = keep_entry.embedding / norm
                
                # Merge camera history (preserve order, avoid duplicates)
                for cam in remove_entry.camera_history:
                    if cam not in keep_entry.camera_history:
                        keep_entry.camera_history.append(cam)
                
                # Update appearance count
                keep_entry.appearance_count += remove_entry.appearance_count
                
                # Update last seen if remove_id was more recent
                if remove_entry.last_seen > keep_entry.last_seen:
                    keep_entry.last_seen = remove_entry.last_seen
                    keep_entry.last_camera_id = remove_entry.last_camera_id
            
            # Remove from VisualMatcher
            del visual_matcher.gallery[remove_id]
            
            # Also remove from face gallery if present
            if remove_id in visual_matcher.face_gallery:
                del visual_matcher.face_gallery[remove_id]
        
        self.stats['merges_performed'] += 1
        self.stats['last_merge_time'] = datetime.now()
        
        logger.info(f"MERGE COMPLETE: {keep_id[:8]} now has {len(keep_gallery.captures)} captures")
        
        return True
    
    def run_merge_pass(self) -> int:
        """
        Run a single merge pass - find and merge all candidates.
        
        Returns:
            Number of merges performed
        """
        candidates = self.find_merge_candidates()
        
        if not candidates:
            return 0
        
        merged_count = 0
        merged_ids: Set[str] = set()
        
        for id1, id2, similarity in candidates:
            # Skip if either ID was already merged in this pass
            if id1 in merged_ids or id2 in merged_ids:
                continue
            
            # Keep the one with more captures or higher quality
            from app.services.gallery_store import get_gallery_store
            gallery_store = get_gallery_store()
            
            if id1 not in gallery_store._galleries or id2 not in gallery_store._galleries:
                continue
            
            g1 = gallery_store._galleries[id1]
            g2 = gallery_store._galleries[id2]
            
            # Decision: keep the one with more captures, tie-break by quality
            keep_id, remove_id = (id1, id2) if (
                len(g1.captures) > len(g2.captures) or
                (len(g1.captures) == len(g2.captures) and g1.best_score >= g2.best_score)
            ) else (id2, id1)
            
            if self.merge_identities(keep_id, remove_id):
                merged_ids.add(remove_id)
                merged_count += 1
        
        if merged_count > 0:
            logger.info(f"Merge pass complete: {merged_count} identities merged")
        
        return merged_count
    
    def get_stats(self) -> Dict:
        """Get merger statistics."""
        return self.stats.copy()


# Singleton instance
_identity_merger: Optional[IdentityMerger] = None


def get_identity_merger() -> IdentityMerger:
    """Get or create singleton IdentityMerger."""
    global _identity_merger
    if _identity_merger is None:
        from app.config import settings
        # Use a higher threshold for safety - avoid false merges
        # Default 0.70 requires BOTH max and avg similarity to be >= 0.70
        threshold = getattr(settings, 'reid_merge_threshold', 0.70)
        _identity_merger = IdentityMerger(
            merge_threshold=threshold,
            min_captures_for_merge=2,  # Require at least 2 captures per identity
        )
    return _identity_merger


def reset_identity_merger():
    """Reset the singleton (useful for testing or reconfiguration)."""
    global _identity_merger
    _identity_merger = None
