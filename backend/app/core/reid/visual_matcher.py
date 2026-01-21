"""
Visual ReID Matcher

Handles visual similarity matching with gallery management
for cross-camera person re-identification.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from loguru import logger


@dataclass
class CameraTransition:
    """Records a camera-to-camera transition with timing."""
    from_camera: int
    to_camera: int
    transition_time: datetime
    time_at_from_camera: float = 0.0  # seconds spent at from_camera before leaving


@dataclass
class GalleryEntry:
    """Entry in the identity gallery."""
    global_id: str
    embedding: np.ndarray  # Average feature embedding
    last_camera_id: int
    last_seen: datetime
    first_seen: datetime = None  # When this identity was first observed
    appearance_count: int = 1
    embeddings_history: List[np.ndarray] = field(default_factory=list)
    camera_history: List[int] = field(default_factory=list)  # All cameras where seen
    camera_timestamps: Dict[int, datetime] = field(default_factory=dict)  # camera_id -> first seen at that camera
    transitions: List[CameraTransition] = field(default_factory=list)  # Ordered list of camera transitions


class VisualMatcher:
    """
    Visual similarity matcher for cross-camera ReID.
    
    Maintains a gallery of known identities and matches new
    detections against them using cosine similarity.
    """
    
    def __init__(
        self,
        match_threshold: float = 0.3,
        candidate_threshold: float = 0.25,  # Lower threshold for candidate pre-filtering
        max_gallery_size: int = 1000,
        embedding_history_size: int = 10,
    ):
        """
        Initialize visual matcher.
        
        Args:
            match_threshold: Minimum similarity for valid match
            candidate_threshold: Lower threshold for candidate pre-filtering (for two-threshold systems)
            max_gallery_size: Maximum identities to track
            embedding_history_size: Embeddings to keep per identity for averaging
        """
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.max_gallery_size = max_gallery_size
        self.embedding_history_size = embedding_history_size
        
        # Gallery: global_id -> GalleryEntry (fused/body embeddings)
        self.gallery: Dict[str, GalleryEntry] = {}
        
        # Face-only gallery: global_id -> face_embedding (for face-based search)
        self.face_gallery: Dict[str, np.ndarray] = {}
        
        logger.info(f"VisualMatcher initialized (threshold={match_threshold})")
    
    def add_to_gallery(
        self,
        global_id: str,
        embedding: np.ndarray,
        camera_id: int,
        timestamp: Optional[datetime] = None,
        quality_score: Optional[float] = None,
        bbox_confidence: Optional[float] = None,
        occlusion_rate: Optional[float] = None,
    ) -> bool:
        """
        Add or update identity in gallery with quality gating.
        
        Args:
            global_id: Unique identity ID
            embedding: Feature embedding
            camera_id: Current camera
            timestamp: Observation time
            quality_score: Overall quality score (0-1)
            bbox_confidence: Detection confidence (0-1)
            occlusion_rate: Occlusion level (0-1, 0=no occlusion)
            
        Returns:
            True if embedding was added, False if rejected
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Quality gating (Phase 1 enhancement)
        from app.config import settings
        
        if quality_score is not None and quality_score < settings.reid_quality_threshold:
            logger.debug(f"Rejected low-quality feature: quality={quality_score:.3f} < {settings.reid_quality_threshold}")
            return False
        
        if bbox_confidence is not None and bbox_confidence < settings.reid_bbox_confidence_threshold:
            logger.debug(f"Rejected low-confidence feature: conf={bbox_confidence:.3f}")
            return False
        
        if occlusion_rate is not None and occlusion_rate > 0.3:
            logger.debug(f"Rejected occluded feature: occlusion_rate={occlusion_rate:.3f}")
            return False
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if global_id in self.gallery:
            entry = self.gallery[global_id]
            
            # Diversity constraint: avoid redundant embeddings (Phase 1 enhancement)
            if entry.embeddings_history:
                similarities = [
                    float(np.dot(embedding, hist_emb))
                    for hist_emb in entry.embeddings_history
                ]
                max_sim = max(similarities)
                
                if max_sim > settings.reid_diversity_threshold:
                    logger.debug(
                        f"Rejected redundant embedding for {global_id[:8]}: "
                        f"max_sim={max_sim:.3f} > {settings.reid_diversity_threshold}"
                    )
                    return False
            
            entry.embeddings_history.append(embedding)
            
            # Keep only recent embeddings (increased to 50 in Phase 1)
            max_size = settings.reid_feature_bank_size  # 50
            if len(entry.embeddings_history) > max_size:
                entry.embeddings_history = entry.embeddings_history[-max_size:]
            
            # Update average embedding
            entry.embedding = np.mean(entry.embeddings_history, axis=0)
            entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)
            
            # Track camera transition if camera changed
            if entry.last_camera_id != camera_id:
                time_at_prev = (timestamp - entry.last_seen).total_seconds()
                transition = CameraTransition(
                    from_camera=entry.last_camera_id,
                    to_camera=camera_id,
                    transition_time=timestamp,
                    time_at_from_camera=time_at_prev
                )
                entry.transitions.append(transition)
                logger.debug(f"Camera transition: {entry.last_camera_id} -> {camera_id} (after {time_at_prev:.1f}s)")
            
            entry.last_camera_id = camera_id
            entry.last_seen = timestamp
            entry.appearance_count += 1
            
            # Track camera history (avoid duplicates in sequence)
            if not entry.camera_history or entry.camera_history[-1] != camera_id:
                entry.camera_history.append(camera_id)
            
            # Track first time at this camera
            if camera_id not in entry.camera_timestamps:
                entry.camera_timestamps[camera_id] = timestamp
        else:
            # Check gallery size limit
            if len(self.gallery) >= self.max_gallery_size:
                self._evict_oldest()
            
            self.gallery[global_id] = GalleryEntry(
                global_id=global_id,
                embedding=embedding.copy(),
                last_camera_id=camera_id,
                last_seen=timestamp,
                first_seen=timestamp,
                embeddings_history=[embedding.copy()],
                camera_history=[camera_id],
                camera_timestamps={camera_id: timestamp},
                transitions=[],
            )
        
        logger.debug(f"Gallery updated: {global_id} (size={len(self.gallery)})")
        return True  # Successfully added
    
    def _evict_oldest(self):
        """Remove oldest entry from gallery."""
        if not self.gallery:
            return
        
        oldest_id = min(self.gallery, key=lambda k: self.gallery[k].last_seen)
        del self.gallery[oldest_id]
        logger.debug(f"Evicted oldest gallery entry: {oldest_id}")
    
    def match(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        exclude_ids: Optional[List[str]] = None,
        use_gallery_store: bool = True,
    ) -> List[Tuple[str, float, GalleryEntry]]:
        """
        Find best matching identities from gallery.
        
        Uses hybrid matching:
        1. Compares against averaged embedding in VisualMatcher (fast)
        2. If GalleryStore available, also computes MAX similarity across 
           all stored embeddings (more robust for cross-camera)
        3. Takes the higher of the two scores
        
        Args:
            query_embedding: Query feature embedding
            top_k: Number of top matches to return
            exclude_ids: IDs to exclude from matching
            use_gallery_store: Whether to also check GalleryStore embeddings
            
        Returns:
            List of (global_id, similarity, entry) sorted by similarity
        """
        if len(self.gallery) == 0:
            return []
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        exclude_ids = exclude_ids or []
        
        # Get GalleryStore for multi-embedding matching
        gallery_store = None
        if use_gallery_store:
            try:
                from app.services.gallery_store import get_gallery_store
                gallery_store = get_gallery_store()
            except Exception:
                pass
        
        # Compute similarities
        results = []
        for global_id, entry in self.gallery.items():
            if global_id in exclude_ids:
                continue
            
            # Score 1: Similarity against averaged embedding (fast)
            avg_similarity = float(np.dot(query_embedding, entry.embedding))
            
            # Score 2: MAX similarity against all GalleryStore embeddings (robust)
            max_similarity = avg_similarity  # default to avg if GalleryStore not available
            if gallery_store:
                gs_max = gallery_store.compute_max_similarity(query_embedding, global_id)
                if gs_max > 0:
                    max_similarity = gs_max
            
            # Use the higher of the two for matching decision
            similarity = max(avg_similarity, max_similarity)
            
            logger.debug(
                f"Match check: {global_id[:8]} cam={entry.last_camera_id} "
                f"avg_sim={avg_similarity:.3f} max_sim={max_similarity:.3f} "
                f"final={similarity:.3f}"
            )
            results.append((global_id, similarity, entry))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Use candidate_threshold for pre-filtering (let caller decide with stricter threshold)
        results = [r for r in results if r[1] >= self.candidate_threshold]
        return results[:top_k]
    
    def match_best(
        self,
        query_embedding: np.ndarray,
        exclude_ids: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, float, GalleryEntry]]:
        """
        Find single best match from gallery.
        
        Returns:
            (global_id, similarity, entry) or None if no match
        """
        matches = self.match(query_embedding, top_k=1, exclude_ids=exclude_ids)
        return matches[0] if matches else None
    
    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """
        Get all gallery embeddings.
        
        Returns:
            Tuple of (list of IDs, embeddings array of shape (N, D))
        """
        if not self.gallery:
            return [], np.empty((0, 256))  # NVIDIA ReID embedding dimension
        
        ids = list(self.gallery.keys())
        embeddings = np.array([self.gallery[id].embedding for id in ids])
        return ids, embeddings
    
    def remove_from_gallery(self, global_id: str):
        """Remove identity from gallery."""
        if global_id in self.gallery:
            del self.gallery[global_id]
            logger.debug(f"Removed from gallery: {global_id}")
        if global_id in self.face_gallery:
            del self.face_gallery[global_id]
    
    def clear_gallery(self):
        """Clear all gallery entries with proper memory cleanup."""
        # Explicitly delete numpy arrays before clearing dictionaries
        for entry in self.gallery.values():
            # Delete main embedding
            if hasattr(entry, 'embedding') and entry.embedding is not None:
                del entry.embedding
            # Delete embedding history list
            if hasattr(entry, 'embeddings_history') and entry.embeddings_history:
                for emb in entry.embeddings_history:
                    if emb is not None:
                        del emb
                entry.embeddings_history.clear()
        
        # Delete face embeddings
        for face_emb in self.face_gallery.values():
            if face_emb is not None:
                del face_emb
        
        # Clear the dictionaries
        self.gallery.clear()
        self.face_gallery.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Gallery cleared with memory cleanup")
    
    def add_face_embedding(self, global_id: str, face_embedding: np.ndarray):
        """
        Add or update face embedding for a person.
        
        Used for face-only search matching.
        """
        # Normalize
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm
        self.face_gallery[global_id] = face_embedding.copy()
    
    def match_face(
        self,
        face_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Optional[GalleryEntry]]]:
        """
        Match a face embedding against face gallery.
        
        Returns matches with gallery entry if exists.
        """
        if len(self.face_gallery) == 0:
            return []
        
        threshold = threshold or self.match_threshold
        
        # Normalize query
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm
        
        results = []
        for global_id, stored_face in self.face_gallery.items():
            similarity = float(np.dot(face_embedding, stored_face))
            entry = self.gallery.get(global_id)
            results.append((global_id, similarity, entry))
        
        # Sort and filter
        results.sort(key=lambda x: x[1], reverse=True)
        results = [r for r in results if r[1] >= threshold]
        return results[:top_k]
    
    @property
    def gallery_size(self) -> int:
        """Current gallery size."""
        return len(self.gallery)


def compute_reranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    K-reciprocal re-ranking for improved ReID accuracy.
    
    Based on: "Re-ranking Person Re-identification with k-Reciprocal Encoding"
    
    Args:
        query_features: Query embeddings (M, D)
        gallery_features: Gallery embeddings (N, D)
        k1: Parameter for initial ranking
        k2: Parameter for local query expansion
        lambda_value: Balance between original and jaccard distance
        
    Returns:
        Re-ranked distance matrix (M, N)
    """
    # Compute initial distance matrix
    query_num = query_features.shape[0]
    gallery_num = gallery_features.shape[0]
    all_features = np.vstack([query_features, gallery_features])
    
    # Cosine distance
    original_dist = 1 - np.dot(all_features, all_features.T)
    
    # Get initial ranking
    initial_rank = np.argsort(original_dist, axis=1)
    
    # Compute k-reciprocal neighbors
    all_num = query_num + gallery_num
    V = np.zeros((all_num, all_num), dtype=np.float32)
    
    for i in range(all_num):
        # Forward k-nearest neighbors
        forward_k = initial_rank[i, :k1 + 1]
        
        # Compute k-reciprocal set
        k_reciprocal = []
        for j in forward_k:
            backward_k = initial_rank[j, :k1 + 1]
            if i in backward_k:
                k_reciprocal.append(j)
        
        k_reciprocal = np.array(k_reciprocal)
        
        # Local query expansion
        if k_reciprocal.shape[0] > k2:
            k_recip_exp = k_reciprocal
            for candidate in k_reciprocal[:k2]:
                candidate_back_k = initial_rank[candidate, :k1 // 2 + 1]
                candidate_recip = []
                for c in candidate_back_k:
                    c_back = initial_rank[c, :k1 // 2 + 1]
                    if candidate in c_back:
                        candidate_recip.append(c)
                
                candidate_recip = np.array(candidate_recip)
                overlap = len(np.intersect1d(candidate_recip, k_reciprocal))
                if overlap > 2/3 * len(candidate_recip):
                    k_recip_exp = np.union1d(k_recip_exp, candidate_recip)
            
            k_reciprocal = k_recip_exp
        
        # Gaussian kernel weights
        weights = np.exp(-original_dist[i, k_reciprocal])
        V[i, k_reciprocal] = weights / np.sum(weights)
    
    # Jaccard distance
    tmp_V = V[:query_num]
    jaccard_dist = 1 - np.dot(tmp_V, V.T[:, query_num:])
    
    # Final distance
    final_dist = (1 - lambda_value) * original_dist[:query_num, query_num:] + lambda_value * jaccard_dist
    
    return final_dist
