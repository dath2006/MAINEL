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

from app.config import settings


@dataclass
class GalleryEntry:
    """
    Entry in the identity gallery.
    
    Uses Exemplar Buffer pattern: stores the BEST embedding seen,
    not an average of all embeddings. This prevents feature drift
    from low-quality frames (occlusions, back views).
    """
    global_id: str
    embedding: np.ndarray  # Best exemplar embedding (not average)
    best_quality_score: float  # Quality score of current best embedding
    last_camera_id: int
    last_seen: datetime
    appearance_count: int = 1
    camera_history: List[int] = field(default_factory=list)  # All cameras where seen


class VisualMatcher:
    """
    Visual similarity matcher for cross-camera ReID.
    
    Maintains a gallery of known identities and matches new
    detections against them using cosine similarity.
    """
    
    def __init__(
        self,
        match_threshold: float = None,
        max_gallery_size: int = None,
        embedding_history_size: int = 10,
    ):
        """
        Initialize visual matcher.
        
        Args:
            match_threshold: Minimum similarity for valid match (from config if None)
            max_gallery_size: Maximum identities to track (from config if None)
            embedding_history_size: Embeddings to keep per identity for averaging
        """
        self.match_threshold = match_threshold if match_threshold is not None else settings.reid_match_threshold
        self.max_gallery_size = max_gallery_size if max_gallery_size is not None else settings.gallery_max_size
        self.embedding_history_size = embedding_history_size
        
        # Gallery: global_id -> GalleryEntry (fused/body embeddings)
        self.gallery: Dict[str, GalleryEntry] = {}
        
        # Face-only gallery: global_id -> face_embedding (for face-based search)
        self.face_gallery: Dict[str, np.ndarray] = {}
        
        logger.info(f"VisualMatcher initialized (threshold={self.match_threshold})")
    
    def add_to_gallery(
        self,
        global_id: str,
        embedding: np.ndarray,
        camera_id: int,
        timestamp: Optional[datetime] = None,
        quality_score: float = 0.5,
    ):
        """
        Add or update identity in gallery using Quality-Priority Buffer.
        
        Uses Exemplar pattern: only updates the stored embedding if the
        new observation is of HIGHER QUALITY than what we already have.
        This prevents "feature drift" from low-quality frames.
        
        Args:
            global_id: Unique identity ID
            embedding: Feature embedding
            camera_id: Current camera
            timestamp: Observation time
            quality_score: Quality score of this frame (0.0-1.0, higher is better)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Normalize embedding (CRITICAL for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if global_id in self.gallery:
            entry = self.gallery[global_id]
            
            # Compute similarity between new embedding and stored exemplar
            sim = float(np.dot(entry.embedding, embedding))
            
            # === QUALITY-PRIORITY UPDATE LOGIC (More Conservative) ===
            # Rule 1: Strict Replace - Only update if new frame is SIGNIFICANTLY BETTER
            if quality_score > entry.best_quality_score + settings.gallery_quality_delta:  # Configurable delta
                # Found a much better view! Replace the exemplar completely.
                logger.info(f"Gallery: Replaced exemplar for {global_id[:8]} (quality: {entry.best_quality_score:.2f} -> {quality_score:.2f}, sim={sim:.3f})")
                entry.embedding = embedding.copy()
                entry.best_quality_score = quality_score
            elif sim > settings.gallery_merge_threshold:  # Configurable merge threshold
                # Rule 2: Smart Merge - Only blend if EXTREMELY similar
                # Same person, slightly different pose - gentle blend
                alpha = 0.05  # Reduced from 0.1 - even gentler blend
                entry.embedding = (1 - alpha) * entry.embedding + alpha * embedding
                # Re-normalize after blend
                entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)
                logger.debug(f"Gallery: Blended embedding for {global_id[:8]} (sim={sim:.3f})")
            else:
                # Rule 3: REJECT - similarity too low or quality not better
                # This prevents wrong person's features from polluting gallery
                logger.debug(f"Gallery: REJECTED embedding update for {global_id[:8]} (sim={sim:.3f} < {settings.gallery_merge_threshold}, quality={quality_score:.2f} vs stored={entry.best_quality_score:.2f})")
            
            # Always update metadata
            entry.last_camera_id = camera_id
            entry.last_seen = timestamp
            entry.appearance_count += 1
            
            # Track camera history (avoid duplicates in sequence)
            if not entry.camera_history or entry.camera_history[-1] != camera_id:
                entry.camera_history.append(camera_id)

        else:
            # New identity - check gallery size limit first
            if len(self.gallery) >= self.max_gallery_size:
                self._evict_oldest()
            
            self.gallery[global_id] = GalleryEntry(
                global_id=global_id,
                embedding=embedding.copy(),
                best_quality_score=quality_score,
                last_camera_id=camera_id,
                last_seen=timestamp,
                camera_history=[camera_id],
            )
        
        logger.debug(f"Gallery updated: {global_id[:8]} (size={len(self.gallery)}, quality={quality_score:.2f})")
    
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
        use_rerank: bool = True,  # "Magic Bullet" enabled by default
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, GalleryEntry]]:
        """
        Find best matching identities from gallery.
        
        Args:
            query_embedding: Query feature embedding
            top_k: Number of top matches to return
            exclude_ids: IDs to exclude from matching
            use_rerank: Enable k-reciprocal reranking (The "Un-Cook" Strategy)
            
        Returns:
            List of (global_id, similarity, entry) sorted by similarity
            Note: similarity is ALWAYS raw cosine similarity for consistent thresholds
        """
        if len(self.gallery) == 0:
            return []
        
        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        exclude_ids = exclude_ids or []
        
        # Compute raw cosine similarities for ALL gallery entries first
        # This ensures consistent threshold comparison regardless of reranking
        raw_similarities = {}
        for global_id, entry in self.gallery.items():
            if global_id in exclude_ids:
                continue
            raw_similarities[global_id] = float(np.dot(query_embedding, entry.embedding))
        
        # Fast path or Not enough data for reranking
        if not use_rerank or len(self.gallery) < 10:
            results = [
                (gid, sim, self.gallery[gid]) 
                for gid, sim in raw_similarities.items()
            ]
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Log ALL candidates before filtering
            logger.info(f"=== Gallery Search: {len(results)} candidates (threshold={threshold or self.match_threshold}) ===")
            for i, (gid, sim, entry) in enumerate(results[:10]):  # Log top 10
                status = "✓ PASS" if sim >= (threshold or self.match_threshold) else "✗ BELOW"
                logger.info(f"  #{i+1}: {gid[:8]}... sim={sim:.4f} {status}")
            
            min_thresh = threshold if threshold is not None else self.match_threshold
            filtered = [r for r in results if r[1] >= min_thresh]
            
            logger.info(f"=== {len(filtered)} matches above threshold ===")
            
            return filtered[:top_k]
            
        # Reranking path: Use reranking for ORDERING but return RAW COSINE for similarity
        ids, gallery_embs = self.get_all_embeddings()
        query_expanded = query_embedding.reshape(1, -1)
        
        # Compute reranked distances (lower is better) for ordering
        dists = compute_reranking(query_expanded, gallery_embs)
        
        # Build results with rerank ordering but RAW COSINE similarity values
        results = []
        for i, dist in enumerate(dists[0]):
            gid = ids[i]
            if gid in exclude_ids:
                continue
            
            # Use RAW cosine similarity (not exp(-dist)) for consistent thresholds
            raw_sim = raw_similarities.get(gid, 0.0)
            rerank_score = float(np.exp(-dist))  # Keep for logging
            
            results.append((gid, raw_sim, self.gallery[gid], dist, rerank_score))
        
        # Sort by rerank distance (lower is better) for ordering
        results.sort(key=lambda x: x[3])
        
        # Log ALL candidates before filtering
        min_thresh = threshold if threshold is not None else self.match_threshold
        logger.info(f"=== Gallery Search (reranked): {len(results)} candidates (threshold={min_thresh}) ===")
        for i, (gid, raw_sim, entry, dist, rerank_score) in enumerate(results[:10]):  # Log top 10
            status = "✓ PASS" if raw_sim >= min_thresh else "✗ BELOW"
            logger.info(f"  #{i+1}: {gid[:8]}... cosine={raw_sim:.4f} rerank_dist={dist:.3f} {status}")
        
        # Filter by raw cosine threshold
        filtered = []
        for gid, raw_sim, entry, dist, rerank_score in results:
            if raw_sim >= min_thresh:
                filtered.append((gid, raw_sim, entry))
        
        logger.info(f"=== {len(filtered)} matches above threshold ===")
        
        return filtered[:top_k]
    

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
            return [], np.empty((0, 512))
        
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
        """Clear all gallery entries."""
        self.gallery.clear()
        self.face_gallery.clear()
        logger.info("Gallery cleared")
    
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
    k1: int = None,
    k2: int = None,
    lambda_value: float = None,
) -> np.ndarray:
    """
    K-reciprocal re-ranking for improved ReID accuracy.
    
    Based on: "Re-ranking Person Re-identification with k-Reciprocal Encoding"
    
    Args:
        query_features: Query embeddings (M, D)
        gallery_features: Gallery embeddings (N, D)
        k1: Parameter for initial ranking (from config if None)
        k2: Parameter for local query expansion (from config if None)
        lambda_value: Balance between original and jaccard distance (from config if None)
        
    Returns:
        Re-ranked distance matrix (M, N)
    """
    # Use config defaults if not provided
    k1 = k1 if k1 is not None else settings.rerank_k1
    k2 = k2 if k2 is not None else settings.rerank_k2
    lambda_value = lambda_value if lambda_value is not None else settings.rerank_lambda
    
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
