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
class GalleryEntry:
    """Entry in the identity gallery."""
    global_id: str
    embedding: np.ndarray  # Average feature embedding
    last_camera_id: int
    last_seen: datetime
    appearance_count: int = 1
    embeddings_history: List[np.ndarray] = field(default_factory=list)


class VisualMatcher:
    """
    Visual similarity matcher for cross-camera ReID.
    
    Maintains a gallery of known identities and matches new
    detections against them using cosine similarity.
    """
    
    def __init__(
        self,
        match_threshold: float = 0.6,
        max_gallery_size: int = 1000,
        embedding_history_size: int = 10,
    ):
        """
        Initialize visual matcher.
        
        Args:
            match_threshold: Minimum similarity for valid match
            max_gallery_size: Maximum identities to track
            embedding_history_size: Embeddings to keep per identity for averaging
        """
        self.match_threshold = match_threshold
        self.max_gallery_size = max_gallery_size
        self.embedding_history_size = embedding_history_size
        
        # Gallery: global_id -> GalleryEntry
        self.gallery: Dict[str, GalleryEntry] = {}
        
        logger.info(f"VisualMatcher initialized (threshold={match_threshold})")
    
    def add_to_gallery(
        self,
        global_id: str,
        embedding: np.ndarray,
        camera_id: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Add or update identity in gallery.
        
        Args:
            global_id: Unique identity ID
            embedding: Feature embedding
            camera_id: Current camera
            timestamp: Observation time
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if global_id in self.gallery:
            entry = self.gallery[global_id]
            entry.embeddings_history.append(embedding)
            
            # Keep only recent embeddings
            if len(entry.embeddings_history) > self.embedding_history_size:
                entry.embeddings_history = entry.embeddings_history[-self.embedding_history_size:]
            
            # Update average embedding
            entry.embedding = np.mean(entry.embeddings_history, axis=0)
            entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)
            
            entry.last_camera_id = camera_id
            entry.last_seen = timestamp
            entry.appearance_count += 1
        else:
            # Check gallery size limit
            if len(self.gallery) >= self.max_gallery_size:
                self._evict_oldest()
            
            self.gallery[global_id] = GalleryEntry(
                global_id=global_id,
                embedding=embedding.copy(),
                last_camera_id=camera_id,
                last_seen=timestamp,
                embeddings_history=[embedding.copy()],
            )
        
        logger.debug(f"Gallery updated: {global_id} (size={len(self.gallery)})")
    
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
    ) -> List[Tuple[str, float, GalleryEntry]]:
        """
        Find best matching identities from gallery.
        
        Args:
            query_embedding: Query feature embedding
            top_k: Number of top matches to return
            exclude_ids: IDs to exclude from matching
            
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
        
        # Compute similarities
        results = []
        for global_id, entry in self.gallery.items():
            if global_id in exclude_ids:
                continue
            
            similarity = float(np.dot(query_embedding, entry.embedding))
            logger.debug(f"Match check: {global_id[:8]} similarity={similarity:.3f} threshold={self.match_threshold}")
            results.append((global_id, similarity, entry))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply threshold and limit
        results = [r for r in results if r[1] >= self.match_threshold]
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
            return [], np.empty((0, 512))
        
        ids = list(self.gallery.keys())
        embeddings = np.array([self.gallery[id].embedding for id in ids])
        return ids, embeddings
    
    def remove_from_gallery(self, global_id: str):
        """Remove identity from gallery."""
        if global_id in self.gallery:
            del self.gallery[global_id]
            logger.debug(f"Removed from gallery: {global_id}")
    
    def clear_gallery(self):
        """Clear all gallery entries."""
        self.gallery.clear()
        logger.info("Gallery cleared")
    
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
