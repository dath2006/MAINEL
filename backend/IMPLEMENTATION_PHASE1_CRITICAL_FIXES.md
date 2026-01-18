# MTMCT Critical Fixes Implementation Plan
## Phase 1: TensorRT Optimization, Vector Database & Proactive Global Association

> **Objective**: Address the three most critical performance and reliability issues that prevent production-grade deployment.

---

## Table of Contents
1. [TensorRT Execution Provider Integration](#1-tensorrt-execution-provider-integration)
2. [Vector Database Integration (FAISS/Qdrant)](#2-vector-database-integration)
3. [Proactive Global Association Algorithm](#3-proactive-global-association-algorithm)
4. [Testing & Verification](#4-testing--verification)

---

## 1. TensorRT Execution Provider Integration

### 1.1 Problem Statement
Current system uses basic CUDA Execution Provider, missing 3-5x potential speedup from TensorRT optimizations including FP16 precision, kernel fusion, and engine caching.

### 1.2 Prerequisites
```bash
# Verify TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# If not installed, install via pip (requires CUDA 11.8+)
pip install tensorrt==8.6.1
```

### 1.3 Implementation Steps

#### Step 1: Create TensorRT Configuration Module

**File**: `app/core/inference/trt_config.py` [NEW]

```python
"""
TensorRT Execution Provider Configuration

Provides optimized ONNX Runtime session configuration for both
PeopleNet and ReIDNet models with FP16, engine caching, and
dynamic shape support.
"""

import os
from pathlib import Path
from loguru import logger

# Cache directory for TensorRT engines
TRT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "trt_cache"
TRT_CACHE_DIR.mkdir(exist_ok=True)


def get_tensorrt_providers(
    device_id: int = 0,
    fp16: bool = True,
    int8: bool = False,
    cache_prefix: str = "model",
) -> list:
    """
    Get ONNX Runtime execution providers with TensorRT optimization.
    
    Args:
        device_id: GPU device ID
        fp16: Enable FP16 precision (recommended for Tensor Cores)
        int8: Enable INT8 quantization (requires calibration)
        cache_prefix: Prefix for cached TensorRT engine files
        
    Returns:
        List of execution providers in priority order
    """
    cache_path = str(TRT_CACHE_DIR / cache_prefix)
    
    trt_options = {
        'device_id': device_id,
        'trt_fp16_enable': fp16,
        'trt_int8_enable': int8,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_path,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': str(TRT_CACHE_DIR / "timing_cache"),
        # Dynamic shape support
        'trt_profile_min_shapes': 'input:1x3x256x128',
        'trt_profile_opt_shapes': 'input:8x3x256x128',
        'trt_profile_max_shapes': 'input:32x3x256x128',
    }
    
    cuda_options = {
        'device_id': device_id,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }
    
    providers = [
        ('TensorrtExecutionProvider', trt_options),
        ('CUDAExecutionProvider', cuda_options),
        'CPUExecutionProvider',
    ]
    
    logger.info(f"TensorRT providers configured: FP16={fp16}, cache={cache_path}")
    return providers


def get_peoplenet_providers(device_id: int = 0) -> list:
    """Get optimized providers for PeopleNet (960x544 input)."""
    cache_path = str(TRT_CACHE_DIR / "peoplenet")
    
    trt_options = {
        'device_id': device_id,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_path,
        # PeopleNet uses fixed input shape
        'trt_profile_min_shapes': 'input_1:1x3x544x960',
        'trt_profile_opt_shapes': 'input_1:1x3x544x960',
        'trt_profile_max_shapes': 'input_1:4x3x544x960',
    }
    
    return [
        ('TensorrtExecutionProvider', trt_options),
        ('CUDAExecutionProvider', {'device_id': device_id}),
        'CPUExecutionProvider',
    ]


def get_reidnet_providers(device_id: int = 0) -> list:
    """Get optimized providers for ReIDNet (256x128 input, dynamic batch)."""
    cache_path = str(TRT_CACHE_DIR / "reidnet")
    
    trt_options = {
        'device_id': device_id,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_path,
        # Dynamic batch for variable detections per frame
        'trt_profile_min_shapes': 'input:1x3x256x128',
        'trt_profile_opt_shapes': 'input:8x3x256x128',
        'trt_profile_max_shapes': 'input:32x3x256x128',
    }
    
    return [
        ('TensorrtExecutionProvider', trt_options),
        ('CUDAExecutionProvider', {'device_id': device_id}),
        'CPUExecutionProvider',
    ]
```

#### Step 2: Update NvidiaReIDExtractor

**File**: `app/core/features/nvidia_reid_extractor.py` [MODIFY]

```diff
# At imports section
+ from app.core.inference.trt_config import get_reidnet_providers

# In __init__ method, replace providers configuration
- providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
- if device == 'cpu':
-     providers = ['CPUExecutionProvider']

+ if device == 'cpu':
+     providers = ['CPUExecutionProvider']
+ else:
+     try:
+         providers = get_reidnet_providers(device_id=0)
+         logger.info("Using TensorRT-optimized providers for ReIDNet")
+     except Exception as e:
+         logger.warning(f"TensorRT unavailable, falling back to CUDA: {e}")
+         providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

#### Step 3: Update PeopleNetDetector

**File**: `preprocessor/peoplenet_detector.py` [MODIFY]

```diff
# At imports section (add near top)
+ import sys
+ sys.path.insert(0, str(Path(__file__).parent.parent))
+ from app.core.inference.trt_config import get_peoplenet_providers

# In _init_session method, replace providers
- providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
- if self.device == 'cpu':
-     providers = ['CPUExecutionProvider']

+ if self.device == 'cpu':
+     providers = ['CPUExecutionProvider']
+ else:
+     try:
+         providers = get_peoplenet_providers(device_id=0)
+         logger.info("Using TensorRT-optimized providers for PeopleNet")
+     except Exception as e:
+         logger.warning(f"TensorRT unavailable: {e}")
+         providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 1.4 Configuration

Add to `app/config.py`:
```python
# TensorRT settings
trt_fp16_enable: bool = True
trt_cache_enable: bool = True
trt_cache_dir: str = "./trt_cache"
```

### 1.5 First-Run Behavior

> [!NOTE]
> First inference will be slow (1-3 minutes) as TensorRT builds and caches optimized engines. Subsequent runs use cached engines for instant startup.

---

## 2. Vector Database Integration

### 2.1 Problem Statement
Current O(N²) brute-force matching in `visual_matcher.py` won't scale beyond ~100 identities. Need O(log N) approximate nearest neighbor search.

### 2.2 Approach: FAISS (In-Memory)
FAISS provides the simplest integration path with excellent performance. For persistent storage, Qdrant can be added later.

### 2.3 Implementation Steps

#### Step 1: Install FAISS

```bash
# GPU version (recommended)
pip install faiss-gpu

# CPU fallback
pip install faiss-cpu
```

#### Step 2: Create Vector Search Module

**File**: `app/core/reid/vector_search.py` [NEW]

```python
"""
Vector Search using FAISS

Provides O(log N) approximate nearest neighbor search for ReID embeddings.
Supports dynamic add/remove operations and maintains ID mapping.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from loguru import logger

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Using brute-force fallback.")


class VectorIndex:
    """
    FAISS-based vector index for ReID embeddings.
    
    Uses Inner Product similarity (equivalent to cosine for normalized vectors).
    Supports dynamic insertions and deletions with ID remapping.
    """
    
    def __init__(
        self,
        dimension: int = 256,
        use_gpu: bool = True,
        index_type: str = "IVF",  # "Flat" for exact, "IVF" for approximate
        nlist: int = 100,  # Number of clusters for IVF
    ):
        """
        Initialize vector index.
        
        Args:
            dimension: Embedding dimension (256 for NVIDIA ReIDNet)
            use_gpu: Use GPU acceleration if available
            index_type: "Flat" (exact) or "IVF" (approximate, faster for large galleries)
            nlist: Number of IVF clusters (only for IVF type)
        """
        self.dimension = dimension
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        self.index_type = index_type
        self.nlist = nlist
        
        # ID management
        self.id_to_idx: Dict[str, int] = {}  # global_id -> FAISS index
        self.idx_to_id: Dict[int, str] = {}  # FAISS index -> global_id
        self.next_idx: int = 0
        self.deleted_indices: set = set()  # Track deleted for reuse
        
        # Initialize index
        self._init_index()
        
        logger.info(f"VectorIndex initialized: dim={dimension}, type={index_type}, GPU={self.use_gpu}")
    
    def _init_index(self):
        """Initialize FAISS index."""
        if not FAISS_AVAILABLE:
            self.index = None
            return
        
        if self.index_type == "Flat":
            # Exact search - Inner Product for cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # IVF for approximate search (faster for large galleries)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            # IVF requires training on initial vectors
            self._needs_training = True
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"GPU FAISS unavailable: {e}")
    
    def add(self, global_id: str, embedding: np.ndarray) -> bool:
        """
        Add embedding to index.
        
        Args:
            global_id: Unique identity ID
            embedding: L2-normalized embedding vector (256,)
            
        Returns:
            True if added, False if already exists
        """
        if not FAISS_AVAILABLE:
            return False
            
        if global_id in self.id_to_idx:
            # Update existing - remove old, add new
            self.remove(global_id)
        
        # Ensure embedding is correct shape and normalized
        embedding = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Add to index
        self.index.add(embedding)
        
        # Update mappings
        idx = self.next_idx
        self.id_to_idx[global_id] = idx
        self.idx_to_id[idx] = global_id
        self.next_idx += 1
        
        return True
    
    def remove(self, global_id: str) -> bool:
        """
        Remove embedding from index.
        
        Note: FAISS doesn't support true deletion. We mark as deleted
        and rebuild periodically if deletion ratio gets high.
        """
        if global_id not in self.id_to_idx:
            return False
        
        idx = self.id_to_idx[global_id]
        self.deleted_indices.add(idx)
        del self.id_to_idx[global_id]
        del self.idx_to_id[idx]
        
        # Rebuild if too many deletions (>20% of index)
        if len(self.deleted_indices) > self.next_idx * 0.2:
            self._rebuild_index()
        
        return True
    
    def _rebuild_index(self):
        """Rebuild index to reclaim deleted space."""
        if not FAISS_AVAILABLE or self.index is None:
            return
            
        # Get all valid embeddings
        valid_ids = list(self.id_to_idx.keys())
        if not valid_ids:
            self._init_index()
            self.next_idx = 0
            self.deleted_indices.clear()
            return
        
        # Extract embeddings from current index
        all_embeddings = []
        for global_id in valid_ids:
            idx = self.id_to_idx[global_id]
            # Note: This is slow but necessary for rebuild
            emb = faiss.rev_swig_ptr(
                self.index.reconstruct(idx), 
                self.dimension
            ).copy()
            all_embeddings.append(emb)
        
        # Reinitialize
        self._init_index()
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.next_idx = 0
        self.deleted_indices.clear()
        
        # Re-add all embeddings
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        self.index.add(embeddings)
        
        for i, global_id in enumerate(valid_ids):
            self.id_to_idx[global_id] = i
            self.idx_to_id[i] = global_id
        
        self.next_idx = len(valid_ids)
        logger.info(f"VectorIndex rebuilt: {len(valid_ids)} embeddings")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query embedding (256,) - should be L2-normalized
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (global_id, similarity) tuples, sorted by similarity desc
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query is correct shape
        query = query.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        # Search (request more than k to handle deleted entries)
        search_k = min(k * 2, self.index.ntotal)
        similarities, indices = self.index.search(query, search_k)
        
        # Filter and map results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx in self.deleted_indices:
                continue
            if idx not in self.idx_to_id:
                continue
            if sim < threshold:
                continue
            
            global_id = self.idx_to_id[idx]
            results.append((global_id, float(sim)))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Query embeddings (N, 256)
            k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of result lists, one per query
        """
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(queries))]
        
        # Normalize queries
        queries = queries.astype(np.float32)
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms[norms == 0] = 1
        queries = queries / norms
        
        # Batch search
        search_k = min(k * 2, self.index.ntotal)
        similarities, indices = self.index.search(queries, search_k)
        
        # Process each query result
        all_results = []
        for q_sims, q_indices in zip(similarities, indices):
            results = []
            for sim, idx in zip(q_sims, q_indices):
                if idx < 0 or idx in self.deleted_indices:
                    continue
                if idx not in self.idx_to_id:
                    continue
                if sim < threshold:
                    continue
                
                global_id = self.idx_to_id[idx]
                results.append((global_id, float(sim)))
                
                if len(results) >= k:
                    break
            
            all_results.append(results)
        
        return all_results
    
    @property
    def size(self) -> int:
        """Number of active embeddings in index."""
        return len(self.id_to_idx)
    
    def clear(self):
        """Clear all embeddings from index."""
        self._init_index()
        self.id_to_idx.clear()
        self.idx_to_id.clear()
        self.next_idx = 0
        self.deleted_indices.clear()
        logger.info("VectorIndex cleared")


# Singleton instance
_vector_index: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    """Get or create singleton VectorIndex."""
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex()
    return _vector_index


def reset_vector_index():
    """Reset the singleton (for testing)."""
    global _vector_index
    if _vector_index:
        _vector_index.clear()
    _vector_index = None
```

#### Step 3: Integrate with VisualMatcher

**File**: `app/core/reid/visual_matcher.py` [MODIFY]

Add vector index integration:

```diff
# At imports
+ from app.core.reid.vector_search import get_vector_index, VectorIndex

# In VisualMatcher.__init__
  def __init__(self, ...):
      ...
      self.gallery: Dict[str, GalleryEntry] = {}
      self.face_gallery: Dict[str, np.ndarray] = {}
+     self._vector_index: Optional[VectorIndex] = None
      
+     # Try to initialize vector index
+     try:
+         self._vector_index = get_vector_index()
+         logger.info("Vector index enabled for O(log N) search")
+     except Exception as e:
+         logger.warning(f"Vector index unavailable, using brute-force: {e}")

# In add_to_gallery method, after updating gallery dict
+ # Sync with vector index
+ if self._vector_index:
+     self._vector_index.add(global_id, entry.embedding)

# In remove_from_gallery method
+ if self._vector_index:
+     self._vector_index.remove(global_id)

# In clear_gallery method
+ if self._vector_index:
+     self._vector_index.clear()

# Replace the match method's search logic
  def match(self, query_embedding: np.ndarray, top_k: int = 5, ...):
      ...
-     # Compute similarities
-     results = []
-     for global_id, entry in self.gallery.items():
-         if global_id in exclude_ids:
-             continue
-         avg_similarity = float(np.dot(query_embedding, entry.embedding))
-         ...
-         results.append((global_id, similarity, entry))
-     results.sort(key=lambda x: x[1], reverse=True)

+     # Use vector index for O(log N) search if available
+     if self._vector_index and self._vector_index.size > 0:
+         vector_results = self._vector_index.search(
+             query_embedding, 
+             k=top_k * 2,  # Get extra for filtering
+             threshold=self.candidate_threshold
+         )
+         results = []
+         for global_id, similarity in vector_results:
+             if global_id in exclude_ids:
+                 continue
+             entry = self.gallery.get(global_id)
+             if entry is None:
+                 continue
+             # Optionally enhance with GalleryStore max similarity
+             if gallery_store:
+                 gs_max = gallery_store.compute_max_similarity(query_embedding, global_id)
+                 similarity = max(similarity, gs_max) if gs_max > 0 else similarity
+             results.append((global_id, similarity, entry))
+     else:
+         # Fallback to brute-force
+         results = []
+         for global_id, entry in self.gallery.items():
+             if global_id in exclude_ids:
+                 continue
+             avg_similarity = float(np.dot(query_embedding, entry.embedding))
+             ...
+             results.append((global_id, similarity, entry))
+         results.sort(key=lambda x: x[1], reverse=True)
```

---

## 3. Proactive Global Association Algorithm

### 3.1 Problem Statement
Current system creates new IDs eagerly and merges reactively every 100 frames. This causes ID fragmentation that's difficult to fix later.

### 3.2 Solution: Query-on-Maturation Pattern
Wait for track to accumulate evidence before querying global gallery.

### 3.3 Implementation Steps

#### Step 1: Create Global Associator Module

**File**: `app/core/reid/global_associator.py` [NEW]

```python
"""
Proactive Global Association

Associates local camera tracks to global identities using a
query-on-maturation pattern that prevents ID fragmentation.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from app.config import settings


@dataclass
class PendingTrack:
    """Track waiting for enough evidence before global query."""
    local_id: int
    camera_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    first_seen: datetime = None
    last_seen: datetime = None
    frame_count: int = 0
    global_id: Optional[UUID] = None  # Assigned after maturation
    query_performed: bool = False


class GlobalAssociator:
    """
    Proactive global identity association.
    
    Key principles:
    1. Wait for track maturation (N frames) before querying
    2. Use averaged embedding for query (more stable)
    3. Apply topology constraints to reduce false positives
    4. Only create new ID when confident no match exists
    """
    
    def __init__(
        self,
        maturation_frames: int = 5,
        maturation_timeout: float = 2.0,  # seconds
        match_threshold: float = 0.50,
        new_id_threshold: float = 0.40,
        embedding_window: int = 10,
    ):
        """
        Initialize global associator.
        
        Args:
            maturation_frames: Frames needed before global query
            maturation_timeout: Max wait time before query
            match_threshold: Minimum similarity for match
            new_id_threshold: Below this, definitely create new ID
            embedding_window: Embeddings to average for query
        """
        self.maturation_frames = maturation_frames
        self.maturation_timeout = timedelta(seconds=maturation_timeout)
        self.match_threshold = match_threshold
        self.new_id_threshold = new_id_threshold
        self.embedding_window = embedding_window
        
        # Pending tracks per camera: (camera_id, local_id) -> PendingTrack
        self._pending: Dict[Tuple[int, int], PendingTrack] = {}
        
        # Local to global ID mapping
        self._local_to_global: Dict[Tuple[int, int], UUID] = {}
        
        logger.info(
            f"GlobalAssociator initialized: maturation={maturation_frames} frames / "
            f"{maturation_timeout}s, thresholds=({new_id_threshold}, {match_threshold})"
        )
    
    def update_track(
        self,
        camera_id: int,
        local_id: int,
        embedding: np.ndarray,
        timestamp: datetime,
    ) -> Optional[UUID]:
        """
        Update track and return global ID if available.
        
        Args:
            camera_id: Camera identifier
            local_id: Local track ID from DeepSORT
            embedding: Current frame's embedding
            timestamp: Frame timestamp
            
        Returns:
            Global UUID if assigned, None if still pending
        """
        key = (camera_id, local_id)
        
        # Check if already assigned
        if key in self._local_to_global:
            # Update gallery with new embedding (reinforce identity)
            global_id = self._local_to_global[key]
            self._update_existing_identity(global_id, embedding, camera_id, timestamp)
            return global_id
        
        # Get or create pending track
        if key not in self._pending:
            self._pending[key] = PendingTrack(
                local_id=local_id,
                camera_id=camera_id,
                first_seen=timestamp,
            )
        
        pending = self._pending[key]
        pending.embeddings.append(embedding)
        pending.last_seen = timestamp
        pending.frame_count += 1
        
        # Keep only recent embeddings
        if len(pending.embeddings) > self.embedding_window:
            pending.embeddings = pending.embeddings[-self.embedding_window:]
        
        # Check if ready for global query
        if self._is_mature(pending, timestamp):
            global_id = self._perform_global_query(pending)
            self._local_to_global[key] = global_id
            del self._pending[key]
            return global_id
        
        return None
    
    def _is_mature(self, track: PendingTrack, current_time: datetime) -> bool:
        """Check if track has enough evidence for global query."""
        # Frame count threshold
        if track.frame_count >= self.maturation_frames:
            return True
        
        # Timeout threshold (even with fewer frames)
        if track.first_seen and (current_time - track.first_seen) >= self.maturation_timeout:
            return True
        
        return False
    
    def _perform_global_query(self, track: PendingTrack) -> UUID:
        """
        Query global gallery for matching identity.
        
        Uses averaged embedding for more stable matching.
        """
        # Compute averaged embedding
        embeddings = np.array(track.embeddings)
        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Query visual matcher
        from app.services.reid_service import get_reid_service
        reid_service = get_reid_service()
        
        # Get topology-filtered candidates
        topology = reid_service.topology
        plausible_cameras = topology.get_reachable(track.camera_id, max_hops=2)
        
        # Query with vector index
        matches = reid_service.visual_matcher.match(
            avg_embedding,
            top_k=5,
            use_gallery_store=True,
        )
        
        # Filter by topology if cameras are known
        if plausible_cameras:
            matches = [
                (gid, sim, entry) for gid, sim, entry in matches
                if entry.last_camera_id in plausible_cameras or 
                   entry.last_camera_id == track.camera_id
            ]
        
        # Evaluate best match
        if matches:
            best_id, best_sim, best_entry = matches[0]
            
            if best_sim >= self.match_threshold:
                # Confident match
                logger.info(
                    f"GlobalAssoc: Track {track.camera_id}:{track.local_id} -> "
                    f"MATCH {best_id[:8]} (sim={best_sim:.3f})"
                )
                return UUID(best_id)
            
            elif best_sim >= self.new_id_threshold:
                # Tentative zone - check spatio-temporal plausibility
                time_delta = (track.first_seen - best_entry.last_seen).total_seconds()
                if time_delta > 0:
                    st_score = reid_service.st_scorer.calculate_score(
                        best_entry.last_camera_id,
                        track.camera_id,
                        time_delta
                    )
                    if st_score > 0.3:
                        logger.info(
                            f"GlobalAssoc: Track {track.camera_id}:{track.local_id} -> "
                            f"TENTATIVE MATCH {best_id[:8]} (sim={best_sim:.3f}, st={st_score:.3f})"
                        )
                        return UUID(best_id)
        
        # No match - create new identity
        new_id = uuid4()
        reid_service.visual_matcher.add_to_gallery(
            str(new_id),
            avg_embedding,
            track.camera_id,
            track.first_seen
        )
        
        logger.info(
            f"GlobalAssoc: Track {track.camera_id}:{track.local_id} -> "
            f"NEW ID {str(new_id)[:8]}"
        )
        return new_id
    
    def _update_existing_identity(
        self,
        global_id: UUID,
        embedding: np.ndarray,
        camera_id: int,
        timestamp: datetime,
    ):
        """Update existing identity with new observation."""
        from app.services.reid_service import get_reid_service
        reid_service = get_reid_service()
        
        reid_service.visual_matcher.add_to_gallery(
            str(global_id),
            embedding,
            camera_id,
            timestamp
        )
    
    def end_track(self, camera_id: int, local_id: int):
        """Mark local track as ended."""
        key = (camera_id, local_id)
        
        # Clean up pending
        if key in self._pending:
            pending = self._pending[key]
            # If track had some frames but didn't mature, force query
            if pending.frame_count >= 2:
                global_id = self._perform_global_query(pending)
                self._local_to_global[key] = global_id
            del self._pending[key]
        
        # Note: We keep _local_to_global mapping for potential resume
    
    def get_global_id(self, camera_id: int, local_id: int) -> Optional[UUID]:
        """Get global ID for local track if assigned."""
        return self._local_to_global.get((camera_id, local_id))
    
    def clear(self):
        """Clear all state."""
        self._pending.clear()
        self._local_to_global.clear()


# Singleton
_global_associator: Optional[GlobalAssociator] = None


def get_global_associator() -> GlobalAssociator:
    """Get or create singleton GlobalAssociator."""
    global _global_associator
    if _global_associator is None:
        _global_associator = GlobalAssociator(
            maturation_frames=settings.association_maturation_frames if hasattr(settings, 'association_maturation_frames') else 5,
            match_threshold=settings.reid_match_threshold,
            new_id_threshold=settings.reid_new_threshold if hasattr(settings, 'reid_new_threshold') else 0.40,
        )
    return _global_associator
```

#### Step 2: Integrate with TrackingService

**File**: `app/services/tracking_service.py` [MODIFY]

```diff
# At imports
+ from app.core.reid.global_associator import get_global_associator

# In process_frame method, after extracting features
+ # Proactive global association
+ global_associator = get_global_associator()
+ 
+ for track in active_tracks:
+     if track.embedding is not None:
+         global_id = global_associator.update_track(
+             camera_id=camera_id,
+             local_id=track.track_id,
+             embedding=track.embedding,
+             timestamp=timestamp,
+         )
+         if global_id:
+             track.global_id = global_id
```

---

## 4. Testing & Verification

### 4.1 TensorRT Verification

```python
# test_tensorrt.py
import time
import numpy as np
from app.core.features.nvidia_reid_extractor import get_nvidia_extractor

extractor = get_nvidia_extractor()

# Warmup
dummy = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
for _ in range(10):
    extractor.extract(dummy)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    extractor.extract(dummy)
    times.append(time.perf_counter() - start)

print(f"Avg inference time: {np.mean(times)*1000:.2f}ms")
print(f"FPS potential: {1/np.mean(times):.0f}")
```

Expected: **< 5ms per extraction** (vs ~20ms without TensorRT)

### 4.2 Vector Search Verification

```python
# test_vector_search.py
import time
import numpy as np
from app.core.reid.vector_search import VectorIndex

index = VectorIndex(dimension=256)

# Add 1000 embeddings
for i in range(1000):
    emb = np.random.randn(256).astype(np.float32)
    emb /= np.linalg.norm(emb)
    index.add(f"id_{i}", emb)

# Benchmark search
query = np.random.randn(256).astype(np.float32)
query /= np.linalg.norm(query)

times = []
for _ in range(1000):
    start = time.perf_counter()
    results = index.search(query, k=5)
    times.append(time.perf_counter() - start)

print(f"Avg search time: {np.mean(times)*1000:.3f}ms")
print(f"Gallery size: {index.size}")
```

Expected: **< 1ms per search** (vs ~10ms for brute-force at 1000 identities)

### 4.3 Integration Test

```bash
# Run backend and observe logs
python -m uvicorn app.main:app --reload

# Look for:
# - "Using TensorRT-optimized providers for ReIDNet"
# - "Vector index enabled for O(log N) search"
# - "GlobalAssoc: Track X:Y -> MATCH/NEW ID"
```

---

## Summary Checklist

| Step | File | Action | Status |
|------|------|--------|--------|
| 1.1 | `app/core/inference/trt_config.py` | Create new file | ⬜ |
| 1.2 | `nvidia_reid_extractor.py` | Add TensorRT providers | ⬜ |
| 1.3 | `peoplenet_detector.py` | Add TensorRT providers | ⬜ |
| 2.1 | Install FAISS | `pip install faiss-gpu` | ⬜ |
| 2.2 | `app/core/reid/vector_search.py` | Create new file | ⬜ |
| 2.3 | `visual_matcher.py` | Integrate vector index | ⬜ |
| 3.1 | `app/core/reid/global_associator.py` | Create new file | ⬜ |
| 3.2 | `tracking_service.py` | Integrate global associator | ⬜ |
| 4.1 | Test TensorRT | Run benchmark | ⬜ |
| 4.2 | Test Vector Search | Run benchmark | ⬜ |
| 4.3 | Integration Test | End-to-end validation | ⬜ |
