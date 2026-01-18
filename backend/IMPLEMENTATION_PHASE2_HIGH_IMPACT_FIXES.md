# MTMCT High-Impact Fixes Implementation Plan
## Phase 2: Multi-Process Architecture, Topology-Filtered Search & DeepSORT Optimization

> **Objective**: Address high-impact architectural issues that limit scalability, reduce false positives, and ensure consistent embedding space usage.

---

## Table of Contents
1. [Multi-Process Stream Architecture](#1-multi-process-stream-architecture)
2. [Topology-Constrained Search](#2-topology-constrained-search)
3. [DeepSORT External Embedder Integration](#3-deepsort-external-embedder-integration)
4. [Testing & Verification](#4-testing--verification)

---

## 1. Multi-Process Stream Architecture

### 1.1 Problem Statement
Current single-threaded `_run_loop` in `stream_processor.py` processes cameras sequentially:
- One slow camera blocks all others
- GPU underutilized during I/O operations
- No parallel frame decoding

### 1.2 Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Multi-Process Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                          │
│  │ Camera  │  │ Camera  │  │ Camera  │   Ingestion Workers      │
│  │    1    │  │    2    │  │    N    │   (per-camera process)   │
│  └────┬────┘  └────┬────┘  └────┬────┘                          │
│       │            │            │                                │
│       ▼            ▼            ▼                                │
│  ┌──────────────────────────────────────┐                       │
│  │         Redis Stream / Queue          │   Frame Buffer       │
│  │    (camera_id, frame, timestamp)      │                      │
│  └──────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────┐                       │
│  │         GPU Inference Worker          │   Batch Processing   │
│  │   (Detection + Tracking + ReID)       │   (TensorRT)         │
│  └──────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────┐                       │
│  │          Results Queue                │   Track Events       │
│  └──────────────────────────────────────┘                       │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────┐                       │
│  │         FastAPI Broadcaster           │   WebSocket Push     │
│  └──────────────────────────────────────┘                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Implementation Steps

#### Step 1: Create Frame Queue Module

**File**: `app/workers/frame_queue.py` [NEW]

```python
"""
Frame Queue using Redis Streams

Provides high-performance frame buffering between ingestion and inference workers.
Uses Redis Streams for persistence and multi-consumer support.
"""

import pickle
import time
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False


@dataclass
class QueuedFrame:
    """Frame data in queue."""
    camera_id: int
    frame: np.ndarray  # BGR image
    timestamp: datetime
    frame_number: int
    source_id: str


class FrameQueue:
    """
    Redis Stream-based frame queue.
    
    Features:
    - Per-camera streams for isolation
    - Automatic trimming to prevent memory overflow
    - Consumer groups for scaling inference workers
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_stream_length: int = 100,  # Max frames per camera
        consumer_group: str = "inference_workers",
    ):
        self.max_length = max_stream_length
        self.consumer_group = consumer_group
        
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory queue")
            self._fallback_queues = {}
            self.redis = None
        else:
            self.redis = redis.from_url(redis_url, decode_responses=False)
            self._init_consumer_groups()
    
    def _init_consumer_groups(self):
        """Initialize consumer groups for each known camera."""
        # Groups are created lazily when cameras are added
        pass
    
    def _stream_key(self, camera_id: int) -> str:
        """Get Redis stream key for camera."""
        return f"frames:camera:{camera_id}"
    
    def push(self, frame_data: QueuedFrame) -> bool:
        """
        Push frame to queue.
        
        Args:
            frame_data: Frame data to queue
            
        Returns:
            True if successful
        """
        if self.redis is None:
            # Fallback to in-memory
            key = frame_data.camera_id
            if key not in self._fallback_queues:
                from queue import Queue
                self._fallback_queues[key] = Queue(maxsize=self.max_length)
            q = self._fallback_queues[key]
            if q.full():
                try:
                    q.get_nowait()  # Drop oldest
                except:
                    pass
            q.put(frame_data)
            return True
        
        try:
            stream_key = self._stream_key(frame_data.camera_id)
            
            # Serialize frame data
            data = {
                b'camera_id': str(frame_data.camera_id).encode(),
                b'frame': pickle.dumps(frame_data.frame),
                b'timestamp': frame_data.timestamp.isoformat().encode(),
                b'frame_number': str(frame_data.frame_number).encode(),
                b'source_id': frame_data.source_id.encode(),
            }
            
            # Add to stream with auto-trim
            self.redis.xadd(
                stream_key,
                data,
                maxlen=self.max_length,
                approximate=True,
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to push frame: {e}")
            return False
    
    def pop(
        self,
        camera_id: int,
        consumer_name: str = "worker_0",
        timeout_ms: int = 1000,
    ) -> Optional[QueuedFrame]:
        """
        Pop frame from queue.
        
        Uses consumer groups for reliable delivery.
        """
        if self.redis is None:
            # Fallback
            if camera_id in self._fallback_queues:
                try:
                    return self._fallback_queues[camera_id].get(timeout=timeout_ms/1000)
                except:
                    return None
            return None
        
        try:
            stream_key = self._stream_key(camera_id)
            
            # Ensure consumer group exists
            try:
                self.redis.xgroup_create(stream_key, self.consumer_group, id='0', mkstream=True)
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            
            # Read from stream
            result = self.redis.xreadgroup(
                self.consumer_group,
                consumer_name,
                {stream_key: '>'},
                count=1,
                block=timeout_ms,
            )
            
            if not result:
                return None
            
            stream_name, messages = result[0]
            if not messages:
                return None
            
            msg_id, data = messages[0]
            
            # Deserialize
            frame_data = QueuedFrame(
                camera_id=int(data[b'camera_id']),
                frame=pickle.loads(data[b'frame']),
                timestamp=datetime.fromisoformat(data[b'timestamp'].decode()),
                frame_number=int(data[b'frame_number']),
                source_id=data[b'source_id'].decode(),
            )
            
            # Acknowledge message
            self.redis.xack(stream_key, self.consumer_group, msg_id)
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Failed to pop frame: {e}")
            return None
    
    def pop_batch(
        self,
        camera_ids: List[int],
        batch_size: int = 8,
        consumer_name: str = "worker_0",
        timeout_ms: int = 100,
    ) -> List[QueuedFrame]:
        """
        Pop batch of frames from multiple cameras.
        
        Used for batch GPU inference.
        """
        frames = []
        
        for camera_id in camera_ids:
            while len(frames) < batch_size:
                frame = self.pop(camera_id, consumer_name, timeout_ms=10)
                if frame is None:
                    break
                frames.append(frame)
        
        return frames
    
    def get_queue_lengths(self) -> dict:
        """Get current queue length per camera."""
        if self.redis is None:
            return {k: v.qsize() for k, v in self._fallback_queues.items()}
        
        result = {}
        for key in self.redis.scan_iter(match="frames:camera:*"):
            camera_id = int(key.decode().split(":")[-1])
            result[camera_id] = self.redis.xlen(key)
        return result


# Singleton
_frame_queue: Optional[FrameQueue] = None


def get_frame_queue() -> FrameQueue:
    """Get or create singleton FrameQueue."""
    global _frame_queue
    if _frame_queue is None:
        from app.config import settings
        _frame_queue = FrameQueue(
            redis_url=settings.redis_url if hasattr(settings, 'redis_url') else "redis://localhost:6379"
        )
    return _frame_queue
```

#### Step 2: Create Ingestion Worker

**File**: `app/workers/ingestion_worker.py` [NEW]

```python
"""
Camera Ingestion Worker

Runs as a separate process per camera to capture and queue frames.
Decouples frame capture from GPU inference.
"""

import multiprocessing as mp
import time
from datetime import datetime
from typing import Optional
import cv2
from loguru import logger


def run_ingestion_worker(
    camera_id: int,
    source_url: str,
    source_id: str,
    redis_url: str,
    target_fps: int = 30,
    stop_event: Optional[mp.Event] = None,
):
    """
    Ingestion worker process entry point.
    
    Args:
        camera_id: Camera identifier
        source_url: RTSP URL or video file path
        source_id: Source UUID
        redis_url: Redis connection URL
        target_fps: Target capture FPS
        stop_event: Event to signal shutdown
    """
    from app.workers.frame_queue import FrameQueue, QueuedFrame
    
    logger.info(f"Ingestion worker starting: camera={camera_id}, source={source_url}")
    
    # Initialize queue
    queue = FrameQueue(redis_url=redis_url)
    
    # Open video source
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source_url}")
        return
    
    # Get source properties
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / min(target_fps, source_fps)
    
    frame_number = 0
    last_frame_time = 0
    
    try:
        while stop_event is None or not stop_event.is_set():
            # Rate limiting
            current_time = time.time()
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)
                continue
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                # Video ended or connection lost
                if isinstance(source_url, str) and source_url.startswith("rtsp"):
                    # RTSP - try reconnect
                    logger.warning(f"RTSP reconnecting: {camera_id}")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(source_url)
                    continue
                else:
                    # Video file - loop or stop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            
            # Queue frame
            frame_data = QueuedFrame(
                camera_id=camera_id,
                frame=frame,
                timestamp=datetime.utcnow(),
                frame_number=frame_number,
                source_id=source_id,
            )
            
            queue.push(frame_data)
            
            frame_number += 1
            last_frame_time = current_time
            
    except Exception as e:
        logger.error(f"Ingestion worker error: {e}")
    finally:
        cap.release()
        logger.info(f"Ingestion worker stopped: camera={camera_id}")


class IngestionManager:
    """
    Manages ingestion workers for multiple cameras.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._workers: dict = {}  # camera_id -> (process, stop_event)
    
    def start_camera(
        self,
        camera_id: int,
        source_url: str,
        source_id: str,
        target_fps: int = 30,
    ):
        """Start ingestion worker for camera."""
        if camera_id in self._workers:
            self.stop_camera(camera_id)
        
        stop_event = mp.Event()
        process = mp.Process(
            target=run_ingestion_worker,
            args=(camera_id, source_url, source_id, self.redis_url, target_fps, stop_event),
            daemon=True,
        )
        process.start()
        
        self._workers[camera_id] = (process, stop_event)
        logger.info(f"Started ingestion worker: camera={camera_id}, pid={process.pid}")
    
    def stop_camera(self, camera_id: int):
        """Stop ingestion worker for camera."""
        if camera_id not in self._workers:
            return
        
        process, stop_event = self._workers[camera_id]
        stop_event.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
        
        del self._workers[camera_id]
        logger.info(f"Stopped ingestion worker: camera={camera_id}")
    
    def stop_all(self):
        """Stop all ingestion workers."""
        for camera_id in list(self._workers.keys()):
            self.stop_camera(camera_id)


# Singleton
_ingestion_manager: Optional[IngestionManager] = None


def get_ingestion_manager() -> IngestionManager:
    """Get or create singleton IngestionManager."""
    global _ingestion_manager
    if _ingestion_manager is None:
        from app.config import settings
        _ingestion_manager = IngestionManager(
            redis_url=settings.redis_url if hasattr(settings, 'redis_url') else "redis://localhost:6379"
        )
    return _ingestion_manager
```

#### Step 3: Create GPU Inference Worker

**File**: `app/workers/inference_worker.py` [NEW]

```python
"""
GPU Inference Worker

Consumes frames from queue, runs batch inference (detection + tracking + ReID),
and publishes results to broadcast queue.
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from loguru import logger

from app.workers.frame_queue import get_frame_queue, QueuedFrame


class InferenceWorker:
    """
    GPU-bound inference worker with batch processing.
    
    Accumulates frames from multiple cameras and processes
    them in batches for better GPU utilization.
    """
    
    def __init__(
        self,
        batch_size: int = 8,
        batch_timeout_ms: int = 50,
        worker_name: str = "gpu_worker_0",
    ):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.worker_name = worker_name
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._active_cameras: List[int] = []
        
        # Stats
        self._frames_processed = 0
        self._batches_processed = 0
        self._last_batch_time = 0
    
    def start(self, camera_ids: List[int]):
        """Start inference worker."""
        if self._running:
            return
        
        self._active_cameras = camera_ids
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"InferenceWorker started: cameras={camera_ids}")
    
    def stop(self):
        """Stop inference worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("InferenceWorker stopped")
    
    def _run_loop(self):
        """Main inference loop with batch accumulation."""
        from app.services.tracking_service import get_tracking_service
        from app.core.reid.global_associator import get_global_associator
        
        tracking_service = get_tracking_service()
        global_associator = get_global_associator()
        frame_queue = get_frame_queue()
        
        batch: List[QueuedFrame] = []
        batch_start_time = time.time()
        
        while self._running:
            try:
                # Collect frames into batch
                for camera_id in self._active_cameras:
                    frame = frame_queue.pop(
                        camera_id,
                        consumer_name=self.worker_name,
                        timeout_ms=10,
                    )
                    if frame:
                        batch.append(frame)
                        if len(batch) >= self.batch_size:
                            break
                
                # Process batch if full or timeout
                current_time = time.time()
                batch_age_ms = (current_time - batch_start_time) * 1000
                
                if len(batch) >= self.batch_size or (batch and batch_age_ms >= self.batch_timeout_ms):
                    self._process_batch(batch, tracking_service, global_associator)
                    batch = []
                    batch_start_time = current_time
                elif not batch:
                    # No frames, short sleep
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                time.sleep(0.1)
    
    def _process_batch(
        self,
        batch: List[QueuedFrame],
        tracking_service,
        global_associator,
    ):
        """Process a batch of frames."""
        start_time = time.perf_counter()
        
        for frame_data in batch:
            try:
                # Run detection + tracking
                tracks, features = tracking_service.process_frame(
                    camera_id=frame_data.camera_id,
                    frame=frame_data.frame,
                    timestamp=frame_data.timestamp,
                    extract_features=True,
                )
                
                # Proactive global association
                for track_id, embedding in features.items():
                    global_id = global_associator.update_track(
                        camera_id=frame_data.camera_id,
                        local_id=track_id,
                        embedding=embedding,
                        timestamp=frame_data.timestamp,
                    )
                    
                    # Attach global ID to track for broadcasting
                    for track in tracks:
                        if track.track_id == track_id and global_id:
                            track.global_id = global_id
                
                # Broadcast results
                self._broadcast_results(frame_data, tracks)
                
                self._frames_processed += 1
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
        
        self._batches_processed += 1
        self._last_batch_time = time.perf_counter() - start_time
    
    def _broadcast_results(self, frame_data: QueuedFrame, tracks: List):
        """Broadcast results via WebSocket."""
        # Delegate to existing broadcast mechanism
        from app.services.stream_manager import get_stream_manager
        
        try:
            stream_manager = get_stream_manager()
            # Format and broadcast...
            # (Integration with existing WebSocket broadcasting)
        except Exception as e:
            logger.debug(f"Broadcast skipped: {e}")
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            "frames_processed": self._frames_processed,
            "batches_processed": self._batches_processed,
            "last_batch_time_ms": self._last_batch_time * 1000,
            "avg_batch_fps": self._frames_processed / max(1, self._batches_processed) * (1 / max(0.001, self._last_batch_time)),
        }


# Singleton
_inference_worker: Optional[InferenceWorker] = None


def get_inference_worker() -> InferenceWorker:
    global _inference_worker
    if _inference_worker is None:
        _inference_worker = InferenceWorker()
    return _inference_worker
```

#### Step 4: Update StreamManager Integration

**File**: `app/services/stream_manager.py` [MODIFY]

Add integration with new multi-process architecture:

```diff
+ from app.workers.ingestion_worker import get_ingestion_manager
+ from app.workers.inference_worker import get_inference_worker

  class StreamManager:
      def start_source(self, source_id: str, ...):
          ...
+         # Start ingestion worker for this camera
+         ingestion_manager = get_ingestion_manager()
+         ingestion_manager.start_camera(
+             camera_id=camera_id,
+             source_url=source.url,
+             source_id=str(source.id),
+         )
          
      def stop_source(self, source_id: str):
          ...
+         # Stop ingestion worker
+         ingestion_manager = get_ingestion_manager()
+         ingestion_manager.stop_camera(camera_id)
```

---

## 2. Topology-Constrained Search

### 2.1 Problem Statement
Current matching considers ALL identities regardless of physical feasibility. This causes false matches between cameras that are far apart.

### 2.2 Implementation Steps

#### Step 1: Enhance CameraTopology

**File**: `app/core/reid/topology.py` [MODIFY]

Add methods for search filtering:

```python
def get_plausible_sources(
    self,
    target_camera_id: int,
    time_window: float,
    max_hops: int = 2,
) -> set:
    """
    Get cameras that could plausibly feed into target camera.
    
    Args:
        target_camera_id: Camera where new detection appeared
        time_window: Time since last observation (seconds)
        max_hops: Maximum transition hops to consider
        
    Returns:
        Set of camera IDs that could be sources
    """
    plausible = {target_camera_id}  # Same camera always plausible
    
    if target_camera_id not in self.cameras:
        return plausible
    
    target_pos = self.cameras[target_camera_id]
    
    for camera_id, pos in self.cameras.items():
        if camera_id == target_camera_id:
            continue
        
        # Check if transition is possible within time window
        distance = self._haversine_distance(
            pos.lat, pos.lon,
            target_pos.lat, target_pos.lon
        )
        
        # Assume walking speed ~1.4 m/s (5 km/h) with 50% margin
        min_travel_time = distance / 2.1  # meters / (m/s)
        max_travel_time = distance / 0.7  # Allow for slow walking
        
        if min_travel_time <= time_window <= max_travel_time * 2:
            plausible.add(camera_id)
    
    return plausible


def filter_candidates_by_topology(
    self,
    candidates: List[Tuple[str, float, Any]],
    target_camera_id: int,
    time_window: float,
) -> List[Tuple[str, float, Any]]:
    """
    Filter ReID candidates by topological plausibility.
    
    Args:
        candidates: List of (global_id, similarity, entry) tuples
        target_camera_id: Current camera
        time_window: Time since candidate's last observation
        
    Returns:
        Filtered candidates list
    """
    plausible_cameras = self.get_plausible_sources(target_camera_id, time_window)
    
    filtered = []
    for global_id, sim, entry in candidates:
        if entry.last_camera_id in plausible_cameras:
            filtered.append((global_id, sim, entry))
        else:
            # Penalize but don't exclude (might be edge case)
            filtered.append((global_id, sim * 0.7, entry))
    
    # Re-sort after penalty
    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered
```

#### Step 2: Integrate with ReIDService

**File**: `app/services/reid_service.py` [MODIFY]

```diff
  async def match_identity(self, camera_id: int, embedding: np.ndarray, timestamp: datetime, ...):
      ...
      # Get candidate matches from visual matcher
      visual_matches = self.visual_matcher.match(embedding, top_k=top_k)
      
+     # Apply topology filtering
+     if visual_matches:
+         time_window = 300  # 5 minutes default
+         visual_matches = self.topology.filter_candidates_by_topology(
+             visual_matches,
+             target_camera_id=camera_id,
+             time_window=time_window,
+         )
```

---

## 3. DeepSORT External Embedder Integration

### 3.1 Problem Statement
DeepSORT may use its own internal embedding network, causing:
- Wasted GPU cycles (duplicate feature extraction)
- Embedding space mismatch (different models = different similarity)
- Inconsistent appearance matching between local and global tracking

### 3.2 Implementation Steps

#### Step 1: Create Custom DeepSORT Wrapper

**File**: `app/core/tracking/deepsort_wrapper.py` [NEW]

```python
"""
DeepSORT Wrapper with External Embeddings

Wraps DeepSORT to accept pre-computed embeddings from NVIDIA ReIDNet
instead of using its internal feature extractor.
"""

from typing import List, Optional, Tuple
import numpy as np
from loguru import logger

# Import DeepSORT components
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from deep_sort_realtime.deep_sort.track import Track
    DEEPSORT_AVAILABLE = True
except ImportError:
    try:
        from boxmot import DeepOCSORT
        DEEPSORT_AVAILABLE = True
    except ImportError:
        DEEPSORT_AVAILABLE = False
        logger.warning("DeepSORT not available")


class ExternalEmbedderDeepSORT:
    """
    DeepSORT wrapper that accepts external embeddings.
    
    Instead of extracting features internally, this wrapper
    accepts pre-computed embeddings from NVIDIA ReIDNet.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
    ):
        """
        Initialize DeepSORT with external embedder.
        
        Args:
            max_age: Max frames to keep track without detection
            n_init: Frames needed to confirm track
            max_iou_distance: Max IOU distance for matching
            max_cosine_distance: Max cosine distance for appearance matching
            nn_budget: Max embeddings to store per track
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        
        # Initialize tracker without internal embedder
        self._init_tracker()
        
        logger.info(
            f"ExternalEmbedderDeepSORT initialized: "
            f"max_age={max_age}, n_init={n_init}"
        )
    
    def _init_tracker(self):
        """Initialize the underlying tracker."""
        try:
            # Option 1: deep-sort-realtime
            self.tracker = DeepSort(
                max_age=self.max_age,
                n_init=self.n_init,
                max_iou_distance=self.max_iou_distance,
                max_cosine_distance=self.max_cosine_distance,
                nn_budget=self.nn_budget,
                embedder=None,  # Disable internal embedder
                embedder_gpu=False,
            )
            self._tracker_type = "deep_sort_realtime"
        except Exception as e:
            logger.warning(f"deep-sort-realtime init failed: {e}")
            # Fallback to boxmot
            try:
                self.tracker = DeepOCSORT(
                    model_weights=None,  # No internal ReID
                    device='cuda',
                    fp16=True,
                )
                self._tracker_type = "boxmot"
            except Exception as e2:
                logger.error(f"All tracker backends failed: {e2}")
                self.tracker = None
                self._tracker_type = None
    
    def update(
        self,
        detections: List[Tuple[List[float], float, str]],
        embeddings: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracker with new detections and pre-computed embeddings.
        
        Args:
            detections: List of (bbox, confidence, class) tuples
                       bbox format: [x1, y1, x2, y2]
            embeddings: Pre-computed embeddings from ReIDNet, shape (N, 256)
            frame: Current frame (optional, for visualization)
            
        Returns:
            List of active tracks
        """
        if self.tracker is None:
            return []
        
        if len(detections) == 0:
            # No detections - just predict
            if self._tracker_type == "deep_sort_realtime":
                return self.tracker.update_tracks([], frame=frame)
            else:
                return []
        
        # Format detections for tracker
        if self._tracker_type == "deep_sort_realtime":
            # deep-sort-realtime format: list of (bbox, conf, class, embedding)
            formatted = []
            for i, (bbox, conf, cls) in enumerate(detections):
                emb = embeddings[i] if embeddings is not None and i < len(embeddings) else None
                formatted.append((bbox, conf, cls, emb))
            
            tracks = self.tracker.update_tracks(formatted, frame=frame)
            
        else:  # boxmot
            # boxmot format: numpy array
            dets_array = np.array([
                [*bbox, conf] for bbox, conf, cls in detections
            ], dtype=np.float32)
            
            tracks = self.tracker.update(dets_array, frame)
        
        return tracks
    
    def get_active_tracks(self) -> List[Track]:
        """Get currently active (confirmed) tracks."""
        if self.tracker is None:
            return []
        
        if self._tracker_type == "deep_sort_realtime":
            return [t for t in self.tracker.tracks if t.is_confirmed()]
        else:
            return list(self.tracker.active_tracks)
    
    def reset(self):
        """Reset tracker state."""
        self._init_tracker()


def create_tracker(
    max_age: int = 30,
    n_init: int = 3,
    max_iou_distance: float = 0.7,
) -> ExternalEmbedderDeepSORT:
    """Factory function to create external embedder tracker."""
    return ExternalEmbedderDeepSORT(
        max_age=max_age,
        n_init=n_init,
        max_iou_distance=max_iou_distance,
    )
```

#### Step 2: Update TrackingService

**File**: `app/services/tracking_service.py` [MODIFY]

```diff
- from deep_sort_realtime.deepsort_tracker import DeepSort as DeepSORTTracker
+ from app.core.tracking.deepsort_wrapper import ExternalEmbedderDeepSORT, create_tracker

  class CameraState:
      def __init__(self, camera_id: int):
          self.camera_id = camera_id
-         self.tracker = DeepSORTTracker(...)
+         self.tracker = create_tracker(
+             max_age=settings.deepsort_max_age,
+             n_init=settings.deepsort_n_init,
+             max_iou_distance=settings.deepsort_max_iou_distance,
+         )

  # In process_frame method:
  def process_frame(self, camera_id, frame, timestamp, extract_features=True):
      ...
      # Extract features FIRST
      if extract_features and len(person_detections) > 0:
          embeddings = self._extract_embeddings(frame, person_detections)
      else:
          embeddings = None
      
      # Update tracker with EXTERNAL embeddings
-     raw_tracks = state.tracker.update_tracks(...)
+     raw_tracks = state.tracker.update(
+         detections=formatted_detections,
+         embeddings=embeddings,
+         frame=frame,
+     )
```

---

## 4. Testing & Verification

### 4.1 Multi-Process Architecture Test

```python
# test_multiprocess.py
import time
from app.workers.ingestion_worker import get_ingestion_manager
from app.workers.inference_worker import get_inference_worker
from app.workers.frame_queue import get_frame_queue

# Start ingestion for test camera
manager = get_ingestion_manager()
manager.start_camera(
    camera_id=1,
    source_url="test_video.mp4",
    source_id="test-source-001",
)

# Check queue is receiving frames
time.sleep(2)
queue = get_frame_queue()
lengths = queue.get_queue_lengths()
print(f"Queue lengths: {lengths}")
assert lengths.get(1, 0) > 0, "No frames queued"

# Start inference worker
worker = get_inference_worker()
worker.start(camera_ids=[1])

time.sleep(5)
stats = worker.get_stats()
print(f"Inference stats: {stats}")
assert stats["frames_processed"] > 0, "No frames processed"

# Cleanup
worker.stop()
manager.stop_all()
```

### 4.2 Topology Filtering Test

```python
# test_topology.py
from app.core.reid.topology import CameraTopology

topo = CameraTopology()

# Add cameras at known locations
topo.add_camera(1, 12.9716, 77.5946)  # Bangalore
topo.add_camera(2, 12.9720, 77.5950)  # 50m away
topo.add_camera(3, 12.9800, 77.5946)  # 1km away

# Test plausible sources
plausible_10s = topo.get_plausible_sources(2, time_window=10)
plausible_120s = topo.get_plausible_sources(2, time_window=120)
plausible_600s = topo.get_plausible_sources(2, time_window=600)

print(f"Plausible in 10s: {plausible_10s}")   # Should include 1, 2
print(f"Plausible in 120s: {plausible_120s}") # Should include 1, 2
print(f"Plausible in 600s: {plausible_600s}") # Should include all

assert 1 in plausible_10s
assert 3 not in plausible_10s  # Too far for 10s
assert 3 in plausible_600s     # Reachable in 10min
```

### 4.3 DeepSORT External Embedder Test

```python
# test_deepsort_external.py
import numpy as np
from app.core.tracking.deepsort_wrapper import create_tracker

tracker = create_tracker(max_age=30, n_init=3)

# Simulate detections
detections = [
    ([100, 100, 200, 300], 0.9, 'person'),
    ([400, 100, 500, 300], 0.85, 'person'),
]

# Pre-computed embeddings (256-dim)
embeddings = np.random.randn(2, 256).astype(np.float32)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Update tracker
tracks = tracker.update(detections, embeddings)
print(f"Tracks after frame 1: {len(tracks)}")

# Second frame - same positions
tracks = tracker.update(detections, embeddings)
tracks = tracker.update(detections, embeddings)
tracks = tracker.update(detections, embeddings)

# After n_init frames, tracks should be confirmed
confirmed = tracker.get_active_tracks()
print(f"Confirmed tracks: {len(confirmed)}")
assert len(confirmed) == 2
```

---

## Summary Checklist

| Step | File | Action | Status |
|------|------|--------|--------|
| 1.1 | `app/workers/frame_queue.py` | Create new file | ⬜ |
| 1.2 | `app/workers/ingestion_worker.py` | Create new file | ⬜ |
| 1.3 | `app/workers/inference_worker.py` | Create new file | ⬜ |
| 1.4 | `stream_manager.py` | Integrate multi-process | ⬜ |
| 2.1 | `app/core/reid/topology.py` | Add filtering methods | ⬜ |
| 2.2 | `reid_service.py` | Integrate topology filter | ⬜ |
| 3.1 | `app/core/tracking/deepsort_wrapper.py` | Create new file | ⬜ |
| 3.2 | `tracking_service.py` | Use external embedder | ⬜ |
| 4.1 | Test multi-process | Run test script | ⬜ |
| 4.2 | Test topology | Run test script | ⬜ |
| 4.3 | Test DeepSORT | Run test script | ⬜ |

---

## Dependencies to Install

```bash
# For Redis Streams (already likely installed)
pip install redis

# For multiprocessing enhancements (optional)
pip install cloudpickle

# Verify DeepSORT installation
pip install deep-sort-realtime  # or boxmot
```

---

## Configuration Updates

Add to `app/config.py`:

```python
# Multi-process settings
redis_url: str = "redis://localhost:6379"
ingestion_target_fps: int = 30
inference_batch_size: int = 8
inference_batch_timeout_ms: int = 50

# Global association settings
association_maturation_frames: int = 5
association_maturation_timeout: float = 2.0

# Topology settings
topology_max_hops: int = 2
topology_walking_speed: float = 1.4  # m/s
```
