# MCMT-ReID System Documentation

> **Multi-Camera Multi-Target Re-Identification System**  
> Complete technical documentation of all modules, thresholds, and methodologies.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [File Structure](#file-structure)
3. [Configuration & Thresholds](#configuration--thresholds)
4. [Detection Module](#detection-module)
5. [Tracking Module](#tracking-module)
6. [Feature Extraction Module](#feature-extraction-module)
7. [Re-Identification Module](#re-identification-module)
8. [Services Layer](#services-layer)
9. [API Endpoints](#api-endpoints)
10. [Data Flow](#data-flow)

---

## System Overview

The MCMT-ReID system is a **real-time person tracking and re-identification** platform designed for surveillance applications. It can:

- Detect people in video streams using YOLO
- Track individuals within a single camera using DeepSORT
- Re-identify the same person across different cameras using visual similarity and spatial-temporal constraints
- Search for people by uploading an image
- Display movement paths on a map

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Camera Support** | Track the same person across unlimited cameras |
| **Real-Time Processing** | Live video stream processing with GPU acceleration |
| **Face + Body Fusion** | Combines face and body features for robust identification |
| **Spatial-Temporal Scoring** | Uses camera topology and transition times to improve matching |
| **Quality-Aware Gallery** | Only stores high-quality embeddings to prevent feature drift |
| **K-Reciprocal Reranking** | Advanced matching algorithm for better accuracy |
| **Image Search** | Upload a photo to find matching person in the system |

---

## File Structure

```
backend/
├── app/
│   ├── config.py                 # All configurable thresholds and settings
│   ├── main.py                   # FastAPI application entry point
│   │
│   ├── api/v1/                   # REST API endpoints
│   │   ├── cameras.py            # Camera management endpoints
│   │   ├── health.py             # Health check endpoint
│   │   ├── realtime.py           # WebSocket for real-time updates
│   │   ├── streams.py            # Video stream management
│   │   └── tracks.py             # Track query and image search
│   │
│   ├── core/                     # Core ML components
│   │   ├── detection/            # Person detection
│   │   │   └── yolo_detector.py  # YOLOv8/v10 and ONNX detectors
│   │   │
│   │   ├── tracking/             # Single-camera tracking
│   │   │   ├── deepsort.py       # DeepSORT tracker implementation
│   │   │   └── kalman.py         # Kalman filter for motion prediction
│   │   │
│   │   ├── features/             # Feature extraction
│   │   │   ├── osnet_extractor.py    # Body feature extraction (OSNet)
│   │   │   └── face_extractor.py     # Face detection and embedding
│   │   │
│   │   ├── reid/                 # Cross-camera re-identification
│   │   │   ├── visual_matcher.py # Gallery and visual similarity matching
│   │   │   ├── st_scorer.py      # Spatial-temporal scoring
│   │   │   └── topology.py       # Camera network graph
│   │   │
│   │   └── utils/
│   │       └── quality.py        # Image quality assessment
│   │
│   ├── services/                 # Business logic layer
│   │   ├── reid_service.py       # ReID orchestration
│   │   ├── tracking_service.py   # Tracking orchestration
│   │   ├── stream_manager.py     # Video stream management
│   │   └── track_store.py        # Track data storage
│   │
│   └── workers/                  # Background processors
│       ├── stream_processor.py   # Main video processing loop
│       └── frame_processor.py    # Batch frame processor
│
└── model_weights/                # Model files location
    ├── yolov8n.pt               # YOLOv8-Nano weights
    ├── yolov10n.pt              # YOLOv10-Nano weights (NMS-free)
    └── osnet_x1_0.onnx          # OSNet ONNX model
```

---

## Configuration & Thresholds

All thresholds are configurable via environment variables or `.env` file.

### File: `app/config.py`

#### Detection Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `yolo_model_path` | `model_weights/yolov8n.pt` | Path to YOLO weights |
| `yolo_confidence` | `0.5` | Minimum detection confidence (0-1) |
| `yolo_iou_threshold` | `0.45` | NMS IoU threshold for overlapping boxes |
| `yolo_version` | `auto` | YOLO version: `v8`, `v10`, or `auto` |

#### Tracking Settings (DeepSORT)

| Setting | Default | Description |
|---------|---------|-------------|
| `deepsort_max_age` | `30` | Frames before deleting lost track |
| `deepsort_n_init` | `3` | Consecutive detections to confirm track |
| `deepsort_max_iou_distance` | `0.7` | Max IoU distance for unconfirmed tracks |
| `deepsort_backbone` | `resnet50` | Appearance backbone: `resnet50`, `osnet`, `resnet18` |

#### ReID Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `reid_match_threshold` | `0.3` | Base threshold for visual matching |
| `reid_embedding_dim` | `512` | Feature vector dimension |
| `use_multi_scale_reid` | `True` | Extract from multiple OSNet layers |

#### Face ReID Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_face_reid` | `True` | Enable face+body fusion |
| `face_weight` | `0.4` | Face contribution in fusion |
| `body_weight` | `0.6` | Body contribution in fusion |

#### Spatial-Temporal Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `st_weight` | `0.5` | Weight of ST score in joint matching |
| `max_transition_time` | `300.0` | Max seconds for cross-camera transition |

#### Quality Control Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `quality_min_sharpness` | `60.0` | Laplacian variance threshold for blur |
| `quality_min_size` | `40` | Minimum crop size (pixels) |
| `min_thumbnail_quality` | `0.3` | Minimum quality to save thumbnail |
| `search_threshold` | `0.4` | Threshold for image search results |

#### ONNX Runtime Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `use_onnx` | `True` | Use ONNX Runtime for inference |
| `onnx_gpu_mem_limit` | `4` | GPU memory limit (GB) |

---

## Detection Module

### File: `app/core/detection/yolo_detector.py`

Provides person detection using YOLO models.

### Classes

#### `Detection` (dataclass)
```python
@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float                         # 0.0 - 1.0
    class_id: int = 0                         # 0 = person (COCO)
```

Properties:
- `x1`, `y1`, `x2`, `y2`: Bounding box corners
- `width`, `height`: Box dimensions
- `center`: Center point (x, y)
- `area`: Box area in pixels
- `to_tlwh()`: Convert to (top-left-x, top-left-y, width, height)
- `to_xyah()`: Convert to (center-x, center-y, aspect-ratio, height)

---

#### `YOLOv10Detector`
**NMS-Free detector using YOLOv10.**

YOLOv10 eliminates Non-Maximum Suppression through one-to-one matching, resulting in:
- 15-20x faster end-to-end inference
- More stable bounding boxes (less flicker)
- Better for tracking (cleaner detections)

```python
detector = YOLOv10Detector(
    model_path="yolov10n.pt",
    confidence=0.5,
    device="cuda"
)
detections = detector.detect(frame)
```

---

#### `YOLODetector`
**PyTorch-based YOLOv8 detector.**

```python
detector = YOLODetector(
    model_path="yolov8n.pt",
    confidence=0.5,
    iou_threshold=0.45,
    device="cuda"
)
```

Methods:
- `detect(frame)`: Detect persons in single frame
- `detect_batch(frames)`: Batch detection
- `crop_detections(frame, detections, padding=0.1)`: Crop detected persons

---

#### `YOLOOnnxDetector`
**ONNX Runtime detector with CUDA Execution Provider.**

Faster than PyTorch on NVIDIA GPUs.

```python
detector = YOLOOnnxDetector(
    model_path="yolov8n.onnx",
    confidence=0.5,
    iou_threshold=0.45,
    device="cuda"
)
```

---

#### `get_detector()` Factory Function
Automatically selects the best available detector:

1. If YOLOv10 model detected → `YOLOv10Detector`
2. If ONNX available → `YOLOOnnxDetector`
3. Fallback → `YOLODetector`

---

## Tracking Module

### File: `app/core/tracking/deepsort.py`

Implements DeepSORT algorithm for single-camera multi-object tracking.

### Track States

```python
class TrackState(Enum):
    TENTATIVE = 1   # New, unconfirmed track
    CONFIRMED = 2   # Confirmed after n_init hits
    DELETED = 3     # Marked for removal
```

### Classes

#### `Track` (dataclass)
Represents a single tracked object.

```python
@dataclass
class Track:
    track_id: int              # Unique ID within camera
    mean: np.ndarray           # Kalman state (8,): [x, y, a, h, vx, vy, va, vh]
    covariance: np.ndarray     # Kalman covariance (8, 8)
    n_init: int                # Frames to confirm
    max_age: int               # Max frames without update
    state: TrackState          # Current lifecycle state
    hits: int                  # Successful update count
    time_since_update: int     # Frames since last match
    features: List[np.ndarray] # Appearance feature history
    global_id: Optional[str]   # Cross-camera ID (from ReID)
```

Methods:
- `to_tlwh()`: Get bounding box as (x, y, w, h)
- `to_tlbr()`: Get bounding box as (x1, y1, x2, y2)
- `predict(kf)`: Kalman prediction step
- `update(kf, detection, feature)`: Kalman update with new detection
- `mark_missed()`: Mark track as missed for this frame

---

#### `DeepSORTTracker`
Main tracker class.

```python
tracker = DeepSORTTracker(
    max_iou_distance=0.7,      # IoU threshold for unconfirmed tracks
    max_age=30,                 # Delete after 30 missed frames
    n_init=3,                   # Confirm after 3 consecutive hits
    metric="cosine",            # Distance metric for appearance
    matching_threshold=0.4,     # Max appearance distance
)
```

**Tracking Loop:**
```python
# Each frame:
tracker.predict()                              # Kalman predict all tracks
active_tracks = tracker.update(detections, features)  # Match and update
```

**Matching Algorithm:**
1. **Cascade Matching**: Match confirmed tracks using appearance features + motion gating
2. **IoU Matching**: Match remaining detections using bounding box overlap
3. **Track Management**: Create new tracks, mark missed, delete stale

**Chi-squared Gating**: Uses Mahalanobis distance to reject impossible matches based on motion model. Threshold: `9.4877` (95% confidence, 4 degrees of freedom).

---

### File: `app/core/tracking/kalman.py`

Kalman filter for motion prediction.

**State Space (8-dimensional):**
```
[x, y, a, h, vx, vy, va, vh]
```
- `(x, y)`: Bounding box center
- `a`: Aspect ratio (width/height)
- `h`: Height
- `(vx, vy, va, vh)`: Velocities

**Motion Model:** Constant velocity

**Methods:**
- `initiate(measurement)`: Create initial state from first detection
- `predict(mean, covariance)`: Predict next state
- `update(mean, covariance, measurement)`: Update with observation
- `gating_distance(mean, covariance, measurements)`: Mahalanobis distance for gating

---

## Feature Extraction Module

### File: `app/core/features/osnet_extractor.py`

Extracts 512-dimensional embeddings from person crops for re-identification.

### Classes

#### `OSNetExtractor`
Standard OSNet feature extractor.

```python
extractor = OSNetExtractor(
    model_name="osnet_x1_0",   # OSNet variant
    model_path=None,           # Custom weights (optional)
    device="cuda",
    pretrained=True
)

embedding = extractor.extract(person_crop)  # Shape: (512,)
```

**Input Size:** 256 x 128 (height x width)

**Preprocessing:**
1. Resize to 256x128
2. Convert BGR → RGB
3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

#### `MultiScaleOSNetExtractor`
**Advanced extractor that pools from multiple OSNet layers.**

Captures both coarse features (color, shape) and fine features (textures, logos) to reduce false positives when people wear similar clothes.

**OSNet Architecture Layers:**
| Layer | Channels | Captured Features |
|-------|----------|-------------------|
| conv3 | 256 | Edges, textures, logos |
| conv4 | 384 | Body parts, accessories |
| conv5 | 512 | Overall appearance (color, shape) |

**Fusion Process:**
1. Extract features from conv3, conv4, conv5
2. Global Average Pool each layer
3. Concatenate: 256 + 384 + 512 = 1152 dims
4. Project to 512 dims via MLP

```python
extractor = MultiScaleOSNetExtractor(
    model_name="osnet_x1_0",
    device="cuda"
)
```

Enabled when `use_multi_scale_reid=True` in config.

---

#### `OSNetOnnxExtractor`
ONNX Runtime version for faster inference.

```python
extractor = OSNetOnnxExtractor(
    model_path="osnet_x1_0.onnx",
    device="cuda"
)
```

---

#### `ResNet18Extractor`
Fallback when torchreid is not installed.

---

### File: `app/core/features/face_extractor.py`

Face detection and embedding extraction using InsightFace.

#### `InsightFaceExtractor`
Uses Buffalo_L model with ArcFace-style training.

```python
extractor = InsightFaceExtractor(
    model_name="buffalo_l",     # Model pack
    det_size=(640, 640),        # Detection input size
    det_thresh=0.5,             # Detection confidence threshold
    device="cuda"
)

faces = extractor.detect_faces(image)  # List[FaceResult]
```

**FaceResult:**
```python
@dataclass
class FaceResult:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray             # 512-dim face embedding
    confidence: float                 # Detection confidence
    landmarks: Optional[np.ndarray]   # 5 facial landmarks
```

---

#### `QualityScorer`
Scores image quality for thumbnail selection.

**Scoring Factors:**
| Factor | Weight | Measurement |
|--------|--------|-------------|
| Sharpness | 30% | Laplacian variance (blur detection) |
| Face visibility | 40% | Face size relative to body + confidence |
| Image size | 30% | Larger is better up to 256x512 |

```python
scorer = QualityScorer()
quality = scorer.score(crop, face_bbox, face_confidence)  # 0.0 - 1.0
```

---

#### `create_fused_embedding()`
**Gated Dynamic Fusion of face and body embeddings.**

Uses quality tiers to prevent noise from bad faces:

| Quality Tier | Range | Face Weight | Body Weight |
|--------------|-------|-------------|-------------|
| HIGH | > 0.7 | 70% | 30% |
| MEDIUM | 0.4 - 0.7 | 30% | 70% |
| LOW | < 0.4 | 0% | 100% |

```python
fused = create_fused_embedding(
    body_embedding,
    face_embedding,
    face_weight=0.4,
    body_weight=0.6,
    face_quality=0.8
)
```

**Critical:** Embeddings are L2-normalized both **before** and **after** fusion to ensure proper cosine similarity.

---

### File: `app/core/utils/quality.py`

Quality gate to filter out low-quality frames.

```python
is_good, score, reason = is_quality_frame(
    image,
    min_resolution=40,      # Minimum width/height
    min_sharpness=60.0      # Laplacian variance threshold
)
```

**Returns:**
- `is_good`: Boolean - pass/fail
- `score`: Float - blur score (higher = sharper)
- `reason`: String - failure reason if applicable

**Filtering:**
- `"empty"`: Image is None or has no pixels
- `"too_small_WxH"`: Below minimum resolution
- `"blurry_X.X"`: Laplacian variance below threshold
- `"ok"`: Passed all checks

---

## Re-Identification Module

### File: `app/core/reid/visual_matcher.py`

Manages the identity gallery and visual similarity matching.

#### `GalleryEntry` (dataclass)
```python
@dataclass
class GalleryEntry:
    global_id: str              # UUID string
    embedding: np.ndarray       # Best exemplar (not average)
    best_quality_score: float   # Quality of current embedding
    last_camera_id: int         # Where last seen
    last_seen: datetime         # When last seen
    appearance_count: int       # Total sightings
    camera_history: List[int]   # All cameras visited
```

---

#### `VisualMatcher`

```python
matcher = VisualMatcher(
    match_threshold=0.5,         # Minimum similarity for match
    max_gallery_size=1000,       # Maximum identities
    embedding_history_size=10    # Embeddings per identity
)
```

**Galleries:**
- `gallery`: Main gallery with fused/body embeddings
- `face_gallery`: Face-only embeddings for face search

---

**`add_to_gallery()`** - Quality-Priority Buffer Pattern

**Update Rules:**
1. **Quality Replacement:** Only replace embedding if new quality > stored quality + 10%
2. **Smart Merge:** Blend if similarity > 0.85 (extremely similar pose)
3. **Reject:** Do nothing if similarity < 0.85 AND quality not significantly better

**Parameters:**
| Threshold | Value | Purpose |
|-----------|-------|---------|
| Quality margin | +10% | Must be significantly better to replace |
| Merge threshold | 0.85 | Must be extremely similar to blend |
| Blend alpha | 0.05 | Gentle blending (5% new, 95% old) |

This prevents **gallery pollution** where wrong person's features contaminate an identity.

---

**`match()`** - Visual Similarity Matching

```python
matches = matcher.match(
    query_embedding,
    top_k=5,
    threshold=0.3,
    use_rerank=True      # K-reciprocal reranking
)
# Returns: List[(global_id, similarity, GalleryEntry)]
```

**Matching Process:**
1. Compute raw cosine similarity to all gallery entries
2. If `use_rerank=True`:
   - Run k-reciprocal reranking
   - Use reranked distances for **ordering** only
   - Return **raw cosine similarity** for threshold comparison
3. Filter by threshold and return top-k

**K-Reciprocal Reranking:** Improves matching by checking if query and gallery item are mutual nearest neighbors. Reduces false positives.

**Reranking Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| k1 | 20 | Initial k-nearest neighbors |
| k2 | 6 | Local query expansion |
| lambda | 0.3 | Balance original vs jaccard distance |

---

### File: `app/core/reid/st_scorer.py`

Spatial-temporal probability scoring.

#### `ParzenEstimator`
Non-parametric density estimation for transition times.

```python
estimator = ParzenEstimator(bandwidth=5.0)  # Kernel bandwidth in seconds
estimator.add_observation(25.3)              # Add observed transition time
prob = estimator.pdf(30.0)                   # Get probability at 30 seconds
```

---

#### `SpatioTemporalScorer`

```python
scorer = SpatioTemporalScorer(
    bandwidth=5.0,               # Parzen window bandwidth
    max_transition_time=300.0,   # 5 minutes maximum
    use_parzen=True              # Use Parzen vs Gaussian
)
```

**Camera Positions:**
```python
scorer.set_camera_position(camera_id=1, lat=12.9716, lon=77.5946)
```

**Update Transition Distribution:**
```python
scorer.update_ttd(from_camera=1, to_camera=2, time_delta=45.2)
```

**Calculate Score:**
```python
score = scorer.calculate_score(
    from_camera=1,
    to_camera=2,
    time_delta=50.0
)  # Returns 0.0 - 1.0
```

**Scoring Methods:**
1. **Parzen Window** (if >= 5 observations): Non-parametric density
2. **Gaussian** (alternative): Based on mean/std of observations
3. **Distance-based** (fallback): Uses walking speed assumptions

**Walking Speed Assumptions:**
| Speed | Value | Description |
|-------|-------|-------------|
| MIN_SPEED | 0.5 m/s | Slow walk |
| AVG_SPEED | 1.4 m/s | Normal walk |
| MAX_SPEED | 3.0 m/s | Fast walk / jog |

---

**`joint_score()`** - Combine Visual and ST Scores

```python
joint = scorer.joint_score(
    visual_score=0.75,
    st_score=0.8,
    alpha=0.5,            # ST weight
    use_logistic=True     # Apply smoothing
)
```

**Logistic Smoothing:** Prevents zero ST scores from killing good visual matches.

```
st_factor = 1 / (1 + exp(-10 * (st_score - 0.3)))
joint = visual * (1 - alpha) + visual * alpha * st_factor
```

---

### File: `app/core/reid/topology.py`

Camera network graph management.

#### `CameraNode` (dataclass)
```python
@dataclass
class CameraNode:
    camera_id: int
    lat: float
    lon: float
    zone_id: Optional[int]
    neighbors: Set[int]        # Connected cameras
    entry_zones: List[str]     # e.g., ["left", "bottom"]
    exit_zones: List[str]      # e.g., ["right", "top"]
```

#### `TopologyEdge` (dataclass)
```python
@dataclass
class TopologyEdge:
    from_camera: int
    to_camera: int
    distance: float            # Meters (Haversine)
    avg_transit_time: float    # Learned from data
    transition_count: int      # Number of observations
    is_bidirectional: bool
```

---

#### `CameraTopology`

```python
topology = CameraTopology(
    auto_connect_radius=500.0   # Auto-connect cameras within 500m
)
```

**Methods:**
- `add_camera(camera_id, lat, lon)`: Add camera, auto-connect to nearby
- `connect_cameras(cam1, cam2)`: Manual connection
- `get_neighbors(camera_id)`: Get directly connected cameras
- `get_reachable(camera_id, max_hops=2)`: BFS to find reachable cameras
- `update_transition(from_cam, to_cam, transit_time)`: Learn transit times
- `infer_topology_from_transitions(transitions)`: Build graph from observations

---

## Services Layer

### File: `app/services/reid_service.py`

Orchestrates cross-camera re-identification.

#### `ReIDService`

```python
service = ReIDService(
    match_threshold=0.3,
    st_weight=0.5,
    max_transition_time=300.0
)
```

**Components:**
- `visual_matcher`: Gallery management and matching
- `st_scorer`: Spatial-temporal scoring
- `topology`: Camera network graph

---

**Two-Threshold Matching System:**

| Threshold | Value | Action |
|-----------|-------|--------|
| `CONFIRM_THRESHOLD` | 0.60 | Above = confident match |
| `NEW_IDENTITY_THRESHOLD` | 0.40 | Below = definitely new person |
| Between thresholds | - | Conservative: create new identity |

**`match_identity()`** - Main Matching Logic

```python
result = await service.match_identity(
    camera_id=1,
    embedding=feature_vector,
    timestamp=datetime.now(),
    top_k=5
)
```

**Returns:**
```python
@dataclass
class MatchResult:
    global_track_id: UUID
    visual_similarity: float    # Raw cosine similarity
    st_probability: float       # Spatial-temporal score
    joint_score: float          # Combined score
    is_new: bool                # True if new identity created
```

**Matching Algorithm:**
1. Get visual candidates from gallery (threshold=0.3)
2. Score each with spatial-temporal probability
3. Compute joint score: `visual * 0.8 + st * 0.2`
4. If best visual >= 0.60 → CONFIRM match
5. If best visual < 0.40 → CREATE new identity
6. Otherwise → CREATE new identity (conservative)

---

**`search_by_image()`** - Image Search

```python
matches = service.search_by_image(
    image_bytes=file_content,
    top_k=5,
    threshold=0.4
)
```

**Process:**
1. Decode uploaded image
2. If large scene → run YOLO to detect persons
3. Extract face embedding (if visible)
4. Extract body embedding
5. Fuse embeddings based on face quality
6. Search gallery and face_gallery
7. Return matches above threshold

---

**Thumbnail Management:**

```python
service.set_thumbnail(global_id, base64_image, quality=0.8)
```

Only updates if new quality > existing quality.

---

### File: `app/services/tracking_service.py`

Orchestrates detection, tracking, and feature extraction.

#### `TrackingService`

```python
service = TrackingService()
```

**Lazy Loading:** Detector, extractor, and face extractor are loaded on first use.

---

**`process_frame()`** - Main Processing Loop

```python
tracks, new_features = await service.process_frame(
    camera_id=1,
    frame=numpy_array,
    timestamp=datetime.now(),
    extract_features=True
)
```

**Processing Pipeline:**
1. **Detection:** YOLO detect persons
2. **Crop:** Extract person crops with 10% padding
3. **Quality Gate:** Filter out blurry/small crops
4. **Feature Extraction:**
   - Extract body features (OSNet)
   - Extract face features (InsightFace) if enabled
   - Fuse based on face quality tier
5. **Tracking:** DeepSORT predict + update
6. **Lifecycle:** Handle tracklet start/end events

---

**Per-Camera State:**

Each camera has independent:
- DeepSORT tracker instance
- Active tracklet mapping
- Last frame timestamp

---

## API Endpoints

### File: `app/api/v1/tracks.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tracks/active` | GET | Get all currently active tracks |
| `/tracks/{track_id}` | GET | Get full trajectory for a track |
| `/tracks/search` | POST | Search tracks by criteria (time, camera, status) |
| `/tracks/search/image` | POST | Search by uploaded image |
| `/tracks/{track_id}/interpolate` | GET | Get OSRM-interpolated path |
| `/tracks/{track_id}/transits` | GET | Get cross-camera transitions |
| `/tracks/{track_id}/finish` | POST | Mark track as finished |
| `/tracks/{track_id}` | DELETE | Delete track and data |

---

**Image Search Response:**
```json
[
  {
    "track": {
      "global_track_id": "uuid",
      "status": "active",
      "first_seen": "2024-01-01T00:00:00",
      "last_seen": "2024-01-01T00:05:00",
      "camera_sequence": [1, 2, 3]
    },
    "score": 0.85,
    "path_points": [
      {"camera_id": 1, "latitude": 12.97, "longitude": 77.59, "name": "Entrance"}
    ]
  }
]
```

---

### File: `app/api/v1/realtime.py`

WebSocket endpoint for real-time updates.

**Endpoint:** `/ws/tracks`

**Event Types:**
| Event | Description |
|-------|-------------|
| `connected` | Initial connection message |
| `heartbeat` | Keep-alive (every 30s) |
| `detection` | New person detected |
| `track_update` | Track position updated |
| `transit` | Cross-camera transition |
| `reid_match` | ReID match found |
| `track_lost` | Track lost |
| `track_finished` | Track completed |
| `track_path_update` | Path updated for map |

---

## Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Video Frame │───►│   YOLO      │───►│ Detections  │
└─────────────┘    │  Detector   │    │ (bboxes)    │
                   └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────────────────┼─────────────────────────┐
                   │                         ▼                         │
                   │  ┌─────────────┐   ┌─────────────┐                │
                   │  │   Crop +    │───►│  Quality    │───► Skip if   │
                   │  │   Padding   │   │   Gate      │     low quality│
                   │  └─────────────┘   └──────┬──────┘                │
                   │                           │ Pass                  │
                   │                           ▼                       │
                   │  ┌─────────────┐   ┌─────────────┐                │
                   │  │   OSNet     │   │ InsightFace │                │
                   │  │  (body)     │   │   (face)    │                │
                   │  └──────┬──────┘   └──────┬──────┘                │
                   │         │                 │                       │
                   │         ▼                 ▼                       │
                   │  ┌───────────────────────────────┐                │
                   │  │   Gated Dynamic Fusion        │                │
                   │  │   (quality-tier weighted)     │                │
                   │  └──────────────┬────────────────┘                │
                   │                 │                                 │
                   │ Tracking Service│                                 │
                   └─────────────────┼─────────────────────────────────┘
                                     ▼
                   ┌─────────────────────────────────┐
                   │         DeepSORT                │
                   │  (Kalman + Appearance Match)    │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼ Confirmed Tracks
                   ┌───────────────────────────────────────────────────┐
                   │                  ReID Service                      │
                   │                                                    │
                   │  ┌─────────────┐   ┌─────────────┐                 │
                   │  │   Visual    │   │  Spatial-   │                 │
                   │  │  Matcher    │   │  Temporal   │                 │
                   │  │  (Gallery)  │   │   Scorer    │                 │
                   │  └──────┬──────┘   └──────┬──────┘                 │
                   │         │                 │                        │
                   │         ▼                 ▼                        │
                   │  ┌───────────────────────────────┐                 │
                   │  │     Two-Threshold Matching    │                 │
                   │  │  CONFIRM (>0.6) / NEW (<0.4)  │                 │
                   │  └──────────────┬────────────────┘                 │
                   │                 │                                  │
                   └─────────────────┼──────────────────────────────────┘
                                     ▼
                   ┌─────────────────────────────────┐
                   │       MatchResult               │
                   │  - global_track_id              │
                   │  - visual_similarity            │
                   │  - st_probability               │
                   │  - is_new                       │
                   └─────────────────────────────────┘
                                     │
                                     ▼
                   ┌─────────────────────────────────┐
                   │      WebSocket Broadcast        │
                   │   (real-time frontend update)   │
                   └─────────────────────────────────┘
```

---

## Threshold Summary Table

| Location | Threshold | Value | Purpose |
|----------|-----------|-------|---------|
| Detection | `yolo_confidence` | 0.5 | Min detection confidence |
| Detection | `yolo_iou_threshold` | 0.45 | NMS overlap threshold |
| Tracking | `deepsort_max_age` | 30 | Frames before track deletion |
| Tracking | `deepsort_n_init` | 3 | Frames to confirm track |
| Tracking | `deepsort_max_iou_distance` | 0.7 | Max IoU for unconfirmed tracks |
| Tracking | `matching_threshold` | 0.4 | Max appearance distance |
| Tracking | `CHI2_THRESHOLD` | 9.4877 | Mahalanobis gating (95% CI) |
| Quality | `quality_min_sharpness` | 60.0 | Blur detection threshold |
| Quality | `quality_min_size` | 40 | Minimum crop size (px) |
| ReID | `reid_match_threshold` | 0.3 | Base similarity threshold |
| ReID | `CONFIRM_THRESHOLD` | 0.60 | High-confidence match |
| ReID | `NEW_IDENTITY_THRESHOLD` | 0.40 | Definitely new person |
| Gallery | Quality margin | +10% | Required for embedding replacement |
| Gallery | Merge threshold | 0.85 | Required for embedding blend |
| Gallery | Blend alpha | 0.05 | Blend ratio (5% new) |
| Reranking | k1 | 20 | K-nearest neighbors |
| Reranking | k2 | 6 | Local expansion |
| Reranking | lambda | 0.3 | Distance blend factor |
| Face Fusion | HIGH tier | > 0.7 | Trust face 70% |
| Face Fusion | MEDIUM tier | 0.4 - 0.7 | Trust face 30% |
| Face Fusion | LOW tier | < 0.4 | Ignore face |
| ST Scoring | `max_transition_time` | 300s | Maximum allowed gap |
| ST Scoring | Parzen min observations | 5 | Before using learned distribution |

---

*Last updated: 2026-01-09*
