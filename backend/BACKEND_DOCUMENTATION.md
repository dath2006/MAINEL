# MCMT-ReID Backend Comprehensive Documentation

## 1. System Overview

The Multi-Camera Multi-Target Re-Identification (MCMT-ReID) backend is a high-performance, real-time surveillance system built with **FastAPI**. It integrates state-of-the-art computer vision models for detection (YOLOv8), tracking (DeepSORT), and cross-camera re-identification (OSNet).

The system features a **Spatio-Temporal (ST) ReID** engine that learns the topology of the camera network (transition times between cameras) to improve matching accuracy, reducing false positives by validating if a transition is physically plausible.

### key Technologies
- **Framework**: FastAPI (Async, Python 3.10+)
- **Database**: PostgreSQL (Metadata) + Redis (Cache/Hot State)
- **Computer Vision**: PyTorch, Ultralytics YOLO, OpenMMLab (via wrappers)
- **Real-time**: WebSockets for frame and event streaming
- **Inference**: ONNX Runtime (CPU/GPU) or PyTorch
- **Processing**: Background worker threads with asyncio integration

---

## 2. Configuration & Constants

Configuration is managed via `app/config.py` using Pydantic Settings, loadable from `.env`.

### Core Settings
- **App Name**: `MCMT-ReID API`
- **Device**: `cuda` (default) or `cpu`
- **Database URL**: `postgresql+asyncpg://...`
- **Redis URL**: `redis://localhost:6379/0`

### Model Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_model_path` | `model_weights/yolov8n.pt` | Detection model source |
| `yolo_confidence` | `0.5` | Minimum confidence for detection |
| `reid_match_threshold` | **0.3** | Cosine similarity threshold (lower = more lenient) |
| `reid_embedding_dim` | `512` | Dimension of feature vectors |
| `use_face_reid` | `True` | Enable multi-modal (Face + Body) matching |
| `face_weight` | `0.4` | Weight of face embedding in fusion |

### Tracking & ReID Logic
| Parameter | Default | Description |
|-----------|---------|-------------|
| `deepsort_max_age` | `30` | Max frames to keep lost track alive |
| `st_weight` | **0.5** | Weight of Spatial-Temporal score vs Visual score |
| `max_transition_time` | `300.0`s | Max allowed time between camera transitions |
| `quality_min_sharpness` | `60.0` | Laplacian variance threshold for blur detection |

---

## 3. Architecture & Core Logic

### 3.1. Stream Processing Pipeline
**File**: `app/workers/stream_processor.py`

The `StreamProcessor` is a singleton background worker that drives the entire pipeline.
1.  **Ingestion**: Pulls raw frames from `StreamManager` (OpenCV capture).
2.  **Detection & Tracking**: Runs every `detection_interval` frames.
    - **YOLOv8**: Detects persons.
    - **DeepSORT**: Associates detections to local tracks (`track_id`).
3.  **Feature Extraction**:
    - **Body**: OSNet extracts 512-dim embedding from person crop.
    - **Face**: InsightFace (optional) extracts face embedding if visible.
4.  **ReID Matching** (`ReIDService`):
    - Matches local tracks to Global Identities (`global_id`).
    - Uses **Visual Similarity** + **Spatial-Temporal Probability**.
5.  **Broadcasting**:
    - Sends annotated frames (JPEG) and events (Active Tracks, Transits) via WebSockets.

### 3.2. DeepSORT Tracking
**File**: `app/core/tracking/deepsort.py`

Custom implementation of DeepSORT:
- **Kalman Filter**: Predicts future position to handle occlusions.
- **Cascade Matching**:
    - Priority given to recently seen tracks (Age 1..MaxAge).
    - **Gating**: Uses Mahalanobis distance to reject physically impossible associations.
    - **Visual Cost**: Cosine distance of appearance features.
- **IOU Fallback**: Matches remaining unconfirmed tracks using intersection-over-union.
- **State Machine**: `TENTATIVE`  (3 hits) -> `CONFIRMED` -> `DELETED` (30 frames lost).

### 3.3. Re-Identification Engine (The "Brain")

#### A. Visual Matcher
**File**: `app/core/reid/visual_matcher.py`

Maintains a **Gallery** of known identities.
- **Matching Strategy**:
    1.  **Cosine Similarity**: Basic visual resemblance.
    2.  **K-Reciprocal Reranking**: (Optional) "Un-Cook" strategy. Re-ranks results based on whether the query and candidate share the same nearest neighbors. Improves accuracy significantly on difficult queries.
- **Gallery Management**:
    - Stores `embeddings_history` (last 10) to compute a robust `mean accuracy`.
    - **Face Gallery**: Separate index for face-only lookups.

#### B. Spatial-Temporal (ST) Scorer
**File**: `app/core/reid/st_scorer.py`

Calculates the probability $P(Transition | \Delta t)$ that a person moved from Camera A to Camera B in time $\Delta t$.
- **Transition Time Distribution (TTD)**:
    - Uses **Parzen Window Estimation** (Kernel Density) to learn non-parametric distributions.
    - Capable of learning multi-modal stats (e.g., two different paths between cameras).
- **Fallback**: If no data exists, uses physics-based heuristics (Walking speed 0.5 - 3.0 m/s).

#### C. Joint Scoring (The "Secret Sauce")
**File**: `app/services/reid_service.py` -> `match_identity`

Combines visual and physical world constraints:
```python
# Simplified Logic Snippet
visual_sim = dot_product(query, gallery)
st_prob = st_scorer.calculate(from_cam, to_cam, time_delta)

# Joint Score Calculation
# Logistic smoothing prevents 0.0 scores from killing good visual matches violently
st_factor = 1 / (1 + exp(-10 * (st_prob - 0.3)))
joint_score = visual_sim * (1 - alpha) + visual_sim * alpha * st_factor
```

### 3.4. Dynamic Topology Learning
**File**: `app/core/reid/topology.py`

- **Auto-Discovery**: Cameras within `auto_connect_radius` (500m) are automatically linked in the graph.
- **Online Learning**: As people move between cameras, the system records transition times and updates the edge statistics (average time, variance).
- **Inference**: Can reconstruct the map topology purely from observed tracking data.

---

### 3.5. Quality Control & Multi-Modal Fusion Strategy

#### A. The "Garbage Collection" (Quality Gating)
**File**: `app/core/utils/quality.py`

Before any feature extraction occurs, crops must pass a quality gate to prevent "garbage in, garbage out" (feature collapse).
- **Resolution Gate**: Crops must be at least `quality_min_size` (default 40px) in both dimensions.
- **Blur Detection**: Uses **Laplacian Variance** to measure sharpness.
  - Formula: $\text{var}(\nabla^2 \text{Image}_{\text{gray}})$
  - Threshold: `quality_min_sharpness` (default 60.0). Scores < 60 are rejected as too blurry.

#### B. Dynamic Face-Body Fusion
**File**: `app/core/features/face_extractor.py` -> `create_fused_embedding`

MCMT-ReID uses a **Dynamic Weighting** mechanism to fuse Face (InsightFace) and Body (OSNet) embeddings. It doesn't just average them; it adjusts weights based on face quality.

1.  **Normalization**: Both 512-dim vectors are L2-normalized first.
2.  **Quality Assessment**:
    - **Sharpness**: Laplacian variance / 200.0 (capped at 1.0).
    - **Face Visibility**: Ratio of face height to body crop height (Ideal: 10-50%).
    - **Confidence**: Detections < 0.5 confidence are ignored.
3.  **Fusion Logic**:
    - **Standard**: $E_{\text{fused}} = 0.4 \cdot E_{\text{face}} + 0.6 \cdot E_{\text{body}}$
    - **High Quality Boost**: If `face_quality > 0.8`, face weight is boosted by 50% (up to 0.8 max), effectively trusting the face more than the body clothes (which can change).
    - **Fallback**: If face is missing or low quality (< 0.4), only Body embedding is used.

### 3.6. Search Logic Specifics
**File**: `app/services/reid_service.py` -> `search_by_image`

The search endpoint is smart enough to distinguish between "Person Crop" and "Full Scene" uploads.
1.  **Scene Analysis**:
    - If image is large (>300px) and aspect ratio is not "person-like" (too wide/squarish), it's treated as a **Scene**.
    - **Action**: Runs YOLOv8 on the uploaded image to detect standard person crops first.
2.  **Embedding Extraction**:
    - Extracts both Face and Body features from the target crop.
    - **Fusion**: Applies the same Dynamic Fusion logic as the realtime pipeline.
3.  **Dual-Gallery Search**:
    - If a valid face is found, it searches the **Face Gallery** (face-only index) separately from the **Main Gallery** (fused index).
    - Results are merged, with duplicate Global IDs favored if they appear in both.

---

## 4. API Reference

### 4.1. Real-time WebSocket
**URL**: `ws://<host>/api/v1/realtime/tracks`
... (WebSocket content remains same) ...

### 4.2. Track & Search Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tracks/active` | `GET` | Get currently tracked people. Supports filtering by `camera_id`. |
| `/tracks/{id}` | `GET` | Get full history including timestamps and camera sequence. |
| `/tracks/search` | `POST` | Advanced query: filter by time range, status, or specific cameras. |
| `/tracks/search/image` | `POST` | **Visual Search**: Upload image -> Get matched Global IDs + Path on map. |
| `/tracks/{id}/finish` | `POST` | Manually mark a track as finished (left area). |

### 4.3. Camera Management
**File**: `app/api/v1/cameras.py`
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cameras/` | `POST` | Register a new camera (name, stream_url, lat/lon stats). |
| `/cameras/` | `GET` | List all cameras. Filters: `zone_id`, `is_active`. |
| `/cameras/{id}/activate` | `POST` | Enable tracking on this camera. |
| `/cameras/{id}/stats` | `GET` | Get debug stats (fps, active tracks count). |

---

## 5. Data Structures

### GalleryEntry (`visual_matcher.py`)
In-memory representation of an identity.
```python
@dataclass
class GalleryEntry:
    global_id: str
    embedding: np.ndarray      # Average of last N sightings
    last_camera_id: int
    last_seen: datetime
    appearance_count: int
    embeddings_history: List[np.ndarray]
    camera_history: List[int]  # Sequence of cameras visited
```

### Track (`deepsort.py`)
Local single-camera state.
```python
@dataclass
class Track:
    track_id: int              # Local ID (reset on restart)
    mean: np.ndarray           # Kalman State (x, y, a, h, dx, dy, da, dh)
    state: TrackState          # TENTATIVE, CONFIRMED, DELETED
    features: List[np.ndarray]
    global_id: str             # Assigned by ReIDService
    face_bbox: Tuple[int, int, int, int] # For UI visualization
```

---

## 6. Directory Structure
```
backend/app/
├── api/v1/          # Endpoints (realtime.py, tracks.py)
├── core/
│   ├── detection/   # YOLO Wrappers
│   ├── features/    # OSNet & Face Extractors
│   ├── reid/        # VisualMatcher, STScorer, Topology
│   └── tracking/    # DeepSORT, Kalman Filter
├── workers/         # StreamProcessor (Background Loop)
├── services/        # ReIDService (Orchestration)
├── db/              # Database Models & Session
└── config.py        # Settings
```
