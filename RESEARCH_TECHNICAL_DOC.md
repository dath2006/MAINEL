# Multi-Camera Multi-Target Re-Identification (MCMT-ReID) System Technical Documentation

## 1. System Architecture Overview

The system is a high-performance, real-time Multi-Camera Multi-Target Re-Identification (MCMT-ReID) platform designed for tracking individuals across non-overlapping camera views. The architecture is modular, consisting of distinct stages: Detection, Tracking, Quality Assessment, Feature Extraction, and Global Re-Identification (ReID). The system employs a hybrid approach combining visual appearance features with spatial-temporal constraints to robustly identify individuals.

## 2. Detection Module

The detection subsystem utilizes a specialized deep learning model for identifying persons, bags, and faces.

### 2.1 Model Specification
- **Architecture**: NVIDIA PeopleNet (ResNet34-based backbone).
- **Inference Runtime**: ONNX Runtime with CUDA execution provider.
- **Input Resolution**: 960x544 pixels (RGB).
- **Input Preprocessing**:
  - Color space conversion: BGR to RGB.
  - Normalization: Scale [0, 255] to [0, 1].
  - Layout: Channel-First (NCHW).

### 2.2 Post-Processing
The raw model outputs (coverage map and bounding boxes) are processed using a vectorized decoding algorithm:
- **Coverage Threshold**: Detections with confidence scores below **0.4** are discarded.
- **Non-Maximum Suppression (NMS)**: Applied with an Intersection over Union (IoU) threshold of **0.5** to eliminate duplicate detections.
- **Class Filtering**: The system is configured to prioritize "person" class detections while optionally tracking associated objects (bags).

## 3. Single-Camera Tracking (SCT)

Within each camera view, a lightweight tracker maintains object identity across frames to generate coherent tracklets.

### 3.1 Tracking Algorithm
The system uses a geometric IOU-based tracker for high-speed performance during the preprocessing phase.
- **Matching Metric**: Intersection over Union (IoU) of bounding boxes.
- **Assignment Strategy**: Greedy matching based on IoU descent.
- **Velocity Smoothing**: An Exponential Moving Average (EMA) with $\alpha=0.7$ updates the velocity vector $(dx, dy)$ for each track, aiding in motion prediction.

### 3.2 Key Thresholds
- **IoU Match Threshold**: **0.3** (Minimum overlap to associate a detection with a track).
- **Track Confirmation**: Requires **3** consecutive "hits" (detections) to confirm a track.
- **Max Age**: Tracks are maintained for **30** frames without detection before deletion (handling temporary occlusions).
- **Duplicate Prevention**: New tracks are not created if the detection IoU with any existing track exceeds **0.05**.

## 4. Quality Assessment Module

To ensure high-performance ReID, the system actively filters poor-quality crops using a composite scoring mechanism. Only high-quality samples are used for feature bank updates.

### 4.1 Scoring Components
The quality score ($Q_{total}$) is a weighted sum of three sub-metrics:
$$ Q_{total} = w_{sharp} \cdot S_{sharp} + w_{pose} \cdot S_{pose} + w_{occ} \cdot S_{occ} $$
Weights: Sharpness (0.30), Pose (0.40), Occlusion (0.30).

#### 4.1.1 Sharpness Score ($S_{sharp}$)
- **Method**: Laplacian variance of the grayscale image.
- **Normalization**: Sigmoid-like scaling where variance $\ge 50.0$ is considered sharp.
- **Blur Threshold**: Variance $< 50.0$ is classified as blurry.

#### 4.1.2 Pose Estimation ($S_{pose}$)
- **Method**: Multi-stage classifier using Haar Cascades (Primary) and Motion Vectors (Secondary).
- **Classification**:
  - **Frontal**: Detected by frontal face cascade. Score: $60-100$ (based on face-to-body ratio).
  - **Side**: Detected by profile face cascade. Score: $40-60$.
  - **Back**: Inferred from motion (moving away/up) or lack of facial features. Score: $< 40$.
- **Motion Heuristic**: Downward motion ($+dy > 0.5$) implies moving towards the camera (Front bias). Upward motion implies moving away (Back bias).

#### 4.1.3 Occlusion Detection ($S_{occ}$)
Combines three indicators to detect obstruction:
1.  **Aspect Ratio**: Ideal range $[0.25, 0.55]$. Deviations penalize the score.
2.  **Edge Density**: Canny edge detection density. Low density implies smooth obstructions (e.g., walls).
3.  **Color Variance**: Standard deviation of HSV channels. Low variance implies uniform occlusion.

### 4.2 Quality Gating
- **Minimum Acceptable Score**: **30.0** (out of 100).
- **Minimum Bounding Box**: 32x64 pixels.
- **Minimum Aspect Ratio**: 0.25.
- **Minimum Frame Coverage**: 0.5% of total frame area.

## 5. Global Re-Identification (ReID)

The core engine responsible for matching identities across different cameras.

### 5.1 Feature Extraction
- **Model**: ResNet50 (Market1501/AICity pre-trained) or OSNet (Legacy).
- **Embedding Dimension**: **256** floating-point values.
- **Normalization**: L2 normalization applied to all embeddings before comparison.

### 5.2 Matching Logic
The system uses a hybrid similarity function combining visual and spatial-temporal signals.

#### 5.2.1 Visual Similarity
- **Metric**: Cosine similarity (dot product of normalized embeddings).
- **Gallery Search**: Matches are searched against a dynamic gallery of global identities.

#### 5.2.2 Spatio-Temporal (ST) Constraints
- **Topology Modeling**: The system learns transition times between camera pairs.
- **ST Scoring**: Probability $P_{st}$ is calculated based on the time difference $\Delta t$ and the expected transition distribution.
- **Weighting**: The final joint score is calculated as:
  $$ S_{joint} = 0.8 \cdot S_{visual} + 0.2 \cdot P_{st} $$

#### 5.2.3 Matching Decisions
A dual-threshold logic governs identity assignment:
1.  **Confident Match** ($S_{visual} \ge 0.55$): Identity is confirmed.
2.  **Tentative Match** ($0.60 > S_{visual} \ge 0.55$): Accepted *only if* backed by strong Spatio-Temporal probability ($P_{st} > 0.3$) or if occurring within the same camera view.
3.  **New Identity**: If best match $S_{visual} < 0.60$, a new global identity ID (UUID) is generated.

### 5.3 Retrospective Identity Merging
To handle identity fragmentation (one person split into multiple IDs), the system performs retrospective analysis during search:
- **Pairwise Check**: All candidate matches are cross-compared.
- **Merge Threshold**: If two distinct Global IDs have a mutual visual similarity $\ge 0.65$, they are treated as the same identity.
- **Result Fusion**: Metadata (camera history, timestamps) is merged into a unified "canonical" identity for the final report.

### 5.4 Post-Occlusion Recovery
To correct ID switches after long occlusions:
- The system compares the post-occlusion embedding against both the tracked identity's history AND the history of known "occluders" (nearby tracks).
- **Correction Trigger**: If the current appearance matches an occluder better than the self-history (Similarity $> 0.5$), an ID swap is performed.

## 6. Configuration Summary (Key Thresholds)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Detection** | | |
| YOLO/PeopleNet Confidence | 0.5 | Minimum confidence for valid detection |
| NMS IoU | 0.45 | Non-Maximum Suppression overlap threshold |
| **Tracking (DeepSORT/IoU)** | | |
| Max Age | 30 frames | Time to keep track alive without detection |
| N Init | 3 frames | Detections needed to confirm track |
| **Re-Identification** | | |
| Match Threshold | 0.55 | Similarity to match existing ID |
| New ID Threshold | 0.60 | Similarity below this creates new ID |
| Merge Threshold | 0.65 | Similarity to merge two existing IDs |
| Quality Threshold | 0.7 | Min quality score to update feature bank |
| Feature Bank Size | 50 | Max embeddings stored per identity |
| **Quality Scorer** | | |
| Blur Variance Threshold | 50.0 | Laplacian variance limit for sharpness |
| Min Aspect Ratio | 0.25 | Reject crops wider than this |
| Max Aspect Ratio | 3.5 | Reject crops taller than this |
| **Weights** | | |
| ReID Visual Weight | 0.8 | Importance of visual similarity |
| ReID ST Weight | 0.2 | Importance of Spatio-Temporal score |
