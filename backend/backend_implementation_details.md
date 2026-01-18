# Backend Implementation Details & Infrastructure Map

This document provides a comprehensive technical overview of the backend infrastructure, logic, and implementation details for the Multi-Camera Multi-Target Re-Identification (MCMT-ReID) system.

---

## 1. System Infrastructure Map

The backend is built as a modular microservices-like architecture running in Docker.

### Component Diagram

```mermaid
graph TD
    Client[Frontend / Client] <-->|WebSocket & HTTP| API[FastAPI Backend]
    
    subgraph "Core Backend Services"
        API --> StreamMgr[Stream Manager]
        StreamMgr --> StreamProc[Stream Processor]
        StreamProc --> Detect[Detection (PeopleNet)]
        StreamProc --> Track[Tracking (DeepSORT)]
        StreamProc --> ReID[ReID Service]
        ReID --> Merger[Identity Merger]
    end
    
    subgraph "Data & State"
        API --> DB[(PostgreSQL)]
        API --> Redis[(Redis Cache)]
        StreamProc --> Redis
        ReID --> DB
    end
    
    subgraph "AI / ML Models"
        Detect --> Model1[ResNet34 PeopleNet]
        ReID --> Model2[ResNet50 ReID]
    end
```

### Infrastructure Components

1.  **Application Server (API)**
    *   **Technology**: Python 3.9+, FastAPI.
    *   **Role**: Handles efficient async web requests, manages WebSocket connections for real-time video streaming, and orchestrates the AI pipeline.
    *   **Port**: 8000.

2.  **Database (DB)**
    *   **Technology**: PostgreSQL 15.
    *   **Role**: Stores persistent data like Camera configurations, Global Identity Tracks, and Transition events. It uses standard Float columns for coordinates (Latitude/Longitude) instead of complex PostGIS geometry types.
    *   **Port**: 5432.

3.  **Cache & Message Broker**
    *   **Technology**: Redis 7 (Alpine).
    *   **Role**: High-speed temporary storage. It manages video frame queues to prevent processing lag and stores volatile "hot" data like active tracklets before they are saved to the database.
    *   **Port**: 6379.

4.  **Routing Engine (External API)**
    *   **Technology**: OpenRouteService (ORS) API.
    *   **Role**: Used by the Frontend to calculate and visualize realistic walking paths on the map.
    *   **Configuration**: Requires a valid `NEXT_PUBLIC_ORS_API_KEY` in the frontend environment variables. This is mandatory for map visualization features.

---

## 2. The AI Processing Storage Pipeline

This is how a video frame travels from a camera to becoming a tracked identity in the database.

### Step 1: Ingestion (Stream Manager)
*   **Input**: Real-time RTSP/CCTV streams or MP4 files.
*   **Action**: Frames are captured and placed into a memory queue (managed by Redis/Python Queue).
*   **Why?**: Decouples "reading" video from "processing" it. If the AI is slow, we drop frames intelligently instead of crashing the stream.

### Step 2: Detection (Finding People)
*   **Model**: **NVIDIA PeopleNet** (ResNet34 backbone).
*   **Action**: Scans the frame for objects.
*   **Thresholds**:
    *   **Person Confidence > 0.4**: We only trust detections that are at least 40% sure to be a person.
    *   **Face Confidence > 0.3**: We look for faces with a lower threshold because faces are small and hard to see.
*   **Face Association**:
    *   If a face box is **inside** a person box, or overlaps significantly (IoU > 0.1), we link them. This gives us a "Person with Face" object.

### Step 3: Tracking (DeepSORT)
*   **Technology**: DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric).
*   **Role**: Assigns a temporary "Local ID" to each person in a **single camera**.
*   **Logic**:
    *   Predicts where a person will move in the next frame (Kalman Filter).
    *   Matches new detections to existing tracks using box overlap (IoU) and appearance.
*   **Configuration**:
    *   **Max Age (30 frames)**: If a person is hidden for 1 second (30 frames), we remember them. After that, we forget them.
    *   **N_Init (3 frames)**: A person must be seen for 3 consecutive frames to be confirmed as a real track (filters out noise/glitches).

### Step 4: Feature Extraction (The "Fingerprint")
*   **Model**: **NVIDIA ResNet50 ReID** (Market1501 dataset trained model).
*   **Action**: Crops the image of the person and converts it into a mathematical list of 256 numbers called an **Embedding**.
*   **Concept**: This "Embedding" is a digital fingerprint. Two images of the same person will have very similar numbers, even if the lighting changes.

### Step 5: Re-Identification (Matching Across Cameras)
*   **Service**: `ReIDService`.
*   **Role**: Connects the "Local ID" from Step 3 to a "Global ID" that persists across the entire city.
*   **Quality Filter**:
    *   **Score > 55.0**: We only use "High Quality" images (good lighting, clear pose) to update the database. Blurry images are ignored to prevent bad data.
*   **Matching Logic (The "Smart" Match)**:
    *   It calculates a **Joint Score**: `Visual Score * 0.8 + Time Score * 0.2`.
    *   **Visual Score**: How consistently the person looks like the target.
    *   **Time Score (Spatial-Temporal)**: Is it physically possible for the person to move from Camera A to Camera B in this time? (e.g., You can't travel 1km in 5 seconds).

### Step 6: The Gallery & Storage
*   **Gallery**: A collection of the best images for each person.
*   **Action**: If a match is found, the global ID is updated. If not, a **New Global ID** is created (UUID).

---

## 3. Advanced Technical Logic

### A. The Identity Merger (Fixing Mistakes)
Sometimes, the system makes a mistake and creates two IDs for the same person (e.g., if they changed clothes or walked in shadow). The **Identity Merger** runs in the background to fix this.

*   **Frequency**: Every 100 frames.
*   **Logic**: It compares **every** global identity against **every other** global identity.
*   **Merge Condition**:
    *   **Average Similarity > 0.70**: The average match score between all images of Person A and Person B must be very high.
    *   It's a conservative system: It prefers to keep IDs separate rather than wrongly merging two different people.

### B. Spatial-Temporal Topology (The "Map")
The system "learns" the map of cameras.
*   **Nodes**: Cameras (Latitude, Longitude).
*   **Edges**: Transitions (paths between cameras).
*   **Learning**: When a person moves from Cam A -> Cam B, the system records the time taken. Over time, it builds a statistical distribution (Bell curve) of travel times.
*   **Usage**: If a match is visually weak but the travel time is *perfect*, the system boosts the score, allowing for matches even if the person looks slightly different (e.g., removed a jacket).

### C. Thresholds Reference Table

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Detection Confidence** | `0.4` | Min probability to count as a person. |
| **ReID Match Threshold** | `0.40` | Min similarity score to suggest a match. |
| **New ID Threshold** | `0.50` | If best match is below this, create a NEW ID. |
| **Merge Threshold** | `0.70` | Min similarity to force-merge two existing IDs. |
| **Max Transition Time** | `300s` | 5 Minutes. Max allowed time between cameras. |
| **Quality Cutoff** | `55.0` | Min image quality (0-100) to occur in gallery. |

---

## 4. Database Schema (Simplified)

*   **`cameras`**: Stores ID, Name, GPS Location, RTSP Stream URL.
*   **`global_tracks`**: The "Folder" for a person. Contains Status (Active/Left), First Seen, Last Seen.
*   **`tracklets`**: A single sighting in one camera. Linked to `global_tracks`. Contains the specific video timestamp and feature vector.
*   **`transit_events`**: A record of movement. "Person X moved from Camera 1 to Camera 2 in 45 seconds".
*   **`camera_transitions`**: The learned statistics. "Average time from Cam 1 to Cam 2 is 50s +/- 5s".
