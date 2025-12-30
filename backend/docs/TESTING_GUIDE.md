# MCMT-ReID API Testing Guide

## Quick Start

1. **Start the server:**
   ```powershell
   cd d:\MAINEL\backend
   uvicorn app.main:app --reload
   ```

2. **Open Swagger UI:** http://127.0.0.1:8000/docs

---

## Testing Endpoints

### 1. Health Check
```
GET /health
GET /health/system
```
Verify CUDA is available and system status.

---

### 2. Camera Management

**Create a Camera:**
```json
POST /api/v1/cameras
{
  "name": "Entrance Camera",
  "latitude": 12.9716,
  "longitude": 77.5946,
  "fov_angle": 90,
  "stream_url": "rtsp://camera1/stream"
}
```

**List Cameras:**
```
GET /api/v1/cameras
```

---

### 3. Real-Time WebSocket

Connect to: `ws://localhost:8000/api/v1/ws/tracks`

**Subscribe to cameras:**
```json
{"action": "subscribe", "cameras": [1, 2]}
```

**Expected events:**
```json
{"type": "detection", "camera_id": 1, "track_count": 3, ...}
{"type": "reid_match", "global_track_id": "...", "score": 0.85}
```

---

## Testing with Real Images

### Option A: Run Test Script with Real Person Image

```powershell
python -m tests.test_real_images
```

### Option B: Use cURL/PowerShell

```powershell
# Base64 encode an image and POST to detection endpoint
$imageBytes = [System.IO.File]::ReadAllBytes("path/to/person.jpg")
$base64 = [Convert]::ToBase64String($imageBytes)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/detect" -Method POST -Body (@{
    camera_id = 1
    frame_base64 = $base64
} | ConvertTo-Json) -ContentType "application/json"
```

---

## Testing with Video Stream

### Option 1: Submit frames via Python

```python
import cv2
import base64
import requests

cap = cv2.VideoCapture("path/to/video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode()
    
    requests.post("http://localhost:8000/api/v1/detect", json={
        "camera_id": 1,
        "frame_base64": frame_b64
    })
cap.release()
```

### Option 2: Use RTSP stream (via workers)

Configure in `.env`:
```
CAMERA_1_URL=rtsp://username:password@ip:port/stream
```

---

## Sample Test Data

Download test videos from:
- MOT Dataset: https://motchallenge.net/
- Market-1501: https://www.kaggle.com/datasets/pengcw1/market-1501

---

## Expected Results

| Test | Success Criteria |
|------|-----------------|
| Health | `{"status": "healthy", "cuda": true}` |
| Camera Create | Returns camera with `id` |
| Detection | Returns bounding boxes |
| Tracking | Track IDs persist across frames |
| ReID | Same person gets same global ID across cameras |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not available | Reinstall torch with CUDA |
| No detections | Check confidence threshold in `.env` |
| Database error | Start PostgreSQL via Docker |
| WebSocket disconnects | Check Redis is running |
