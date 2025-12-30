# MCMT-ReID Backend

Multi-Camera Multi-Target Person Re-Identification API

## Quick Start

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model Weights
```bash
# YOLOv8 (auto-downloads on first run, or manually)
pip install ultralytics
yolo export model=yolov8n.pt format=onnx  # Downloads weights

# OSNet (from torchreid model zoo)
# Will auto-download pretrained weights on first use
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 4. Run Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Access API
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api
```

## API Endpoints

### Cameras
- `POST /api/v1/cameras/` - Register camera
- `GET /api/v1/cameras/` - List cameras
- `GET /api/v1/cameras/{id}` - Get camera
- `PUT /api/v1/cameras/{id}` - Update camera
- `DELETE /api/v1/cameras/{id}` - Delete camera

### Tracks
- `GET /api/v1/tracks/active` - Get active tracks
- `GET /api/v1/tracks/{id}` - Get track details
- `POST /api/v1/tracks/search` - Search tracks

### WebSocket
- `WS /api/v1/ws/tracks` - Real-time tracking stream

## Project Structure

```
backend/
├── app/
│   ├── api/v1/          # API endpoints
│   ├── core/            # ML pipeline
│   │   ├── detection/   # YOLOv8 detector
│   │   ├── tracking/    # DeepSORT tracker
│   │   ├── features/    # OSNet extractor
│   │   └── reid/        # ReID matching
│   ├── services/        # Business logic
│   ├── db/              # Database layer
│   └── schemas/         # Pydantic models
├── tests/               # Test suite
├── model_weights/       # ML model files
├── Dockerfile
└── docker-compose.yml
```

## Tech Stack

- **FastAPI** - High-performance async web framework
- **YOLOv8** - Real-time person detection
- **DeepSORT** - Multi-object tracking
- **OSNet** - Person re-identification features
- **PostgreSQL/PostGIS** - Geospatial database
- **Redis** - Message queue and caching
- **WebSocket** - Real-time updates
