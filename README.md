# MCMT-ReID: Multi-Camera Multi-Target Person Re-Identification System

A real-time surveillance system that tracks individuals across multiple cameras using deep learning-based person re-identification. The system combines YOLOv8 detection, DeepSORT tracking, and OSNet feature extraction for seamless cross-camera tracking.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [Docker Deployment (Alternative)](#docker-deployment-alternative)
- [Accessing the Application](#accessing-the-application)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before setting up the project, ensure you have the following installed:

### Required Software

| Software      | Version  | Download Link                                                                 |
|---------------|----------|-------------------------------------------------------------------------------|
| **Python**    | 3.11+    | [python.org](https://www.python.org/downloads/)                              |
| **Node.js**   | 18.x+    | [nodejs.org](https://nodejs.org/)                                            |
| **PostgreSQL**| 15+      | [postgresql.org](https://www.postgresql.org/download/)                       |
| **Redis**     | 7+       | [redis.io](https://redis.io/download/) or use Docker                         |
| **Git**       | Latest   | [git-scm.com](https://git-scm.com/downloads)                                 |

### Optional (For GPU Acceleration)

| Software          | Version | Notes                                    |
|-------------------|---------|------------------------------------------|
| **NVIDIA CUDA**   | 11.8+   | Required for GPU-accelerated ML models   |
| **cuDNN**         | 8.6+    | NVIDIA Deep Neural Network library       |
| **NVIDIA GPU**    | -       | CUDA-capable GPU recommended             |

---

## Project Structure

```
MAINEL/
├── backend/               # FastAPI backend (Python)
│   ├── app/               # Application source code
│   │   ├── api/           # REST API endpoints
│   │   ├── core/          # ML pipeline (detection, tracking, ReID)
│   │   ├── db/            # Database models and utilities
│   │   ├── schemas/       # Pydantic request/response models
│   │   ├── services/      # Business logic
│   │   └── workers/       # Background processors
│   ├── model_weights/     # ML model files (YOLOv8, OSNet)
│   ├── requirements.txt   # Python dependencies
│   ├── Dockerfile         # Docker configuration
│   └── docker-compose.yml # Docker Compose configuration
│
├── frontend/              # Next.js frontend (TypeScript)
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Node.js dependencies
│
└── README.md              # This file
```

---

## Backend Setup

### Step 1: Navigate to the Backend Directory

```bash
cd backend
```

### Step 2: Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### Step 4: Install TorchReID (Required for OSNet)

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

### Step 5: Configure Environment Variables

Copy the example environment file and edit it with your settings:

```bash
copy .env.example .env
```

Edit the `.env` file with your configuration:

```env
# Application
APP_NAME=MCMT-ReID API
DEBUG=true

# Database (Update with your PostgreSQL credentials)
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/mcmt_reid

# Redis
REDIS_URL=redis://localhost:6379/0

# OSRM (Optional - for path interpolation)
OSRM_URL=http://localhost:5000

# ML Models
YOLO_MODEL_PATH=model_weights/yolov8n.pt
YOLO_CONFIDENCE=0.5
OSNET_MODEL_PATH=model_weights/osnet_x1_0.pth
REID_MATCH_THRESHOLD=0.6

# Device (use 'cuda' for GPU or 'cpu' for CPU-only)
DEVICE=cuda

# Tracking
DEEPSORT_MAX_AGE=30
DEEPSORT_N_INIT=3

# Spatial-Temporal
ST_WEIGHT=0.5
MAX_TRANSITION_TIME=300.0
```

### Step 6: Setup PostgreSQL Database

Create the database:

```sql
CREATE DATABASE mcmt_reid;
```

Or via command line:

```bash
psql -U postgres -c "CREATE DATABASE mcmt_reid;"
```

### Step 7: Run Database Migrations

```bash
alembic upgrade head
```

### Step 8: Download Model Weights (Auto-downloads on first run)

**YOLOv8** automatically downloads on first run. Alternatively, manually download:

```bash
mkdir model_weights
cd model_weights
# YOLOv8 nano model will auto-download
```

---

## Frontend Setup

### Step 1: Navigate to the Frontend Directory

```bash
cd frontend
```

### Step 2: Install Node.js Dependencies

```bash
npm install
```

Or using other package managers:

```bash
# Using yarn
yarn install

# Using pnpm
pnpm install
```

---

## Running the Application

### Start the Backend Server

Open a terminal in the `backend` directory and run:

```bash
# Activate virtual environment first (if not already active)
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at: `http://localhost:8000`

### Start the Frontend Development Server

Open another terminal in the `frontend` directory and run:

```bash
npm run dev
```

The frontend will be available at: `http://localhost:3000`

---

## Docker Deployment (Alternative)

For quick deployment using Docker:

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Build and Run with Docker Compose

```bash
docker-compose up -d
```

This will start:
- **API Server** on port `8000`
- **PostgreSQL** (with PostGIS) on port `5432`
- **Redis** on port `6379`

### View Logs

```bash
docker-compose logs -f api
```

### Stop Services

```bash
docker-compose down
```

---

## Accessing the Application

| Service             | URL                           | Description                          |
|---------------------|-------------------------------|--------------------------------------|
| **Frontend**        | http://localhost:3000         | Web interface                        |
| **Backend API**     | http://localhost:8000         | REST API                             |
| **API Documentation** | http://localhost:8000/docs  | Swagger UI (interactive docs)        |
| **ReDoc**           | http://localhost:8000/redoc   | Alternative API documentation        |
| **Health Check**    | http://localhost:8000/health  | Server health status                 |

---

## Troubleshooting

### Common Issues

#### 1. Webcam Not Working / Black Frames

If your webcam shows black frames:

1. **Check Windows Privacy Settings:**
   - Go to **Settings → Privacy & Security → Camera**
   - Ensure **Camera access** is ON
   - Enable **Let desktop apps access your camera**

2. **Close Other Camera Apps:**
   - Close Zoom, Teams, Discord, or any app using the camera

3. **Check Physical Privacy Shutter:**
   - Some laptops have a physical camera cover/slider

4. **Antivirus Software:**
   - Temporarily disable "Webcam Protection" in your antivirus

#### 2. CUDA/GPU Not Detected

- Ensure NVIDIA drivers are installed
- Verify CUDA installation: `nvidia-smi`
- Set `DEVICE=cpu` in `.env` to use CPU instead

#### 3. Database Connection Error

- Ensure PostgreSQL is running
- Verify credentials in `.env`
- Check if the database `mcmt_reid` exists

#### 4. Redis Connection Error

- Ensure Redis server is running
- On Windows, use Docker or WSL for Redis:
  ```bash
  docker run -d -p 6379:6379 redis:7-alpine
  ```

#### 5. Module Not Found Errors

- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Quick Start Summary

```bash
# Terminal 1 - Backend
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env       # Configure your settings
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm run dev
```

Open your browser and navigate to `http://localhost:3000` to use the application.

---

## License

This project is for educational and research purposes.
