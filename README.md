# MCMT-ReID: Multi-Camera Multi-Target Person Re-Identification System

A real-time surveillance system that tracks individuals across multiple cameras using deep learning-based person re-identification. The system combines YOLOv8 detection, DeepSORT tracking, and OSNet feature extraction for seamless cross-camera tracking.


---

- First Install docker and docker compose in your system, see any yt video for it if needed.
- Install cuda-toolkit for GPU support using: https://developer.nvidia.com/cuda-12-1-0-download-archive
- Clone this repository from github: git clone https://github.com/dath2006/MAINEL.git
- Then for testing the system,you download the videos from:https://drive.google.com/file/d/1e3ZhyrqKd-E9KbJIixlGB_vnsZB-zNsd/view?usp=drive_link , https://drive.google.com/file/d/1dOoTA8qq7hZEdc4E4lXu4r-PQQWWzbL2/view?usp=drive_link , https://drive.google.com/file/d/1Rbj4jWvF00kxSu9Vv85WRQKrva5igPLm/view?usp=drive_link. and place them in the backend/app/videos folder. NOTE that you need to use the path as "app/videos/<video-name>" while adding in the frontend.
Also take a pic of a person from the video, so that it can be used for searching the person. 
- see this video: https://drive.google.com/file/d/14qRnaSkuR0EZ1vJIKzrFeGK2pu9BbgYn/view?usp=sharing

## Docker Development (Recommended)

The easiest way to set up the entire project is using Docker. This provides:
- **Hot-reloading** for both backend and frontend
- **Automatic package installation** when you modify `requirements.txt` or `package.json`
- **No local dependency conflicts** - everything runs in isolated containers

### Quick Start

```bash
# Clone and navigate to project
cd MAINEL

# Copy environment template
copy .env.docker backend\.env

# Start all services (CPU mode)
docker compose up --build

# OR for GPU/CUDA support (requires NVIDIA Container Toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

This will start:
- **Frontend** at `http://localhost:2999`
- **Backend API** at `http://localhost:7999`
- **PostgreSQL** (PostGIS) at port `5431`
- **Redis** at port `6378`

### CPU vs GPU Mode

Configure in `backend/.env`:
```env
# For CPU-only (default, works on all machines)
DEVICE=cpu

# For GPU acceleration (requires NVIDIA GPU + CUDA)
DEVICE=cuda
```

When using `DEVICE=cuda`, also include the GPU compose file:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Installing New Python Packages

```bash
# Option 0: Add to requirements.txt and restart
echo "new-package==0.0.0" >> backend/requirements.txt
docker compose restart backend

# Option 1: Install directly in running container
docker compose exec backend pip install new-package
```

### Installing New npm Packages

```bash
# Install in running container
docker compose exec frontend npm install new-package

# The package.json will be updated automatically
```

- See after running the docker, the frontend will be running at `http://localhost:3000` and the backend will be running at `http://localhost:8000`
- And if you want to stop the docker, you can use the following command:
```bash
docker compose down or press ctrl + c in terminal
```
- Then again to start the docker, you can use the following command:
```bash
docker compose up
```

### Useful Docker Commands

```bash
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Rebuild a specific service
docker compose up --build backend

# Enter a container shell
docker compose exec backend bash
docker compose exec frontend sh

# Stop all services
docker compose down

# Stop and remove all data (clean slate)
docker compose down -v
```

## Docker Development END

--- 
## The following setup is to run the project without docker
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
