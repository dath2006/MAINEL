# ============================================
# MCMT-ReID Backend - GPU-enabled Dockerfile
# ============================================
# Uses NVIDIA CUDA base image for GPU acceleration
# Supports PeopleNet + DeepSORT + Nvidia ReID pipeline

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==2.2.6 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY preprocessor/ preprocessor/
COPY alembic/ alembic/
COPY alembic.ini .

# Create directories for model weights and uploads
RUN mkdir -p model_weights uploads videos

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV DEVICE=cuda

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
