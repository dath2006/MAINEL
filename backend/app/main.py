"""
Multi-Camera Multi-Target Re-Identification API

This is the main FastAPI application entry point.
Provides REST endpoints and WebSocket connections for real-time tracking.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.api.v1 import router as api_v1_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Device: {settings.device}")
    
    # Initialize database
    from app.db import init_db, close_db
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")
    
    # Initialize Redis
    from app.workers import get_redis_client, close_redis
    try:
        await get_redis_client()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection skipped: {e}")
    
    # Initialize ML models (lazy loading)
    logger.info("ML models will be loaded on first request (lazy loading)")
    
    # Start stream processor
    from app.workers.stream_processor import start_stream_processor
    start_stream_processor()
    logger.info("Stream processor started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop stream processor
    from app.workers.stream_processor import stop_stream_processor
    stop_stream_processor()
    
    # Cleanup stream manager
    from app.services.stream_manager import get_stream_manager
    get_stream_manager().cleanup()
    
    try:
        await close_db()
    except Exception:
        pass
    
    try:
        await close_redis()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
        ## Multi-Camera Multi-Target Person Re-Identification API
        
        This API provides endpoints for:
        - **Camera Management**: Register and configure surveillance cameras
        - **Real-time Tracking**: WebSocket streams for live tracking updates
        - **Track Queries**: Search and retrieve person trajectories
        - **ReID Operations**: Feature extraction and identity matching
        
        ### Architecture
        - YOLOv8 for person detection
        - DeepSORT for single-camera tracking
        - OSNet for visual feature extraction
        - Spatial-Temporal scoring for cross-camera ReID
        - OSRM for blind-zone path interpolation
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
    
    return app


# Create application instance
app = create_app()


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
