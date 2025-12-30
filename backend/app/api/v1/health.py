"""
Health check endpoints.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
import torch

from app.config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    cuda_available: bool
    cuda_device: str | None


class SystemInfo(BaseModel):
    """System information response."""
    app_name: str
    version: str
    debug: bool
    device: str
    cuda_available: bool
    cuda_device_name: str | None
    database_configured: bool
    redis_configured: bool
    osrm_configured: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    cuda_available = torch.cuda.is_available()
    cuda_device = None
    if cuda_available:
        cuda_device = torch.cuda.get_device_name(0)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
    )


@router.get("/info", response_model=SystemInfo)
async def system_info():
    """Get detailed system information."""
    cuda_available = torch.cuda.is_available()
    cuda_device_name = None
    if cuda_available:
        cuda_device_name = torch.cuda.get_device_name(0)
    
    return SystemInfo(
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        device=settings.device,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        database_configured=bool(settings.database_url),
        redis_configured=bool(settings.redis_url),
        osrm_configured=bool(settings.osrm_url),
    )
