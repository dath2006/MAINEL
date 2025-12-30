"""
Camera Management API Endpoints

CRUD operations for camera registration and configuration.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraWithStats,
)


router = APIRouter()

# In-memory storage (replace with database in production)
_cameras: dict[int, dict] = {}
_next_camera_id = 1


@router.post("/", response_model=CameraResponse, status_code=201)
async def create_camera(camera: CameraCreate):
    """
    Register a new camera.
    
    Creates a camera node in the tracking topology with GPS coordinates.
    """
    global _next_camera_id
    
    camera_id = _next_camera_id
    _next_camera_id += 1
    
    camera_data = {
        "id": camera_id,
        "name": camera.name,
        "latitude": camera.latitude,
        "longitude": camera.longitude,
        "zone_id": camera.zone_id,
        "fov_angle": camera.fov_angle,
        "stream_url": camera.stream_url,
        "description": camera.description,
        "is_active": True,
        "created_at": datetime.utcnow(),
        "updated_at": None,
    }
    
    _cameras[camera_id] = camera_data
    logger.info(f"Created camera {camera_id}: {camera.name}")
    
    return CameraResponse(**camera_data)


@router.get("/", response_model=List[CameraResponse])
async def list_cameras(
    zone_id: Optional[int] = Query(None, description="Filter by zone"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
):
    """
    List all registered cameras.
    
    Optionally filter by zone or active status.
    """
    cameras = list(_cameras.values())
    
    if zone_id is not None:
        cameras = [c for c in cameras if c["zone_id"] == zone_id]
    
    if is_active is not None:
        cameras = [c for c in cameras if c["is_active"] == is_active]
    
    return [CameraResponse(**c) for c in cameras]


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: int):
    """Get camera details by ID."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    return CameraResponse(**_cameras[camera_id])


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(camera_id: int, camera: CameraUpdate):
    """Update camera configuration."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    camera_data = _cameras[camera_id]
    update_data = camera.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        camera_data[field] = value
    
    camera_data["updated_at"] = datetime.utcnow()
    
    logger.info(f"Updated camera {camera_id}")
    return CameraResponse(**camera_data)


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: int):
    """Remove camera from system."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    del _cameras[camera_id]
    logger.info(f"Deleted camera {camera_id}")


@router.post("/{camera_id}/activate", response_model=CameraResponse)
async def activate_camera(camera_id: int):
    """Activate camera for tracking."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    _cameras[camera_id]["is_active"] = True
    _cameras[camera_id]["updated_at"] = datetime.utcnow()
    
    logger.info(f"Activated camera {camera_id}")
    return CameraResponse(**_cameras[camera_id])


@router.post("/{camera_id}/deactivate", response_model=CameraResponse)
async def deactivate_camera(camera_id: int):
    """Deactivate camera from tracking."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    _cameras[camera_id]["is_active"] = False
    _cameras[camera_id]["updated_at"] = datetime.utcnow()
    
    logger.info(f"Deactivated camera {camera_id}")
    return CameraResponse(**_cameras[camera_id])


@router.get("/{camera_id}/stats", response_model=CameraWithStats)
async def get_camera_stats(camera_id: int):
    """Get camera with tracking statistics."""
    if camera_id not in _cameras:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    camera_data = _cameras[camera_id].copy()
    
    # TODO: Get actual stats from tracking service
    camera_data.update({
        "total_detections": 0,
        "active_tracks": 0,
        "last_detection_at": None,
    })
    
    return CameraWithStats(**camera_data)
