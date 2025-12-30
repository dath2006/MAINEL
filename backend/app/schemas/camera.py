"""
Pydantic schemas for Camera models.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CameraBase(BaseModel):
    """Base camera schema with common fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Camera name")
    latitude: float = Field(..., ge=-90, le=90, description="GPS latitude")
    longitude: float = Field(..., ge=-180, le=180, description="GPS longitude")
    zone_id: Optional[int] = Field(None, description="Zone/area identifier")
    fov_angle: Optional[float] = Field(None, ge=0, le=360, description="Field of view angle")
    stream_url: Optional[str] = Field(None, description="RTSP/HTTP stream URL")
    description: Optional[str] = Field(None, description="Camera description")


class CameraCreate(CameraBase):
    """Schema for creating a new camera."""
    pass


class CameraUpdate(BaseModel):
    """Schema for updating camera (all fields optional)."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    zone_id: Optional[int] = None
    fov_angle: Optional[float] = Field(None, ge=0, le=360)
    stream_url: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class CameraResponse(CameraBase):
    """Schema for camera response."""
    id: int
    is_active: bool = True
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class CameraWithStats(CameraResponse):
    """Camera response with tracking statistics."""
    total_detections: int = 0
    active_tracks: int = 0
    last_detection_at: Optional[datetime] = None
