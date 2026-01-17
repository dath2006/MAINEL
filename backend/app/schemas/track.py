"""
Pydantic schemas for Track models.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field


class TrackStatus(str, Enum):
    """Track lifecycle status."""
    ACTIVE = "active"
    LOST = "lost"
    FINISHED = "finished"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: float = Field(..., description="Top-left x coordinate")
    y: float = Field(..., description="Top-left y coordinate")
    width: float = Field(..., ge=0, description="Box width")
    height: float = Field(..., ge=0, description="Box height")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")


class Detection(BaseModel):
    """Single detection result."""
    camera_id: int
    timestamp: datetime
    bbox: BoundingBox
    class_id: int = 0
    class_name: str = "person"
    embedding: Optional[List[float]] = Field(None, description="Feature embedding vector")

    @property
    def x1(self) -> float: return self.bbox.x
    @property
    def y1(self) -> float: return self.bbox.y
    @property
    def x2(self) -> float: return self.bbox.x + self.bbox.width
    @property
    def y2(self) -> float: return self.bbox.y + self.bbox.height
    @property
    def confidence(self) -> float: return self.bbox.confidence

    def to_xyah(self) -> List[float]:
        """Convert to (center x, center y, aspect ratio, height)."""
        center_x = self.bbox.x + self.bbox.width / 2
        center_y = self.bbox.y + self.bbox.height / 2
        aspect_ratio = self.bbox.width / max(1e-6, self.bbox.height)
        return [center_x, center_y, aspect_ratio, self.bbox.height]


class TrackletBase(BaseModel):
    """Base tracklet (single-camera track) schema."""
    camera_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    exit_zone: Optional[str] = Field(None, description="Exit direction: left, right, top, bottom")


class TrackletCreate(TrackletBase):
    """Schema for creating a tracklet."""
    feature_vector: List[float] = Field(..., description="Average feature embedding")


class TrackletResponse(TrackletBase):
    """Tracklet response schema."""
    id: UUID
    duration_seconds: Optional[float] = None
    
    class Config:
        from_attributes = True


class GlobalTrackBase(BaseModel):
    """Base global track (cross-camera identity) schema."""
    status: TrackStatus = TrackStatus.ACTIVE


class GlobalTrackCreate(GlobalTrackBase):
    """Schema for creating a global track."""
    first_camera_id: int
    first_tracklet_id: UUID


class GlobalTrackResponse(GlobalTrackBase):
    """Global track response schema."""
    id: UUID
    first_seen: datetime
    last_seen: datetime
    camera_sequence: List[int] = Field(default_factory=list, description="Cameras visited in order")
    tracklet_count: int = 0
    
    class Config:
        from_attributes = True


class GlobalTrackDetail(GlobalTrackResponse):
    """Detailed global track with trajectory."""
    tracklets: List[TrackletResponse] = Field(default_factory=list)
    interpolated_path: Optional[List[dict]] = Field(None, description="OSRM interpolated path")


class TransitEvent(BaseModel):
    """Cross-camera transition event."""
    global_track_id: UUID
    from_camera_id: int
    to_camera_id: int
    from_tracklet_id: UUID
    to_tracklet_id: UUID
    transit_time_seconds: float
    visual_similarity: float
    st_probability: float
    joint_score: float
    interpolated_path: Optional[List[dict]] = None
    timestamp: datetime


class TrackSearchQuery(BaseModel):
    """Query parameters for track search."""
    camera_ids: Optional[List[int]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[TrackStatus] = None
    min_duration_seconds: Optional[float] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)
