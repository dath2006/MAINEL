"""Pydantic schemas package."""

from app.schemas.camera import (
    CameraBase,
    CameraCreate,
    CameraUpdate,
    CameraResponse,
    CameraWithStats,
)
from app.schemas.track import (
    TrackStatus,
    BoundingBox,
    Detection,
    TrackletBase,
    TrackletCreate,
    TrackletResponse,
    GlobalTrackBase,
    GlobalTrackCreate,
    GlobalTrackResponse,
    GlobalTrackDetail,
    TransitEvent,
    TrackSearchQuery,
)
from app.schemas.detection import (
    DetectionRequest,
    DetectionResult,
    FrameDetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ReIDMatchRequest,
    ReIDMatchResult,
    ReIDMatchResponse,
)

__all__ = [
    # Camera
    "CameraBase",
    "CameraCreate", 
    "CameraUpdate",
    "CameraResponse",
    "CameraWithStats",
    # Track
    "TrackStatus",
    "BoundingBox",
    "Detection",
    "TrackletBase",
    "TrackletCreate",
    "TrackletResponse",
    "GlobalTrackBase",
    "GlobalTrackCreate",
    "GlobalTrackResponse",
    "GlobalTrackDetail",
    "TransitEvent",
    "TrackSearchQuery",
    # Detection
    "DetectionRequest",
    "DetectionResult",
    "FrameDetectionResponse",
    "BatchDetectionRequest",
    "BatchDetectionResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ReIDMatchRequest",
    "ReIDMatchResult",
    "ReIDMatchResponse",
]
