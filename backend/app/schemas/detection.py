"""
Pydantic schemas for Detection models.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request for processing a single frame."""
    camera_id: int
    timestamp: datetime
    frame_base64: str = Field(..., description="Base64 encoded frame image")


class DetectionResult(BaseModel):
    """Single person detection result."""
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    confidence: float
    track_id: Optional[int] = Field(None, description="Local track ID within camera")
    embedding: Optional[List[float]] = Field(None, description="Feature embedding if extracted")


class FrameDetectionResponse(BaseModel):
    """Response from frame detection."""
    camera_id: int
    timestamp: datetime
    frame_number: int
    detections: List[DetectionResult]
    processing_time_ms: float


class BatchDetectionRequest(BaseModel):
    """Request for processing multiple frames."""
    camera_id: int
    frames: List[dict] = Field(..., description="List of {timestamp, frame_base64}")


class BatchDetectionResponse(BaseModel):
    """Response from batch detection."""
    camera_id: int
    results: List[FrameDetectionResponse]
    total_processing_time_ms: float


class EmbeddingRequest(BaseModel):
    """Request for feature extraction from a person crop."""
    image_base64: str = Field(..., description="Base64 encoded person crop image")


class EmbeddingResponse(BaseModel):
    """Feature embedding response."""
    embedding: List[float]
    embedding_dim: int
    processing_time_ms: float


class ReIDMatchRequest(BaseModel):
    """Request to find matching identity."""
    query_embedding: List[float]
    camera_id: int
    timestamp: datetime
    top_k: int = Field(default=5, ge=1, le=100)


class ReIDMatchResult(BaseModel):
    """Single ReID match result."""
    global_track_id: str
    visual_similarity: float
    st_probability: Optional[float] = None
    joint_score: float
    last_camera_id: int
    last_seen: datetime


class ReIDMatchResponse(BaseModel):
    """ReID matching response."""
    matches: List[ReIDMatchResult]
    best_match: Optional[ReIDMatchResult] = None
    is_new_identity: bool
    processing_time_ms: float
