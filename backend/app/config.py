"""
Application configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "MCMT-ReID API"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/mcmt_reid",
        description="PostgreSQL connection string"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # OSRM (Routing)
    osrm_url: str = Field(
        default="http://localhost:5000",
        description="OSRM server URL for route interpolation"
    )
    
    # ML Models
    yolo_model_path: str = Field(
        default="model_weights/yolov8n.pt",
        description="Path to YOLOv8 model weights"
    )
    yolo_confidence: float = Field(
        default=0.7,  # Higher threshold reduces false positives (shadows, ground patterns)
        ge=0.0,
        le=1.0,
        description="YOLO detection confidence threshold"
    )
    yolo_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="YOLO NMS IoU threshold"
    )
    
    osnet_model_path: Optional[str] = Field(
        default=None,
        description="Path to OSNet model weights (None for auto-download)"
    )
    reid_embedding_dim: int = Field(
        default=1024,
        description="Dimension of ReID feature embeddings"
    )
    reid_match_threshold: float = Field(
        default=0.5,  # Raised from 0.4 to reduce duplicate gallery entries
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for ReID matching (higher = more lenient)"
    )
    
    # Tracking
    deepsort_max_age: int = Field(
        default=30,
        description="Max frames before track deletion"
    )
    deepsort_n_init: int = Field(
        default=3,
        description="Frames to confirm a track"
    )
    deepsort_max_iou_distance: float = Field(
        default=0.7,
        description="Max IOU distance for matching"
    )
    
    # Spatial-Temporal
    st_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight of ST score in joint matching"
    )
    max_transition_time: float = Field(
        default=300.0,
        description="Max seconds between camera transitions"
    )
    
    # Device
    device: str = Field(
        default="cuda",
        description="Compute device: 'cuda' or 'cpu'"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export singleton
settings = get_settings()
