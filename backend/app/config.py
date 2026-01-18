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
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "https://mainel.vercel.app",
            "https://*.vercel.app",  # Allow all Vercel preview deployments
        ],
        description="Allowed CORS origins"
    )
    
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
        default=0.5,
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
    
    
    # NVIDIA TAO ReID Settings (Preferred)
    nvidia_reid_onnx_path: str = Field(
        default="model_weights/resnet50_market1501_aicity156.onnx",
        description="Path to NVIDIA TAO ReIdentificationNet ONNX model"
    )
    use_nvidia_reid: bool = Field(
        default=True,
        description="Use NVIDIA TAO model for ReID (recommended)"
    )

    # OSNet (Legacy) Settings
    osnet_model_path: Optional[str] = Field(
        default=None,
        description="Path to OSNet model weights (fallback if NVIDIA unavailable)"
    )
    reid_embedding_dim: int = Field(
        default=256,  # Changed from 2048/512 for NVIDIA model
        description="Dimension of ReID feature embeddings (256 for NVIDIA)"
    )
    reid_match_threshold: float = Field(
        default=0.40,  # Lower threshold for cross-camera matching (domain shift)
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for matching to existing identity"
    )
    reid_new_threshold: float = Field(
        default=0.50,  # Higher threshold - only create new ID if no match above this
        ge=0.0,
        le=1.0,
        description="Threshold for creating new identity (if best match below this, create new)"
    )
    reid_merge_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Threshold for merging fragmented identities in search results"
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
    
    # ONNX Runtime Settings
    use_onnx: bool = Field(
        default=True,
        description="Use ONNX Runtime for inference (faster on supported GPUs)"
    )
    yolo_onnx_path: Optional[str] = Field(
        default="model_weights/yolov8n.onnx",
        description="Path to ONNX YOLO model (exported from .pt)"
    )
    osnet_onnx_path: Optional[str] = Field(
        default="model_weights/osnet_x1_0.onnx",
        description="Path to ONNX OSNet model"
    )
    onnx_gpu_mem_limit: int = Field(
        default=4,
        description="GPU memory limit for ONNX Runtime in GB"
    )
    
    # Gallery & Search Settings
    search_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for image search (lower = more results)"
    )
    gallery_quality_threshold: float = Field(
        default=55.0,
        ge=0.0,
        le=100.0,
        description="Minimum quality score (0-100) to save captures to gallery"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export singleton
settings = get_settings()
