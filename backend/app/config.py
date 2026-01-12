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
    
    # FastReID Settings (SOTA ReID) - Using BoT R50-ibn for better generalization
    fastreid_config_path: str = Field(
        default="fastreid_lib/configs/Market1501/bagtricks_R50-ibn.yml",
        description="Path to FastReID config YAML"
    )
    fastreid_weights_path: str = Field(
        default="model_weights/market_bot_R50-ibn.pth",
        description="Path to FastReID pretrained weights"
    )
    fastreid_onnx_path: Optional[str] = Field(
        default="model_weights/fastreid_bot_R50-ibn.onnx",
        description="Path to FastReID ONNX model"
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
        description="Path to OSNet model weights (fallback if FastReID/NVIDIA unavailable)"
    )
    reid_embedding_dim: int = Field(
        default=256,  # Changed from 2048/512 for NVIDIA model
        description="Dimension of ReID feature embeddings (256 for NVIDIA, 2048 for FastReID)"
    )
    reid_match_threshold: float = Field(
        default=0.45,  # Adjusted for NVIDIA model (0.45=Low, 0.65=Medium, 0.80=High)
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for ReID matching"
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
    
    # Thumbnail Settings
    min_thumbnail_quality: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to save/update thumbnail"
    )
    search_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for image search (lower = more results)"
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
