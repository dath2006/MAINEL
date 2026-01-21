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
        default=0.50,  # INCREASED from 0.40 to reduce false merges
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for matching to existing identity"
    )
    reid_new_threshold: float = Field(
        default=0.50,  # INCREASED from 0.50 to create new IDs more readily
        ge=0.0,
        le=1.0,
        description="Threshold for creating new identity (if best match below this, create new)"
    )
    reid_merge_threshold: float = Field(
        default=0.65,  # INCREASED from 0.50 to prevent false merges
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
    
    # ReID Enhancement Settings - Phase 1
    # Quality Gating
    reid_quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality score to add feature to bank"
    )
    reid_diversity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Maximum similarity to existing features (diversity constraint)"
    )
    reid_feature_bank_size: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Maximum embeddings to store per identity (increased from 10)"
    )
    reid_bbox_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum bbox confidence for feature quality"
    )
    
    # Occlusion Detection & Handling
    reid_occlusion_iou_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="IoU threshold to consider track occluded"
    )
    reid_enable_occlusion_detection: bool = Field(
        default=True,
        description="Enable occlusion state tracking"
    )
    reid_enable_id_correction: bool = Field(
        default=True,
        description="Enable post-occlusion ID verification and correction"
    )
    reid_post_occlusion_similarity_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for ID swap detection"
    )
    
    # Quality Scoring Weights
    reid_blur_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    reid_occlusion_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    reid_illumination_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    reid_confidence_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Blur Detection Thresholds (Laplacian variance)
    reid_min_blur_variance: float = Field(
        default=50.0,
        description="Laplacian variance below this is considered blurry"
    )
    reid_max_blur_variance: float = Field(
        default=100.0,
        description="Laplacian variance above this is very sharp"
    )
    
    # Gallery Quality Filters (Phase 1+)
    reid_min_bbox_width: int = Field(
        default=32,  # Further relaxed from 48
        description="Minimum person width in pixels"
    )
    reid_min_bbox_height: int = Field(
        default=64,  # Further relaxed from 96
        description="Minimum person height in pixels"
    )
    reid_min_aspect_ratio: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Minimum height/width ratio for person detection"
    )
    reid_max_aspect_ratio: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description="Maximum height/width ratio for person detection"
    )
    reid_min_frame_coverage: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Minimum bbox area as fraction of frame (0.5%)"
    )
    reid_min_detection_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence for gallery"
    )
    
    # Person Presence Validation (Phase 2)
    reid_enable_presence_check: bool = Field(
        default=True,  # Now enabled
        description="Validate crop contains actual person"
    )
    reid_min_crop_variance: float = Field(
        default=200.0,
        description="Minimum pixel variance (empty backgrounds too uniform)"
    )
    reid_min_edge_density: float = Field(
        default=0.02,
        description="Minimum edge pixel ratio (persons have edges)"
    )
    reid_min_color_entropy: float = Field(
        default=1.5,
        description="Minimum color diversity (persons varied colors)"
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
