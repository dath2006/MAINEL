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
    yolo_version: str = Field(
        default="auto",
        description="YOLO version: 'v8', 'v10', or 'auto' (v10 is NMS-free, faster)"
    )
    yolov10_model_path: str = Field(
        default="model_weights/yolov10n.pt",
        description="Path to YOLOv10-Nano weights (NMS-free, fastest)"
    )
    

    osnet_model_path: Optional[str] = Field(
        default=None,
        description="Path to OSNet model weights (None for auto-download)"
    )
    reid_embedding_dim: int = Field(
        default=512,
        description="Dimension of ReID feature embeddings"
    )
    reid_match_threshold: float = Field(
        default=0.3,  # Lowered from 0.4 for better cross-camera matching
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for ReID matching (lower = more lenient)"
    )
    use_multi_scale_reid: bool = Field(
        default=True,
        description="Enable multi-scale feature pooling for more robust ReID (reduces false matches on similar clothing)"
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
    deepsort_backbone: str = Field(
        default="resnet50",
        description="DeepSORT appearance backbone: 'resnet50' (accurate), 'osnet', or 'resnet18' (fast)"
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
    
    # Face ReID Settings (Multi-Modal)
    use_face_reid: bool = Field(
        default=True,
        description="Enable face+body multi-modal ReID matching"
    )
    face_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for face embedding in fusion (body_weight = 1 - face_weight)"
    )
    body_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for body embedding in fusion"
    )
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
    
    # Quality Control (The "Garbage Collector")
    quality_min_sharpness: float = Field(
        default=60.0,
        description="Laplacian variance threshold for blur (higher = stricter, < 60 is usually blurry)"
    )
    quality_min_size: int = Field(
        default=40,
        description="Minimum width/height in pixels for valid feature extraction"
    )
    
    # =============================================
    # ReID Two-Threshold System
    # =============================================
    reid_confirm_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="High-confidence threshold for matching (above = confirmed same person)"
    )
    reid_new_identity_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Low-confidence threshold (below = definitely new person)"
    )
    reid_candidate_threshold: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Threshold for getting initial gallery candidates before two-threshold filtering"
    )
    
    # =============================================
    # Gallery Update Thresholds
    # =============================================
    gallery_merge_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity required to blend (merge) embeddings in gallery"
    )
    gallery_quality_delta: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Quality improvement required to replace exemplar embedding"
    )
    gallery_max_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of identities to track in gallery"
    )
    
    # =============================================
    # K-Reciprocal Reranking Parameters
    # =============================================
    rerank_k1: int = Field(
        default=20,
        ge=1,
        description="K-reciprocal reranking initial neighbor count"
    )
    rerank_k2: int = Field(
        default=6,
        ge=1,
        description="K-reciprocal reranking expansion parameter"
    )
    rerank_lambda: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Balance between original and Jaccard distance in reranking"
    )
    
    # =============================================
    # Face Detection & Fusion Thresholds
    # =============================================
    face_det_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="InsightFace detection confidence threshold"
    )
    face_quality_low_tier: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Below this quality, face is ignored in fusion (body only)"
    )
    face_quality_high_tier: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Above this quality, face is heavily trusted in fusion"
    )
    face_high_tier_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Face weight when quality is above high tier"
    )
    face_medium_tier_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Face weight when quality is between low and high tier"
    )
    
    # =============================================
    # Spatial-Temporal Parameters
    # =============================================
    st_min_speed: float = Field(
        default=0.5,
        ge=0.1,
        description="Minimum pedestrian walking speed (m/s)"
    )
    st_avg_speed: float = Field(
        default=1.4,
        ge=0.1,
        description="Average pedestrian walking speed (m/s)"
    )
    st_max_speed: float = Field(
        default=3.0,
        ge=0.1,
        description="Maximum pedestrian speed (m/s), fast walk or jog"
    )
    st_bandwidth: float = Field(
        default=5.0,
        ge=1.0,
        description="Parzen window bandwidth for transition time estimation (seconds)"
    )
    st_plausibility_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum ST probability for plausible camera transition"
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
