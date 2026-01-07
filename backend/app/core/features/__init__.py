"""Features module."""

from app.core.features.osnet_extractor import (
    OSNetExtractor,
    get_extractor,
)
from app.core.features.face_extractor import (
    InsightFaceExtractor,
    FaceResult,
    QualityScorer,
    get_face_extractor,
    get_quality_scorer,
    create_fused_embedding,
)
from app.core.features.base import BaseFeatureExtractor

__all__ = [
    "OSNetExtractor",
    "get_extractor",
    "BaseFeatureExtractor",
    "InsightFaceExtractor",
    "FaceResult",
    "QualityScorer",
    "get_face_extractor",
    "get_quality_scorer",
    "create_fused_embedding",
]

