"""Features module."""

from app.core.features.osnet_extractor import (
    OSNetExtractor,
    get_extractor,
)
from app.core.features.nvidia_reid_extractor import NvidiaReIDExtractor
from app.core.features.face_extractor import (
    QualityScorer,
    get_quality_scorer,
)
from app.core.features.base import BaseFeatureExtractor

__all__ = [
    "OSNetExtractor",
    "get_extractor",
    "BaseFeatureExtractor",
    "QualityScorer",
    "get_quality_scorer",
]
