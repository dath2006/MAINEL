"""Features module."""

from app.core.features.nvidia_reid_extractor import NvidiaReIDExtractor
from app.core.features.base import BaseFeatureExtractor

__all__ = [
    "BaseFeatureExtractor",
    "NvidiaReIDExtractor",
]
