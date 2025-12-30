"""Features module."""

from app.core.features.osnet_extractor import (
    OSNetExtractor,
    get_extractor,
)
from app.core.features.base import BaseFeatureExtractor

__all__ = [
    "OSNetExtractor",
    "get_extractor",
    "BaseFeatureExtractor",
]
