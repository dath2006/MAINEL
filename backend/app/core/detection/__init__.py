"""Detection module."""

from app.schemas.track import Detection
from app.core.detection.base import BaseDetector

__all__ = [
    "Detection",
    "BaseDetector",
]
