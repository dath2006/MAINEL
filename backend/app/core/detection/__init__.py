"""Detection module."""

from app.core.detection.yolo_detector import (
    Detection,
    YOLODetector,
    get_detector,
)
from app.core.detection.base import BaseDetector

__all__ = [
    "Detection",
    "YOLODetector",
    "get_detector",
    "BaseDetector",
]
