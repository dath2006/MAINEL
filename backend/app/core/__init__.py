"""Core module init."""

from app.core.detection import (
    Detection,
    YOLODetector,
    get_detector,
)
from app.core.tracking import (
    Track,
    TrackState,
    DeepSORTTracker,
    KalmanFilter,
)
from app.core.features import (
    OSNetExtractor,
    get_extractor,
)
from app.core.reid import (
    VisualMatcher,
    SpatioTemporalScorer,
    CameraTopology,
)

__all__ = [
    # Detection
    "Detection",
    "YOLODetector",
    "get_detector",
    # Tracking
    "Track",
    "TrackState",
    "DeepSORTTracker",
    "KalmanFilter",
    # Features
    "OSNetExtractor",
    "get_extractor",
    # ReID
    "VisualMatcher",
    "SpatioTemporalScorer",
    "CameraTopology",
]
