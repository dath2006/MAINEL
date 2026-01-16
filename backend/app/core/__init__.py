"""Core module init."""

from app.core.detection import (
    Detection,
)
from app.core.tracking import (
    Track,
    TrackState,
    DeepSORTTracker,
    KalmanFilter,
)
from app.core.features import (
    NvidiaReIDExtractor,
)
from app.core.reid import (
    VisualMatcher,
    SpatioTemporalScorer,
    CameraTopology,
)

__all__ = [
    # Detection
    "Detection",
    # Tracking
    "Track",
    "TrackState",
    "DeepSORTTracker",
    "KalmanFilter",
    # Features
    "NvidiaReIDExtractor",
    # ReID
    "VisualMatcher",
    "SpatioTemporalScorer",
    "CameraTopology",
]
