"""Tracking module."""

from app.core.tracking.kalman import KalmanFilter
from app.core.tracking.bytetrack import (
    Track,
    TrackState,
    CrossCameraTrackState,
    OcclusionInfo,
    ByteTrackTracker,
    DeepSORTTracker,  # Alias for backward compatibility
)
from app.core.tracking.base import BaseTracker

__all__ = [
    "KalmanFilter",
    "Track",
    "TrackState",
    "CrossCameraTrackState",
    "OcclusionInfo",
    "ByteTrackTracker",
    "DeepSORTTracker",  # Alias pointing to ByteTrackTracker
    "BaseTracker",
]
