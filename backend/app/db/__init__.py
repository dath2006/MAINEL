"""Database package."""

from app.db.session import (
    Base,
    engine,
    async_session_factory,
    get_db,
    get_db_context,
    init_db,
    close_db,
)
from app.db.models import (
    Camera,
    Tracklet,
    GlobalTrack,
    TransitEvent,
    CameraTransition,
    TrackStatus,
)
from app.db.repositories import (
    CameraRepository,
    TrackletRepository,
    GlobalTrackRepository,
)

__all__ = [
    # Session
    "Base",
    "engine",
    "async_session_factory",
    "get_db",
    "get_db_context",
    "init_db",
    "close_db",
    # Models
    "Camera",
    "Tracklet",
    "GlobalTrack",
    "TransitEvent",
    "CameraTransition",
    "TrackStatus",
    # Repositories
    "CameraRepository",
    "TrackletRepository",
    "GlobalTrackRepository",
]
