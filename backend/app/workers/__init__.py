"""Workers package."""

from app.workers.redis_client import (
    RedisClient,
    get_redis_client,
    close_redis,
    STREAM_DETECTIONS,
    STREAM_TRACKLETS,
    STREAM_TRANSITS,
    STREAM_ALERTS,
    CHANNEL_REALTIME,
)
from app.workers.osrm_client import (
    OSRMClient,
    get_osrm_client,
    close_osrm,
    RouteResult,
)
from app.workers.frame_processor import (
    FrameProcessor,
    submit_frame,
)

__all__ = [
    # Redis
    "RedisClient",
    "get_redis_client",
    "close_redis",
    "STREAM_DETECTIONS",
    "STREAM_TRACKLETS",
    "STREAM_TRANSITS",
    "STREAM_ALERTS",
    "CHANNEL_REALTIME",
    # OSRM
    "OSRMClient",
    "get_osrm_client",
    "close_osrm",
    "RouteResult",
    # Frame Processor
    "FrameProcessor",
    "submit_frame",
]
