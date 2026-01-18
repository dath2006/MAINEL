"""
API v1 Router

Aggregates all API v1 endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.cameras import router as cameras_router
from app.api.v1.tracks import router as tracks_router
from app.api.v1.realtime import router as realtime_router
from app.api.v1.streams import router as streams_router
from app.api.v1.video_library import router as video_library_router

router = APIRouter()

# Include all v1 routers
router.include_router(health_router, tags=["Health"])
router.include_router(cameras_router, prefix="/cameras", tags=["Cameras"])
router.include_router(tracks_router, prefix="/tracks", tags=["Tracks"])
router.include_router(realtime_router, prefix="/ws", tags=["WebSocket"])
router.include_router(streams_router, tags=["Streams"])
router.include_router(video_library_router, tags=["Video Library"])

