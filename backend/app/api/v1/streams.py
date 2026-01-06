"""
Stream Control API

REST endpoints for managing video/camera sources and playback control.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
import os
import shutil
from loguru import logger

from app.services.stream_manager import (
    get_stream_manager,
    SourceType,
    PlaybackState,
)


router = APIRouter(prefix="/streams", tags=["streams"])


# Request/Response models
class SourceTypeEnum(str, Enum):
    video_file = "video_file"
    webcam = "webcam"
    rtsp = "rtsp"


class AddSourceRequest(BaseModel):
    camera_id: int
    source_type: SourceTypeEnum
    source_path: str  # File path, camera index, or RTSP URL
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class SourceResponse(BaseModel):
    id: int
    camera_id: int
    name: str
    source_type: str
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    current_frame: int
    is_active: bool
    latitude: Optional[float]
    longitude: Optional[float]


class PlaybackStatus(BaseModel):
    state: str
    source_count: int
    target_fps: float
    queue_size: int
    sources: List[dict]


class FPSRequest(BaseModel):
    fps: float


# Upload directory
UPLOAD_DIR = "uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/sources", response_model=SourceResponse)
async def add_source(request: AddSourceRequest):
    """Add a video/camera source."""
    manager = get_stream_manager()
    
    # Validate source path based on type
    if request.source_type == SourceTypeEnum.video_file:
        if not os.path.exists(request.source_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Video file not found: {request.source_path}"
            )
    
    # Map source type
    source_type_map = {
        SourceTypeEnum.video_file: SourceType.VIDEO_FILE,
        SourceTypeEnum.webcam: SourceType.WEBCAM,
        SourceTypeEnum.rtsp: SourceType.RTSP,
    }
    
    logger.info(f"Adding source: camera_id={request.camera_id}, type={request.source_type}, path={request.source_path}")
    
    try:
        source = await manager.add_source(
            camera_id=request.camera_id,
            source_path=request.source_path,
            source_type=source_type_map[request.source_type],
            name=request.name or "",
            latitude=request.latitude,
            longitude=request.longitude,
        )
    except Exception as e:
        logger.error(f"Failed to add source: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to open source: {str(e)}")
    
    if source is None:
        raise HTTPException(status_code=400, detail="Failed to open source. Check the file path or camera availability.")
    
    return SourceResponse(
        id=source.id,
        camera_id=source.camera_id,
        name=source.name,
        source_type=source.source_type.value,
        source_path=source.source_path,
        fps=source.fps,
        width=source.width,
        height=source.height,
        total_frames=source.total_frames,
        current_frame=source.current_frame,
        is_active=source.is_active,
        latitude=source.latitude,
        longitude=source.longitude,
    )


@router.post("/sources/upload")
async def upload_video(
    camera_id: int = Form(...),
    name: str = Form(""),
    file: UploadFile = File(...),
):
    """Upload a video file and add as source."""
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    logger.info(f"Saved uploaded video: {file_path}")
    
    # Add as source
    manager = get_stream_manager()
    source = await manager.add_source(
        camera_id=camera_id,
        source_path=file_path,
        source_type=SourceType.VIDEO_FILE,
        name=name or file.filename,
        latitude=0.0, # TODO: Add lat/lon to upload form if needed
        longitude=0.0,
    )
    
    if source is None:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Failed to open video file")
    
    return SourceResponse(
        id=source.id,
        camera_id=source.camera_id,
        name=source.name,
        source_type=source.source_type.value,
        source_path=source.source_path,
        fps=source.fps,
        width=source.width,
        height=source.height,
        total_frames=source.total_frames,
        current_frame=source.current_frame,
        is_active=source.is_active,
        latitude=source.latitude,
        longitude=source.longitude,
    )


@router.get("/sources", response_model=List[SourceResponse])
async def list_sources():
    """List all active sources."""
    manager = get_stream_manager()
    return [
        SourceResponse(
            id=s.id,
            camera_id=s.camera_id,
            name=s.name,
            source_type=s.source_type.value,
            source_path=s.source_path,
            fps=s.fps,
            width=s.width,
            height=s.height,
            total_frames=s.total_frames,
            current_frame=s.current_frame,
            is_active=s.is_active,
            latitude=s.latitude,
            longitude=s.longitude,
        )
        for s in manager.sources
    ]


@router.delete("/sources/{source_id}")
async def remove_source(source_id: int):
    """Remove a source."""
    manager = get_stream_manager()
    if not await manager.remove_source(source_id):
        raise HTTPException(status_code=404, detail="Source not found")
    return {"success": True}


@router.post("/play")
async def play():
    """Start or resume playback."""
    manager = get_stream_manager()
    manager.play()
    return {"state": manager.state.value}


@router.post("/pause")
async def pause():
    """Pause playback."""
    manager = get_stream_manager()
    manager.pause()
    return {"state": manager.state.value}


@router.post("/stop")
async def stop():
    """Stop playback and reset."""
    manager = get_stream_manager()
    manager.stop()
    return {"state": manager.state.value}


@router.get("/status", response_model=PlaybackStatus)
async def get_status():
    """Get playback status."""
    manager = get_stream_manager()
    status = manager.get_status()
    return PlaybackStatus(**status)


@router.get("/debug-cam")
async def debug_cam():
    """Debug webcam access from main thread."""
    import cv2
    import base64
    
    # Try to open camera 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return {"status": "error", "detail": "Could not open camera 0"}
    
    # Read frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return {"status": "error", "detail": "Could not read frame"}
    
    # Encode
    _, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": "success",
        "detail": "Camera opened and frame captured",
        "frame_size": len(b64),
        "frame_preview": b64[:50] + "..."
    }


@router.post("/fps")
async def set_fps(request: FPSRequest):
    """Set target FPS."""
    manager = get_stream_manager()
    manager.target_fps = request.fps
    return {"fps": manager.target_fps}


@router.get("/gallery")
async def get_person_gallery():
    """Get all tracked persons with their thumbnails and global IDs."""
    from app.services.reid_service import get_reid_service
    
    reid_service = get_reid_service()
    gallery = reid_service.get_gallery()
    
    return {
        "persons": gallery,
        "total": len(gallery),
    }


@router.delete("/gallery")
async def clear_gallery():
    """Clear the person gallery."""
    from app.services.reid_service import get_reid_service
    
    reid_service = get_reid_service()
    reid_service.clear_gallery()
    
    return {"status": "success", "message": "Gallery cleared"}

