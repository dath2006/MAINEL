"""
Video Library API

REST endpoints for managing the uploaded video library.
Allows uploading videos to a library and reusing them for multiple sources.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
import shutil
import cv2
from uuid import UUID
from loguru import logger

from app.db.session import get_db_context
from app.db.models import VideoMetadata
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError


router = APIRouter(prefix="/video-library", tags=["Video Library"])


# Upload directory  
UPLOAD_DIR = "uploads/videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Response models
class VideoMetadataResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    file_path: str
    file_size: int
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    uploaded_at: datetime
    last_used: Optional[datetime]
    use_count: int
    description: Optional[str]
    tags: List[str]
    
    class Config:
        from_attributes = True


def video_to_response(video: VideoMetadata) -> VideoMetadataResponse:
    """Convert VideoMetadata model to response with proper UUID serialization."""
    return VideoMetadataResponse(
        id=str(video.id),
        filename=video.filename,
        original_filename=video.original_filename,
        file_path=video.file_path,
        file_size=video.file_size,
        duration=video.duration,
        fps=video.fps,
        width=video.width,
        height=video.height,
        total_frames=video.total_frames,
        uploaded_at=video.uploaded_at,
        last_used=video.last_used,
        use_count=video.use_count,
        description=video.description,
        tags=video.tags or [],
    )


@router.get("/videos", response_model=List[VideoMetadataResponse])
async def list_videos():
    """List all videos in the library."""
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(VideoMetadata).order_by(VideoMetadata.uploaded_at.desc())
            )
            videos = result.scalars().all()
            
            return [video_to_response(v) for v in videos]
    except Exception as e:
        logger.error(f"Failed to list videos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")


@router.post("/upload", response_model=VideoMetadataResponse)
async def upload_to_library(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),  # Comma-separated
):
    """
    Upload a video to the library without creating a source.
    The video can later be selected when adding sources.
    """
    import uuid
    
    try:
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.normpath(os.path.join(UPLOAD_DIR, unique_filename)).replace('\\', '/')
        
        # Save file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Get video properties
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Parse tags
        tag_list = [t.strip() for t in tags.split(",")] if tags else []
        
        # Save to database
        async with get_db_context() as db:
            video_meta = VideoMetadata(
                filename=unique_filename,
                original_filename=file.filename,
                file_path=file_path,
                file_size=file_size,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                description=description,
                tags=tag_list,
            )
            
            db.add(video_meta)
            await db.commit()
            await db.refresh(video_meta)
            
            logger.info(f"Uploaded video to library: {file.filename} -\u003e {unique_filename}")
            
            return video_to_response(video_meta)
    
    except Exception as e:
        # Cleanup file if database save fails
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/videos/{video_id}", response_model=VideoMetadataResponse)
async def get_video_info(video_id: str):
    """Get detailed information about a specific video."""
    try:
        async with get_db_context() as db:
            result = await db.execute(
                select(VideoMetadata).where(VideoMetadata.id == UUID(video_id))
            )
            video = result.scalar_one_or_none()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            return video_to_response(video)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video info: {str(e)}")


@router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video from the library and filesystem."""
    try:
        async with get_db_context() as db:
            # Get video metadata
            result = await db.execute(
                select(VideoMetadata).where(VideoMetadata.id == UUID(video_id))
            )
            video = result.scalar_one_or_none()
            
            if not video:
                raise HTTPException(status_code=404, detail="Video not found")
            
            # Check if video is in use
            if video.use_count > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot delete video: currently used by {video.use_count} source(s)"
                )
            
            # Delete file from filesystem (try with normpath too)
            normalized_path = os.path.normpath(video.file_path)
            if os.path.exists(normalized_path):
                os.remove(normalized_path)
                logger.info(f"Deleted video file: {normalized_path}")
            elif os.path.exists(video.file_path):
                os.remove(video.file_path)
                logger.info(f"Deleted video file: {video.file_path}")
            else:
                logger.warning(f"Video file not found for deletion: {video.file_path} (normalized: {normalized_path})")
            
            # Delete from database
            await db.execute(
                delete(VideoMetadata).where(VideoMetadata.id == UUID(video_id))
            )
            await db.commit()
            
            logger.info(f"Deleted video from library: {video.filename}")
            
            return {"success": True, "message": "Video deleted successfully"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")
