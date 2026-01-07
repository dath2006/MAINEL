"""
Stream Manager

Manages multiple video/camera sources for synchronized playback and processing.
Supports: video files, webcams (by index), RTSP/IP cameras.
"""

import asyncio
import threading
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import cv2
import numpy as np
from loguru import logger
from sqlalchemy import select
from app.db.session import get_db_context
from app.db.models import Camera


class SourceType(str, Enum):
    """Type of video source."""
    VIDEO_FILE = "video_file"
    WEBCAM = "webcam"
    RTSP = "rtsp"
    HTTP = "http"


class PlaybackState(str, Enum):
    """Playback state."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass
class VideoSource:
    """Represents a video source (file, webcam, or stream)."""
    id: int
    camera_id: int
    source_type: SourceType
    source_path: str  # File path, camera index as string, or URL
    name: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    fps: float = 30.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    current_frame: int = 0
    is_active: bool = False
    _capture: Optional[cv2.VideoCapture] = field(default=None, repr=False)
    
    def open(self) -> bool:
        """Open the video source with a timeout."""
        def _open_cam():
            try:
                if self.source_type == SourceType.WEBCAM:
                    # Specific backend preference can help avoiding hangs on Windows
                    # CAP_DSHOW is often faster/safer on Windows for webcams
                    self._capture = cv2.VideoCapture(int(self.source_path), cv2.CAP_DSHOW)
                else:
                    # Debug: Check file existence
                    import os
                    if isinstance(self.source_path, str) and not self.source_path.startswith(('rtsp://', 'http://', 'https://')):
                        if not os.path.exists(self.source_path):
                             logger.error(f"FILE NOT FOUND: {self.source_path} - Check path spelling or permissions.")
                        else:
                             logger.info(f"File found: {self.source_path}, Size: {os.path.getsize(self.source_path)}")
                    
                    self._capture = cv2.VideoCapture(self.source_path)
                
                if self._capture.isOpened():
                    self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
                    self.total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
            except Exception as e:
                logger.error(f"Error inside open thread: {e}")

        try:
            # Run open in a thread to allow timeout
            t = threading.Thread(target=_open_cam, daemon=True)
            t.start()
            t.join(timeout=5.0)  # 5 second timeout
            
            if t.is_alive():
                logger.error(f"Timeout opening source: {self.source_path}")
                # We can't easily kill the thread in Python, but we can return False
                return False

            if self._capture is None or not self._capture.isOpened():
                logger.error(f"Failed to open source: {self.source_path}")
                return False
            
            self.is_active = True
            logger.info(f"Opened source {self.id}: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Error opening source {self.source_path}: {e}")
            return False
    
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        if self._capture is None or not self._capture.isOpened():
            return False, None
        
        ret, frame = self._capture.read()
        if ret:
            self.current_frame = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame
    
    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame (video files only)."""
        if self._capture is None or self.source_type != SourceType.VIDEO_FILE:
            return False
        
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
        return True
    
    def close(self):
        """Close the video source."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.is_active = False
    
    def reset(self):
        """Reset to beginning (video files only)."""
        if self.source_type == SourceType.VIDEO_FILE:
            self.seek(0)


@dataclass
class FrameData:
    """Frame data packet."""
    source_id: int
    camera_id: int
    frame: np.ndarray
    frame_number: int
    timestamp: datetime
    source_fps: float


class StreamManager:
    """
    Manages multiple video sources with synchronized playback.
    
    Features:
    - Add/remove sources dynamically
    - Synchronized play/pause/stop
    - Frame queue for processing
    - Callback-based frame delivery
    """
    
    def __init__(
        self,
        target_fps: float = 30.0,
        frame_queue_size: int = 100,
    ):
        self.target_fps = target_fps
        self.frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=frame_queue_size)
        
        self._sources: Dict[int, VideoSource] = {}
        self._next_source_id = 1
        self._state = PlaybackState.STOPPED
        self._threads: Dict[int, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_frame_callbacks: List[Callable[[FrameData], None]] = []
        
        logger.info(f"StreamManager initialized (target_fps={target_fps})")
        # Async loading must be called explicitly via load_initial_state()

    async def load_initial_state(self):
        """Load active cameras from database (Async)."""
        logger.info("Loading initial state from DB...")
        try:
            async with get_db_context() as db:
                result = await db.execute(select(Camera).where(Camera.is_active == True))
                cameras = result.scalars().all()
                
                for cam in cameras:
                    logger.info(f"Restoring camera {cam.id} from DB")
                    # Use internal sync logic to restore state without re-saving to DB
                    self._add_source_internal(
                        camera_id=cam.id,
                        source_path=cam.stream_url or "0",
                        source_type=SourceType.WEBCAM if (cam.stream_url or "").isdigit() else SourceType.VIDEO_FILE,
                        name=cam.name,
                        latitude=cam.latitude,
                        longitude=cam.longitude,
                    )
        except Exception as e:
            logger.error(f"Failed to load sources from DB: {e}")


    
    @property
    def state(self) -> PlaybackState:
        return self._state
    
    @property
    def sources(self) -> List[VideoSource]:
        return list(self._sources.values())
    
    async def add_source(
        self,
        camera_id: int,
        source_path: str,
        source_type: SourceType = SourceType.VIDEO_FILE,
        name: str = "",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Optional[VideoSource]:
        """Add a video source (Async)."""
        # Call internal sync logic to start threads
        source = self._add_source_internal(
            camera_id, source_path, source_type, name, latitude, longitude
        )
        
        if source:
             # Async DB Persist
             await self._persist_source(camera_id, name, source_path, latitude, longitude)
        
        return source

    def _add_source_internal(
        self,
        camera_id: int,
        source_path: str,
        source_type: SourceType,
        name: str,
        latitude: float,
        longitude: float,
    ) -> Optional[VideoSource]:
        """Internal sync add source logic."""
        with self._lock:
            source = VideoSource(
                id=self._next_source_id,
                camera_id=camera_id,
                source_type=source_type,
                source_path=source_path,
                name=name or f"Source {self._next_source_id}",
                latitude=latitude,
                longitude=longitude,
            )
            
            if not source.open():
                return None
            
            self._sources[source.id] = source
            self._next_source_id += 1
            
            logger.info(f"Added source {source.id} for camera {camera_id}: {source_path}")
            return source

    async def _persist_source(self, camera_id: int, name: str, path: str, lat: float, lon: float):
        """Save source to database (Async)."""
        try:
            async with get_db_context() as db:
                result = await db.execute(select(Camera).where(Camera.id == camera_id))
                cam = result.scalar_one_or_none()
                
                if not cam:
                    cam = Camera(
                        id=camera_id,
                        name=name,
                        stream_url=path,
                        latitude=lat or 0.0,
                        longitude=lon or 0.0,
                        is_active=True
                    )
                    db.add(cam)
                else:
                    cam.is_active = True
                    cam.stream_url = path
                    cam.latitude = lat or 0.0
                    cam.longitude = lon or 0.0
                
                await db.commit()
        except Exception as e:
            logger.error(f"DB Persist Error: {e}")
    
    async def remove_source(self, source_id: int) -> bool:
        """Remove a video source (Async)."""
        with self._lock:
            if source_id not in self._sources:
                return False
            
            source = self._sources[source_id]
            
            # Stop the source thread
            source.close()
            del self._sources[source_id]
            
            logger.info(f"Removed source {source_id}")
            
        # Deactivate in DB
        try:
            async with get_db_context() as db:
                result = await db.execute(select(Camera).where(Camera.id == source.camera_id))
                cam = result.scalar_one_or_none()
                if cam:
                    cam.is_active = False
                    await db.commit()
        except Exception as e:
            logger.error(f"DB Removal Error: {e}")

        return True
    
    def get_source(self, source_id: int) -> Optional[VideoSource]:
        """Get source by ID."""
        return self._sources.get(source_id)
    
    def play(self):
        """Start or resume playback of all sources."""
        
        if self._state == PlaybackState.PAUSED:
            self._pause_event.set()
            self._state = PlaybackState.PLAYING
            logger.info("Resumed playback")
            return
        
        if self._state == PlaybackState.STOPPED:
            # Start fresh
            self._stop_event.clear()
            self._pause_event.set()
        
        self._state = PlaybackState.PLAYING
        
        # Start reader threads for each source (including newly added ones)
        for source_id, source in self._sources.items():
            if source_id not in self._threads or not self._threads[source_id].is_alive():
                logger.info(f"Starting thread for source {source_id}")
                thread = threading.Thread(
                    target=self._reader_loop,
                    args=(source,),
                    daemon=True,
                )
                self._threads[source_id] = thread
                thread.start()
        
        logger.info(f"Playback active with {len(self._sources)} sources")
    
    def pause(self):
        """Pause playback."""
        if self._state != PlaybackState.PLAYING:
            return
        
        self._pause_event.clear()
        self._state = PlaybackState.PAUSED
        logger.info("Paused playback")
    
    def stop(self):
        """Stop playback and reset all sources."""
        self._stop_event.set()
        self._pause_event.set()  # Unblock paused threads
        
        # Wait for threads to finish
        for thread in self._threads.values():
            thread.join(timeout=2.0)
        
        self._threads.clear()
        
        # Reset sources
        for source in self._sources.values():
            source.reset()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self._state = PlaybackState.STOPPED
        logger.info("Stopped playback")
    
    def _reader_loop(self, source: VideoSource):
        """Reader thread for a single source."""
        frame_interval = 1.0 / self.target_fps
        
        # Re-open camera/video in this thread
        logger.info(f"Reader thread starting for source {source.id}")
        
        # Close existing capture if any (transfer ownership to thread)
        if source._capture is not None:
            source._capture.release()
            source._capture = None
        
            # Open capture
        try:
            if source.source_type == SourceType.WEBCAM:
                # Force DirectShow backend on Windows (cv2.CAP_DSHOW = 700)
                logger.debug(f"Opening webcam {source.source_path} with CAP_DSHOW")
                source._capture = cv2.VideoCapture(int(source.source_path), cv2.CAP_DSHOW)
                source._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                source._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                logger.debug(f"Opening video file {source.source_path}")
                # Try to use Hardware Acceleration if available
                source._capture = cv2.VideoCapture(source.source_path, cv2.CAP_ANY)
                source._capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            
            if not source._capture.isOpened():
                logger.error(f"Failed to open source {source.id}: {source.source_path}")
                source.is_active = False
                return
                
            # Update source properties from the actual opened capture
            width = source._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = source._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = source._capture.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Smart Resize Target
            PROCESSING_WIDTH = 1280
            if width > PROCESSING_WIDTH:
                scale = PROCESSING_WIDTH / width
                width = PROCESSING_WIDTH
                height = int(height * scale)
                logger.info(f"Source {source.id} will be resized to {int(width)}x{int(height)}")

            if width > 0 and height > 0:
                source.width = int(width)
                source.height = int(height)
                source.fps = fps
                
        except Exception as e:
            logger.error(f"Exception opening source {source.id}: {e}")
            source.is_active = False
            return
        
        logger.info(f"Source {source.id} reader running: {source.width}x{source.height} @ {source.fps:.1f}fps")
        
        frame_count = 0
        consecutive_errors = 0
        MAX_ERRORS = 50  # Stop after 50 consecutive read errors
        
        while not self._stop_event.is_set():
            # Handle pause
            if self._pause_event.wait(timeout=0.1) is False:
                # If wait returns False (timeout), checks loop condition again
                # But here we want to block until set.
                self._pause_event.wait()
            
            if self._stop_event.is_set():
                break
            
            start_time = time.time()
            
            # Read frame
            try:
                ret, frame = source._capture.read()
                if ret:
                    # Smart Resize
                    if frame.shape[1] > PROCESSING_WIDTH:
                        frame = cv2.resize(frame, (source.width, source.height))
            except Exception as e:
                # Catch OpenCV errors (including OOM) and generic errors
                logger.warning(f"Error reading from source {source.id}: {e}")
                import gc
                gc.collect()
                time.sleep(0.1)
                continue

            if not ret:
                consecutive_errors += 1
                if source.source_type == SourceType.VIDEO_FILE:
                    # Video file ended - Loop it
                    logger.debug(f"Video source {source.id} ended, checking loop...")
                    
                    # Verify we aren't in an infinite fail loop
                    if consecutive_errors > 10 and source.total_frames < 2:
                         logger.error(f"Source {source.id} seems broken (read failed repeatedly). Stopping.")
                         break
                         
                    # Reset to beginning
                    source._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01) # output yield
                    continue
                else:
                    # Camera disconnected or failed
                    logger.warning(f"Source {source.id} read failed (attempt {consecutive_errors})")
                    if consecutive_errors > MAX_ERRORS:
                        logger.error(f"Source {source.id} too many errors. Stopping.")
                        break
                    time.sleep(0.1)
                    continue
            
            # Reset error count on success
            consecutive_errors = 0
            frame_count += 1
            source.current_frame = frame_count
            
            # Create frame data
            # Use current time for live, or calculate based on frame for strict file playback?
            # For this system, wall-clock time is better for realtime sync.
            frame_data = FrameData(
                source_id=source.id,
                camera_id=source.camera_id,
                frame=frame,
                frame_number=frame_count,
                timestamp=datetime.now(),
                source_fps=source.fps,
            )
            
            # Put in queue
            try:
                self.frame_queue.put_nowait(frame_data)
            except queue.Full:
                pass # Drop oldest? No, queue handles it.
            
            # Callbacks
            for callback in self._on_frame_callbacks:
                try:
                    callback(frame_data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            # FPS Control
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)
        
        # Cleanup
        if source._capture:
            source._capture.release()
        source.is_active = False
        logger.info(f"Reader thread stopped for source {source.id}")
    
    def on_frame(self, callback: Callable[[FrameData], None]):
        """Register a frame callback."""
        self._on_frame_callbacks.append(callback)
    
    def get_next_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get next frame from queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_status(self) -> dict:
        """Get current status."""
        return {
            "state": self._state.value,
            "source_count": len(self._sources),
            "target_fps": self.target_fps,
            "queue_size": self.frame_queue.qsize(),
            "sources": [
                {
                    "id": s.id,
                    "camera_id": s.camera_id,
                    "name": s.name,
                    "type": s.source_type.value,
                    "active": s.is_active,
                    "current_frame": s.current_frame,
                    "total_frames": s.total_frames,
                    "progress": s.current_frame / s.total_frames if s.total_frames > 0 else 0,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                }
                for s in self._sources.values()
            ],
        }
    
    def cleanup(self):
        """Cleanup all resources."""
        self.stop()
        for source in self._sources.values():
            source.close()
        self._sources.clear()
        logger.info("StreamManager cleaned up")


# Singleton instance
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get or create singleton StreamManager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
