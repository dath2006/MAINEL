"""
Frame Processing Worker

Background worker for processing video frames in batches.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import base64
import numpy as np
import cv2
from loguru import logger

from app.workers.redis_client import (
    get_redis_client,
    STREAM_DETECTIONS,
    STREAM_TRACKLETS,
    CHANNEL_REALTIME,
)
from app.services import get_tracking_service, get_reid_service


class FrameProcessor:
    """
    Background worker for video frame processing.
    
    Consumes frames from Redis stream, runs detection/tracking,
    and publishes results.
    """
    
    def __init__(
        self,
        consumer_group: str = "frame_processors",
        consumer_name: str = "processor_1",
    ):
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name
        self._running = False
    
    async def start(self):
        """Start processing loop."""
        self._running = True
        logger.info(f"Frame processor started: {self.consumer_name}")
        
        redis = await get_redis_client()
        tracking = get_tracking_service()
        reid = get_reid_service()
        
        while self._running:
            try:
                # Consume frame events
                events = await redis.consume_events(
                    stream=STREAM_DETECTIONS,
                    consumer_group=self.consumer_group,
                    consumer_name=self.consumer_name,
                    count=5,
                    block_ms=1000,
                )
                
                for event in events:
                    try:
                        await self._process_event(
                            event, tracking, reid, redis
                        )
                        await redis.ack_event(
                            STREAM_DETECTIONS,
                            self.consumer_group,
                            event["id"],
                        )
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Frame processor stopped: {self.consumer_name}")
    
    def stop(self):
        """Stop processing loop."""
        self._running = False
    
    async def _process_event(
        self,
        event: Dict,
        tracking,
        reid,
        redis,
    ):
        """Process a single frame event."""
        data = event["data"]
        event_type = event["type"]
        
        if event_type == "frame":
            await self._process_frame(data, tracking, reid, redis)
        elif event_type == "tracklet_end":
            await self._process_tracklet_end(data, reid, redis)
    
    async def _process_frame(
        self,
        data: Dict,
        tracking,
        reid,
        redis,
    ):
        """Process video frame."""
        camera_id = data["camera_id"]
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        # Decode frame
        frame_bytes = base64.b64decode(data["frame_base64"])
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.warning("Failed to decode frame")
            return
        
        # Run tracking
        tracks, new_features = await tracking.process_frame(
            camera_id, frame, timestamp
        )
        
        # Process new tracklets for ReID
        for tracklet_id, embedding in new_features:
            match_result = await reid.match_identity(
                camera_id, embedding, timestamp
            )
            
            # Publish match result
            await redis.publish(CHANNEL_REALTIME, {
                "type": "reid_match",
                "camera_id": camera_id,
                "tracklet_id": str(tracklet_id),
                "global_track_id": str(match_result.global_track_id),
                "is_new": match_result.is_new,
                "score": match_result.joint_score,
            })
        
        # Publish detection results
        detection_data = {
            "camera_id": camera_id,
            "timestamp": timestamp.isoformat(),
            "track_count": len(tracks),
            "tracks": [
                {
                    "id": t.track_id,
                    "bbox": t.to_tlbr().tolist(),
                    "state": t.state.value,
                }
                for t in tracks
            ],
        }
        
        await redis.publish(CHANNEL_REALTIME, {
            "type": "detections",
            **detection_data,
        })
    
    async def _process_tracklet_end(
        self,
        data: Dict,
        reid,
        redis,
    ):
        """Process tracklet end event."""
        from uuid import UUID
        
        tracklet_id = UUID(data["tracklet_id"])
        end_time = datetime.fromisoformat(data["end_time"])
        
        reid.end_tracklet(tracklet_id, end_time)
        
        # Publish event
        await redis.publish(CHANNEL_REALTIME, {
            "type": "tracklet_end",
            "tracklet_id": str(tracklet_id),
            "camera_id": data.get("camera_id"),
        })


async def submit_frame(
    camera_id: int,
    frame: np.ndarray,
    timestamp: Optional[datetime] = None,
):
    """
    Submit a frame for async processing.
    
    Args:
        camera_id: Camera identifier
        frame: Frame as numpy array (H, W, C)
        timestamp: Frame timestamp
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    # Encode frame
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode()
    
    redis = await get_redis_client()
    await redis.publish_event(
        STREAM_DETECTIONS,
        "frame",
        {
            "camera_id": camera_id,
            "timestamp": timestamp.isoformat(),
            "frame_base64": frame_base64,
        },
    )
