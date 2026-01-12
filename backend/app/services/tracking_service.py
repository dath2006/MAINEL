"""
Tracking Service

Orchestrates detection, tracking, and feature extraction for video frames.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from uuid import UUID, uuid4
import numpy as np
from loguru import logger

from app.core.detection import YOLODetector, Detection, get_detector
from app.core.tracking import DeepSORTTracker, Track, TrackState
from app.core.features import OSNetExtractor, get_extractor
from app.config import settings


class CameraState:
    """Per-camera tracking state."""
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.tracker = DeepSORTTracker(
            max_age=settings.deepsort_max_age,
            n_init=settings.deepsort_n_init,
            max_iou_distance=settings.deepsort_max_iou_distance,
        )
        self.active_tracklets: Dict[int, UUID] = {}  # local_id -> tracklet_uuid
        self.last_frame_time: Optional[datetime] = None


class TrackingService:
    """
    Service for real-time person tracking.
    
    Manages per-camera trackers and coordinates the detection,
    tracking, and feature extraction pipeline.
    """
    
    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        extractor: Optional[OSNetExtractor] = None,
    ):
        self.detector = detector
        self.extractor = extractor
        
        # Per-camera state
        self.camera_states: Dict[int, CameraState] = {}
        
        # Track lifecycle callbacks (for ReID service integration)
        self._on_tracklet_start: List[callable] = []
        self._on_tracklet_end: List[callable] = []
        
        logger.info("TrackingService initialized")
    
    def _get_detector(self) -> YOLODetector:
        """Lazy load detector."""
        if self.detector is None:
            # Prefer ONNX path if use_onnx is enabled and ONNX model exists
            import os
            model_path = settings.yolo_model_path
            if settings.use_onnx and settings.yolo_onnx_path:
                if os.path.exists(settings.yolo_onnx_path):
                    model_path = settings.yolo_onnx_path
            
            self.detector = get_detector(
                model_path=model_path,
                confidence=settings.yolo_confidence,
                iou_threshold=settings.yolo_iou_threshold,
                device=settings.device,
                use_onnx=settings.use_onnx,
            )
        return self.detector
    
    def _get_extractor(self) -> OSNetExtractor:
        """Lazy load extractor."""
        if self.extractor is None:
            import os
            model_path = None
            # Prefer NVIDIA TAO model
            if settings.use_nvidia_reid and settings.nvidia_reid_onnx_path:
                if os.path.exists(settings.nvidia_reid_onnx_path):
                     model_path = settings.nvidia_reid_onnx_path

            self.extractor = get_extractor(
                model_path=model_path,
                device=settings.device,
                use_onnx=settings.use_onnx,
                use_nvidia=settings.use_nvidia_reid,
            )
        return self.extractor
    
    def _get_camera_state(self, camera_id: int) -> CameraState:
        """Get or create camera state."""
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraState(camera_id)
            logger.info(f"Created camera state for camera {camera_id}")
        return self.camera_states[camera_id]
    
    async def process_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: datetime,
        extract_features: bool = True,
    ) -> Tuple[List[Track], List[Tuple[UUID, np.ndarray]]]:
        """
        Process a single video frame.
        
        Args:
            camera_id: Camera identifier
            frame: Frame as numpy array (H, W, C) in BGR
            timestamp: Frame timestamp
            extract_features: Whether to extract ReID features
            
        Returns:
            Tuple of (active_tracks, new_tracklet_features)
        """
        state = self._get_camera_state(camera_id)
        detector = self._get_detector()
        
        # 1. Detection
        detections = detector.detect(frame)
        
        # 2. Feature extraction using FastReID (if enabled)
        features = None
        if extract_features and len(detections) > 0:
            extractor = self._get_extractor()
            crops = detector.crop_detections(frame, detections)
            if crops:
                # Extract body features using FastReID
                features = extractor.extract_batch(crops)
        
        # 3. Predict and update tracker
        state.tracker.predict()
        active_tracks = state.tracker.update(detections, features)
        
        # 4. Handle track lifecycle
        new_features = await self._handle_track_events(
            state, active_tracks, timestamp
        )
        
        state.last_frame_time = timestamp
        
        return active_tracks, new_features
    
    def extract_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract ReID features from a single image/crop.
        
        Args:
            image: BGR image (H, W, C)
        
        Returns:
            Feature vector (D,) or None if extraction fails
        """
        if image is None or image.size == 0:
            return None
            
        extractor = self._get_extractor()
        # extract_batch expects a list of images
        features = extractor.extract_batch([image])
        
        if features is not None and len(features) > 0:
            return features[0]
        return None
    
    async def _handle_track_events(
        self,
        state: CameraState,
        tracks: List[Track],
        timestamp: datetime,
    ) -> List[Tuple[UUID, np.ndarray]]:
        """Handle tracklet start/end events."""
        new_features = []
        current_track_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            current_track_ids.add(track.track_id)
            
            # New tracklet
            if track.track_id not in state.active_tracklets:
                tracklet_id = uuid4()
                state.active_tracklets[track.track_id] = tracklet_id
                
                # Get average feature if available
                if track.features:
                    avg_feature = np.mean(track.features, axis=0)
                    new_features.append((tracklet_id, avg_feature))
                
                # Notify callbacks
                for callback in self._on_tracklet_start:
                    await callback(
                        state.camera_id,
                        tracklet_id,
                        track.track_id,
                        timestamp,
                    )
        
        # Check for ended tracklets
        ended_ids = set(state.active_tracklets.keys()) - current_track_ids
        for local_id in ended_ids:
            tracklet_id = state.active_tracklets.pop(local_id)
            
            # Notify callbacks
            for callback in self._on_tracklet_end:
                await callback(
                    state.camera_id,
                    tracklet_id,
                    local_id,
                    timestamp,
                )
        
        return new_features
    
    def get_active_tracks(self, camera_id: int) -> List[Track]:
        """Get currently active tracks for a camera."""
        if camera_id not in self.camera_states:
            return []
        
        return [
            t for t in self.camera_states[camera_id].tracker.tracks
            if t.is_confirmed()
        ]
    
    def get_track_count(self, camera_id: int) -> int:
        """Get number of active tracks."""
        return len(self.get_active_tracks(camera_id))
    
    def reset_camera(self, camera_id: int):
        """Reset tracking state for a camera."""
        if camera_id in self.camera_states:
            del self.camera_states[camera_id]
            logger.info(f"Reset camera state for camera {camera_id}")
    
    def on_tracklet_start(self, callback: callable):
        """Register callback for tracklet start events."""
        self._on_tracklet_start.append(callback)
    
    def on_tracklet_end(self, callback: callable):
        """Register callback for tracklet end events."""
        self._on_tracklet_end.append(callback)


# Service singleton
_tracking_service: Optional[TrackingService] = None


def get_tracking_service() -> TrackingService:
    """Get or create tracking service singleton."""
    global _tracking_service
    if _tracking_service is None:
        _tracking_service = TrackingService()
    return _tracking_service
