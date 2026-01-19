"""
Tracking Service

Orchestrates detection, tracking, and feature extraction for video frames.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from uuid import UUID, uuid4
import numpy as np
from loguru import logger

from app.schemas.track import Detection, BoundingBox
from app.core.tracking import DeepSORTTracker, Track, TrackState
from app.core.features import NvidiaReIDExtractor
from app.config import settings

# Import PeopleNet Detector
try:
    from preprocessor.peoplenet_detector import PeopleNetDetector
    PEOPLENET_AVAILABLE = True
except ImportError as e:
    PEOPLENET_AVAILABLE = False
    logger.error(f"Could not import PeopleNetDetector: {e}")



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
        detector: Optional[object] = None,
        extractor: Optional[NvidiaReIDExtractor] = None,
    ):

        self.detector = detector
        self.extractor = extractor
        
        # Per-camera state
        self.camera_states: Dict[int, CameraState] = {}
        
        # Track lifecycle callbacks (for ReID service integration)
        self._on_tracklet_start: List[callable] = []
        self._on_tracklet_end: List[callable] = []
        
        logger.info("TrackingService initialized")
    
    def _get_detector(self) -> object:
        """Lazy load detector (PeopleNet)."""
        if self.detector is None:
            if PEOPLENET_AVAILABLE:
                # Use PeopleNet
                peoplenet_path = getattr(settings, 'peoplenet_model_path', "model_weights/resnet34_peoplenet.onnx")
                
                logger.info(f"Initializing PeopleNetDetector from {peoplenet_path}")
                self.detector = PeopleNetDetector(
                    model_path=peoplenet_path,
                    device=settings.device,
                    confidence_threshold=settings.yolo_confidence, # Reuse yolo confidence
                )
            else:
                raise ImportError("PeopleNetDetector not available and YOLO fallback removed.")
        return self.detector


    
    def _get_extractor(self) -> NvidiaReIDExtractor:
        """Lazy load extractor."""
        if self.extractor is None:
            model_path = settings.nvidia_reid_onnx_path
            logger.info(f"Initializing NvidiaReIDExtractor from {model_path}")
            self.extractor = NvidiaReIDExtractor(
                model_path=model_path,
                device=settings.device
            )
        return self.extractor
    
    def _get_camera_state(self, camera_id: int) -> CameraState:
        """Get or create camera state."""
        if camera_id not in self.camera_states:
            self.camera_states[camera_id] = CameraState(camera_id)
            logger.info(f"Created camera state for camera {camera_id}")
        return self.camera_states[camera_id]
    
    async def process_frames_batch(
        self,
        frames_data: List[Tuple[int, np.ndarray, datetime]],
        extract_features: bool = True,
    ) -> Dict[int, Tuple[List[Track], List[Tuple[UUID, np.ndarray]]]]:
        """
        Process a batch of video frames (TensorRT optimized).
        
        Args:
            frames_data: List of (camera_id, frame, timestamp) tuples
            extract_features: Whether to extract ReID features
            
        Returns:
            Dict mapping camera_id to (active_tracks, new_tracklet_features)
        """
        if not frames_data:
            return {}
        
        detector = self._get_detector()
        
        # Check if detector supports batch processing
        if hasattr(detector, 'detect_batch'):
            # Batch detection (8-12x faster)
            frames = [f[1] for f in frames_data]  # Extract frames
            batch_detections = detector.detect_batch(frames, confidence_threshold=0.3)
            
            # Process each frame's detections individually
            results = {}
            for (camera_id, frame, timestamp), raw_dets in zip(frames_data, batch_detections):
                # Process this frame's detections
                tracks, features = await self._process_single_frame_detections(
                    camera_id, frame, timestamp, raw_dets, extract_features
                )
                results[camera_id] = (tracks, features)
            
            return results
        else:
            # Fallback to sequential processing
            results = {}
            for camera_id, frame, timestamp in frames_data:
                tracks, features = await self.process_frame(
                    camera_id, frame, timestamp, extract_features
                )
                results[camera_id] = (tracks, features)
            return results
    
    async def _process_single_frame_detections(
        self,
        camera_id: int,
        frame: np.ndarray,
        timestamp: datetime,
        raw_detections: List[dict],
        extract_features: bool = True,
    ) -> Tuple[List[Track], List[Tuple[UUID, np.ndarray]]]:
        """
        Process detections from a single frame (helper for batch processing).
        """
        state = self._get_camera_state(camera_id)
        
        # Separate persons and faces
        person_dets = [d for d in raw_detections if d.get('class_name') == 'person' and d.get('confidence', 0) >= 0.4]
        face_dets = [d for d in raw_detections if d.get('class_name') == 'face' and d.get('confidence', 0) >= 0.3]
        
        # Convert to Detection objects
        detections = []
        face_detections = []
        
        for d in person_dets:
            bbox = BoundingBox(
                x=float(d['x1']),
                y=float(d['y1']),
                width=float(d['x2'] - d['x1']),
                height=float(d['y2'] - d['y1']),
                confidence=float(d['confidence'])
            )
            detections.append(Detection(
                camera_id=camera_id,
                timestamp=timestamp,
                bbox=bbox,
                class_id=0,
                class_name="person"
            ))
        
        for f in face_dets:
            face_detections.append({
                'x1': float(f['x1']),
                'y1': float(f['y1']),
                'x2': float(f['x2']),
                'y2': float(f['y2']),
                'confidence': float(f['confidence'])
            })
        
        # Extract features and run tracking (rest of pipeline)
        features = None
        if extract_features and len(detections) > 0:
            extractor = self._get_extractor()
            
            # Crop detections
            h, w = frame.shape[:2]
            crops = []
            for det in detections:
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crops.append(frame[y1:y2, x1:x2])
            
            if crops:
                # Extract body features
                features = extractor.extract_batch(crops)
        
        # Run tracking (DeepSORT only accepts detections and features)
        state.tracker.predict()
        tracks = state.tracker.update(detections, features)
        
        new_features = []
        if features is not None and len(features) > 0:
            for track in tracks:
                if track.is_confirmed() and len(track.features) > 0:
                    new_features.append((track.track_id, track.features[-1]))
        
        return tracks, new_features
    
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
        face_detections = []  # Face boxes from PeopleNet
        
        # Check if detector is PeopleNetDetector by looking for its unique signature
        is_peoplenet = hasattr(detector, 'CLASSES') and hasattr(detector, 'detect')
        
        if is_peoplenet:
             # PeopleNet - get ALL detections (person, face, bag)
             # Use lower threshold (0.3) to capture faces, then filter persons by stricter threshold
             raw_all = detector.detect(frame, confidence_threshold=0.3)
             
             # Separate persons and faces with different confidence thresholds
             # Persons: use 0.4 to capture more detections, faces: use 0.3 for sensitivity
             person_dets = [d for d in raw_all if d.get('class_name') == 'person' and d.get('confidence', 0) >= 0.4]
             face_dets = [d for d in raw_all if d.get('class_name') == 'face' and d.get('confidence', 0) >= 0.3]
             
             logger.info(f"PeopleNet detected {len(person_dets)} persons, {len(face_dets)} faces (raw: {len(raw_all)})")
             
             # Debug: Log class breakdown from raw detections
             if len(raw_all) > 0:
                 class_counts = {}
                 for d in raw_all:
                     cn = d.get('class_name', 'unknown')
                     conf = d.get('confidence', 0)
                     if cn not in class_counts:
                         class_counts[cn] = []
                     class_counts[cn].append(conf)
                 for cn, confs in class_counts.items():
                     logger.info(f"  -> {cn}: {len(confs)} detections, conf range: {min(confs):.2f}-{max(confs):.2f}")
             
             # Convert person detections to Detection objects
             detections = []
             for d in person_dets:
                 bbox = BoundingBox(
                     x=float(d['x1']),
                     y=float(d['y1']),
                     width=float(d['x2'] - d['x1']),
                     height=float(d['y2'] - d['y1']),
                     confidence=float(d['confidence'])
                 )
                 detections.append(Detection(
                     camera_id=camera_id,
                     timestamp=timestamp,
                     bbox=bbox,
                     class_id=0,
                     class_name="person"
                 ))
             
             # Convert face detections for passthrough
             for f in face_dets:
                 face_detections.append({
                     'x1': float(f['x1']),
                     'y1': float(f['y1']),
                     'x2': float(f['x2']),
                     'y2': float(f['y2']),
                     'confidence': float(f['confidence'])
                 })
        elif hasattr(detector, 'detect_persons'):
              # Old PeopleNet API fallback
              raw_detections = detector.detect_persons(frame, confidence_threshold=settings.yolo_confidence)
              detections = []
              for d in raw_detections:
                  detections.append(Detection(
                      bbox=(float(d['x1']), float(d['y1']), float(d['x2']), float(d['y2'])),
                      confidence=float(d['confidence']),
                      class_id=0  # Person
                  ))
        else:
             raise ImportError("Unknown detector type or detector missing detect method.")
        
        # 2. Feature extraction (if enabled)
        features = None
        if extract_features and len(detections) > 0:
            extractor = self._get_extractor()
            # Crop detections
            if hasattr(detector, 'crop_detections'):
                 crops = detector.crop_detections(frame, detections)
            else:
                 # Manually crop if detector doesn't support it (PeopleNetDetector doesn't have crop_detections taking Detection objs)
                 h, w = frame.shape[:2]
                 crops = []
                 for det in detections:
                     x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                     x1, y1 = max(0, x1), max(0, y1)
                     x2, y2 = min(w, x2), min(h, y2)
                     if x2 > x1 and y2 > y1:
                         crops.append(frame[y1:y2, x1:x2])
            
            if crops:
                # Extract body features
                features = extractor.extract_batch(crops)
        
        # 3. Predict and update tracker
        state.tracker.predict()
        active_tracks = state.tracker.update(detections, features)
        
        # 3b. Associate face boxes with person tracks (IOU matching)
        def calc_iou(box1, box2):
            """Calculate IOU between two boxes [x1, y1, x2, y2]."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 < x1 or y2 < y1:
                return 0.0
            inter = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return inter / (area1 + area2 - inter + 1e-6)
        
        for track in active_tracks:
            person_box = track.to_tlbr()  # [x1, y1, x2, y2]
            best_face = None
            best_score = 0.0  # Combined score instead of just IOU
            
            logger.debug(f"Track {track.track_id}: person_box={person_box.tolist()}, checking {len(face_detections)} faces")
            
            for face in face_detections:
                face_box = [face['x1'], face['y1'], face['x2'], face['y2']]
                iou = calc_iou(person_box, face_box)
                
                # Check containment (face center inside person box)
                face_cx = (face_box[0] + face_box[2]) / 2
                face_cy = (face_box[1] + face_box[3]) / 2
                inside = (person_box[0] <= face_cx <= person_box[2] and
                          person_box[1] <= face_cy <= person_box[3])
                
                logger.debug(f"  Face {face_box}: IOU={iou:.3f}, inside={inside}, face_center=({face_cx:.0f}, {face_cy:.0f})")
                
                # Accept face if EITHER inside OR has reasonable IOU
                if inside or iou > 0.1:
                    score = iou + (0.5 if inside else 0.0)  # Bonus for containment
                    if score > best_score:
                        best_score = score
                        best_face = face_box
            
            track.face_bbox = best_face  # Will be None if no face found
            if best_face:
                logger.info(f"Track {track.track_id}: Associated face_bbox={best_face} (score={best_score:.3f})")
        
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
