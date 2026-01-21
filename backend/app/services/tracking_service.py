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
from app.core.tracking import DeepSORTTracker, Track, TrackState, CrossCameraTrackState
from app.core.features import NvidiaReIDExtractor
from app.core.reid import QualityScorer
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
        
        # Quality scorer for feature assessment
        self.quality_scorer = QualityScorer(
            blur_weight=settings.reid_blur_weight,
            occlusion_weight=settings.reid_occlusion_weight,
            illumination_weight=settings.reid_illumination_weight,
            confidence_weight=settings.reid_confidence_weight,
            min_blur_variance=settings.reid_min_blur_variance,
            max_blur_variance=settings.reid_max_blur_variance,
        )
        
        # Track lifecycle callbacks (for ReID service integration)
        self._on_tracklet_start: List[callable] = []
        self._on_tracklet_end: List[callable] = []
        
        # Frame counter for occlusion tracking
        self.frame_counter: Dict[int, int] = {}  # camera_id -> frame_number
        
        logger.info("TrackingService initialized with QualityScorer")
    
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
             
            #  logger.info(f"PeopleNet detected {len(person_dets)} persons, {len(face_dets)} faces (raw: {len(raw_all)})")
             
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
        
        # 2. Feature extraction with quality assessment (if enabled)
        features = None
        quality_scores = []
        crops = []
        all_bboxes = []
        
        if extract_features and len(detections) > 0:
            extractor = self._get_extractor()
            
            # Prepare all bboxes for occlusion assessment
            for det in detections:
                all_bboxes.append(np.array([det.x1, det.y1, det.x2, det.y2]))
            
            # Crop detections
            if hasattr(detector, 'crop_detections'):
                 crops = detector.crop_detections(frame, detections)
            else:
                 # Manually crop if detector doesn't support it
                 h, w = frame.shape[:2]
                 for det in detections:
                     x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                     x1, y1 = max(0, x1), max(0, y1)
                     x2, y2 = min(w, x2), min(h, y2)
                     if x2 > x1 and y2 > y1:
                         crops.append(frame[y1:y2, x1:x2])
                     else:
                         crops.append(None)  # Invalid crop
            
            # Assess quality for each crop
            for i, (crop, det, bbox) in enumerate(zip(crops, detections, all_bboxes)):
                if crop is not None and crop.size > 0:
                    quality_score = self.quality_scorer.compute_quality_score(
                        crop=crop,
                        bbox=bbox,
                        all_bboxes=all_bboxes,
                        confidence=det.bbox.confidence,
                    )
                    quality_scores.append(quality_score)
                else:
                    quality_scores.append(0.0)
            
            # Extract features only from valid crops
            valid_crops = [c for c in crops if c is not None and c.size > 0]
            if valid_crops:
                features = extractor.extract_batch(valid_crops)
        
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
        
        # 3c. Detect occlusions and update track states
        if settings.reid_enable_occlusion_detection:
            # Increment frame counter
            if camera_id not in self.frame_counter:
                self.frame_counter[camera_id] = 0
            self.frame_counter[camera_id] += 1
            current_frame = self.frame_counter[camera_id]
            
            # Detect occlusions
            occlusions = self.detect_occlusions(active_tracks)
            
            # Update each track's occlusion state and quality
            for idx, track in enumerate(active_tracks):
                if not track.is_confirmed():
                    continue
                
                # Update occlusion state
                is_occluded = track.track_id in occlusions
                occluding_ids = occlusions.get(track.track_id, [])
                track.update_occlusion_state(is_occluded, occluding_ids, current_frame)
                
                # Store high-quality embeddings
                if features is not None and idx < len(quality_scores) and idx < len(features):
                    quality = quality_scores[idx]
                    track.quality_history.append(quality)
                    
                    # Keep only recent quality scores
                    if len(track.quality_history) > 50:
                        track.quality_history = track.quality_history[-50:]
                    
                    # Update last high-quality embedding
                    if quality >= settings.reid_quality_threshold:
                        track.last_high_quality_embedding = features[idx].copy()
                        logger.debug(
                            f"Track {track.track_id}: Stored high-quality embedding (quality={quality:.3f})"
                        )
        
        # 4. Handle track lifecycle
        new_features = await self._handle_track_events(
            state, active_tracks, timestamp
        )
        
        state.last_frame_time = timestamp
        
        return active_tracks, new_features
    
    async def process_batch(
        self,
        frames: List[Tuple[int, np.ndarray, datetime]],
        extract_features: bool = True,
    ) -> List[Tuple[List[Track], int]]:
        """
        Process a batch of frames from multiple cameras efficiently.
        
        Uses batched detection and feature extraction to maximize GPU throughput.
        DeepSORT tracking is still per-camera (sequential) as it maintains state.
        
        Args:
            frames: List of (camera_id, frame, timestamp) tuples
            extract_features: Whether to extract ReID features
            
        Returns:
            List of (active_tracks, camera_id) tuples in same order as input
        """
        if not frames:
            return []
        
        detector = self._get_detector()
        
        # Check if detector supports batch inference
        if not hasattr(detector, 'detect_batch'):
            # Fallback to sequential processing
            results = []
            for camera_id, frame, timestamp in frames:
                tracks, _ = await self.process_frame(camera_id, frame, timestamp, extract_features)
                results.append((tracks, camera_id))
            return results
        
        # 1. Batch Detection - Run inference once on all frames
        images = [f[1] for f in frames]  # Extract frames
        batch_detections = detector.detect_batch(images, confidence_threshold=0.3)
        
        # 2. Process each frame with its detections - use local storage
        all_crops = []
        crop_indices = []  # Track which (frame_idx, det_idx) each crop belongs to
        frame_data = {}  # Local storage: frame_idx -> {detections, face_dets, timestamp}
        
        for frame_idx, (camera_id, frame, timestamp) in enumerate(frames):
            raw_all = batch_detections[frame_idx]
            
            # Separate persons and faces
            person_dets = [d for d in raw_all if d.get('class_name') == 'person' and d.get('confidence', 0) >= 0.4]
            face_dets = [d for d in raw_all if d.get('class_name') == 'face' and d.get('confidence', 0) >= 0.3]
            
            # Convert to Detection objects
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
            
            # Crop detections for feature extraction
            h, w = frame.shape[:2]
            for det_idx, det in enumerate(detections):
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    all_crops.append(crop)
                    crop_indices.append((frame_idx, det_idx))
            
            # Store in local dict
            frame_data[frame_idx] = {
                'detections': detections,
                'face_dets': face_dets,
                'timestamp': timestamp,
                'camera_id': camera_id
            }
        
        # 3. Batch Feature Extraction
        all_features = None
        if extract_features and all_crops:
            extractor = self._get_extractor()
            all_features = extractor.extract_batch(all_crops)
        
        # 4. Update trackers and associate features (per-camera, sequential)
        results = []
        for frame_idx, (camera_id, frame, timestamp) in enumerate(frames):
            state = self._get_camera_state(camera_id)
            fd = frame_data[frame_idx]
            detections = fd['detections']
            face_dets = fd['face_dets']
            
            # Get features for this frame's detections
            features = None
            if all_features is not None and len(detections) > 0:
                # Get indices of crops belonging to this frame
                frame_crop_indices = [i for i, (fidx, _) in enumerate(crop_indices) if fidx == frame_idx]
                if frame_crop_indices:
                    features = all_features[frame_crop_indices]
            
            # Update tracker
            state.tracker.predict()
            active_tracks = state.tracker.update(detections, features)
            
            # Associate faces with tracks
            for track in active_tracks:
                person_box = track.to_tlbr()
                best_face = None
                best_score = 0.0
                for face in face_dets:
                    face_box = [face['x1'], face['y1'], face['x2'], face['y2']]
                    face_cx = (face_box[0] + face_box[2]) / 2
                    face_cy = (face_box[1] + face_box[3]) / 2
                    inside = (person_box[0] <= face_cx <= person_box[2] and
                              person_box[1] <= face_cy <= person_box[3])
                    if inside:
                        score = 0.5
                        if score > best_score:
                            best_score = score
                            best_face = face_box
                track.face_bbox = best_face
            
            state.last_frame_time = timestamp
            results.append((active_tracks, camera_id))
        
        return results
    
    def validate_detection_quality(
        self,
        bbox: np.ndarray,
        confidence: float,
        frame_shape: Tuple[int, int],
    ) -> Tuple[bool, str]:
        """
        Validate detection meets minimum quality standards for gallery.
        
        Filters out:
        - Distant/small detections (insufficient detail for ReID)
        - Invalid aspect ratios (non-person shapes like helmets)
        - Tiny detections relative to frame
        - Low confidence detections
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2]
            confidence: Detection confidence score
            frame_shape: Frame dimensions (height, width)
            
        Returns:
            (is_valid, rejection_reason)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 1. Minimum size check (ReID training standard: 64x128)
        if width < settings.reid_min_bbox_width or height < settings.reid_min_bbox_height:
            return False, f"too_small:{width:.0f}x{height:.0f}"
        
        # 2. Aspect ratio check (person-like shape)
        aspect_ratio = height / width if width > 0 else 0
        
        if not (settings.reid_min_aspect_ratio <= aspect_ratio <= settings.reid_max_aspect_ratio):
            return False, f"bad_aspect:{aspect_ratio:.2f}"
        
        # 3. Frame coverage check (reject tiny detections)
        frame_h, frame_w = frame_shape
        coverage = (width * height) / (frame_w * frame_h)
        
        if coverage < settings.reid_min_frame_coverage:
            return False, f"tiny_coverage:{coverage*100:.2f}%"
        
        # 4. Confidence check
        if confidence < settings.reid_min_detection_confidence:
            return False, f"low_conf:{confidence:.2f}"
        
        return True, "OK"
    
    def validate_person_presence(
        self,
        crop: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        Validate that crop actually contains a person (not empty background).
        
        Uses three checks:
        1. Pixel variance: Empty backgrounds have uniform pixels
        2. Edge density: Persons have edges, backgrounds don't
        3. Color entropy: Persons have varied colors
        
        Args:
            crop: Image crop as numpy array (H, W, C)
            
        Returns:
            (is_valid, rejection_reason)
        """
        import cv2
        
        if crop is None or crop.size == 0:
            return False, "empty_crop"
        
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return False, "crop_too_small"
        
        # 1. Check pixel variance (empty backgrounds are uniform)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        variance = float(gray.var())
        
        if variance < settings.reid_min_crop_variance:
            return False, f"low_variance:{variance:.1f}"
        
        # 2. Check edge density (persons have many edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum()) / (edges.size * 255)  # Normalize
        
        if edge_density < settings.reid_min_edge_density:
            return False, f"low_edges:{edge_density:.3f}"
        
        # 3. Check color entropy (persons have varied colors)
        if len(crop.shape) == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist = hist.flatten() / (hist.sum() + 1e-10)
            entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
            
            if entropy < settings.reid_min_color_entropy:
                return False, f"low_entropy:{entropy:.2f}"
        
        return True, "OK"
    
    def detect_occlusions(self, tracks: List[Track]) -> Dict[int, List[int]]:
        """
        Detect which tracks are occluded and by whom.
        
        Uses IoU-based analysis: if two tracks overlap significantly,
        the one with the smaller bbox is considered occluded.
        
        Args:
            tracks: List of active tracks
            
        Returns:
            Dict mapping track_id -> list of occluding track_ids
        """
        occlusions = {}
        
        for i, track_i in enumerate(tracks):
            if not track_i.is_confirmed():
                continue
                
            bbox_i = track_i.to_tlbr()
            
            for j, track_j in enumerate(tracks):
                if i == j or not track_j.is_confirmed():
                    continue
                    
                bbox_j = track_j.to_tlbr()
                
                # Compute IoU
                iou = self._compute_iou(bbox_i, bbox_j)
                
                if iou > settings.reid_occlusion_iou_threshold:
                    # Determine which is in front (larger bbox = closer to camera)
                    area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                    area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                    
                    if area_i > area_j:
                        # track_i is occluding track_j
                        occlusions.setdefault(track_j.track_id, []).append(track_i.track_id)
                    else:
                        # track_j is occluding track_i
                        occlusions.setdefault(track_i.track_id, []).append(track_j.track_id)
        
        return occlusions
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bbox as [x1, y1, x2, y2]
            bbox2: Second bbox as [x1, y1, x2, y2]
            
        Returns:
            IoU value (0.0-1.0)
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        # IoU
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
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
