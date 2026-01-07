import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Any
from loguru import logger

# Import BoxMOT
try:
    from boxmot.trackers.deepocsort.deepocsort import DeepOcSort as DeepOCSORT
except ImportError:
    logger.error("BoxMOT not installed or DeepOcSort not found.")
    DeepOCSORT = None

from app.core.tracking.base import BaseTracker
from app.core.detection.yolo_detector import Detection
from app.core.features.unified_extractor import UnifiedFeatureExtractor
# Import original Track/TrackState to maintain compatibility
from app.core.tracking.deepsort import Track, TrackState
from app.core.tracking.kalman import KalmanFilter # Needed for Track init

class UnifiedModelWrapper:
    """
    Wraps UnifiedFeatureExtractor to behave like a standard ReID model for BoxMOT.
    Also caches face detection info for retrieval when updating Track objects.
    """
    def __init__(self, extractor: UnifiedFeatureExtractor):
        self.extractor = extractor
        # Helper to allow 'model.warmup()' check if it exists in boxmot
        self.training = False
        # Cache face info from last extraction: list of {bbox, det_score} or None per detection
        self.last_face_info: list = []
        self.last_bboxes: np.ndarray = np.empty((0, 4))  # Store bboxes for matching
    
    def eval(self):
        pass
    
    def to(self, device):
        pass
        
    def train(self, mode=True):
        self.training = mode

    def __call__(self, text):
        pass

    def get_features(self, bboxes, img):
        """
        Calculates embeddings and caches face detection info.
        Args:
            bboxes: (N, 4) or (N, 6) numpy array of bounding boxes
            img: Full frame image
        Returns:
            np.ndarray: (N, feature_dim) - BoxMOT expects numpy arrays, not tensors
        """
        # img is passed by DeepOcSort
        if len(bboxes) == 0:
             self.last_face_info = []
             self.last_bboxes = np.empty((0, 4))
             return np.zeros((0, self.extractor.total_dim), dtype=np.float32)
             
        # Extract features AND face info using the enhanced method
        # boxmot passes (N, 4) xyxy usually
        embeddings, face_info = self.extractor.extract_with_faces(img, bboxes[:, :4])
        
        # Cache face info and bboxes for later lookup
        self.last_face_info = face_info
        self.last_bboxes = bboxes[:, :4].copy()
        
        # Return numpy array directly - BoxMOT's association code expects numpy, not tensors
        return embeddings.astype(np.float32)
    
    def get_face_info_for_bbox(self, x1, y1, x2, y2, iou_threshold=0.5) -> dict:
        """
        Get face info for a bounding box by matching against cached detection bboxes.
        Returns face info dict or None if no match.
        """
        if len(self.last_bboxes) == 0 or len(self.last_face_info) == 0:
            return None
        
        query_bbox = np.array([x1, y1, x2, y2])
        
        # Find best matching cached bbox by IOU
        best_iou = 0
        best_idx = -1
        
        for i, cached_bbox in enumerate(self.last_bboxes):
            iou = self._compute_iou(query_bbox, cached_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_iou >= iou_threshold and best_idx >= 0 and best_idx < len(self.last_face_info):
            return self.last_face_info[best_idx]
        
        return None
    
    @staticmethod
    def _compute_iou(bbox1, bbox2):
        """Compute IOU between two bboxes [x1, y1, x2, y2]."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0


class BoxMOTTracker(BaseTracker):
    def __init__(self, 
                 model_name='osnet_x1_0', 
                 device='cuda',
                 reid_weights: Path = None):
        
        if DeepOCSORT is None:
            raise RuntimeError("BoxMOT library not available")

        self.device = device
        
        # 1. Initialize Unified Extractor
        self.extractor = UnifiedFeatureExtractor(
            model_name=model_name,
            device=device
        )
        
        # 2. Initialize BoxMOT Tracker (DeepOCSORT)
        # BoxMOT expects device as '0', 'cpu', etc. not 'cuda'
        boxmot_device = '0' if device == 'cuda' else device
        self.boxmot = DeepOCSORT(
            reid_weights=Path('osnet_x0_25_msmt17.pt'), 
            device=boxmot_device,
            half=True,
            max_age=8,  # Reduce coasting duration from default 30 to 8 frames
        )
        
        # 3. Replace internal model
        self.model_wrapper = UnifiedModelWrapper(self.extractor)
        self.boxmot.model = self.model_wrapper
        
        # 4. State Management for Compatibility
        self.tracks_map: Dict[int, Track] = {} # id -> Track
        self.tracks: List[Track] = [] # List of active tracks (required by TrackingService in some getters)
        
        # Dummy KF for Track init (BoxMOT handles KF internally, but Track constr needs values)
        self.dummy_kf = KalmanFilter()
        
        logger.info("BoxMOTTracker initialized with UnifiedFeatureExtractor (Body+Face)")

    def predict(self):
        pass

    def update(self, detections: List[Detection], frame: np.ndarray = None, features: np.ndarray = None) -> List[Track]:
        """
        Update tracker with detections.
        Arguments match BaseTracker but we need 'frame'.
        Legacy calls might pass 'features' (ignored here as we compute internally).
        """
        if frame is None:
            logger.warning("BoxMOTTracker.update called without frame! Cannot compute features.")
            return []
            
        logger.info(f"BoxMOTTracker: Received {len(detections) if detections else 0} detections")

        if not detections:
             dets_np = np.empty((0, 6))
        else:
            dets_array = []
            for d in detections:
                # BoxMOT expects [x1, y1, x2, y2, conf, cls]
                dets_array.append([d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id])
            dets_np = np.array(dets_array)

        # 1. Update BoxMOT
        # BoxMOT returns np array of shape (N, 8) -> [x1, y1, x2, y2, id, conf, cls, ind]
        # output_results (np.ndarray): [x1, y1, x2, y2, id, conf, cls, ind]
        try:
            results = self.boxmot.update(dets_np, frame)
        except Exception as e:
            # Handle OpenCV optical flow errors (e.g. resolution change)
            if "prevPyr" in str(e) and "size" in str(e):
                logger.warning(f"Tracker optical flow error (likely resolution change). Resetting tracker: {e}")
                
                # Re-initialize BoxMOT
                boxmot_device = '0' if self.device == 'cuda' else self.device
                self.boxmot = DeepOCSORT(
                    reid_weights=Path('osnet_x0_25_msmt17.pt'), 
                    device=boxmot_device,
                    half=True,
                    max_age=8,  # Reduce coasting duration from default 30 to 8 frames
                )
                self.boxmot.model = self.model_wrapper
                self.tracks_map.clear()
                return []
            raise e
        
        # 2. Sync with Track objects
        active_ids = set()
        
        # results might be (N, 8) or (0, 8)
        if len(results) > 0:
            logger.info(f"BoxMOT returned {len(results)} tracks")
            for i, res in enumerate(results):
                # res: x1, y1, x2, y2, id, conf, cls, ind
                x1, y1, x2, y2 = res[:4]
                track_id = int(res[4])
                conf = res[5]
                class_id = int(res[6])
                det_idx = int(res[7]) # Index in original detections usually (if supported by specific tracker output)
                
                # CRITICAL: Only mark as active if MATCHED to a detection
                # Coasted/predicted tracks (det_idx=-1) should NOT be kept in tracks_map
                if det_idx > -1:
                    active_ids.add(track_id)
                
                # Get the embedding for this track
                # 'DeepOCSORT' tracks are stored in self.boxmot.trackers
                # We need to find the internal track object to get the latest feature.
                # BoxMOT trackers list usually contains objects with .id attribute
                
                current_feat = None
                
                # Internal track lookup - optimization: map id to internal track once
                internal_track = None
                # DeepOcSort uses 'active_tracks'
                track_source = getattr(self.boxmot, 'trackers', getattr(self.boxmot, 'active_tracks', []))
                
                for t in track_source:
                    if t.id == track_id:
                        internal_track = t
                        break
                
                if internal_track:
                    try:
                        # BoxMOT / DeepOcSort tracks usually have 'emb' for current feature
                        # or 'features' list for history.
                        if hasattr(internal_track, 'emb') and internal_track.emb is not None:
                             current_feat = internal_track.emb
                        elif hasattr(internal_track, 'features') and len(internal_track.features) > 0:
                            current_feat = internal_track.features[-1]
                        elif hasattr(internal_track, 'curr_feat'):
                             current_feat = internal_track.curr_feat
                        
                        # Cleanup tensor/numpy conversion is done below
                    except Exception as e:
                        logger.error(f"Error accessing features: {e}")
                
                # Map to our Track object (only if matched - coasted tracks are skipped)
                if det_idx == -1:
                    # NUCLEAR OPTION: Immediately delete coasted tracks from our map
                    if track_id in self.tracks_map:
                        print(f"BOXMOT_DEBUG: Track {track_id}: COASTED - DELETING from tracks_map")
                        del self.tracks_map[track_id]
                    else:
                        print(f"BOXMOT_DEBUG: Track {track_id}: COASTED (det_idx={det_idx}), SKIPPING")
                    continue
                    
                if track_id not in self.tracks_map:
                    # Create new Track
                    # Need mean/cov. We can approximate from bbox.
                    # Track expects (8,) mean (cx, cy, a, h, dx, dy, da, dh)
                    # We just fill pos and zeros for vel.
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w/2
                    cy = y1 + h/2
                    mean = np.array([cx, cy, w/h, h, 0, 0, 0, 0])
                    cov = np.eye(8) # dummy
                    
                    self.tracks_map[track_id] = Track(
                        track_id=track_id,
                        mean=mean,
                        covariance=cov,
                        n_init=3, # Config?
                        max_age=30, # Config?
                        state=TrackState.CONFIRMED # If BoxMOT output it, it's usually confirmed/active
                    )
                
                track_obj = self.tracks_map[track_id]
                
                # Update Track properties that changed
                # Position (update mean)
                w = x2 - x1
                h = y2 - y1
                track_obj.mean[:4] = [x1 + w/2, y1 + h/2, w/h, h]
                track_obj.confidence = conf
                track_obj.class_id = class_id
                track_obj.state = TrackState.CONFIRMED # Reset to confirmed
                
                # Try to attach face bounding box if available help
                # We use the wrapper's cached info
                if hasattr(self.model_wrapper, 'get_face_info_for_bbox'):
                    face_info = self.model_wrapper.get_face_info_for_bbox(x1, y1, x2, y2)
                    if face_info and face_info.get('bbox'):
                         track_obj.face_bbox = face_info['bbox']
                
                # CRITICAL FIX: Only mark as "hit" if matched to a valid detection
                # BoxMOT/DeepOCSORT returns det_idx (index 7) as -1 for coasted/predicted tracks
                if det_idx > -1:
                    track_obj.hits += 1
                    track_obj.time_since_update = 0
                    print(f"BOXMOT_DEBUG: Track {track_id}: MATCHED (det_idx={det_idx}), conf={conf:.2f}")
                else:
                    # It's a prediction (coasted), so treat as missed for display purposes
                    track_obj.time_since_update += 1
                    print(f"BOXMOT_DEBUG: Track {track_id}: COASTED (det_idx={det_idx}), age={track_obj.time_since_update}")
                
                if current_feat is not None:
                    # Copy feature
                    f = current_feat.cpu().numpy() if hasattr(current_feat, 'cpu') else current_feat
                    track_obj.features.append(f)
                    if len(track_obj.features) > 100:
                         track_obj.features = track_obj.features[-100:]

        # 3. Cleanup lost tracks
        # Loop over known tracks, if not in active_ids, mark missed/deleted
        for tid in list(self.tracks_map.keys()):
            if tid not in active_ids:
                track = self.tracks_map[tid]
                track.mark_missed()
                if track.is_deleted():
                    del self.tracks_map[tid]
        
        # 4. AGGRESSIVE CLEANUP: Clear stale tracks from BoxMOT's internal state
        # This prevents "ghost" tracks from persisting at frame edges
        self._cleanup_stale_internal_tracks()
        
        # 5. Update public tracks list - ONLY return CONFIRMED tracks that were updated this frame
        # AND pass strict visibility validation
        
        valid_tracks = []
        tracks_to_delete = []  # Track IDs to remove after iteration
        
        if frame is not None:
            H, W = frame.shape[:2]
            MIN_VISIBILITY_RATIO = 0.50  # Track must be at least 50% inside frame (increased from 0.30)
            EDGE_TOLERANCE = 15  # Pixels - if track edge is this close to frame edge, suspect it's stuck (increased from 3)
            
            total_tracks = len(self.tracks_map)
            confirmed_tracks = 0
            fresh_tracks = 0
            visibility_passed = 0
            
            for track in self.tracks_map.values():
                if not track.is_confirmed():
                    continue
                confirmed_tracks += 1
                
                # STRICT VISUALIZATION FILTER:
                # Only show tracks that were matched to a REAL detection in this frame.
                # 'time_since_update == 0' means it was updated this frame.
                # Any track with time_since_update > 0 is a "ghost" / prediction and should be hidden.
                if track.time_since_update > 0:
                     continue
                     
                fresh_tracks += 1
                    
                # Get bounding box
                x1, y1, x2, y2 = track.to_tlbr()
                track_w = x2 - x1
                track_h = y2 - y1
                
                # Skip invalid boxes
                if track_w <= 0 or track_h <= 0:
                    continue
                
                # Calculate visibility: what fraction of the track is inside the frame?
                visible_x1 = max(0, x1)
                visible_y1 = max(0, y1)
                visible_x2 = min(W, x2)
                visible_y2 = min(H, y2)
                
                visible_w = max(0, visible_x2 - visible_x1)
                visible_h = max(0, visible_y2 - visible_y1)
                
                visible_area = visible_w * visible_h
                total_area = track_w * track_h
                visibility_ratio = visible_area / total_area if total_area > 0 else 0
                
                # STRICT FILTER: Must have minimum visibility
                if visibility_ratio < MIN_VISIBILITY_RATIO:
                    logger.debug(f"Track {track.track_id} FILTERED: low visibility ({visibility_ratio:.1%})")
                    # Mark for deletion if stuck at edge
                    if visibility_ratio < 0.1:  # Almost completely outside
                        tracks_to_delete.append(track.track_id)
                    continue
                
                # EDGE-STUCK DETECTION: If track edge is exactly at frame edge, it's likely stuck
                is_stuck_at_edge = (
                    (x1 <= EDGE_TOLERANCE and visibility_ratio < 0.8) or
                    (y1 <= EDGE_TOLERANCE and visibility_ratio < 0.8) or
                    (x2 >= W - EDGE_TOLERANCE and visibility_ratio < 0.8) or
                    (y2 >= H - EDGE_TOLERANCE and visibility_ratio < 0.8)
                )
                
                # CENTER-BASED BOUNDARY CHECK: Delete if center is outside frame
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center_outside_frame = (center_x < 0 or center_x >= W or center_y < 0 or center_y >= H)
                
                if is_stuck_at_edge or center_outside_frame:
                    if center_outside_frame:
                        logger.debug(f"Track {track.track_id} FILTERED: center outside frame (center=[{center_x:.0f},{center_y:.0f}])")
                    else:
                        logger.debug(f"Track {track.track_id} FILTERED: stuck at edge (bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], vis={visibility_ratio:.1%})")
                    tracks_to_delete.append(track.track_id)
                    continue
                
                visibility_passed += 1
                valid_tracks.append(track)
                
            print(f"BOXMOT_DEBUG: Filtering - total={total_tracks}, confirmed={confirmed_tracks}, fresh={fresh_tracks}, visibility_passed={visibility_passed}, RETURNED={len(valid_tracks)}")
            
            # Delete stuck/out-of-bounds tracks
            for tid in tracks_to_delete:
                if tid in self.tracks_map:
                    del self.tracks_map[tid]
                    logger.debug(f"Deleted edge-stuck track {tid}")
        else:
            # Fallback if no frame provided (shouldn't happen per line 110)
            valid_tracks = [t for t in self.tracks_map.values() if t.is_confirmed() and t.time_since_update == 0]

        self.tracks = valid_tracks
        
        return self.tracks
    
    def _cleanup_stale_internal_tracks(self):
        """
        Aggressively clean up stale tracks from BoxMOT's internal tracker state.
        This prevents ghost tracks from accumulating at frame edges.
        """
        try:
            # Get internal trackers list from BoxMOT
            internal_trackers = getattr(self.boxmot, 'trackers', getattr(self.boxmot, 'active_tracks', []))
            
            if not internal_trackers:
                return
            
            # Find stale internal tracks (high age, not matched recently)
            stale_ids = []
            for t in internal_trackers:
                # Internal BoxMOT tracks have different attribute names
                track_age = getattr(t, 'time_since_update', getattr(t, 'age', 0))
                track_id = getattr(t, 'id', getattr(t, 'track_id', -1))
                
                # If track hasn't been updated in 3+ frames, consider it stale
                if track_age >= 3:
                    stale_ids.append(track_id)
            
            # Remove stale tracks from internal list (if mutable)
            if stale_ids and hasattr(internal_trackers, '__delitem__'):
                for i in range(len(internal_trackers) - 1, -1, -1):
                    t = internal_trackers[i]
                    if getattr(t, 'id', getattr(t, 'track_id', -1)) in stale_ids:
                        try:
                            del internal_trackers[i]
                        except:
                            pass  # Some tracker implementations may not allow deletion
                            
        except Exception as e:
            # Don't break tracking if cleanup fails
            logger.debug(f"Internal track cleanup error (non-fatal): {e}")
