"""
Stream Processor Worker

Background worker that processes frames from StreamManager,
runs detection/tracking/ReID, and broadcasts results via WebSocket.
"""

import asyncio
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from loguru import logger

# Try simplejpeg first (fastest CPU encoder), then TurboJPEG, then OpenCV
SIMPLEJPEG_AVAILABLE = False
try:
    import simplejpeg
    SIMPLEJPEG_AVAILABLE = True
    logger.info("simplejpeg available - using fast JPEG encoding")
except ImportError:
    logger.info("simplejpeg not available")

# TurboJPEG fallback
TURBOJPEG = None
TURBOJPEG_AVAILABLE = False
if not SIMPLEJPEG_AVAILABLE:
    try:
        from turbojpeg import TurboJPEG
        lib_paths = [
            r"C:\libjpeg-turbo64\bin\turbojpeg.dll",
            r"C:\Program Files\libjpeg-turbo64\bin\turbojpeg.dll",
        ]
        for path in lib_paths:
            try:
                import os
                if os.path.exists(path):
                    TURBOJPEG = TurboJPEG(path)
                    break
            except:
                continue
        if TURBOJPEG is None:
            TURBOJPEG = TurboJPEG()
        TURBOJPEG_AVAILABLE = True
        logger.info("TurboJPEG available - using fast JPEG encoding")
    except Exception as e:
        logger.info(f"TurboJPEG not available, using OpenCV encoding")

# Thread pool for parallel JPEG encoding
ENCODE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jpeg_encode")

from app.services.stream_manager import get_stream_manager, FrameData, PlaybackState
from app.services.tracking_service import get_tracking_service
from app.services.reid_service import get_reid_service
from app.services.track_store import get_track_store
from app.services.identity_merger import get_identity_merger
from app.api.v1.realtime import broadcast_event
from app.schemas.track import TrackStatus
from app.services.gallery_store import get_gallery_store
from app.config import settings

# Batch processing support
try:
    from app.workers.batch_processor import BatchFrameAccumulator
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    logger.warning("batch_processor not available, using sequential processing")
    BATCH_PROCESSING_AVAILABLE = False

# Function to get scorer (lazy load to avoid heavy imports at root if needed)
_quality_scorer = None
def get_quality_scorer_instance():
    global _quality_scorer
    if _quality_scorer is None:
        try:
            from preprocessor.quality_scorer import QualityScorer
            # Use settings for thresholds if available, otherwise defaults
            _quality_scorer = QualityScorer()
        except ImportError:
            logger.warning("Could not import QualityScorer from preprocessor")
    return _quality_scorer



class StreamProcessor:
    """
    Processes video frames from StreamManager and runs ML pipeline.
    
    Pipeline:
    1. Get frame from StreamManager queue
    2. Run PeopleNet detection
    3. Run DeepSORT tracking
    4. Extract features
    5. Run ReID matching
    6. Broadcast results via WebSocket
    """
    
    def __init__(
        self,
        detection_interval: int = 1,  # Process every N frames
        broadcast_frames: bool = True,  # Send frames to frontend
        frame_quality: int = 50,  # JPEG quality (balanced for ReID)
        target_display_fps: int = 30,  # Target FPS for display
    ):
        self.detection_interval = detection_interval
        self.broadcast_frames = broadcast_frames
        self.frame_quality = frame_quality
        self.target_display_fps = target_display_fps
        self._min_frame_interval = 1.0 / target_display_fps  # Seconds between frames
        self._last_broadcast_time: Dict[int, float] = {}  # Per-camera throttle
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        # Batch processing (TensorRT optimization)
        self.use_batch_processing = BATCH_PROCESSING_AVAILABLE and settings.tensorrt_batch_size > 1
        if self.use_batch_processing:
            self.batch_accumulator = BatchFrameAccumulator(
                batch_size=settings.tensorrt_batch_size,
                timeout=0.05  # 50ms max wait
            )
            logger.info(f"Batch processing enabled (batch_size={settings.tensorrt_batch_size})")
        else:
            self.batch_accumulator = None
            logger.info("Sequential processing (batch disabled)")
        
        # Sticky state for frame skipping
        self._last_detections = []
        self._last_tracks = []
        
        # Identity merger - check every N frames for fragmented identities
        self._merge_check_interval = 100  # Check for merge candidates every 100 frames
        self._last_merge_frame = 0
        
        logger.info("StreamProcessor initialized")
    
    def start(self):
        """Start the processor in a background thread."""
        if self._running:
            return
        
        # Capture the main event loop to schedule async storage/broadcasts
        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback if logic is not started from async context (unlikely in FastAPI)
            logger.warning("StreamProcessor started context without event loop, using new one")
            self._main_loop = asyncio.new_event_loop()
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("StreamProcessor started")
    
    def stop(self):
        """Stop the processor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("StreamProcessor stopped")
    
    def _run_loop(self):
        """Main processing loop with batch support."""
        stream_manager = get_stream_manager()
        
        # Try to get ML services, but don't fail if unavailable
        tracking_service = None
        reid_service = None
        try:
            tracking_service = get_tracking_service()
            reid_service = get_reid_service()
            logger.info("ML services loaded successfully")
        except Exception as e:
            logger.warning(f"ML services not available (frames will still stream): {e}")
        
        # Debug: Log batch processing status
        logger.info(f"🔍 Batch processing check:")
        logger.info(f"  - use_batch_processing: {self.use_batch_processing}")
        logger.info(f"  - batch_accumulator: {self.batch_accumulator is not None}")
        logger.info(f"  - tracking_service: {tracking_service is not None}")
        
        # Batch processing mode
        if self.use_batch_processing and self.batch_accumulator is not None:
            logger.info("🚀 USING BATCH PROCESSING MODE")
            self._run_loop_batched(stream_manager, tracking_service, reid_service)
        else:
            logger.info("⚠️ USING SEQUENTIAL PROCESSING MODE")
            self._run_loop_sequential(stream_manager, tracking_service, reid_service)
    
    def _run_loop_batched(self, stream_manager, tracking_service, reid_service):
        """Batch processing loop (TensorRT optimized)."""
        logger.info("Starting BATCH processing mode")
        
        # Create a single event loop for the entire batch session
        batch_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(batch_loop)
        
        # Get detector once
        detector = tracking_service._get_detector() if tracking_service else None
        if not detector:
            logger.error("No detector available! Falling back to sequential mode")
            self._run_loop_sequential(stream_manager, tracking_service, reid_service)
            return
        
        if not hasattr(detector, 'preprocess'):
            logger.error("Detector missing preprocess method! Falling back to sequential mode")
            self._run_loop_sequential(stream_manager, tracking_service, reid_service)
            return
        
        logger.info(f"✅ Batch mode initialized with detector: {type(detector).__name__}")
        
        while self._running:
            # Check if playing
            if stream_manager.state != PlaybackState.PLAYING:
                time.sleep(0.1)
                continue
            
            # Accumulate frames for batch processing
            while not self.batch_accumulator.should_process() and self._running:
                frame_data = stream_manager.get_next_frame(timeout=0.05)
                if frame_data is None:
                    break
                
                self._frame_count += 1
                
                # Preprocess frame for batch
                preprocessed, _, _ = detector.preprocess(frame_data.frame)
                self.batch_accumulator.add_frame(frame_data, preprocessed)
                
                # Log every 10 frames
                if self._frame_count % 10 == 0:
                    logger.debug(f"Accumulated {len(self.batch_accumulator)} frames (target: {settings.tensorrt_batch_size})")
            
            # Process accumulated batch
            if len(self.batch_accumulator) > 0:
                self._process_batch(
                    self.batch_accumulator,
                    tracking_service,
                    reid_service,
                    stream_manager,
                    batch_loop
                )
    
    def _process_batch(self, accumulator, tracking_service, reid_service, stream_manager, loop):
        """Process accumulated batch of frames."""
        batch_items = accumulator.clear()
        batch_size = len(batch_items)
        
        if batch_size == 0:
            return
        
        try:
            # Prepare batch data
            frames_data = [(item.frame_data.camera_id, item.frame_data.frame, item.frame_data.timestamp) 
                          for item in batch_items]
            
            # Batch detection + tracking (FAST!)
            start_time = time.time()
            
            batch_results = loop.run_until_complete(
                tracking_service.process_frames_batch(frames_data, extract_features=True)
            )
            
            batch_time = (time.time() - start_time) * 1000  # ms
            per_frame_time = batch_time / batch_size
            batch_fps = 1000 / per_frame_time
            
            logger.info(
                f"🚀 Batch inference: {batch_size} frames in {batch_time:.2f}ms "
                f"({batch_fps:.1f} FPS, {per_frame_time:.2f}ms per frame)"
            )
            
            # Update FPS counter
            self._fps_frame_count += batch_size
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._fps = self._fps_frame_count / (now - self._last_fps_time)
                self._fps_frame_count = 0
                self._last_fps_time = now
            
            # Collect last frame per camera for ReID (heavy processing only on last)
            last_frame_per_camera = {}
            for item in batch_items:
                camera_id = item.frame_data.camera_id
                if camera_id in batch_results:
                    tracks, features = batch_results[camera_id]
                    last_frame_per_camera[camera_id] = (item.frame_data, tracks, features)
            
            # Run ReID only on LAST frame per camera (expensive operation)
            for camera_id, (frame_data, tracks, features) in last_frame_per_camera.items():
                self._process_reid_and_broadcast(
                    frame_data,
                    tracks,
                    features,
                    reid_service,
                    loop
                )
            
            # Broadcast ALL frames in batch using PARALLEL encoding
            if self.broadcast_frames:
                self._broadcast_batch_parallel(batch_items, batch_results)
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _run_loop_sequential(self, stream_manager, tracking_service, reid_service):
        """Sequential processing loop (fallback)."""
        logger.info("Starting SEQUENTIAL processing mode")
        
        while self._running:
            # Check if playing
            if stream_manager.state != PlaybackState.PLAYING:
                time.sleep(0.1)
                continue
            
            # Frame Skipping: Drain queue to get the latest frame
            # This prevents lag if processing is slower than capture
            while stream_manager.frame_queue.qsize() > 1:
                try:
                    _ = stream_manager.frame_queue.get_nowait()
                    stream_manager.frame_queue.task_done()
                except:
                    break

            # Get frame from queue
            frame_data = stream_manager.get_next_frame(timeout=0.5)
            if frame_data is None:
                continue
            
            self._frame_count += 1
            self._fps_frame_count += 1
            
            # Check for black frames (common webcam issue)
            if self._frame_count % 100 == 0:
                if np.mean(frame_data.frame) < 1.0:
                    logger.warning(f"Source {frame_data.source_id} is producing black frames (mean < 1.0)")
            
            # Calculate FPS
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self._fps = self._fps_frame_count / (now - self._last_fps_time)
                self._fps_frame_count = 0
                self._last_fps_time = now
            
            # Skip ML processing based on interval
            frame_detections = []
            frame_tracks = []
            frame_results = {}
            
            run_ml = (
                self._frame_count % self.detection_interval == 0 
                and tracking_service is not None
            )
            
            if run_ml:
                try:
                    # Process frame through pipeline
                    frame_results = self._process_frame(
                        frame_data,
                        tracking_service,
                        reid_service,
                    )
                    
                    frame_detections = frame_results.get("detections", [])
                    frame_tracks = frame_results.get("tracks", [])
                    
                    # Update sticky state
                    self._last_detections = frame_detections
                    self._last_tracks = frame_tracks
                    
                    # Broadcast events
                    self._broadcast_events(frame_data, frame_results)
                    
                except RuntimeError as e:
                    if "CUDA" in str(e) and torch:
                        logger.error(f"CUDA Error during ML processing {self.source_id}: {e}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        time.sleep(0.1)
                    else:
                        logger.debug(f"RuntimeError in ML processing {self.source_id}: {e}")
                except Exception as e:
                    logger.debug(f"Frame processing error: {e}")
            else:
                # Reuse last known results for smoothness
                frame_detections = self._last_detections
                frame_tracks = self._last_tracks
            
            # Broadcast frame (with overlays if ML ran)
            if self.broadcast_frames:
                try:
                    self._broadcast_frame(
                        frame_data, 
                        frame_detections, 
                        frame_tracks
                    )
                except MemoryError:
                    logger.error(f"MemoryError in processor for source {frame_data.source_id}. Clearing memory.")
                    gc.collect()
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(1)  # Longer pause to let system recover
                    continue
                except RuntimeError as e:
                    if "CUDA" in str(e) and torch:
                        logger.error(f"CUDA Error in processor for source {frame_data.source_id}: {e}. Attempting to clear cache.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.1) # Short pause
                    else:
                        logger.debug(f"RuntimeError in processor for source {frame_data.source_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error broadcasting frame for source {frame_data.source_id}: {e}")
                    continue
    
    def _process_reid_and_broadcast(self, frame_data, tracks, features, reid_service, loop=None):
        """Process ReID matching and broadcast results (helper for batch mode)."""
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            quality_scorer = get_quality_scorer_instance()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                if conf < 0.5:
                    continue
                
                if not track.features:
                    continue
                
                feature = track.features[-1]
                
                # Quality check
                should_update = False
                quality_score = 0.0
                thumb_b64 = None
                
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame_data.frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1 and quality_scorer:
                    crop = frame_data.frame[y1:y2, x1:x2]
                    q_result = quality_scorer.score(crop)
                    quality_score = q_result.total_score
                    
                    if quality_score > settings.gallery_quality_threshold:
                        should_update = True
                        thumb = cv2.resize(crop, (64, 128))
                        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        thumb_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # ReID matching
                current_global_id = getattr(track, 'global_id', None)
                
                if should_update or current_global_id is None:
                    if reid_service:
                        match = loop.run_until_complete(
                            reid_service.match_identity(
                                camera_id=frame_data.camera_id,
                                embedding=feature,
                                timestamp=frame_data.timestamp,
                            )
                        )
                        track.global_id = str(match.global_track_id)
                        
                        if should_update and thumb_b64:
                            gallery_store = get_gallery_store()
                            gallery_store.add_capture(
                                global_id=track.global_id,
                                image_b64=thumb_b64,
                                quality_score=quality_score,
                                pose=q_result.pose if hasattr(q_result, 'pose') else 'unknown',
                                sharpness=q_result.sharpness_score if hasattr(q_result, 'sharpness_score') else 0.0,
                                frame_number=frame_data.frame_number,
                                timestamp=frame_data.timestamp,
                                embedding=feature
                            )
                
                # Update track store
                if getattr(track, 'global_id', None):
                    try:
                        track_store = get_track_store()
                        track_store.add_or_update_track(
                            track.global_id,
                            {"status": TrackStatus.ACTIVE}
                        )
                        track_store.update_camera_sequence(track.global_id, frame_data.camera_id)
                    except Exception as e:
                        logger.error(f"Failed to update TrackStore: {e}")
        
        except Exception as e:
            logger.error(f"ReID processing error: {e}")
    
    def _process_frame(
        self,
        frame_data: FrameData,
        tracking_service,
        reid_service,
    ) -> Dict[str, Any]:
        """Process a single frame through the ML pipeline."""
        results = {
            "detections": [],
            "tracks": [],
            "reid_matches": [],
        }
        
        # Run tracking (detection + tracking + features)
        # Note: ML services are likely async, but we are in a thread.
        # We should use a local loop for ML ops if they are not bound to main loop.
        # However, for simplicity/safety, we create a new loop for ML ONLY.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tracks, features = loop.run_until_complete(
                tracking_service.process_frame(
                    camera_id=frame_data.camera_id,
                    frame=frame_data.frame,
                    timestamp=frame_data.timestamp,
                    extract_features=True,
                )
            )
            
            # 2. Run ReID matching for confirmed tracks (Smart Filtered)
            if reid_service:
                logger.info(f"ReID: Processing {len(tracks)} tracks")
                quality_scorer = get_quality_scorer_instance()
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    # Filter low confidence tracks for ReID too
                    conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                    if conf < 0.5: 
                         continue

                    # Find feature for this track
                    if not track.features:
                        continue
                        
                    feature = track.features[-1] # Latest feature
                    
                    try:
                        # Smart Filter: Check quality BEFORE matching/updating
                        should_update = False
                        quality_score = 0.0
                        thumb_b64 = None
                        
                        bbox = track.to_tlbr()
                        x1, y1, x2, y2 = map(int, bbox)
                        h, w = frame_data.frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1 and quality_scorer:
                             crop = frame_data.frame[y1:y2, x1:x2]
                             
                             # Assess Quality
                             # Use simple crop scoring.
                             q_result = quality_scorer.score(crop)
                             quality_score = q_result.total_score
                             
                             # Threshold for Gallery Update (High Quality)
                             # QualityScorer returns 0-100, configurable via GALLERY_QUALITY_THRESHOLD
                             if quality_score > settings.gallery_quality_threshold:
                                 should_update = True
                                 
                                 # Prepare thumbnail
                                 thumb = cv2.resize(crop, (64, 128))
                                 _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                 thumb_b64 = base64.b64encode(buffer).decode('utf-8')
                        else:
                            q_result = None
                        
                        # Store quality info on track for visualization
                        track.quality_score = quality_score
                        track.pose = q_result.pose if q_result and hasattr(q_result, 'pose') else 'unknown'
                        track.is_saved = should_update

                        # Run Match Identity
                        # We run matching to maintain ID, but only update gallery if quality is high.
                        # Actually, if we skip matching on low quality, we might lose ID continuity across cameras?
                        # No, DeepSORT maintains ID locally. ReID finds GLOBAL ID.
                        # If we previously found a Global ID, we stick with it unless re-matched.
                        # To minimize compute, we can only run ReID check:
                        # 1. If Update needed (High Quality)
                        # 2. If track has NO Global ID yet (First time seen)
                        
                        current_global_id = getattr(track, 'global_id', None)
                        
                        if should_update or current_global_id is None:
                             match = loop.run_until_complete(
                                reid_service.match_identity(
                                    camera_id=frame_data.camera_id,
                                    embedding=feature,
                                    timestamp=frame_data.timestamp,
                                )
                             )
                             track.global_id = str(match.global_track_id)
                             
                             if should_update and thumb_b64:
                                 # Add to multi-capture gallery (Top-K with diversity)
                                 gallery_store = get_gallery_store()
                                 gallery_store.add_capture(
                                     global_id=track.global_id,
                                     image_b64=thumb_b64,
                                     quality_score=quality_score,
                                     pose=q_result.pose if hasattr(q_result, 'pose') else 'unknown',
                                     sharpness=q_result.sharpness_score if hasattr(q_result, 'sharpness_score') else 0.0,
                                     frame_number=frame_data.frame_number,
                                     timestamp=frame_data.timestamp,
                                     embedding=feature  # Cache embedding for fast search
                                 )
                                 logger.debug(f"Added capture to gallery for {track.global_id} (Q={quality_score:.2f})")
                                 
                                 # Periodic identity merge check
                                 if frame_data.frame_number - self._last_merge_frame >= self._merge_check_interval:
                                     try:
                                         identity_merger = get_identity_merger()
                                         merge_count = identity_merger.run_merge_pass()
                                         if merge_count > 0:
                                             logger.info(f"IdentityMerger: Merged {merge_count} fragmented identities")
                                         self._last_merge_frame = frame_data.frame_number
                                     except Exception as e:
                                         logger.warning(f"Identity merge check failed: {e}")
                        # Store Global ID in track for persistence if needed
                        # (DeepSORT track keeps attributes)
                        
                    except Exception as e:
                        logger.error(f"ReID or Quality check error for track {track.track_id}: {e}")

                    # Sync with TrackStore
                    if getattr(track, 'global_id', None):
                        try:
                            track_store = get_track_store()
                            track_store.add_or_update_track(
                                track.global_id,
                                {"status": TrackStatus.ACTIVE}
                            )
                            track_store.update_camera_sequence(track.global_id, frame_data.camera_id)
                        except Exception as e:
                            logger.error(f"Failed to update TrackStore: {e}")

            # 3. Convert tracks to serializable format (After ReID updates)

            for track in tracks:
                # Filter low confidence tracks to reduce "fake boxes"
                conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                if conf < 0.5 and track.state.name == 'CONFIRMED':
                     pass
                elif conf < 0.4: 
                     continue

                bbox = track.to_tlbr()
                # Get face_bbox and convert to serializable format
                face_bbox = getattr(track, 'face_bbox', None)
                if face_bbox is not None:
                    face_bbox = [int(x) for x in face_bbox]

                # Log face_bbox for debugging
                if face_bbox is not None:
                    logger.info(f"Track {track.track_id} has face_bbox: {face_bbox}")

                results["tracks"].append({
                    "track_id": track.track_id,
                    "global_id": getattr(track, 'global_id', None), # Include Global ID
                    "bbox": bbox.tolist(),
                    "face_bbox": face_bbox,  # Face bbox for visualization
                    "confidence": conf,
                    "class_name": getattr(track, 'class_name', 'unknown'),
                    "state": track.state.name,
                    "quality_score": getattr(track, 'quality_score', 0.0),
                    "pose": getattr(track, 'pose', 'unknown'),
                    "is_saved": getattr(track, 'is_saved', False),
                })
            
        finally:
            loop.close()
        
        return results
    
    def _encode_frame_fast(self, frame: np.ndarray, quality: int = 50) -> bytes:
        """Fast JPEG encoding with simplejpeg > TurboJPEG > OpenCV fallback."""
        # Reduce resolution first (biggest speed gain)
        h, w = frame.shape[:2]
        max_width = 960  # Good balance for ReID quality
        if w > max_width:
            scale = max_width / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if SIMPLEJPEG_AVAILABLE:
            # simplejpeg is ~3-5x faster than OpenCV, expects BGR
            return simplejpeg.encode_jpeg(frame, quality=quality, colorspace='BGR')
        
        if TURBOJPEG_AVAILABLE and TURBOJPEG:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return TURBOJPEG.encode(frame_rgb, quality=quality)
        
        # OpenCV fallback (slowest)
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return buffer.tobytes()
    
    def _prepare_frame_for_broadcast(self, frame: np.ndarray, tracks: List[dict]) -> np.ndarray:
        """Draw overlays on frame (lightweight but informative)."""
        for track in tracks:
            bbox = track["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            track_id = track["track_id"]
            global_id = track.get("global_id", "")
            conf = track.get("confidence", 0.0)
            
            # Color based on confidence (proxy for quality)
            if conf >= 0.7:
                color = (0, 255, 0)      # Green - High confidence
            elif conf >= 0.5:
                color = (0, 255, 255)    # Yellow - Medium
            else:
                color = (0, 0, 255)      # Red - Low
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Simple label with ID
            label = f"{global_id[:8] if global_id else track_id}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _broadcast_batch_parallel(self, batch_items, batch_results):
        """Broadcast all frames in batch using parallel JPEG encoding."""
        from concurrent.futures import as_completed
        
        # Prepare all frames with overlays first
        frames_to_encode = []
        for item in batch_items:
            camera_id = item.frame_data.camera_id
            if camera_id not in batch_results:
                continue
            
            tracks, features = batch_results[camera_id]
            track_list = [
                {
                    "track_id": track.track_id,
                    "bbox": track.to_tlbr().tolist(),
                    "confidence": float(track.confidence) if hasattr(track, 'confidence') else 0.0,
                    "global_id": getattr(track, 'global_id', None),
                }
                for track in tracks if track.is_confirmed()
            ]
            
            # Draw overlays
            frame = item.frame_data.frame.copy()
            frame = self._prepare_frame_for_broadcast(frame, track_list)
            
            frames_to_encode.append((item.frame_data, frame, track_list))
        
        # Encode all frames in parallel using thread pool
        futures = {}
        for frame_data, frame, track_list in frames_to_encode:
            future = ENCODE_EXECUTOR.submit(self._encode_frame_fast, frame, self.frame_quality)
            futures[future] = (frame_data, track_list)
        
        # Collect results and broadcast as they complete
        for future in as_completed(futures):
            frame_data, track_list = futures[future]
            try:
                buffer = future.result()
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                event = {
                    "type": "frame",
                    "camera_id": frame_data.camera_id,
                    "source_id": frame_data.source_id,
                    "frame_number": frame_data.frame_number,
                    "timestamp": frame_data.timestamp.isoformat(),
                    "frame_data": frame_b64,
                    "track_count": len(track_list),
                    "tracks": track_list,
                    "fps": round(self._fps, 1),
                }
                
                if self._main_loop and self._main_loop.is_running():
                    asyncio.run_coroutine_threadsafe(broadcast_event(event), self._main_loop)
            except Exception as e:
                logger.error(f"Encoding error: {e}")
    
    def _broadcast_frame(
        self,
        frame_data: FrameData,
        detections: List[dict],
        tracks: List[dict],
    ):
        """Broadcast frame with overlays to WebSocket."""
        try:
            # Draw overlays
            frame = frame_data.frame.copy()
            
            for track in tracks:
                bbox = track["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track["track_id"]
                
                # Determine color based on quality score
                quality_score = track.get("quality_score", 0.0)
                is_saved = track.get("is_saved", False)
                
                if quality_score >= 70:
                    color = (0, 255, 0)      # Green - Good
                elif quality_score >= 40:
                    color = (0, 255, 255)    # Yellow - Acceptable
                else:
                    color = (0, 0, 255)      # Red - Poor
                
                # Draw person bounding box
                thickness = 3 if is_saved else 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw face bounding box if detected (red)
                face_bbox = track.get("face_bbox")
                if face_bbox:
                    fx1, fy1, fx2, fy2 = map(int, face_bbox)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
                
                # Draw label with ID, Quality, Pose
                pose = track.get("pose", "unknown")
                label = f"ID:{track_id} Q:{quality_score:.0f} {pose}"
                if is_saved:
                    label += " [SAVED]"
                
                # Background for label text
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
            
            # Use fast encoding
            buffer = self._encode_frame_fast(frame, self.frame_quality)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Broadcast
            event = {
                "type": "frame",
                "camera_id": frame_data.camera_id,
                "source_id": frame_data.source_id,
                "frame_number": frame_data.frame_number,
                "timestamp": frame_data.timestamp.isoformat(),
                "frame_data": frame_b64,
                "track_count": len(tracks),
                "tracks": tracks,  # Bundle full track list for frontend sync
                "fps": round(self._fps, 1),
            }
            
            # Use main loop to broadcast
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event(event), self._main_loop)
                
        except MemoryError:
            raise  # Re-raise to be handled by caller (who does gc)
        except Exception as e:
            logger.error(f"Error in _broadcast_frame: {e}")
    
    def _broadcast_events(self, frame_data: FrameData, results: Dict[str, Any]):
        """Broadcast detection and ReID events."""
        # Broadcast track events
        if results["tracks"]:
            event = {
                "type": "tracks",
                "camera_id": frame_data.camera_id,
                "timestamp": frame_data.timestamp.isoformat(),
                "tracks": results["tracks"],
            }
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event(event), self._main_loop)
        
        # Broadcast ReID matches
        for match in results.get("reid_matches", []):
            event = {
                "type": "reid_match",
                "camera_id": frame_data.camera_id,
                "timestamp": frame_data.timestamp.isoformat(),
                **match,
            }
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(broadcast_event(event), self._main_loop)
    
    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            "running": self._running,
            "frame_count": self._frame_count,
            "fps": round(self._fps, 1),
            "detection_interval": self.detection_interval,
        }


# Singleton
_processor: Optional[StreamProcessor] = None


def get_stream_processor() -> StreamProcessor:
    """Get or create singleton StreamProcessor."""
    global _processor
    if _processor is None:
        _processor = StreamProcessor()
    return _processor


def start_stream_processor():
    """Start the stream processor."""
    processor = get_stream_processor()
    processor.start()


def stop_stream_processor():
    """Stop the stream processor."""
    global _processor
    if _processor:
        _processor.stop()
