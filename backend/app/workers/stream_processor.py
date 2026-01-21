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
from datetime import datetime
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from loguru import logger

from app.services.stream_manager import get_stream_manager, FrameData, PlaybackState
from app.services.tracking_service import get_tracking_service
from app.services.reid_service import get_reid_service
from app.services.track_store import get_track_store
from app.services.identity_merger import get_identity_merger
from app.api.v1.realtime import broadcast_event
from app.schemas.track import TrackStatus
from app.services.gallery_store import get_gallery_store
from app.services.gallery_store import get_gallery_store
from app.config import settings
from app.core.gpu_jpeg_encoder import get_gpu_encoder

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
        frame_quality: int = 50,  # JPEG quality for frames
    ):
        self.detection_interval = detection_interval
        self.broadcast_frames = broadcast_frames
        self.frame_quality = frame_quality
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
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
        """Main processing loop - sequential frame processing."""
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
        
        while self._running:
            # Check if playing
            if stream_manager.state != PlaybackState.PLAYING:
                time.sleep(0.1)
                continue
            
            # Get frame from queue
            frame_data = stream_manager.get_next_frame(timeout=0.5)
            if frame_data is None:
                continue
            
            self._frame_count += 1
            self._fps_frame_count += 1
            
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
                    if "CUDA" in str(e):
                        logger.error(f"CUDA Error during ML processing: {e}")
                        import gc
                        gc.collect()
                        time.sleep(0.1)
                    else:
                        logger.debug(f"RuntimeError in ML processing: {e}")
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
                    logger.error(f"MemoryError in processor. Clearing memory.")
                    import gc
                    gc.collect()
                    time.sleep(1)
                    continue
                except Exception as e:
                    logger.error(f"Error broadcasting frame: {e}")
                    continue
    
    def _process_batch(
        self,
        batch_frames: List[FrameData],
        tracking_service,
        reid_service,
    ) -> List[Dict[str, Any]]:
        """Process a batch of frames through the ML pipeline with quality scoring."""
        
        # Get quality scorer
        quality_scorer = get_quality_scorer_instance()
        
        # Prepare frames for batch processing
        frames_tuple = [
            (fd.camera_id, fd.frame, fd.timestamp)
            for fd in batch_frames
        ]
        
        # Run batch tracking
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            batch_track_results = loop.run_until_complete(
                tracking_service.process_batch(frames_tuple)
            )
        finally:
            loop.close()
        
        # Process results for each frame
        all_results = []
        for idx, (tracks, camera_id) in enumerate(batch_track_results):
            frame_data = batch_frames[idx]
            frame = frame_data.frame
            results = {"detections": [], "tracks": [], "reid_matches": []}
            
            # ReID matching and quality scoring for confirmed tracks
            if reid_service:
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    
                    conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                    if conf < 0.5:
                        continue
                    
                    if not track.features:
                        continue
                    
                    feature = track.features[-1]
                    
                    # Quality assessment
                    should_update = False
                    quality_score = 0.0
                    thumb_b64 = None
                    pose = 'unknown'
                    
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Phase 1: Pre-filtering - Validate detection quality
                    if x2 > x1 and y2 > y1:
                        bbox_array = np.array([x1, y1, x2, y2])
                        frame_shape = (h, w)
                        
                        is_valid, reason = tracking_service.validate_detection_quality(
                            bbox=bbox_array,
                            confidence=conf,
                            frame_shape=frame_shape,
                        )
                        
                        if not is_valid:
                            logger.debug(f"[Batch] Rejected detection: {reason}")
                            continue  # Skip this track for gallery
                    
                    if x2 > x1 and y2 > y1 and quality_scorer:
                        crop = frame[y1:y2, x1:x2]
                        
                        # Phase 2: Validate person presence

                        if settings.reid_enable_presence_check:
                            is_person_present, _ = tracking_service.validate_person_presence(crop)
                            if not is_person_present:
                                continue
                        
                        # Assess Quality
                        q_result = quality_scorer.score(crop)
                        quality_score = q_result.total_score
                        pose = q_result.pose if hasattr(q_result, 'pose') else 'unknown'
                        
                        # High quality - save to gallery
                        if quality_score > settings.gallery_quality_threshold:
                            should_update = True
                            
                            # Prepare thumbnail
                            thumb = cv2.resize(crop, (64, 128))
                            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            thumb_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Store quality info on track
                    track.quality_score = quality_score
                    track.pose = pose
                    track.is_saved = should_update
                    
                    # ReID matching
                    current_global_id = getattr(track, 'global_id', None)
                    
                    if should_update or current_global_id is None:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            match = loop.run_until_complete(
                                reid_service.match_identity(
                                    camera_id=camera_id,
                                    embedding=feature,
                                    timestamp=frame_data.timestamp,
                                )
                            )
                            track.global_id = str(match.global_track_id)
                        finally:
                            loop.close()
                        
                        # Save to gallery if high quality
                        if should_update and thumb_b64:
                            gallery_store = get_gallery_store()
                            gallery_store.add_capture(
                                global_id=track.global_id,
                                image_b64=thumb_b64,
                                quality_score=quality_score,
                                pose=pose,
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
                            track_store.update_camera_sequence(track.global_id, camera_id)
                        except Exception as e:
                            logger.error(f"Failed to update TrackStore: {e}")
            
            # Convert tracks to serializable format
            for track in tracks:
                conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                if conf < 0.4:
                    continue
                
                bbox = track.to_tlbr()
                face_bbox = getattr(track, 'face_bbox', None)
                if face_bbox is not None:
                    face_bbox = [int(x) for x in face_bbox]
                
                results["tracks"].append({
                    "track_id": track.track_id,
                    "global_id": getattr(track, 'global_id', None),
                    "bbox": bbox.tolist(),
                    "face_bbox": face_bbox,
                    "confidence": conf,
                    "class_name": getattr(track, 'class_name', 'unknown'),
                    "state": track.state.name,
                    "quality_score": getattr(track, 'quality_score', 0.0),
                    "pose": getattr(track, 'pose', 'unknown'),
                    "is_saved": getattr(track, 'is_saved', False),
                })
            
            all_results.append(results)
        
        return all_results
    
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
                # logger.info(f"ReID: Processing {len(tracks)} tracks")
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
                        
                        # Phase 1: Pre-filtering - Validate detection quality
                        if x2 > x1 and y2 > y1:
                            bbox_array = np.array([x1, y1, x2, y2])
                            frame_shape = (h, w)
                            
                            is_valid, reason = tracking_service.validate_detection_quality(
                                bbox=bbox_array,
                                confidence=conf,
                                frame_shape=frame_shape,
                            )
                            
                            if not is_valid:
                                logger.debug(f"Rejected detection for gallery: {reason}")
                                continue  # Skip this track, don't add to gallery
                        
                        if x2 > x1 and y2 > y1 and quality_scorer:
                             crop = frame_data.frame[y1:y2, x1:x2]
                             
                             # Phase 2: Person presence validation
                             if settings.reid_enable_presence_check:
                                 is_person_present, presence_reason = tracking_service.validate_person_presence(crop)
                                 if not is_person_present:
                                     logger.debug(f"Rejected empty/invalid crop: {presence_reason}")
                                     continue  # Skip, no valid person in crop
                             
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
                                #  logger.debug(f"Added capture to gallery for {track.global_id} (Q={quality_score:.2f})")
                                 
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
            
            # Encode frame as JPEG
            # Uses GPU acceleration if available, falls back to CPU
            try:
                frame_b64 = get_gpu_encoder(self.frame_quality).encode(frame)
                # Convert bytes to base64 string
                frame_b64 = base64.b64encode(frame_b64).decode('utf-8')
            except Exception as e:
                logger.error(f"Encoding failed: {e}")
                return
            
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
