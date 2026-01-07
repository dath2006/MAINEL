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
from app.api.v1.realtime import broadcast_event


class StreamProcessor:
    """
    Processes video frames from StreamManager and runs ML pipeline.
    
    Pipeline:
    1. Get frame from StreamManager queue
    2. Run YOLO detection
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
        """Main processing loop."""
        # Create dedicated loop for this thread
        self._worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._worker_loop)

        stream_manager = get_stream_manager()
        
        # Try to get ML services, but don't fail if unavailable
        tracking_service = None
        reid_service = None
        try:
            print("DEBUG_PRINT: Loading ML services...")
            tracking_service = get_tracking_service()
            print(f"DEBUG_PRINT: tracking_service loaded: {tracking_service}")
            reid_service = get_reid_service()
            print("DEBUG_PRINT: ML services loaded successfully")
            logger.info("ML services loaded successfully")
        except Exception as e:
            print(f"DEBUG_PRINT: FAILED TO LOAD SERVICES: {e}")
            import traceback
            traceback.print_exc()
            logger.warning(f"ML services not available (frames will still stream): {e}")
            
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
                if self._frame_count % 10 == 0:
                     print("DEBUG_PRINT: Waiting for frames... (Queue empty)")
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
            
            print(f"DEBUG_PRINT LOOP: frame={self._frame_count}, run_ml={run_ml}, tracking={tracking_service is not None}, fps={self._fps:.1f}")

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
                        print(f"DEBUG_PRINT: RuntimeError in ML: {e}")
                        logger.error(f"RuntimeError in ML processing {self.source_id}: {e}")
                except Exception as e:
                    print(f"DEBUG_PRINT: Frame processing error: {e}")
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Frame processing error: {e}")
            else:
                # Don't reuse old tracks - prevents stuck bounding boxes
                # Send current frame WITHOUT old bounding boxes
                frame_detections = []
                frame_tracks = []
            
            # Broadcast frame ONLY when ML ran (to avoid stale bounding boxes)
            if self.broadcast_frames and run_ml:
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
        if hasattr(self, '_worker_loop'):
            self._worker_loop.close()
    
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
        # We use the persistent worker loop for async ML ops
        try:
            tracks, features = self._worker_loop.run_until_complete(
                tracking_service.process_frame(
                    camera_id=frame_data.camera_id,
                    frame=frame_data.frame,
                    timestamp=frame_data.timestamp,
                    extract_features=True,
                )
            )
            
            # 2. Run ReID matching for confirmed tracks
            if reid_service:
                logger.info(f"ReID: Processing {len(tracks)} tracks")
                for match_idx, track in enumerate(tracks):
                    if not track.is_confirmed():
                        continue
                        
                    # Find feature for this track
                    # Note: tracking_service returns new_features as (uuid, feat) for NEW tracklets
                    # But we need current feature. Track object stores history.
                    if not track.features:
                        logger.warning(f"ReID: Track {track.track_id} has no features, skipping")
                        continue
                        
                    feature = track.features[-1] # Latest feature
                    
                    # OPTIMIZATION: If track already has global_id, skip expensive matching
                    # This enforces identity consistency for the life of the track
                    if track.global_id:
                        logger.debug(f"ReID: Track {track.track_id} already has Global ID {track.global_id}, skipping match")
                        continue

                    logger.info(f"ReID: Matching track {track.track_id} with feature shape {feature.shape}")
                    
                    try:
                        match = self._worker_loop.run_until_complete(
                            reid_service.match_identity(
                                camera_id=frame_data.camera_id,
                                embedding=feature,
                                timestamp=frame_data.timestamp,
                            )
                        )
                        
                        # Assign Global ID to Track
                        track.global_id = str(match.global_track_id)
                        logger.info(f"ReID: Track {track.track_id} -> Global ID {track.global_id} (is_new={match.is_new}, sim={match.visual_similarity:.2f})")
                        
                        # Update TrackStore
                        track_store = get_track_store()
                        # Don't pass camera_sequence here to avoid overwriting history
                        track_store.add_or_update_track(track.global_id, {
                             # metadata updates if any
                        })
                        track_store.update_camera_sequence(track.global_id, frame_data.camera_id)

                        # Capture thumbnail for new identities OR if no thumbnail exists yet
                        existing_thumb = reid_service.person_thumbnails.get(track.global_id)
                        should_capture = match.is_new or existing_thumb is None
                        
                        if should_capture:
                            try:
                                import cv2
                                import base64
                                
                                bbox = track.to_tlbr()  # [x1, y1, x2, y2]
                                x1, y1, x2, y2 = map(int, bbox)
                                
                                # Clamp to frame bounds
                                h, w = frame_data.frame.shape[:2]
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w, x2), min(h, y2)
                                
                                crop_w = x2 - x1
                                crop_h = y2 - y1
                                
                                # RELAXED Quality check to prevent empty images
                                # Allow smaller crops, especially for faces/far objects
                                MIN_CROP_W, MIN_CROP_H = 24, 48 
                                EDGE_MARGIN = 2 # Reduced from 10 to capture people entering/leaving
                                
                                is_quality_crop = (
                                    crop_w >= MIN_CROP_W and 
                                    crop_h >= MIN_CROP_H and
                                    x1 >= EDGE_MARGIN and 
                                    y1 >= EDGE_MARGIN and
                                    x2 <= w - EDGE_MARGIN and
                                    y2 <= h - EDGE_MARGIN
                                )
                                
                                if is_quality_crop:
                                    crop = frame_data.frame[y1:y2, x1:x2]
                                    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                                         logger.warning(f"Empty crop for {track.global_id}, skipping thumbnail")
                                    else:
                                        # Resize to thumbnail size
                                        thumb = cv2.resize(crop, (64, 128))
                                        _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                        thumb_b64 = base64.b64encode(buffer).decode('utf-8')
                                        
                                        # Double check generated b64
                                        if thumb_b64 and len(thumb_b64) > 100:
                                            reid_service.set_thumbnail(track.global_id, thumb_b64)
                                            logger.info(f"Captured thumbnail for {track.global_id}")
                                        else:
                                            logger.warning(f"Generated empty/invalid b64 for {track.global_id}")
                                else:
                                    logger.debug(f"Skipped low-quality thumbnail for {track.global_id} (size={crop_w}x{crop_h})")
                            except Exception as thumb_err:
                                logger.warning(f"Failed to capture thumbnail: {thumb_err}")
                        
                        results["reid_matches"].append({
                            "local_track_id": track.track_id,
                            "global_track_id": str(match.global_track_id),
                            "visual_similarity": match.visual_similarity,
                            "is_new": match.is_new,
                        })
                    except Exception as e:
                        logger.error(f"ReID error for track {track.track_id}: {e}")

            # 3. Convert tracks to serializable format (After ReID updates)
            # CRITICAL: Only include tracks that were matched to actual detections
            # BoxMOT returns det_idx=-1 for coasted/predicted tracks - these must be filtered
            for track in tracks:
                # Skip if this track wasn't matched to a detection (coasted/predicted)
                # We check time_since_update==0 as a proxy for "was matched this frame"
                if track.time_since_update != 0:
                    continue
                    
                # Filter low confidence tracks to reduce "fake boxes"
                conf = float(track.confidence) if hasattr(track, 'confidence') else 0.0
                if conf < 0.5 and track.state.name == 'CONFIRMED':
                     pass
                elif conf < 0.4: 
                     continue

                bbox = track.to_tlbr()
                results["tracks"].append({
                    "track_id": track.track_id,
                    "global_id": getattr(track, 'global_id', None), # Include Global ID
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                    "class_name": getattr(track, 'class_name', 'unknown'),
                    "state": track.state.name,
                    "face_bbox": getattr(track, 'face_bbox', None),  # Include face bbox
                })
            
            
        finally:
            pass # Loop management is handled in _run_loop
        
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
                
                # Draw body bounding box (green, 2px)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw face bounding box if available (cyan, 1px thin)
                face_bbox = track.get("face_bbox")
                if face_bbox:
                    fx1, fy1, fx2, fy2 = map(int, face_bbox)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)  # Cyan color
                
                # Draw label
                label = f"ID: {track_id}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            
            # Encode frame as JPEG
            # Use lower quality to save bandwidth/memory if needed
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            
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
