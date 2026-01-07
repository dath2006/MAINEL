import sys
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

try:
    from app.core.tracking.boxmot_tracker import BoxMOTTracker
    from app.core.detection import Detection
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def test_boxmot_pipeline():
    logger.info("Initializing BoxMOTTracker...")
    try:
        # Use CPU for testing to minimize VRAM if on dev machine
        tracker = BoxMOTTracker(device='cuda' if False else 'cpu') 
    except Exception as e:
        logger.error(f"Failed to init tracker: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Tracker initialized. Creating dummy data...")
    
    # Dummy frame 640x640 range 0-255
    frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Dummy detection [100, 100, 200, 200]
    # Detection(bbox, conf, class_id)
    detections = [
        Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0)
    ]
    
    logger.info("Running update...")
    try:
        tracks = tracker.update(detections, frame=frame)
        logger.info(f"Update successful. Tracks: {len(tracks)}")
        if len(tracks) > 0:
            t = tracks[0]
            logger.info(f"Track ID: {t.track_id}, Confidence: {t.confidence}, State: {t.state}")
            if hasattr(t, 'features') and len(t.features) > 0:
                logger.info(f"Feature shape: {t.features[0].shape}")
            else:
                 logger.warning("Track has no features.")
    except Exception as e:
        logger.error(f"Update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boxmot_pipeline()
