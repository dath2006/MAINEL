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
        # Use CPU for testing if no GPU
        tracker = BoxMOTTracker(device='cpu')
    except Exception as e:
        logger.error(f"Failed to init tracker: {e}")
        return

    logger.info("Tracker initialized. Creating dummy data...")
    
    # Dummy frame 640x640
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Dummy detection [100, 100, 200, 200]
    detections = [
        Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0)
    ]
    
    logger.info("Running update...")
    try:
        tracks = tracker.update(detections, frame=frame)
        logger.info(f"Update successful. Tracks: {len(tracks)}")
        if len(tracks) > 0:
            t = tracks[0]
            logger.info(f"Track ID: {t.track_id}, Features: {len(t.features)}")
            if len(t.features) > 0:
                logger.info(f"Feature shape: {t.features[0].shape}")
    except Exception as e:
        logger.error(f"Update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boxmot_pipeline()
