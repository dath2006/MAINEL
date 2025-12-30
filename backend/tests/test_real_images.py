"""
Real Image Pipeline Test

Downloads real person images and tests the full detection → tracking → ReID pipeline.
Run with: python -m tests.test_real_images
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import cv2
import urllib.request

# Add parent to path for imports
import sys
sys.path.insert(0, ".")

from app.services import get_tracking_service, get_reid_service


# Real person images from COCO dataset (publicly available)
PERSON_IMAGES = [
    "https://images.pexels.com/photos/1181686/pexels-photo-1181686.jpeg?w=400",  # Person 1
    "https://images.pexels.com/photos/1181690/pexels-photo-1181690.jpeg?w=400",  # Person 2
    "https://images.pexels.com/photos/1181391/pexels-photo-1181391.jpeg?w=400",  # Person 3
]


def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as numpy array."""
    print(f"  Downloading: {url[:50]}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            image_data = response.read()
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("    Failed to decode image")
            return None
            
        print(f"    Downloaded: {img.shape}")
        return img
    except Exception as e:
        print(f"    Error: {e}")
        return None


def create_frame_with_image(person_img: np.ndarray, x: int = 200, y: int = 100,
                            frame_size=(1280, 720)) -> np.ndarray:
    """Place person image on a frame at specified position."""
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Gray background
    
    h, w = person_img.shape[:2]
    
    # Scale down if too large
    max_h = frame_size[1] - 100
    if h > max_h:
        scale = max_h / h
        person_img = cv2.resize(person_img, (int(w * scale), int(h * scale)))
        h, w = person_img.shape[:2]
    
    # Ensure within bounds
    x = min(max(0, x), frame_size[0] - w)
    y = min(max(0, y), frame_size[1] - h)
    
    frame[y:y+h, x:x+w] = person_img
    return frame


async def test_detection_with_real_image():
    """Test YOLOv8 detection on a real person image."""
    print("\n" + "="*60)
    print("TEST 1: Detection with Real Person Image")
    print("="*60)
    
    from app.core.detection import get_detector
    
    # Download a real person image
    img = download_image(PERSON_IMAGES[0])
    if img is None:
        print("❌ Failed to download image, skipping test")
        return
    
    # Create a frame with the person
    frame = create_frame_with_image(img, x=300, y=50)
    
    # Run detection
    detector = get_detector()
    detections = detector.detect(frame)
    
    print(f"\nDetections found: {len(detections)}")
    for det in detections:
        print(f"  - Confidence: {det.confidence:.2f}, BBox: {det.bbox}")
    
    if len(detections) > 0:
        print("\n✅ Successfully detected person in real image!")
    else:
        print("\n⚠️ No person detected (may need lower confidence threshold)")
    
    # Save visualization
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det.confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite("tests/detection_result.jpg", frame)
    print("Saved visualization to: tests/detection_result.jpg")


async def test_tracking_with_moving_person():
    """Test tracking a person moving across frames."""
    print("\n" + "="*60)
    print("TEST 2: Tracking Person Across Frames")
    print("="*60)
    
    tracking = get_tracking_service()
    
    # Download person image
    img = download_image(PERSON_IMAGES[0])
    if img is None:
        print("❌ Failed to download image, skipping test")
        return
    
    # Simulate person walking across frame
    positions = [(50 + i*80, 50) for i in range(8)]  # Moving right
    
    print(f"\nSimulating person walking across {len(positions)} frames...")
    
    track_ids_seen = set()
    
    for i, (x, y) in enumerate(positions):
        frame = create_frame_with_image(img, x=x, y=y)
        timestamp = datetime.now() + timedelta(milliseconds=i*200)
        
        tracks, features = await tracking.process_frame(
            camera_id=10,  # Use camera 10 for this test
            frame=frame,
            timestamp=timestamp,
            extract_features=True,
        )
        
        for t in tracks:
            track_ids_seen.add(t.track_id)
        
        print(f"  Frame {i+1}: {len(tracks)} tracks, IDs: {[t.track_id for t in tracks]}")
    
    if len(track_ids_seen) == 1:
        print(f"\n✅ Successfully tracked person with consistent ID: {track_ids_seen}")
    elif len(track_ids_seen) > 1:
        print(f"\n⚠️ Track ID changed: {track_ids_seen} (may need parameter tuning)")
    else:
        print("\n⚠️ No tracks created")


async def test_reid_across_cameras():
    """Test re-identification across two cameras."""
    print("\n" + "="*60)
    print("TEST 3: Cross-Camera Re-Identification")
    print("="*60)
    
    tracking = get_tracking_service()
    reid = get_reid_service()
    
    # Register two cameras
    reid.register_camera(20, 12.9716, 77.5946)
    reid.register_camera(21, 12.9720, 77.5950)
    print("Registered cameras 20 and 21")
    
    # Download same person image
    img = download_image(PERSON_IMAGES[0])
    if img is None:
        print("❌ Failed to download image, skipping test")
        return
    
    # Camera 20: Person appears
    print("\n📷 Camera 20: Person enters...")
    frame1 = create_frame_with_image(img, x=200, y=50)
    timestamp1 = datetime.now()
    
    tracks1, features1 = await tracking.process_frame(20, frame1, timestamp1)
    
    global_id = None
    if features1:
        _, embedding = features1[0]
        match1 = await reid.match_identity(20, embedding, timestamp1)
        global_id = match1.global_track_id
        print(f"  Global Track ID: {global_id} (new={match1.is_new})")
    
    # Camera 21: Same person appears (simulated transition)
    print("\n📷 Camera 21: Same person appears...")
    tracking.reset_camera(21)
    
    frame2 = create_frame_with_image(img, x=300, y=60)  # Slightly different position
    timestamp2 = datetime.now() + timedelta(seconds=30)  # 30 seconds later
    
    tracks2, features2 = await tracking.process_frame(21, frame2, timestamp2)
    
    if features2:
        _, embedding2 = features2[0]
        match2 = await reid.match_identity(21, embedding2, timestamp2)
        
        print(f"\n  Match Result:")
        print(f"    Global Track ID: {match2.global_track_id}")
        print(f"    Visual Similarity: {match2.visual_similarity:.3f}")
        print(f"    Is Same Person: {match2.global_track_id == global_id}")
        print(f"    Is New Identity: {match2.is_new}")
        
        if match2.global_track_id == global_id:
            print("\n✅ SUCCESS: Same person correctly identified across cameras!")
        else:
            print("\n⚠️ Different identity assigned")


async def main():
    """Run all tests with real images."""
    print("\n" + "="*60)
    print("MCMT ReID - Real Image Test Suite")
    print("="*60)
    print("\nNote: This test downloads real images from the web.")
    
    try:
        await test_detection_with_real_image()
        await test_tracking_with_moving_person()
        await test_reid_across_cameras()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
