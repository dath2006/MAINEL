"""
Pipeline Test Script

Tests the full detection → tracking → ReID pipeline with synthetic data.
Run with: python -m tests.test_pipeline
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
import cv2

# Add parent to path for imports
import sys
sys.path.insert(0, ".")

from app.services import get_tracking_service, get_reid_service

# Synthetic person images (colored rectangles simulating different people)
def create_person_image(person_id: int, size=(128, 256)) -> np.ndarray:
    """Create a synthetic person image with unique color pattern."""
    np.random.seed(person_id)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    # Body color
    body_color = tuple(np.random.randint(50, 200, 3).tolist())
    cv2.rectangle(img, (20, 80), (108, 240), body_color, -1)
    
    # Head
    head_color = (220, 180, 160)  # Skin tone
    cv2.ellipse(img, (64, 50), (30, 40), 0, 0, 360, head_color, -1)
    
    # Add some variation
    for _ in range(5):
        x, y = np.random.randint(20, 100), np.random.randint(80, 220)
        c = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img, (x, y), 10, c, -1)
    
    return img


def create_frame_with_person(person_id: int, x: int, y: int, 
                             frame_size=(1920, 1080)) -> np.ndarray:
    """Create a frame with a person at specified position."""
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # Dark background
    
    person = create_person_image(person_id)
    h, w = person.shape[:2]
    
    # Ensure within bounds
    x = min(max(0, x), frame_size[0] - w)
    y = min(max(0, y), frame_size[1] - h)
    
    frame[y:y+h, x:x+w] = person
    return frame


async def test_single_camera_tracking():
    """Test tracking a person across frames in a single camera."""
    print("\n" + "="*60)
    print("TEST 1: Single Camera Tracking")
    print("="*60)
    
    tracking = get_tracking_service()
    
    # Simulate person walking across frame
    person_id = 1
    positions = [(100 + i*50, 400) for i in range(10)]  # Walking right
    
    print(f"Simulating person {person_id} walking across camera...")
    
    for i, (x, y) in enumerate(positions):
        frame = create_frame_with_person(person_id, x, y)
        timestamp = datetime.now() + timedelta(milliseconds=i*100)
        
        tracks, features = await tracking.process_frame(
            camera_id=1,
            frame=frame,
            timestamp=timestamp,
            extract_features=True,
        )
        
        print(f"  Frame {i+1}: {len(tracks)} active tracks")
        for t in tracks:
            print(f"    Track {t.track_id}: bbox={t.to_tlbr()[:2].astype(int)}, state={t.state}")
    
    print("✅ Single camera tracking complete!")


async def test_cross_camera_reid():
    """Test re-identification across two cameras."""
    print("\n" + "="*60)
    print("TEST 2: Cross-Camera Re-Identification")
    print("="*60)
    
    tracking = get_tracking_service()
    reid = get_reid_service()
    
    # Register cameras
    reid.register_camera(1, 12.9716, 77.5946)  # Camera 1 position
    reid.register_camera(2, 12.9720, 77.5950)  # Camera 2 (nearby)
    
    print("Registered 2 cameras for ReID")
    
    # Same person appears in both cameras
    person_id = 42
    
    # Camera 1: Person enters and exits
    print("\n📷 Camera 1: Person enters...")
    camera1_frames = [(100 + i*30, 400) for i in range(5)]
    camera1_embedding = None
    
    for i, (x, y) in enumerate(camera1_frames):
        frame = create_frame_with_person(person_id, x, y)
        timestamp = datetime.now() + timedelta(seconds=i)
        
        tracks, features = await tracking.process_frame(1, frame, timestamp)
        
        if features:
            _, camera1_embedding = features[0]
            print(f"  Extracted embedding: shape={camera1_embedding.shape}")
    
    # Match in Camera 1 (first appearance)
    if camera1_embedding is not None:
        match1 = await reid.match_identity(
            camera_id=1,
            embedding=camera1_embedding,
            timestamp=datetime.now(),
        )
        print(f"  Global Track: {match1.global_track_id} (new={match1.is_new})")
        global_id = match1.global_track_id
    
    # Camera 2: Same person appears after 30 seconds
    print("\n📷 Camera 2: Same person appears (30s later)...")
    tracking.reset_camera(2)  # Fresh tracker for camera 2
    
    camera2_frames = [(200 + i*30, 350) for i in range(5)]
    camera2_embedding = None
    
    for i, (x, y) in enumerate(camera2_frames):
        frame = create_frame_with_person(person_id, x, y)
        timestamp = datetime.now() + timedelta(seconds=30+i)
        
        tracks, features = await tracking.process_frame(2, frame, timestamp)
        
        if features:
            _, camera2_embedding = features[0]
    
    # Match in Camera 2
    if camera2_embedding is not None:
        match2 = await reid.match_identity(
            camera_id=2,
            embedding=camera2_embedding,
            timestamp=datetime.now() + timedelta(seconds=30),
        )
        
        print(f"\n  Match result:")
        print(f"    Global Track: {match2.global_track_id}")
        print(f"    Visual Similarity: {match2.visual_similarity:.3f}")
        print(f"    ST Probability: {match2.st_probability:.3f}")
        print(f"    Joint Score: {match2.joint_score:.3f}")
        print(f"    Is New Identity: {match2.is_new}")
        
        if match2.global_track_id == global_id:
            print("\n✅ SUCCESS: Same person correctly identified across cameras!")
        else:
            print("\n⚠️ Different identity assigned (may need tuning)")


async def test_feature_extractor():
    """Test the feature extractor directly."""
    print("\n" + "="*60)
    print("TEST 3: Feature Extractor")
    print("="*60)
    
    from app.core.features import get_extractor
    
    extractor = get_extractor()
    print(f"Extractor type: {type(extractor).__name__}")
    print(f"Embedding dimension: {extractor.EMBEDDING_DIM}")
    
    # Create two images of same "person" and one different
    img1 = create_person_image(1)
    img2 = create_person_image(1)  # Same person
    img3 = create_person_image(2)  # Different person
    
    emb1 = extractor.extract(img1)
    emb2 = extractor.extract(img2)
    emb3 = extractor.extract(img3)
    
    sim_same = np.dot(emb1, emb2)
    sim_diff = np.dot(emb1, emb3)
    
    print(f"\nSimilarity (same person): {sim_same:.4f}")
    print(f"Similarity (diff person): {sim_diff:.4f}")
    
    if sim_same > sim_diff:
        print("✅ Feature extractor correctly distinguishes people!")
    else:
        print("⚠️ Feature extractor needs improvement")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("MCMT ReID Pipeline Test Suite")
    print("="*60)
    
    try:
        await test_feature_extractor()
        await test_single_camera_tracking()
        await test_cross_camera_reid()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
