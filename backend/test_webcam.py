"""
Quick webcam test to verify OpenCV can access the camera.
"""
import cv2
import sys

def test_webcam(camera_index: int = 0):
    print(f"Testing camera index: {camera_index}")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}")
        print("Try:")
        print("  1. Close any apps using the camera (Zoom, Teams, etc)")
        print("  2. Try a different index (1, 2, etc)")
        print("  3. Check camera permissions")
        return False
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"SUCCESS! Camera opened: {width}x{height} @ {fps}fps")
    
    # Read one frame
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured successfully! Shape: {frame.shape}")
    else:
        print("WARNING: Could not read frame")
    
    cap.release()
    return True


if __name__ == "__main__":
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_webcam(index)
