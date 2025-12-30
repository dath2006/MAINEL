
import cv2
import time
import os

def test_webcam(index=0):
    print(f"Testing Webcam {index}...")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    # Warmup
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            print(f"Frame Read! Shape: {frame.shape}, Mean Color: {frame.mean()}")
        else:
            print("Failed to read frame")
        time.sleep(0.05)
    
    # Save one
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("debug_webcam.jpg", frame)
        print("Saved debug_webcam.jpg")
    
    cap.release()

def test_video_file(path):
    print(f"Testing Video File: {path}")
    if not os.path.exists(path):
        print("File does not exist!")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Failed to open video file")
        return

    ret, frame = cap.read()
    if ret:
        print(f"Frame Read! Shape: {frame.shape}")
        cv2.imwrite("debug_video.jpg", frame)
        print("Saved debug_video.jpg")
    else:
        print("Failed to read video frame")
    
    cap.release()

if __name__ == "__main__":
    # Test Webcam 0
    test_webcam(0)
    
    # Test dummy file if exists (user can specify path)
    # test_video_file("C:/videos/test.mp4")
