
import cv2
import sys

def test_camera_display(index=0):
    print(f"Attempting to open camera {index}...")
    # CAP_DSHOW is often needed for Windows webcams to initialize quickly
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {index}")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened successfully.")
    print("Press 'q' in the window to quit.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        cv2.imshow(f'Test Camera {index}', frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} - Shape: {frame.shape}")

        # Check for 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")

if __name__ == "__main__":
    idx = 0
    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
    test_camera_display(idx)
