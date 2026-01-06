import cv2
import numpy as np

# 1. SETUP: Your GPS Coordinates (Destination)
# Format: [Longitude (x), Latitude (y)] 
# Note: We use Long/Lat to map to X/Y. 
# ORDER: Bottom-Left, Bottom-Right, Top-Left, Top-Right (As per your list)
gps_dst = np.array([
    [77.496482, 12.922103], # Bottom Left
    [77.496479, 12.922112], # Bottom Right
    [77.496493, 12.922095], # Top Left
    [77.496488, 12.922115]  # Top Right
], dtype='float32')

# List to store image pixel points you click
pixel_src = []

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Phase 1: Calibration (Collecting the 4 points)
        if len(pixel_src) < 4:
            pixel_src.append([x, y])
            print(f"Point {len(pixel_src)} Captured: {x}, {y}")
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) # Red dot
            cv2.imshow("CCTV Calibration", image)
            
            if len(pixel_src) == 4:
                print("\n--- Calibration Complete! Calculating Matrix... ---")
                calculate_and_test(image)

        # Phase 2: Testing (Simulating a person detection)
        else:
            print(f"\n[Test] Clicked Pixel: {x}, {y}")
            # This is the logic you will put inside your bounding box loop
            predict_gps(x, y)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1) # Green dot
            cv2.imshow("CCTV Calibration", image)

def calculate_and_test(img):
    global h_matrix
    
    # Convert list to numpy array
    pts_src = np.array(pixel_src, dtype='float32')
    
    # Calculate Homography Matrix
    h_matrix, status = cv2.findHomography(pts_src, gps_dst)
    
    print("Matrix Ready. Now click anywhere to see the GPS coordinate.")

def predict_gps(x, y):
    # Prepare the point (Must be shape 1,1,2)
    point = np.array([[[x, y]]], dtype='float32')
    
    # Transform
    gps_point = cv2.perspectiveTransform(point, h_matrix)
    
    lat = gps_point[0][0][1]
    long = gps_point[0][0][0]
    
    print(f"PREDICTED GPS -> Lat: {lat:.6f}, Long: {long:.6f}")

# --- MAIN EXECUTION ---
# REPLACE 'your_image.jpg' with your actual image filename
image_path = 'test.jpeg' 
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Check the filename.")
else:
    print("INSTRUCTIONS:")
    print("1. Click the Bottom-Left reference point.")
    print("2. Click the Bottom-Right reference point.")
    print("3. Click the Top-Left reference point.")
    print("4. Click the Top-Right reference point.")
    
    cv2.imshow("CCTV Calibration", image)
    cv2.setMouseCallback("CCTV Calibration", mouse_handler)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()