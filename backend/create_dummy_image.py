
import cv2
import numpy as np
import os

def create_image():
    # Create a red image 100x200
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255) # BGR Red
    
    path = os.path.abspath("test_person.jpg")
    cv2.imwrite(path, img)
    print(f"Created {path}")

if __name__ == "__main__":
    create_image()
