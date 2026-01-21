
import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.cuda

# Add backend to path so we can import app code
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from app.core.gpu_jpeg_encoder import GPUJpegEncoder

def test_encoder():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Create a dummy image (1080p random noise)
    print("\nCreating dummy 1080p image...")
    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Initialize encoder
    print("Initializing GPUJpegEncoder...")
    encoder = GPUJpegEncoder(quality=75)
    
    # Run a few warmup passes
    print("Warming up...")
    for _ in range(5):
        try:
            encoder.encode(img)
        except Exception as e:
            print(f"Warmup failed: {e}")
            
    # Benchmark
    print("Benchmarking 100 encodes...")
    start_time = time.time()
    for _ in range(100):
        encoded_bytes = encoder.encode(img)
        
    end_time = time.time()
    avg_time = (end_time - start_time) / 100 * 1000
    fps = 1000 / avg_time
    
    print(f"Average encode time: {avg_time:.2f} ms")
    print(f"Approx FPS: {fps:.2f}")
    
    # Verify correctness
    print("\nVerifying correctness...")
    encoded_bytes = encoder.encode(img)
    decoded_img = cv2.imdecode(np.frombuffer(encoded_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    if decoded_img is None:
        print("FAIL: Could not decode generated JPEG")
    else:
        print(f"Success! Decoded image shape: {decoded_img.shape}")
        
    # Stats
    stats = encoder.get_stats()
    print("\nEncoder Stats:")
    print(stats)
    
    if torch.cuda.is_available() and stats['gpu_encodes'] == 0:
        print("\nWARNING: CUDA is available but no GPU encodes were performed!")
        print("This likely means torchvision was not compiled with nvJPEG support.")

if __name__ == "__main__":
    test_encoder()
