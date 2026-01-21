"""
Benchmark for GPU JPEG Encoder
"""

import time
import numpy as np
import cv2
from app.core.gpu_jpeg_encoder import get_gpu_encoder

def create_test_frame(width=1920, height=1080):
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add structure
    cv2.rectangle(frame, (100, 100), (400, 500), (0, 255, 0), 2)
    return frame

def main():
    print("="*60)
    print("GPU JPEG Encoder Benchmark")
    print("="*60)
    
    frame = create_test_frame()
    encoder = get_gpu_encoder(quality=50)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        _ = encoder.encode(frame)
    
    # Benchmark
    iterations = 100
    print(f"Running {iterations} iterations...")
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = encoder.encode(frame)
    end = time.perf_counter()
    
    total_time = (end - start) * 1000
    avg_time = total_time / iterations
    
    stats = encoder.get_stats()
    
    print("\nResults:")
    print(f"  Average Time: {avg_time:.3f} ms")
    print(f"  Total Time: {total_time:.3f} ms")
    print(f"  FPS Potential: {1000/avg_time:.1f}")
    
    print("\nEncoder Stats:")
    print(f"  GPU Available: {stats['gpu_available']}")
    print(f"  GPU Encodes: {stats['gpu_encodes']}")
    print(f"  CPU Fallbacks: {stats['cpu_fallbacks']}")
    
    if stats['gpu_available']:
        print("\n✅ GPU Acceleration is WORKING")
    else:
        print("\n⚠️  GPU Acceleration is NOT working (using CPU fallback)")

if __name__ == "__main__":
    main()
