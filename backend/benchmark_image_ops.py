"""
Benchmark Script for Image Processing Operations

This script measures the performance of:
1. cv2.imencode (JPEG encoding)
2. cv2.resize (image resizing)
3. base64 encoding
4. Full broadcast pipeline simulation

These are the suspected CPU bottlenecks in _broadcast_frame.
"""

import time
import base64
import statistics
import numpy as np
import cv2

# Try to import GPU acceleration libraries
try:
    import nvjpeg
    HAS_NVJPEG = True
except ImportError:
    HAS_NVJPEG = False

HAS_CUDA = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0


def create_test_frame(width=1920, height=1080):
    """Create a realistic test frame with random content."""
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add some structure (simulates real video content)
    cv2.rectangle(frame, (100, 100), (400, 500), (0, 255, 0), 2)
    cv2.putText(frame, "Test Frame", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return frame


def benchmark_imencode(frame, iterations=100, quality=50):
    """Benchmark cv2.imencode for JPEG encoding."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "operation": f"cv2.imencode (JPEG Q={quality})",
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "buffer_size_kb": len(buffer) / 1024,
    }


def benchmark_resize(frame, target_size=(64, 128), iterations=100):
    """Benchmark cv2.resize for thumbnail generation."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = cv2.resize(frame, target_size)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "operation": f"cv2.resize to {target_size}",
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_base64(buffer, iterations=100):
    """Benchmark base64 encoding."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = base64.b64encode(buffer).decode('utf-8')
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "operation": "base64 encode + decode",
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_full_broadcast_pipeline(frame, iterations=100, quality=50):
    """Benchmark the full broadcast pipeline as in _broadcast_frame."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 1. Copy frame (as in original code)
        frame_copy = frame.copy()
        
        # 2. Draw some overlays (simulating track boxes)
        cv2.rectangle(frame_copy, (100, 100), (300, 500), (0, 255, 0), 2)
        cv2.rectangle(frame_copy, (400, 200), (600, 600), (0, 255, 255), 2)
        cv2.putText(frame_copy, "ID:1 Q:75", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 3. JPEG encode
        _, buffer = cv2.imencode('.jpg', frame_copy, encode_params)
        
        # 4. Base64 encode
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "operation": "Full broadcast pipeline (copy+draw+encode+b64)",
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "base64_size_kb": len(frame_b64) / 1024,
    }


def benchmark_cuda_resize(frame, target_size=(64, 128), iterations=100):
    """Benchmark GPU-accelerated resize using cv2.cuda."""
    if not HAS_CUDA:
        return None
    
    # Upload frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = cv2.cuda.resize(gpu_frame, target_size)
        # Force synchronization
        _ = result.download()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        "operation": f"cv2.cuda.resize to {target_size}",
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_cuda_full_upload_download(frame, iterations=100):
    """Benchmark GPU upload/download time (important for hybrid pipeline)."""
    if not HAS_CUDA:
        return None
    
    times_upload = []
    times_download = []
    
    gpu_frame = cv2.cuda_GpuMat()
    
    for _ in range(iterations):
        start = time.perf_counter()
        gpu_frame.upload(frame)
        end = time.perf_counter()
        times_upload.append((end - start) * 1000)
        
        start = time.perf_counter()
        _ = gpu_frame.download()
        end = time.perf_counter()
        times_download.append((end - start) * 1000)
    
    return {
        "operation": "GPU upload + download round-trip",
        "upload_mean_ms": statistics.mean(times_upload),
        "download_mean_ms": statistics.mean(times_download),
        "total_mean_ms": statistics.mean(times_upload) + statistics.mean(times_download),
    }


def print_result(result):
    """Pretty print benchmark result."""
    if result is None:
        return
    print(f"\n{'='*60}")
    print(f"Operation: {result['operation']}")
    print(f"{'='*60}")
    
    for key, value in result.items():
        if key != "operation":
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")


def calculate_max_fps(mean_ms):
    """Calculate theoretical max FPS from mean latency."""
    if mean_ms > 0:
        return 1000 / mean_ms
    return float('inf')


def main():
    print("="*60)
    print("Image Processing Bottleneck Benchmark")
    print("="*60)
    print(f"\nSystem Info:")
    print(f"  OpenCV version: {cv2.__version__}")
    print(f"  CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"  CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
        cv2.cuda.printShortCudaDeviceInfo(0)
    print(f"  nvJPEG available: {HAS_NVJPEG}")
    
    # Create test frame
    print("\n\nCreating test frame (1920x1080)...")
    frame = create_test_frame()
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")
    print(f"  Frame size: {frame.nbytes / 1024 / 1024:.2f} MB")
    
    iterations = 100
    print(f"\n\nRunning benchmarks ({iterations} iterations each)...")
    
    # Run benchmarks
    results = []
    
    # CPU benchmarks
    results.append(benchmark_imencode(frame, iterations, quality=50))
    results.append(benchmark_imencode(frame, iterations, quality=30))
    results.append(benchmark_resize(frame, (64, 128), iterations))
    results.append(benchmark_resize(frame, (640, 480), iterations))
    
    # Get JPEG buffer for base64 test
    _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    results.append(benchmark_base64(jpeg_buffer, iterations))
    
    # Full pipeline
    results.append(benchmark_full_broadcast_pipeline(frame, iterations, quality=50))
    
    # GPU benchmarks
    if HAS_CUDA:
        results.append(benchmark_cuda_resize(frame, (64, 128), iterations))
        results.append(benchmark_cuda_resize(frame, (640, 480), iterations))
        results.append(benchmark_cuda_full_upload_download(frame, iterations))
    
    # Print results
    for result in results:
        print_result(result)
    
    # Calculate FPS implications
    print("\n\n" + "="*60)
    print("FPS IMPLICATIONS")
    print("="*60)
    
    pipeline_result = [r for r in results if "Full broadcast" in r.get("operation", "")][0]
    pipeline_ms = pipeline_result["mean_ms"]
    
    print(f"\nFull broadcast pipeline: {pipeline_ms:.2f}ms")
    print(f"  Theoretical max FPS (encoding only): {calculate_max_fps(pipeline_ms):.1f}")
    print(f"  At 30 FPS, encoding overhead: {pipeline_ms * 30 / 1000 * 100:.1f}% of CPU time")
    print(f"  At 60 FPS, encoding overhead: {pipeline_ms * 60 / 1000 * 100:.1f}% of CPU time")
    
    # Bottleneck breakdown
    imencode_ms = [r for r in results if "imencode" in r.get("operation", "") and "Q=50" in r.get("operation", "")][0]["mean_ms"]
    base64_ms = [r for r in results if "base64" in r.get("operation", "")][0]["mean_ms"]
    
    print(f"\n  Breakdown:")
    print(f"    cv2.imencode: {imencode_ms:.2f}ms ({imencode_ms/pipeline_ms*100:.1f}%)")
    print(f"    base64 encode: {base64_ms:.2f}ms ({base64_ms/pipeline_ms*100:.1f}%)")
    print(f"    Other (copy+draw): {pipeline_ms - imencode_ms - base64_ms:.2f}ms")
    
    print("\n\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if pipeline_ms > 20:
        print("\n⚠️  WARNING: Broadcasting is CPU-bound and may cause frame drops at 30+ FPS")
        print("\nSuggested optimizations:")
        print("  1. Use nvJPEG for GPU-accelerated JPEG encoding")
        print("  2. Reduce JPEG quality for faster encoding")
        print("  3. Use cv2.cuda.resize for thumbnail generation")
        print("  4. Consider WebP format (often smaller than JPEG)")
        print("  5. Implement frame skipping for broadcast (send every 2nd frame)")
    else:
        print("\n✅ Broadcasting overhead is acceptable for real-time streaming")


if __name__ == "__main__":
    main()
