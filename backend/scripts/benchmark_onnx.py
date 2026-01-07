"""
ONNX vs PyTorch Benchmark Script

Compares inference performance between ONNX Runtime (with CUDA EP)
and PyTorch backends for YOLO detection and OSNet feature extraction.

Usage:
    python scripts/benchmark_onnx.py
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_test_frames(count: int = 100, size: tuple = (1920, 1080)):
    """Create list of random test frames."""
    frames = []
    for i in range(count):
        # Create a frame with random noise + some structure
        frame = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        
        # Add some rectangles to simulate person-like regions
        for _ in range(np.random.randint(1, 5)):
            x = np.random.randint(0, size[0] - 100)
            y = np.random.randint(0, size[1] - 200)
            w = np.random.randint(50, 100)
            h = np.random.randint(100, 200)
            color = tuple(np.random.randint(100, 200, 3).tolist())
            frame[y:y+h, x:x+w] = color
        
        frames.append(frame)
    return frames


def create_test_crops(count: int = 100):
    """Create list of random person crops."""
    crops = []
    for _ in range(count):
        # Random person-like crop (height > width)
        h = np.random.randint(200, 400)
        w = int(h * 0.4)  # Person aspect ratio
        crop = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


def benchmark_yolo_pytorch(frames: list, warmup: int = 10, iterations: int = 100):
    """Benchmark PyTorch YOLO detector."""
    try:
        # Force PyTorch backend
        from app.core.detection.yolo_detector import YOLODetector
        
        model_path = "model_weights/yolov8n.pt"
        if not os.path.exists(model_path):
            logger.warning(f"PyTorch YOLO model not found: {model_path}")
            return None
        
        detector = YOLODetector(model_path=model_path)
        
        # Warmup
        for i in range(warmup):
            _ = detector.detect(frames[i % len(frames)])
        
        # Benchmark
        times = []
        for i in range(iterations):
            frame = frames[i % len(frames)]
            start = time.perf_counter()
            _ = detector.detect(frame)
            times.append(time.perf_counter() - start)
        
        return times
    except Exception as e:
        logger.error(f"PyTorch YOLO benchmark failed: {e}")
        return None


def benchmark_yolo_onnx(frames: list, warmup: int = 10, iterations: int = 100):
    """Benchmark ONNX YOLO detector."""
    try:
        from app.core.detection.yolo_detector import YOLOOnnxDetector
        
        model_path = "model_weights/yolov8n.onnx"
        if not os.path.exists(model_path):
            logger.warning(f"ONNX YOLO model not found: {model_path}")
            return None
        
        detector = YOLOOnnxDetector(model_path=model_path)
        
        # Warmup
        for i in range(warmup):
            _ = detector.detect(frames[i % len(frames)])
        
        # Benchmark
        times = []
        for i in range(iterations):
            frame = frames[i % len(frames)]
            start = time.perf_counter()
            _ = detector.detect(frame)
            times.append(time.perf_counter() - start)
        
        return times
    except Exception as e:
        logger.error(f"ONNX YOLO benchmark failed: {e}")
        return None


def benchmark_osnet_pytorch(crops: list, warmup: int = 10, iterations: int = 100):
    """Benchmark PyTorch OSNet extractor."""
    try:
        from app.core.features.osnet_extractor import OSNetExtractor
        
        extractor = OSNetExtractor()
        
        # Warmup
        for i in range(warmup):
            _ = extractor.extract(crops[i % len(crops)])
        
        # Benchmark
        times = []
        for i in range(iterations):
            crop = crops[i % len(crops)]
            start = time.perf_counter()
            _ = extractor.extract(crop)
            times.append(time.perf_counter() - start)
        
        return times
    except Exception as e:
        logger.error(f"PyTorch OSNet benchmark failed: {e}")
        return None


def benchmark_osnet_onnx(crops: list, warmup: int = 10, iterations: int = 100):
    """Benchmark ONNX OSNet extractor."""
    try:
        from app.core.features.osnet_extractor import OSNetOnnxExtractor
        
        model_path = "model_weights/osnet_x1_0.onnx"
        if not os.path.exists(model_path):
            logger.warning(f"ONNX OSNet model not found: {model_path}")
            return None
        
        extractor = OSNetOnnxExtractor(model_path=model_path)
        
        # Warmup
        for i in range(warmup):
            _ = extractor.extract(crops[i % len(crops)])
        
        # Benchmark
        times = []
        for i in range(iterations):
            crop = crops[i % len(crops)]
            start = time.perf_counter()
            _ = extractor.extract(crop)
            times.append(time.perf_counter() - start)
        
        return times
    except Exception as e:
        logger.error(f"ONNX OSNet benchmark failed: {e}")
        return None


def print_results(name: str, times: list):
    """Print benchmark results."""
    if times is None:
        logger.warning(f"{name}: N/A (model not available)")
        return
    
    avg = sum(times) / len(times) * 1000
    fps = 1000 / avg
    min_t = min(times) * 1000
    max_t = max(times) * 1000
    std = np.std(times) * 1000
    
    print(f"  {name}:")
    print(f"    Average: {avg:.2f} ms ({fps:.1f} FPS)")
    print(f"    Min/Max: {min_t:.2f} / {max_t:.2f} ms")
    print(f"    Std Dev: {std:.2f} ms")


def check_gpu_memory():
    """Check GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n📊 GPU Memory:")
            print(f"   Allocated: {allocated:.2f} GB")
            print(f"   Reserved:  {reserved:.2f} GB")
    except:
        pass
    
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"   Total:     {info.total / 1024**3:.2f} GB")
        print(f"   Used:      {info.used / 1024**3:.2f} GB")
        print(f"   Free:      {info.free / 1024**3:.2f} GB")
        pynvml.nvmlShutdown()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX vs PyTorch inference")
    parser.add_argument("--frames", type=int, default=100, help="Number of test frames")
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ONNX vs PyTorch Inference Benchmark")
    print("=" * 60)
    
    # Create test data
    print(f"\n🔧 Creating {args.frames} test frames and crops...")
    frames = create_test_frames(args.frames)
    crops = create_test_crops(args.frames)
    
    # YOLO Benchmarks
    print("\n📦 YOLO Detection Benchmarks:")
    print("-" * 40)
    
    yolo_pt_times = benchmark_yolo_pytorch(frames, args.warmup, args.iterations)
    print_results("PyTorch", yolo_pt_times)
    
    yolo_onnx_times = benchmark_yolo_onnx(frames, args.warmup, args.iterations)
    print_results("ONNX Runtime", yolo_onnx_times)
    
    if yolo_pt_times and yolo_onnx_times:
        speedup = (sum(yolo_pt_times) / sum(yolo_onnx_times))
        print(f"\n  ⚡ ONNX Speedup: {speedup:.2f}x")
    
    # OSNet Benchmarks
    print("\n👤 OSNet Feature Extraction Benchmarks:")
    print("-" * 40)
    
    osnet_pt_times = benchmark_osnet_pytorch(crops, args.warmup, args.iterations)
    print_results("PyTorch", osnet_pt_times)
    
    osnet_onnx_times = benchmark_osnet_onnx(crops, args.warmup, args.iterations)
    print_results("ONNX Runtime", osnet_onnx_times)
    
    if osnet_pt_times and osnet_onnx_times:
        speedup = (sum(osnet_pt_times) / sum(osnet_onnx_times))
        print(f"\n  ⚡ ONNX Speedup: {speedup:.2f}x")
    
    # GPU Memory
    check_gpu_memory()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
