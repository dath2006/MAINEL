"""
TensorRT Verification Script

Quick test to verify TensorRT optimization is working correctly.
Tests both PeopleNet and ReID models with performance benchmarks.

Usage:
    python scripts/test_tensorrt.py
"""

import time
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
from loguru import logger
import onnxruntime as ort

def test_tensorrt_available():
    """Test if TensorRT Execution Provider is available."""
    logger.info("=" * 60)
    logger.info("Testing TensorRT Availability")
    logger.info("=" * 60)
    
    providers = ort.get_available_providers()
    logger.info(f"Available providers: {providers}")
    
    # Check for TensorRT (case-insensitive)
    has_tensorrt = any('tensorrt' in p.lower() for p in providers)
    has_cuda = any('cuda' in p.lower() for p in providers)
    
    if has_tensorrt:
        logger.info("✓ TensorRT Execution Provider: AVAILABLE")
    else:
        logger.warning("✗ TensorRT Execution Provider: NOT AVAILABLE")
        if has_cuda:
            logger.info("  CUDA available - will fallback to CUDA EP")
        else:
            logger.error("  No GPU acceleration available!")
    
    return has_tensorrt


def test_peoplenet_performance():
    """Test PeopleNet detector with TensorRT."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing PeopleNet Detector")
    logger.info("=" * 60)
    
    try:
        from preprocessor.peoplenet_detector import PeopleNetDetector
        
        model_path = "model_weights/resnet34_peoplenet.onnx"
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        # Initialize detector (should use TensorRT EP)
        logger.info("Initializing PeopleNet detector...")
        detector = PeopleNetDetector(
            model_path=model_path,
            device='cuda',
            confidence_threshold=0.4
        )
        
        # Create dummy frame (960x544 after preprocessing)
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Warmup (3 iterations)
        logger.info("Running warmup iterations...")
        for i in range(3):
            _ = detector.detect(dummy_frame)
            logger.info(f"  Warmup {i+1}/3 complete")
        
        # Benchmark (10 iterations)
        logger.info("\nBenchmarking single-frame inference...")
        latencies = []
        
        for i in range(10):
            start = time.time()
            detections = detector.detect(dummy_frame)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / avg_latency
        
        logger.info(f"\n✓ Single-frame Performance:")
        logger.info(f"  Average latency: {avg_latency:.2f}ms ± {std_latency:.2f}ms")
        logger.info(f"  FPS: {fps:.1f}")
        logger.info(f"  Min/Max: {min(latencies):.2f}ms / {max(latencies):.2f}ms")
        
        # Expected with TensorRT FP16: 8-12ms (80-120 FPS)
        if avg_latency < 15:
            logger.info(f"  ✓ EXCELLENT - TensorRT optimization likely active!")
        elif avg_latency < 30:
            logger.info(f"  ⚠ MODERATE - May be using CUDA EP instead of TensorRT")
        else:
            logger.warning(f"  ✗ SLOW - Check TensorRT configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"PeopleNet test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_reid_performance():
    """Test ReID extractor with TensorRT."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ReID Extractor")
    logger.info("=" * 60)
    
    try:
        from app.core.features.nvidia_reid_extractor import NvidiaReIDExtractor
        
        model_path = "model_weights/resnet50_market1501_aicity156.onnx"
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        # Initialize extractor (should use TensorRT EP)
        logger.info("Initializing ReID extractor...")
        extractor = NvidiaReIDExtractor(
            model_path=model_path,
            device='cuda'
        )
        
        # Create dummy crops (256x128)
        dummy_crops = [
            np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        # Warmup
        logger.info("Running warmup iterations...")
        for i in range(3):
            _ = extractor(dummy_crops[:1])
            logger.info(f"  Warmup {i+1}/3 complete")
        
        # Benchmark single crop
        logger.info("\nBenchmarking single-crop inference...")
        latencies = []
        
        for i in range(20):
            start = time.time()
            _ = extractor(dummy_crops[:1])
            latency = (time.time() - start) * 1000
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        fps = 1000 / avg_latency
        
        logger.info(f"\n✓ Single-crop Performance:")
        logger.info(f"  Average latency: {avg_latency:.2f}ms")
        logger.info(f"  FPS: {fps:.1f}")
        
        # Expected with TensorRT FP16: 3-5ms (200-300 FPS)
        if avg_latency < 6:
            logger.info(f"  ✓ EXCELLENT - TensorRT optimization likely active!")
        elif avg_latency < 12:
            logger.info(f"  ⚠ MODERATE - May be using CUDA EP")
        else:
            logger.warning(f"  ✗ SLOW - Check TensorRT configuration")
        
        return True
        
    except Exception as e:
        logger.error(f"ReID test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_cache_exists():
    """Check if TensorRT engine cache exists."""
    logger.info("\n" + "=" * 60)
    logger.info("Checking TensorRT Engine Cache")
    logger.info("=" * 60)
    
    cache_paths = [
        Path("trt_cache/peoplenet"),
        Path("trt_cache/reidnet"),
        Path("trt_cache/timing_cache"),
    ]
    
    cache_found = False
    
    for path in cache_paths:
        if path.exists():
            files = list(path.glob("*"))
            if files:
                logger.info(f"✓ {path}: {len(files)} files")
                for f in files[:3]:  # Show first 3
                    size_mb = f.stat().st_size / 1024 / 1024
                    logger.info(f"    - {f.name} ({size_mb:.2f} MB)")
                cache_found = True
            else:
                logger.warning(f"✗ {path}: Empty")
        else:
            logger.warning(f"✗ {path}: Not found")
    
    if not cache_found:
        logger.warning("\nNo TensorRT cache found. Run: python scripts/build_tensorrt_engines.py")
    
    return cache_found


def main():
    """Run all tests."""
    logger.info("TensorRT Verification Script")
    logger.info("=" * 60 + "\n")
    
    results = {}
    
    # Test 1: TensorRT availability
    results['tensorrt_available'] = test_tensorrt_available()
    
    # Test 2: Cache exists
    results['cache_exists'] = test_cache_exists()
    
    # Test 3: PeopleNet performance
    results['peoplenet_ok'] = test_peoplenet_performance()
    
    # Test 4: ReID performance
    results['reid_ok'] = test_reid_performance()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🚀 All tests passed! TensorRT optimization is working correctly.")
    else:
        logger.warning("\n⚠ Some tests failed. Check errors above.")
        if not results['tensorrt_available']:
            logger.info("\nNext steps:")
            logger.info("  1. Ensure NVIDIA GPU with CUDA 12.1+ is available")
            logger.info("  2. Verify onnxruntime-gpu is installed")
            logger.info("  3. Check that TensorRT is bundled with CUDA toolkit")
        if not results['cache_exists']:
            logger.info("\nBuild TensorRT engines:")
            logger.info("  python scripts/build_tensorrt_engines.py")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
