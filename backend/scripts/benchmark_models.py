"""
Benchmark Script - Compare inference speed of PyTorch vs TensorRT/ONNX models.

Run with: python scripts/benchmark_models.py
"""

import time
import numpy as np
import sys
from pathlib import Path
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def benchmark_yolo(iterations: int = 100):
    """Benchmark YOLO detection with different backends."""
    from app.core.detection.yolo_detector import YOLODetector
    
    logger.info("="*60)
    logger.info("YOLO BENCHMARK")
    logger.info("="*60)
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Test with auto-detection (uses best available)
    logger.info("\nLoading detector (auto-detect best model)...")
    detector = YOLODetector()
    
    # Warmup
    for _ in range(10):
        detector.detect(frame)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        detector.detect(frame)
    elapsed = time.perf_counter() - start
    
    fps = iterations / elapsed
    ms_per_frame = (elapsed / iterations) * 1000
    
    logger.info(f"\nResults ({detector.model_type.upper()}):")
    logger.info(f"  Average: {ms_per_frame:.2f} ms/frame")
    logger.info(f"  FPS: {fps:.1f}")
    
    return {"model_type": detector.model_type, "fps": fps, "ms": ms_per_frame}


def benchmark_osnet(iterations: int = 100):
    """Benchmark OSNet feature extraction with different backends."""
    logger.info("\n" + "="*60)
    logger.info("OSNET BENCHMARK")
    logger.info("="*60)
    
    # Create dummy crops (batch of person images)
    crops = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(8)]
    
    results = {}
    
    # Test ONNX if available
    try:
        from app.core.features.osnet_onnx import get_onnx_osnet_extractor
        
        onnx_extractor = get_onnx_osnet_extractor(
            model_path="model_weights/osnet_x1_0.onnx",
            device="cuda",
            use_tensorrt=False,  # Disabled - TensorRT DLLs not in PATH, use CUDA EP
        )
        
        if onnx_extractor:
            logger.info("\nTesting ONNX Runtime...")
            
            # Warmup
            for _ in range(10):
                onnx_extractor.extract_batch(crops)
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                onnx_extractor.extract_batch(crops)
            elapsed = time.perf_counter() - start
            
            fps = iterations / elapsed
            ms_per_batch = (elapsed / iterations) * 1000
            
            logger.info(f"\nResults (ONNX Runtime):")
            logger.info(f"  Batch size: {len(crops)}")
            logger.info(f"  Average: {ms_per_batch:.2f} ms/batch")
            logger.info(f"  Batches/sec: {fps:.1f}")
            
            results["onnx"] = {"fps": fps, "ms": ms_per_batch}
        else:
            logger.info("ONNX model not found. Export with: python scripts/export_models.py --osnet")
    except ImportError:
        logger.warning("ONNX OSNet extractor not available")
    
    # Test PyTorch
    try:
        import torch
        from boxmot.reid.backbones.osnet import osnet_x1_0
        
        logger.info("\nTesting PyTorch...")
        
        model = osnet_x1_0(num_classes=1000, loss='softmax', pretrained=True)
        model.eval()
        model.cuda()
        
        # Prepare batch
        batch = []
        for crop in crops:
            import cv2
            img = cv2.resize(crop, (128, 256))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / \
                  np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            batch.append(img)
        batch = torch.tensor(np.stack(batch)).float().cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(batch)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        fps = iterations / elapsed
        ms_per_batch = (elapsed / iterations) * 1000
        
        logger.info(f"\nResults (PyTorch):")
        logger.info(f"  Batch size: {len(crops)}")
        logger.info(f"  Average: {ms_per_batch:.2f} ms/batch")
        logger.info(f"  Batches/sec: {fps:.1f}")
        
        results["pytorch"] = {"fps": fps, "ms": ms_per_batch}
    except Exception as e:
        logger.error(f"PyTorch benchmark failed: {e}")
    
    # Compare
    if "onnx" in results and "pytorch" in results:
        speedup = results["pytorch"]["ms"] / results["onnx"]["ms"]
        logger.info(f"\n🚀 ONNX Speedup: {speedup:.2f}x faster than PyTorch")
    
    return results


def main():
    logger.info("="*60)
    logger.info("MODEL BENCHMARK SUITE")
    logger.info("="*60)
    
    iterations = 50  # Reduce for faster testing
    
    try:
        yolo_results = benchmark_yolo(iterations)
    except Exception as e:
        logger.error(f"YOLO benchmark failed: {e}")
        yolo_results = None
    
    try:
        osnet_results = benchmark_osnet(iterations)
    except Exception as e:
        logger.error(f"OSNet benchmark failed: {e}")
        osnet_results = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    if yolo_results:
        logger.info(f"YOLO ({yolo_results['model_type']}): {yolo_results['fps']:.1f} FPS")
    
    if osnet_results:
        for backend, data in osnet_results.items():
            logger.info(f"OSNet ({backend}): {data['fps']:.1f} batches/sec")


if __name__ == "__main__":
    main()
