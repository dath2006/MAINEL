"""
ONNX Model Export Script

Exports PyTorch models (YOLO, OSNet) to ONNX format for optimized inference
with ONNX Runtime and CUDA Execution Provider.

Usage:
    python scripts/export_models.py --yolo model_weights/yolov8n.pt
    python scripts/export_models.py --osnet osnet_x1_0
    python scripts/export_models.py --all
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_yolo(pt_path: str, output_dir: str = "model_weights"):
    """Export YOLOv8 model to ONNX format."""
    from app.core.detection.yolo_detector import export_yolo_to_onnx
    
    pt_path = Path(pt_path)
    if not pt_path.exists():
        logger.error(f"PyTorch model not found: {pt_path}")
        return None
    
    onnx_path = Path(output_dir) / f"{pt_path.stem}.onnx"
    
    logger.info(f"Exporting YOLO: {pt_path} -> {onnx_path}")
    result = export_yolo_to_onnx(str(pt_path), str(onnx_path))
    
    if os.path.exists(result):
        logger.success(f"✅ YOLO exported successfully: {result}")
        return result
    else:
        logger.error("YOLO export failed")
        return None


def export_osnet(model_name: str = "osnet_x1_0", output_dir: str = "model_weights"):
    """Export OSNet model to ONNX format."""
    from app.core.features.osnet_extractor import export_osnet_to_onnx
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = Path(output_dir) / f"{model_name}.onnx"
    
    logger.info(f"Exporting OSNet ({model_name}) -> {onnx_path}")
    result = export_osnet_to_onnx(str(onnx_path), model_name)
    
    if os.path.exists(result):
        logger.success(f"✅ OSNet exported successfully: {result}")
        return result
    else:
        logger.error("OSNet export failed")
        return None


def verify_onnx(onnx_path: str) -> bool:
    """Verify ONNX model is valid."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"✓ ONNX model verified: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def benchmark_model(model_type: str, model_path: str, iterations: int = 100):
    """Benchmark ONNX model inference speed."""
    import time
    import numpy as np
    
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed")
        return
    
    # Create session with CUDA EP
    providers = [
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]
    
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return
    
    # Get active provider
    active_provider = session.get_providers()[0]
    logger.info(f"Running on: {active_provider}")
    
    # Get input shape
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    input_name = input_info.name
    
    # Handle dynamic dimensions
    if model_type == 'yolo':
        input_shape = [1, 3, 640, 640]
    elif model_type == 'osnet':
        input_shape = [1, 3, 256, 128]
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - start)
    
    avg_time = sum(times) / len(times) * 1000  # ms
    fps = 1000 / avg_time
    
    logger.info(f"📊 {model_type.upper()} Benchmark Results:")
    logger.info(f"   Average latency: {avg_time:.2f} ms")
    logger.info(f"   Throughput: {fps:.1f} FPS")
    logger.info(f"   Min/Max: {min(times)*1000:.2f} / {max(times)*1000:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument("--yolo", type=str, help="Path to YOLOv8 .pt model")
    parser.add_argument("--osnet", type=str, default="osnet_x1_0", 
                        help="OSNet model name (default: osnet_x1_0)")
    parser.add_argument("--output-dir", type=str, default="model_weights",
                        help="Output directory for ONNX models")
    parser.add_argument("--all", action="store_true", 
                        help="Export both YOLO and OSNet with defaults")
    parser.add_argument("--verify", action="store_true",
                        help="Verify exported ONNX models")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark after export")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    exported_models = []
    
    if args.all:
        # Export YOLO with default path
        yolo_pt = "model_weights/yolov8n.pt"
        if os.path.exists(yolo_pt):
            result = export_yolo(yolo_pt, args.output_dir)
            if result:
                exported_models.append(('yolo', result))
        else:
            logger.warning(f"YOLO model not found at {yolo_pt}, skipping")
        
        # Export OSNet
        result = export_osnet(args.osnet, args.output_dir)
        if result:
            exported_models.append(('osnet', result))
    else:
        if args.yolo:
            result = export_yolo(args.yolo, args.output_dir)
            if result:
                exported_models.append(('yolo', result))
        
        if args.osnet or not args.yolo:  # Export OSNet by default if no YOLO specified
            result = export_osnet(args.osnet, args.output_dir)
            if result:
                exported_models.append(('osnet', result))
    
    # Verify if requested
    if args.verify:
        logger.info("\n🔍 Verifying exported models...")
        for model_type, path in exported_models:
            verify_onnx(path)
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("\n⏱️ Running benchmarks...")
        for model_type, path in exported_models:
            benchmark_model(model_type, path)
    
    logger.info("\n✅ Export complete!")
    for model_type, path in exported_models:
        logger.info(f"   {model_type}: {path}")


if __name__ == "__main__":
    main()
