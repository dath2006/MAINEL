"""
Model Export Script - Export YOLO and OSNet to TensorRT/ONNX formats.

Run this script once to generate optimized model files:
    python scripts/export_models.py

The exported models will be saved in the model_weights/ directory.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_yolo_tensorrt(
    model_name: str = "yolov8n.pt",
    output_dir: str = "model_weights",
    half: bool = True,
    imgsz: int = 640,
):
    """
    Export YOLOv8 model to TensorRT engine.
    
    Args:
        model_name: Name of YOLO model (will download if not exists)
        output_dir: Directory to save exported model
        half: Use FP16 precision (faster, minimal accuracy loss)
        imgsz: Input image size
    """
    from ultralytics import YOLO
    
    logger.info(f"Exporting {model_name} to TensorRT engine...")
    
    # Load model
    model = YOLO(model_name)
    
    # Export to TensorRT
    output_path = model.export(
        format="engine",
        half=half,
        device=0,  # GPU device ID
        imgsz=imgsz,
        simplify=True,  # Simplify ONNX graph
        workspace=4,  # GB of GPU memory for TensorRT workspace
    )
    
    logger.info(f"TensorRT engine exported: {output_path}")
    
    # Move to output directory if needed
    output_file = Path(output_path)
    target_path = Path(output_dir) / output_file.name
    
    if output_file != target_path:
        import shutil
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_file), str(target_path))
        logger.info(f"Moved to: {target_path}")
    
    return str(target_path)


def export_yolo_onnx(
    model_name: str = "yolov8n.pt",
    output_dir: str = "model_weights",
    half: bool = True,
    imgsz: int = 640,
):
    """
    Export YOLOv8 model to ONNX format (for ONNX Runtime with TensorRT EP).
    
    Args:
        model_name: Name of YOLO model
        output_dir: Directory to save exported model
        half: Use FP16 precision
        imgsz: Input image size
    """
    from ultralytics import YOLO
    
    logger.info(f"Exporting {model_name} to ONNX format...")
    
    model = YOLO(model_name)
    
    output_path = model.export(
        format="onnx",
        half=half,
        imgsz=imgsz,
        simplify=True,
        opset=17,  # ONNX opset version
    )
    
    logger.info(f"ONNX model exported: {output_path}")
    
    # Move to output directory
    output_file = Path(output_path)
    target_path = Path(output_dir) / output_file.name
    
    if output_file != target_path:
        import shutil
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_file), str(target_path))
        logger.info(f"Moved to: {target_path}")
    
    return str(target_path)


def export_osnet_onnx(
    model_name: str = "osnet_x1_0",
    output_dir: str = "model_weights",
    input_height: int = 256,
    input_width: int = 128,
):
    """
    Export OSNet body ReID model to ONNX format.
    
    Args:
        model_name: OSNet variant name
        output_dir: Directory to save exported model
        input_height: Input image height (standard: 256)
        input_width: Input image width (standard: 128)
    """
    try:
        from boxmot.reid.backbones.osnet import osnet_x1_0, osnet_x0_75, osnet_x0_5, osnet_x0_25
    except ImportError:
        logger.error("BoxMOT not installed. Install with: pip install boxmot")
        return None
    
    logger.info(f"Exporting {model_name} to ONNX format...")
    
    # Select model variant
    model_map = {
        "osnet_x1_0": osnet_x1_0,
        "osnet_x0_75": osnet_x0_75,
        "osnet_x0_5": osnet_x0_5,
        "osnet_x0_25": osnet_x0_25,
    }
    
    if model_name not in model_map:
        logger.error(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
        return None
    
    # Load model
    model = model_map[model_name](num_classes=1000, loss='softmax', pretrained=True)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height, width)
    dummy_input = torch.randn(1, 3, input_height, input_width)
    
    # Output path
    output_path = Path(output_dir) / f"{model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    logger.info(f"OSNet ONNX model exported: {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed!")
    except ImportError:
        logger.warning("onnx package not installed, skipping validation")
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Export models to TensorRT/ONNX formats")
    parser.add_argument("--yolo", action="store_true", help="Export YOLO to TensorRT")
    parser.add_argument("--yolo-onnx", action="store_true", help="Export YOLO to ONNX")
    parser.add_argument("--osnet", action="store_true", help="Export OSNet to ONNX")
    parser.add_argument("--all", action="store_true", help="Export all models")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--osnet-model", default="osnet_x1_0", help="OSNet model variant")
    parser.add_argument("--output-dir", default="model_weights", help="Output directory")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    
    args = parser.parse_args()
    
    # If no specific model selected, export all
    if args.all or not (args.yolo or args.yolo_onnx or args.osnet):
        args.yolo = True
        args.osnet = True
    
    half = not args.fp32
    
    results = {}
    
    if args.yolo:
        try:
            results["yolo_tensorrt"] = export_yolo_tensorrt(
                model_name=args.yolo_model,
                output_dir=args.output_dir,
                half=half,
            )
        except Exception as e:
            logger.error(f"YOLO TensorRT export failed: {e}")
            # Fallback to ONNX
            logger.info("Falling back to ONNX export...")
            args.yolo_onnx = True
    
    if args.yolo_onnx:
        try:
            results["yolo_onnx"] = export_yolo_onnx(
                model_name=args.yolo_model,
                output_dir=args.output_dir,
                half=half,
            )
        except Exception as e:
            logger.error(f"YOLO ONNX export failed: {e}")
    
    if args.osnet:
        try:
            results["osnet_onnx"] = export_osnet_onnx(
                model_name=args.osnet_model,
                output_dir=args.output_dir,
            )
        except Exception as e:
            logger.error(f"OSNet ONNX export failed: {e}")
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("Export Summary:")
    logger.info("="*50)
    for name, path in results.items():
        status = "✅" if path else "❌"
        logger.info(f"{status} {name}: {path or 'FAILED'}")
    
    return results


if __name__ == "__main__":
    main()
