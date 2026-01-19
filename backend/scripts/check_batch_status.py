"""
Check if batch processing is actually running
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
from app.config import settings
from app.services.tracking_service import get_tracking_service

logger.info("Checking batch processing configuration...")

# 1. Check settings
logger.info(f"Settings:")
logger.info(f"  - use_tensorrt: {settings.use_tensorrt}")
logger.info(f"  - tensorrt_batch_size: {settings.tensorrt_batch_size}")
logger.info(f"  - tensorrt_fp16: {settings.tensorrt_fp16}")

# 2. Check if batch processing is available
try:
    from app.workers.batch_processor import BatchFrameAccumulator
    logger.info("✅ BatchFrameAccumulator available")
except ImportError as e:
    logger.error(f"❌ BatchFrameAccumulator NOT available: {e}")

# 3. Check if detector has batch method
try:
    tracking_service = get_tracking_service()
    detector = tracking_service._get_detector()
    
    logger.info(f"Detector type: {type(detector).__name__}")
    
    if hasattr(detector, 'detect_batch'):
        logger.info("✅ Detector has detect_batch method")
    else:
        logger.error("❌ Detector MISSING detect_batch method")
    
    if hasattr(detector, 'preprocess'):
        logger.info("✅ Detector has preprocess method")
    else:
        logger.error("❌ Detector MISSING preprocess method")
        
except Exception as e:
    logger.error(f"❌ Failed to check detector: {e}")
    import traceback
    logger.error(traceback.format_exc())

# 4. Check provider
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    logger.info(f"Available ONNX providers: {providers}")
    
    if any('tensorrt' in p.lower() for p in providers):
        logger.info("✅ TensorRT provider available")
    else:
        logger.warning("❌ TensorRT provider NOT available")
except Exception as e:
    logger.error(f"Failed to check providers: {e}")
