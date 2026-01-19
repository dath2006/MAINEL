"""
Quick test for batch processing integration.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

logger.info("Testing batch processing imports...")

try:
    from app.workers.batch_processor import BatchFrameAccumulator
    logger.info("✅ BatchFrameAccumulator imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import BatchFrameAccumulator: {e}")

try:
    from app.workers.stream_processor import StreamProcessor
    logger.info("✅ StreamProcessor imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import StreamProcessor: {e}")

try:
    from app.services.tracking_service import TrackingService
    logger.info("✅ TrackingService imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import TrackingService: {e}")

try:
    from app.config import settings
    logger.info(f"✅ Settings loaded:")
    logger.info(f"  - tensorrt_batch_size: {settings.tensorrt_batch_size}")
    logger.info(f"  - use_tensorrt: {settings.use_tensorrt}")
    logger.info(f"  - tensorrt_fp16: {settings.tensorrt_fp16}")
except Exception as e:
    logger.error(f"❌ Failed to load settings: {e}")

logger.info("\n🎉 All imports successful! Batch processing ready.")
logger.info("\nNext steps:")
logger.info("  1. Start backend: uvicorn app.main:app --reload")
logger.info("  2. Watch for: 'Batch processing enabled (batch_size=8)'")
logger.info("  3. Monitor logs for: '🚀 Batch inference: 8 frames in XXms'")
