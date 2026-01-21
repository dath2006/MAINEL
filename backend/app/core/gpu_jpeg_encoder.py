"""
GPU-Accelerated JPEG Encoder

Uses NVIDIA nvJPEG for hardware-accelerated JPEG encoding via pynvjpeg.
Falls back to CPU (cv2.imencode) if GPU encoding is unavailable.
"""

import threading
from typing import Optional
import numpy as np
import cv2
from loguru import logger

# Try to import pynvjpeg for GPU acceleration
try:
    from nvjpeg import NvJpeg
    HAS_NVJPEG = True
except ImportError:
    HAS_NVJPEG = False
    logger.warning("pynvjpeg not available - using CPU JPEG encoding")


class GPUJpegEncoder:
    """
    Thread-safe GPU JPEG encoder with automatic CPU fallback.
    
    Uses NVIDIA nvJPEG for hardware-accelerated encoding when available,
    otherwise falls back to cv2.imencode.
    """
    
    def __init__(self, quality: int = 50, use_gpu: bool = True):
        """
        Initialize the encoder.
        
        Args:
            quality: JPEG quality (1-100, higher = better quality, larger file)
            use_gpu: Whether to attempt GPU encoding (will fallback to CPU if unavailable)
        """
        self.quality = quality
        self._use_gpu = use_gpu and HAS_NVJPEG
        self._encoder: Optional[NvJpeg] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._gpu_available = False
        
        # Stats for monitoring
        self._encode_count = 0
        self._gpu_encode_count = 0
        self._cpu_fallback_count = 0
    
    def _ensure_initialized(self) -> bool:
        """Lazily initialize the GPU encoder."""
        if self._initialized:
            return self._gpu_available
        
        with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return self._gpu_available
            
            if self._use_gpu:
                try:
                    self._encoder = NvJpeg()
                    self._gpu_available = True
                    logger.info("GPU JPEG encoder initialized (nvJPEG)")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU JPEG encoder: {e}")
                    self._gpu_available = False
            
            self._initialized = True
            return self._gpu_available
    
    def encode(self, frame: np.ndarray) -> bytes:
        """
        Encode a frame to JPEG bytes.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            
        Returns:
            JPEG encoded bytes
        """
        self._encode_count += 1
        
        # Try GPU encoding first
        if self._ensure_initialized() and self._encoder is not None:
            try:
                # nvJPEG expects BGR format (same as OpenCV)
                jpeg_bytes = self._encoder.encode(frame, self.quality)
                self._gpu_encode_count += 1
                return jpeg_bytes
            except Exception as e:
                # GPU encoding failed, fall back to CPU
                logger.debug(f"GPU JPEG encode failed, using CPU fallback: {e}")
                self._cpu_fallback_count += 1
        
        # CPU fallback using cv2.imencode
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            raise RuntimeError("Failed to encode frame to JPEG")
        
        return buffer.tobytes()
    
    def get_stats(self) -> dict:
        """Get encoding statistics."""
        return {
            "total_encodes": self._encode_count,
            "gpu_encodes": self._gpu_encode_count,
            "cpu_fallbacks": self._cpu_fallback_count,
            "gpu_available": self._gpu_available,
            "gpu_usage_pct": (
                self._gpu_encode_count / self._encode_count * 100
                if self._encode_count > 0 else 0
            ),
        }


# Singleton instance
_encoder: Optional[GPUJpegEncoder] = None
_encoder_lock = threading.Lock()


def get_gpu_encoder(quality: int = 50) -> GPUJpegEncoder:
    """
    Get or create the singleton GPU JPEG encoder.
    
    Args:
        quality: JPEG quality (only used on first call)
        
    Returns:
        GPUJpegEncoder instance
    """
    global _encoder
    
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                _encoder = GPUJpegEncoder(quality=quality)
    
    return _encoder


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 50) -> bytes:
    """
    Convenience function to encode a frame to JPEG bytes.
    
    Uses GPU acceleration when available.
    
    Args:
        frame: BGR image as numpy array
        quality: JPEG quality (1-100)
        
    Returns:
        JPEG encoded bytes
    """
    encoder = get_gpu_encoder(quality)
    return encoder.encode(frame)
