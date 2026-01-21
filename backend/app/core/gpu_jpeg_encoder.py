"""
GPU-Accelerated JPEG Encoder

Uses TorchVision for hardware-accelerated JPEG encoding.
Falls back to CPU (cv2.imencode) if GPU encoding is unavailable or fails.
"""

import threading
from typing import Optional
import numpy as np
import cv2
import torch
import torchvision.io
from loguru import logger

class GPUJpegEncoder:
    """
    Thread-safe GPU JPEG encoder with automatic CPU fallback.
    
    Uses TorchVision for encoding. Accepts OpenCV BGR images, converts to 
    PyTorch tensors (RGB, CHW), and encodes.
    """
    
    def __init__(self, quality: int = 50, use_gpu: bool = True):
        """
        Initialize the encoder.
        
        Args:
            quality: JPEG quality (1-100, higher = better quality, larger file)
            use_gpu: Whether to attempt GPU encoding (will fallback to CPU if unavailable)
        """
        self.quality = quality
        self._use_gpu = use_gpu and torch.cuda.is_available()
        self._lock = threading.Lock()
        
        # Stats for monitoring
        self._encode_count = 0
        self._gpu_encode_count = 0
        self._cpu_fallback_count = 0
        
        if self._use_gpu:
            logger.info(f"GPU JPEG encoder initialized (TorchVision). GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU not available for JPEG encoding - using CPU fallback")

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
        if self._use_gpu:
            try:
                # 1. Convert Numpy (BGR, HWC) to Tensor (CHW) on GPU
                # We can optimize this transfer
                # copy=True to ensure we don't have issues if the numpy array is not writeable or strided oddly
                tensor = torch.from_numpy(frame).to(device='cuda', non_blocking=True)
                
                # 2. Permute HWC -> CHW
                tensor = tensor.permute(2, 0, 1)
                
                # 3. Convert BGR -> RGB (Swap channels 0 and 2)
                tensor = tensor[[2, 1, 0], :, :]
                
                # 4. Encode
                # torchvision.io.encode_jpeg supports CUDA tensors if nvJPEG is linked
                encoded_tensor = torchvision.io.encode_jpeg(tensor, quality=self.quality)
                
                # 5. Move back to CPU and convert to bytes
                jpeg_bytes = encoded_tensor.cpu().numpy().tobytes()
                
                self._gpu_encode_count += 1
                return jpeg_bytes
                
            except Exception as e:
                # GPU encoding failed, fall back to CPU
                # This could happen if nvJPEG is not available in the PyTorch build
                # or if OOM occurs
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
            "gpu_available": self._use_gpu,
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

