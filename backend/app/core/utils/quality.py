"""
Image Quality Utilities

Provides functions to assess image quality for ReID "Garbage Collection".
Filters out blurry, low-resolution, or otherwise noisy frames before they enter the ML pipeline.
"""

import cv2
import numpy as np
from app.config import settings

def is_quality_frame(
    image: np.ndarray,
    min_resolution: int = None,
    min_sharpness: float = None
) -> tuple[bool, float, str]:
    """
    Check if an image crop is of sufficient quality for feature extraction.
    
    Args:
        image: BGR numpy array
        min_resolution: Minimum width/height (defaults to settings.quality_min_size)
        min_sharpness: Minimum Laplacian variance (defaults to settings.quality_min_sharpness)
        
    Returns:
        tuple(is_good, score, reason)
    """
    if image is None or image.size == 0:
        return False, 0.0, "empty"
        
    h, w = image.shape[:2]
    min_res = min_resolution or settings.quality_min_size
    min_sharp = min_sharpness or settings.quality_min_sharpness
    
    # 1. Resolution Check
    if w < min_res or h < min_res:
        return False, 0.0, f"too_small_{w}x{h}"

    # 2. Blur Check (Laplacian Variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < min_sharp:
        return False, blur_score, f"blurry_{blur_score:.1f}"

    return True, blur_score, "ok"
