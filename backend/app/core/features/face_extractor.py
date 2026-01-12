"""
Quality Scorer for Thumbnail Selection

Simple quality scoring for selecting the best thumbnail from person crops.
InsightFace has been removed - using FastReID for body-only ReID.
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from loguru import logger


class QualityScorer:
    """
    Image quality scorer for thumbnail selection.
    
    Scores images based on sharpness and size.
    Face detection has been removed.
    """
    
    def __init__(
        self,
        sharpness_weight: float = 0.5,
        size_weight: float = 0.5,
    ):
        self.sharpness_weight = sharpness_weight
        self.size_weight = size_weight
    
    def score(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_confidence: float = 0.0,
    ) -> float:
        """
        Compute overall quality score.
        
        Args:
            image: Person crop as BGR numpy array
            face_bbox: Unused (kept for backward compatibility)
            face_confidence: Unused (kept for backward compatibility)
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        h, w = image.shape[:2]
        
        # 1. Sharpness score (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: 100 is decent, 500+ is very sharp
        sharpness = min(1.0, laplacian_var / 200.0)
        
        # 2. Image size score (larger is better for thumbnails)
        # Normalize: 128x256 is baseline, larger is better up to 256x512
        size_score = min(1.0, (h * w) / (256 * 512))
        
        # Combine scores (no face component now)
        total = (
            self.sharpness_weight * sharpness +
            self.size_weight * size_score
        )
        
        return min(1.0, max(0.0, total))


# Quality scorer singleton
_quality_scorer: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    """Get or create singleton quality scorer."""
    global _quality_scorer
    
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    
    return _quality_scorer
