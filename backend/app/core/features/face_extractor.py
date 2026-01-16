"""
Quality Scorer for Thumbnail Selection

Simple quality scoring for selecting the best thumbnail from person crops.
InsightFace has been removed - using ReID for body-only features.
"""

from typing import Optional, Tuple
import os
import numpy as np
import cv2
from loguru import logger
from app.core.features.pose_estimator import PoseEstimator


class QualityScorer:
    """
    Scores image quality for ReID gallery selection using MediaPipe 3D Landmarks.
    
    Formula (ISO-Aligned): S = w1*Pose + w2*Sharpness + w3*Size + w4*Illum
    Weights:
        - Pose: 0.40 (3D Yaw/Pitch - Crucial)
        - Sharpness: 0.30 (Tenengrad)
        - Size: 0.20 (Resolution)
        - Illum: 0.10 (Brightness)
    """
    
    def __init__(self):
        # Weights
        self.w_pose = 0.40
        self.w_sharp = 0.30
        self.w_size = 0.20
        self.w_illum = 0.10
        
        # Initialize MediaPipe Pose Estimator
        try:
            self.pose_estimator = PoseEstimator()
            logger.info("Initialized MediaPipe PoseEstimator for Quality Scoring")
        except Exception as e:
            logger.error(f"Failed to initialize PoseEstimator: {e}")
            self.pose_estimator = None

    def score(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_confidence: float = 1.0,
    ) -> float:
        """
        Calculate quality score (0.0 to 1.0).
        """
        if image is None or image.size == 0:
            return 0.0
            
        if self.pose_estimator is None:
            # Fallback if MP fails (simple size/blur)
            return 0.5

        # 1. Pose Score (3D Yaw/Pitch)
        q_pose, pose_details = self.pose_estimator.get_pose_score(image)

        # 2. Sharpness Score (Tenengrad)
        q_sharp = self.pose_estimator.get_sharpness_score(image)

        # 3. Size Score
        area = image.shape[0] * image.shape[1]
        q_size = min(1.0, area / (128 * 256))
        
        # 4. Illumination Score
        q_illum = self.pose_estimator.get_illumination_score(image)

        # Weighted Sum
        final_score = (
            self.w_pose * q_pose +
            self.w_sharp * q_sharp +
            self.w_size * q_size +
            self.w_illum * q_illum
        )
        final_score = min(max(final_score, 0.0), 1.0)
        
        # Debug Log breakdown
        if final_score > 0.4:
            yaw = pose_details.get("yaw", 0)
            logger.debug(
                f"Quality: {final_score:.3f} "
                f"[Pose={q_pose:.2f} (Yaw={yaw:.1f}) Sharp={q_sharp:.2f} Size={q_size:.2f} Illum={q_illum:.2f}]"
            )

        return final_score


# Quality scorer singleton
_quality_scorer: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    """Get or create singleton quality scorer."""
    global _quality_scorer
    
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    
    return _quality_scorer
