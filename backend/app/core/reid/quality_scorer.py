"""
ReID Feature Quality Scorer

Assesses the quality of person crops for re-identification feature extraction.
Filters out low-quality features (blurry, occluded, poorly-lit) from the feature bank.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from loguru import logger


class QualityScorer:
    """
    Assess quality of person crops for ReID feature extraction.
    
    Evaluates multiple quality dimensions:
    - Blur: Sharpness using Laplacian variance
    - Occlusion: Overlap with other detections
    - Illumination: Lighting quality via histogram analysis
    - Confidence: Detection confidence score
    """
    
    def __init__(
        self,
        blur_weight: float = 0.3,
        occlusion_weight: float = 0.4,
        illumination_weight: float = 0.2,
        confidence_weight: float = 0.1,
        min_blur_variance: float = 50.0,
        max_blur_variance: float = 100.0,
    ):
        """
        Initialize quality scorer.
        
        Args:
            blur_weight: Weight for blur score in final quality
            occlusion_weight: Weight for occlusion score
            illumination_weight: Weight for illumination score
            confidence_weight: Weight for detection confidence
            min_blur_variance: Laplacian variance threshold for blurry images
            max_blur_variance: Laplacian variance threshold for very sharp images
        """
        self.blur_weight = blur_weight
        self.occlusion_weight = occlusion_weight
        self.illumination_weight = illumination_weight
        self.confidence_weight = confidence_weight
        self.min_blur_variance = min_blur_variance
        self.max_blur_variance = max_blur_variance
        
        # Normalize weights
        total = blur_weight + occlusion_weight + illumination_weight + confidence_weight
        self.blur_weight /= total
        self.occlusion_weight /= total
        self.illumination_weight /= total
        self.confidence_weight /= total
    
    def assess_blur(self, crop: np.ndarray) -> float:
        """
        Assess image sharpness using Laplacian variance.
        
        Args:
            crop: BGR image crop (H, W, 3)
            
        Returns:
            Blur quality score (0.0-1.0), where 1.0 is sharp
        """
        if crop is None or crop.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range
        # variance < min_blur_variance -> blurry (0.0)
        # variance > max_blur_variance -> very sharp (1.0)
        score = (variance - self.min_blur_variance) / (
            self.max_blur_variance - self.min_blur_variance
        )
        
        return np.clip(score, 0.0, 1.0)
    
    def assess_occlusion(
        self,
        bbox: np.ndarray,
        all_bboxes: List[np.ndarray],
    ) -> float:
        """
        Assess occlusion level based on overlap with other detections.
        
        Args:
            bbox: Target bounding box [x1, y1, x2, y2]
            all_bboxes: All detected bounding boxes in the frame
            
        Returns:
            Occlusion quality score (0.0-1.0), where 1.0 is no occlusion
        """
        if len(all_bboxes) <= 1:
            # Only one detection, no occlusion possible
            return 1.0
        
        max_overlap = 0.0
        
        for other_bbox in all_bboxes:
            # Skip self-comparison
            if np.array_equal(bbox, other_bbox):
                continue
            
            iou = self._compute_iou(bbox, other_bbox)
            max_overlap = max(max_overlap, iou)
        
        # Convert overlap to quality score
        # 0% overlap -> 1.0 quality
        # 100% overlap -> 0.0 quality
        return 1.0 - max_overlap
    
    def assess_illumination(self, crop: np.ndarray) -> float:
        """
        Assess illumination quality using histogram analysis.
        
        Checks for:
        - Underexposure (too dark)
        - Overexposure (too bright)
        - Low contrast (washed out)
        
        Args:
            crop: BGR image crop (H, W, 3)
            
        Returns:
            Illumination quality score (0.0-1.0), where 1.0 is good lighting
        """
        if crop is None or crop.size == 0:
            return 0.0
        
        # Convert to grayscale for histogram analysis
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        # Compute histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Calculate mean intensity
        mean_intensity = np.average(np.arange(256), weights=hist)
        
        # Calculate contrast (standard deviation)
        std_intensity = np.sqrt(np.average((np.arange(256) - mean_intensity)**2, weights=hist))
        
        # Score based on mean intensity (penalize too dark/bright)
        # Ideal range: 60-200 (on 0-255 scale)
        if mean_intensity < 60:
            # Too dark (underexposed)
            intensity_score = mean_intensity / 60.0
        elif mean_intensity > 200:
            # Too bright (overexposed)
            intensity_score = (255 - mean_intensity) / 55.0
        else:
            # Good range
            intensity_score = 1.0
        
        # Score based on contrast
        # Low contrast (<20) is bad, good contrast (>40) is ideal
        contrast_score = min(std_intensity / 40.0, 1.0)
        
        # Combined illumination score (weighted average)
        illumination_score = 0.6 * intensity_score + 0.4 * contrast_score
        
        return np.clip(illumination_score, 0.0, 1.0)
    
    def compute_quality_score(
        self,
        crop: np.ndarray,
        bbox: np.ndarray,
        all_bboxes: List[np.ndarray],
        confidence: float,
    ) -> float:
        """
        Compute overall quality score combining all metrics.
        
        Args:
            crop: BGR image crop (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            all_bboxes: All bounding boxes in the frame
            confidence: Detection confidence score (0.0-1.0)
            
        Returns:
            Overall quality score (0.0-1.0)
        """
        # Compute individual scores
        blur_score = self.assess_blur(crop)
        occlusion_score = self.assess_occlusion(bbox, all_bboxes)
        illumination_score = self.assess_illumination(crop)
        
        # Weighted combination
        quality = (
            self.blur_weight * blur_score +
            self.occlusion_weight * occlusion_score +
            self.illumination_weight * illumination_score +
            self.confidence_weight * confidence
        )
        
        return np.clip(quality, 0.0, 1.0)
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bbox [x1, y1, x2, y2]
            bbox2: Second bbox [x1, y1, x2, y2]
            
        Returns:
            IoU value (0.0-1.0)
        """
        # Intersection coordinates
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Intersection area
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union area
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        # IoU
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def get_quality_breakdown(
        self,
        crop: np.ndarray,
        bbox: np.ndarray,
        all_bboxes: List[np.ndarray],
        confidence: float,
    ) -> dict:
        """
        Get detailed quality breakdown for debugging/logging.
        
        Returns:
            Dict with individual scores and overall quality
        """
        blur_score = self.assess_blur(crop)
        occlusion_score = self.assess_occlusion(bbox, all_bboxes)
        illumination_score = self.assess_illumination(crop)
        overall = self.compute_quality_score(crop, bbox, all_bboxes, confidence)
        
        return {
            'blur': blur_score,
            'occlusion': occlusion_score,
            'illumination': illumination_score,
            'confidence': confidence,
            'overall': overall,
        }
