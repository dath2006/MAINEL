"""
Quality Scorer for Person Crops.

Assesses quality of person bounding box crops for ReID preprocessing.
Uses three metrics:
1. Sharpness (Laplacian variance)
2. Pose/Orientation (OpenCV-based front/side/back detection)
3. Occlusion (aspect ratio + edge density)
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass



@dataclass
class QualityResult:
    """Quality assessment result for a person crop."""
    total_score: float  # 0-100
    sharpness_score: float  # 0-100
    pose: str  # 'front', 'side', 'back', 'unknown'
    pose_score: float  # 0-100
    occlusion_score: float  # 0-100
    is_acceptable: bool  # Whether quality meets minimum threshold
    
    def to_dict(self) -> Dict:
        return {
            'total_score': self.total_score,
            'sharpness_score': self.sharpness_score,
            'pose': self.pose,
            'pose_score': self.pose_score,
            'occlusion_score': self.occlusion_score,
            'is_acceptable': self.is_acceptable
        }


class SharpnessScorer:
    """
    Measures image sharpness using Laplacian variance.
    Higher variance = sharper image.
    """
    
    def __init__(self, blur_threshold: float = 50.0):
        """
        Args:
            blur_threshold: Minimum Laplacian variance to consider sharp
        """
        self.blur_threshold = blur_threshold
    
    def score(self, image: np.ndarray) -> float:
        """
        Compute sharpness score.
        
        Args:
            image: BGR image (person crop)
            
        Returns:
            Sharpness score 0-100 (100 = very sharp)
        """
        if image is None or image.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-100 (typical variance range: 0-500+)
        # Use sigmoid-like normalization for better distribution
        normalized = min(100.0, (variance / 5.0) ** 0.5 * 10)
        
        return normalized
    
    def is_blurry(self, image: np.ndarray) -> bool:
        """Check if image is too blurry."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold


class PoseEstimator:
    """
    Estimates body pose orientation using face detection.
    Uses OpenCV's face cascade for front/back classification.
    Classifies as: front, side, back, or unknown.
    """
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Args:
            min_confidence: Minimum confidence for detection
        """
        self.min_confidence = min_confidence
        
        # Load OpenCV face cascade (built-in, no download needed)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
    
    def estimate(
        self,
        image: np.ndarray,
        velocity: Tuple[float, float] = (0.0, 0.0)
    ) -> Tuple[str, float, dict]:
        """
        Estimate pose orientation using face detection and motion.
        
        Args:
            image: BGR person crop
            velocity: (dx, dy) motion vector from tracker
            
        Returns:
            Tuple of (pose_type, score, info)
            - pose_type: 'front', 'side', 'back', 'unknown'
            - score: 0-100
            - info: dict with detection info
        """
        if image is None or image.size == 0:
            return 'unknown', 0.0, {}
        
        h, w = image.shape[:2]
        if h < 32 or w < 16:
            return 'unknown', 30.0, {'reason': 'too_small'}
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase sensitivity for low-res faces by lowering neighbors default
        # Detect frontal faces
        frontal_faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15)
        )
        
        # Detect profile faces
        profile_faces = self._profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=2, minSize=(15, 15)
        )
        
        info = {
            'frontal_faces': len(frontal_faces),
            'profile_faces': len(profile_faces),
            'velocity_dy': velocity[1]
        }
        
        # 1. Visual Classification (Primary)
        if len(frontal_faces) > 0:
            face_area = frontal_faces[0][2] * frontal_faces[0][3]
            image_area = h * w
            face_ratio = face_area / image_area
            score = min(100.0, 60.0 + face_ratio * 200)
            return 'front', score, info
        
        elif len(profile_faces) > 0:
            face_area = profile_faces[0][2] * profile_faces[0][3]
            image_area = h * w
            face_ratio = face_area / image_area
            score = 40.0 + face_ratio * 100
            if abs(velocity[1]) > 1.5 and velocity[1] > 0:
                 # If visual says side but moving strongly down (front), trust motion slightly
                 return 'front', 55.0, info
            return 'side', score, info
        
        # 2. Motion Classification (Fallback)
        # Assuming typical camera: moving down (+y) = towards camera (front)
        # Moving up (-y) = away from camera (back)
        dy = velocity[1]
        motion_threshold = 0.5
        
        if dy > motion_threshold:
            # Moving down -> Front
            return 'front', 50.0 + min(20.0, dy * 2), info
        elif dy < -motion_threshold:
            # Moving up -> Back
            return 'back', 40.0 + min(20.0, abs(dy) * 2), info
            
        # 3. Brightness/Structure (Last Resort)
        upper_third = gray[:h//3, :]
        brightness = np.mean(upper_third)
        
        if brightness > 60: # Light hair/skin?
            return 'front', 30.0, info # Weak bias
        else:
            return 'back', 25.0, info
    
    def close(self):
        """Release resources (no-op for OpenCV)."""
        pass


class OcclusionDetector:
    """
    Detects potential occlusion in person crops using:
    1. Aspect ratio analysis (normal person ~0.4)
    2. Edge density (occluded = less internal edges)
    3. Color distribution (uniform = potential occlusion)
    """
    
    def __init__(self):
        self.ideal_aspect_min = 0.25  # Narrowest acceptable
        self.ideal_aspect_max = 0.55  # Widest acceptable
    
    def score(self, image: np.ndarray, bbox: Tuple[int, int, int, int] = None) -> float:
        """
        Compute occlusion score.
        
        Args:
            image: BGR person crop
            bbox: Optional (x1, y1, x2, y2) for aspect ratio calculation
            
        Returns:
            Occlusion-free score 0-100 (100 = no occlusion detected)
        """
        if image is None or image.size == 0:
            return 0.0
        
        h, w = image.shape[:2]
        
        # 1. Aspect ratio score
        aspect = w / h if h > 0 else 0
        if self.ideal_aspect_min <= aspect <= self.ideal_aspect_max:
            aspect_score = 100.0
        elif aspect < self.ideal_aspect_min:
            aspect_score = max(0, 100 - (self.ideal_aspect_min - aspect) * 200)
        else:
            aspect_score = max(0, 100 - (aspect - self.ideal_aspect_max) * 150)
        
        # 2. Edge density score (more edges = more visible body parts)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0 * 100
        edge_score = min(100.0, edge_density * 2.5)  # Normalize
        
        # 3. Color variance score (uniform color = potential occlusion)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        color_variance = (h_std + s_std) / 2
        color_score = min(100.0, color_variance * 2)
        
        # Combine scores
        total = (aspect_score * 0.4) + (edge_score * 0.35) + (color_score * 0.25)
        
        return total


class QualityScorer:
    """
    Combined quality scorer for person crops.
    Aggregates sharpness, pose, and occlusion scores.
    """
    
    # Score weights
    WEIGHT_SHARPNESS = 0.30
    WEIGHT_POSE = 0.40
    WEIGHT_OCCLUSION = 0.30
    
    # Minimum acceptable score
    MIN_ACCEPTABLE_SCORE = 30.0
    
    def __init__(
        self,
        blur_threshold: float = 50.0,
        min_acceptable_score: float = 30.0
    ):
        """
        Initialize quality scorer.
        
        Args:
            blur_threshold: Laplacian variance threshold for blur detection
            min_acceptable_score: Minimum total score to accept a crop
        """
        self.sharpness_scorer = SharpnessScorer(blur_threshold)
        self.pose_estimator = PoseEstimator()
        self.occlusion_detector = OcclusionDetector()
        self.min_acceptable_score = min_acceptable_score
    
    def score(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
        quick_mode: bool = False
    ) -> QualityResult:
        """
        Compute comprehensive quality score.
        
        Args:
            image: BGR person crop
            bbox: Optional bounding box (x1, y1, x2, y2)
            velocity: motion vector (dx, dy)
            quick_mode: Skip pose estimation for speed
            
        Returns:
            QualityResult with all scores
        """
        if image is None or image.size == 0:
            return QualityResult(
                total_score=0.0,
                sharpness_score=0.0,
                pose='unknown',
                pose_score=0.0,
                occlusion_score=0.0,
                is_acceptable=False
            )
        
        # Compute individual scores
        sharpness_score = self.sharpness_scorer.score(image)
        
        if quick_mode:
            pose_type = 'unknown'
            pose_score = 50.0  # Neutral
        else:
            pose_type, pose_score, _ = self.pose_estimator.estimate(image, velocity)
        
        occlusion_score = self.occlusion_detector.score(image, bbox)
        
        # Weighted combination
        total_score = (
            sharpness_score * self.WEIGHT_SHARPNESS +
            pose_score * self.WEIGHT_POSE +
            occlusion_score * self.WEIGHT_OCCLUSION
        )
        
        is_acceptable = total_score >= self.min_acceptable_score
        
        return QualityResult(
            total_score=total_score,
            sharpness_score=sharpness_score,
            pose=pose_type,
            pose_score=pose_score,
            occlusion_score=occlusion_score,
            is_acceptable=is_acceptable
        )
    
    def quick_score(self, image: np.ndarray) -> float:
        """
        Fast scoring using only sharpness and occlusion.
        Skips pose estimation for speed.
        
        Args:
            image: BGR person crop
            
        Returns:
            Quick quality score 0-100
        """
        if image is None or image.size == 0:
            return 0.0
        
        sharpness = self.sharpness_scorer.score(image)
        occlusion = self.occlusion_detector.score(image)
        
        # Simple average for quick mode
        return (sharpness + occlusion) / 2
    
    def close(self):
        """Release resources."""
        pass


def test_quality_scorer():
    """Test quality scorer on sample images."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_scorer.py <image_path>")
        return
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Cannot read image: {image_path}")
        return
    
    print(f"Testing quality scorer on: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    scorer = QualityScorer()
    result = scorer.score(image)
    
    print(f"\n=== Quality Assessment ===")
    print(f"Total Score: {result.total_score:.1f}/100")
    print(f"  Sharpness: {result.sharpness_score:.1f}/100")
    print(f"  Pose: {result.pose} ({result.pose_score:.1f}/100)")
    print(f"  Occlusion-free: {result.occlusion_score:.1f}/100")
    print(f"  Acceptable: {result.is_acceptable}")
    
    scorer.close()


if __name__ == '__main__':
    test_quality_scorer()
