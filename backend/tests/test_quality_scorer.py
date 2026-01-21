"""
Unit tests for QualityScorer.

Tests blur detection, occlusion assessment, illumination quality,
and overall quality scoring.
"""

import pytest
import numpy as np
import cv2
from app.core.reid.quality_scorer import QualityScorer


@pytest.fixture
def scorer():
    """Create a QualityScorer instance for testing."""
    return QualityScorer()


class TestBlurAssessment:
    """Tests for blur detection using Laplacian variance."""
    
    def test_sharp_image(self, scorer):
        """Test that sharp images get high blur scores."""
        # Create a sharp image with strong edges
        sharp_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sharp_image[45:55, :] = 255  # Strong horizontal edge
        sharp_image[:, 45:55] = 0     # Strong vertical edge
        
        score = scorer.assess_blur(sharp_image)
        assert score > 0.5, f"Sharp image should have high blur score, got {score}"
    
    def test_blurry_image(self, scorer):
        """Test that blurry images get low blur scores."""
        # Create a blurry image (smooth gradient)
        blurry_image = np.linspace(0, 255, 10000).reshape(100, 100).astype(np.uint8)
        blurry_image = np.stack([blurry_image] * 3, axis=-1)
        blurry_image = cv2.GaussianBlur(blurry_image, (15, 15), 5)
        
        score = scorer.assess_blur(blurry_image)
        assert score < 0.5, f"Blurry image should have low blur score, got {score}"
    
    def test_empty_image(self, scorer):
        """Test that empty images return 0."""
        score = scorer.assess_blur(np.array([]))
        assert score == 0.0
    
    def test_grayscale_image(self, scorer):
        """Test that grayscale images work."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        gray[45:55, :] = 255
        
        score = scorer.assess_blur(gray)
        assert 0.0 <= score <= 1.0


class TestOcclusionAssessment:
    """Tests for occlusion detection using IoU."""
    
    def test_no_occlusion(self, scorer):
        """Test single bbox with no occlusion."""
        bbox = np.array([10, 10, 50, 50])
        all_bboxes = [bbox]
        
        score = scorer.assess_occlusion(bbox, all_bboxes)
        assert score == 1.0, "Single detection should have no occlusion"
    
    def test_partial_occlusion(self, scorer):
        """Test partially overlapping bboxes."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([30, 30, 70, 70])  # Overlaps with bbox1
        all_bboxes = [bbox1, bbox2]
        
        score = scorer.assess_occlusion(bbox1, all_bboxes)
        assert 0.0 < score < 1.0, f"Partial occlusion should give score between 0-1, got {score}"
    
    def test_heavy_occlusion(self, scorer):
        """Test heavily overlapping bboxes."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([15, 15, 55, 55])  # Heavy overlap
        all_bboxes = [bbox1, bbox2]
        
        score = scorer.assess_occlusion(bbox1, all_bboxes)
        assert score < 0.5, f"Heavy occlusion should give low score, got {score}"
    
    def test_no_overlap(self, scorer):
        """Test completely separate bboxes."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([100, 100, 150, 150])  # No overlap
        all_bboxes = [bbox1, bbox2]
        
        score = scorer.assess_occlusion(bbox1, all_bboxes)
        assert score == 1.0, "No overlap should give perfect score"


class TestIlluminationAssessment:
    """Tests for illumination quality."""
    
    def test_good_illumination(self, scorer):
        """Test well-lit image with good contrast."""
        # Create image with good brightness and contrast
        img = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
        
        score = scorer.assess_illumination(img)
        assert score > 0.5, f"Well-lit image should have high score, got {score}"
    
    def test_underexposed(self, scorer):
        """Test dark/underexposed image."""
        # Very dark image
        dark_img = np.random.randint(0, 50, (100, 100, 3), dtype=np.uint8)
        
        score = scorer.assess_illumination(dark_img)
        assert score < 0.7, f"Dark image should have lower score, got {score}"
    
    def test_overexposed(self, scorer):
        """Test bright/overexposed image."""
        # Very bright image
        bright_img = np.random.randint(210, 255, (100, 100, 3), dtype=np.uint8)
        
        score = scorer.assess_illumination(bright_img)
        assert score < 0.7, f"Overexposed image should have lower score, got {score}"
    
    def test_low_contrast(self, scorer):
        """Test low contrast (washed out) image."""
        # All pixels similar value (low contrast)
        low_contrast = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        score = scorer.assess_illumination(low_contrast)
        # Low contrast will have decent intensity but very low std, resulting in ~0.6 score
        assert score < 0.8, f"Low contrast image should have moderate score, got {score}"


class TestOverallQualityScore:
    """Tests for combined quality scoring."""
    
    def test_high_quality_crop(self, scorer):
        """Test high-quality crop gets high score."""
        # Sharp, well-lit, no occlusion
        high_quality = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
        high_quality[45:55, :] = 255  # Add sharp edges
        bbox = np.array([10, 10, 50, 50])
        all_bboxes = [bbox]
        
        score = scorer.compute_quality_score(
            crop=high_quality,
            bbox=bbox,
            all_bboxes=all_bboxes,
            confidence=0.9
        )
        
        assert score > 0.6, f"High quality crop should score >0.6, got {score}"
    
    def test_low_quality_crop(self, scorer):
        """Test low-quality crop gets low score."""
        # Blurry, dark, occluded
        low_quality = np.full((100, 100, 3), 30, dtype=np.uint8)  # Dark
        low_quality = cv2.GaussianBlur(low_quality, (15, 15), 5)  # Blurry
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([15, 15, 55, 55])  # Heavy occlusion
        
        score = scorer.compute_quality_score(
            crop=low_quality,
            bbox=bbox1,
            all_bboxes=[bbox1, bbox2],
            confidence=0.5
        )
        
        assert score < 0.5, f"Low quality crop should score <0.5, got {score}"
    
    def test_score_range(self, scorer):
        """Test that scores are always in [0, 1] range."""
        # Random image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])
        
        score = scorer.compute_quality_score(
            crop=img,
            bbox=bbox,
            all_bboxes=[bbox],
            confidence=0.7
        )
        
        assert 0.0 <= score <= 1.0, f"Score must be in [0,1], got {score}"
    
    def test_quality_breakdown(self, scorer):
        """Test detailed quality breakdown."""
        img = np.random.randint(60, 200, (100, 100, 3), dtype=np.uint8)
        bbox = np.array([10, 10, 50, 50])
        
        breakdown = scorer.get_quality_breakdown(
            crop=img,
            bbox=bbox,
            all_bboxes=[bbox],
            confidence=0.8
        )
        
        assert 'blur' in breakdown
        assert 'occlusion' in breakdown
        assert 'illumination' in breakdown
        assert 'confidence' in breakdown
        assert 'overall' in breakdown
        
        # Check all scores are in valid range
        for key, value in breakdown.items():
            assert 0.0 <= value <= 1.0, f"{key} score out of range: {value}"


class TestIoUComputation:
    """Tests for IoU calculation."""
    
    def test_identical_boxes(self, scorer):
        """Test IoU of identical boxes is 1.0."""
        bbox = np.array([10, 10, 50, 50])
        iou = scorer._compute_iou(bbox, bbox)
        assert iou == 1.0
    
    def test_non_overlapping_boxes(self, scorer):
        """Test IoU of non-overlapping boxes is 0.0."""
        bbox1 = np.array([10, 10, 50, 50])
        bbox2 = np.array([100, 100, 150, 150])
        iou = scorer._compute_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_partial_overlap(self, scorer):
        """Test IoU of partially overlapping boxes."""
        bbox1 = np.array([0, 0, 10, 10])    # Area = 100
        bbox2 = np.array([5, 5, 15, 15])    # Area = 100
        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        iou = scorer._compute_iou(bbox1, bbox2)
        assert 0.14 <= iou <= 0.15, f"Expected IoU ≈0.143, got {iou}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
