"""
Gallery Manager for Person Captures.

Manages per-person image galleries, keeping only the best quality captures.
Implements:
- Top-K selection per person
- Diversity bonus (avoid duplicate poses)
- "Something is better than nothing" fallback
- Temporal spacing to avoid redundant captures
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from quality_scorer import QualityResult


@dataclass
class CaptureInfo:
    """Information about a captured image."""
    frame_idx: int
    quality_score: float
    pose: str
    sharpness: float
    filename: str = ""
    saved: bool = False


@dataclass
class PersonGallery:
    """Gallery of captures for a single person."""
    track_id: int
    captures: List[CaptureInfo] = field(default_factory=list)
    best_score: float = 0.0
    pose_distribution: Dict[str, int] = field(default_factory=lambda: {'front': 0, 'side': 0, 'back': 0, 'unknown': 0})
    
    def add_capture(self, info: CaptureInfo):
        """Add a capture to the gallery."""
        self.captures.append(info)
        self.pose_distribution[info.pose] = self.pose_distribution.get(info.pose, 0) + 1
        if info.quality_score > self.best_score:
            self.best_score = info.quality_score


class GalleryManager:
    """
    Manages per-person image galleries.
    
    Keeps top-K quality images per tracked person with diversity.
    """
    
    def __init__(
        self,
        output_dir: str,
        max_captures_per_person: int = 5,
        min_frame_gap: int = 5,  # Minimum frames between captures
        min_quality_for_save: float = 25.0,  # Minimum score to save
        diversity_bonus: float = 10.0  # Bonus for different pose
    ):
        """
        Initialize gallery manager.
        
        Args:
            output_dir: Directory to save per-person galleries
            max_captures_per_person: Maximum images to keep per person
            min_frame_gap: Minimum frames between saves for same person
            min_quality_for_save: Minimum quality score to consider saving
            diversity_bonus: Score bonus for underrepresented pose
        """
        self.output_dir = Path(output_dir)
        self.max_captures = max_captures_per_person
        self.min_frame_gap = min_frame_gap
        self.min_quality = min_quality_for_save
        self.diversity_bonus = diversity_bonus
        
        # Per-person galleries
        self.galleries: Dict[int, PersonGallery] = {}
        
        # Last capture frame per person
        self.last_capture_frame: Dict[int, int] = {}
        
        # Statistics
        self.stats = {
            'total_captures': 0,
            'saved_captures': 0,
            'rejected_low_quality': 0,
            'rejected_timing': 0,
            'persons_seen': 0
        }
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def should_capture(
        self,
        track_id: int,
        frame_idx: int,
        quality: QualityResult
    ) -> Tuple[bool, str]:
        """
        Determine if this detection should be captured.
        
        Args:
            track_id: Person track ID
            frame_idx: Current frame index
            quality: Quality assessment result
            
        Returns:
            Tuple of (should_capture, reason)
        """
        self.stats['total_captures'] += 1
        
        # Initialize gallery if new person
        if track_id not in self.galleries:
            self.galleries[track_id] = PersonGallery(track_id=track_id)
            self.stats['persons_seen'] += 1
        
        gallery = self.galleries[track_id]
        
        # Check timing (temporal spacing)
        last_frame = self.last_capture_frame.get(track_id, -999)
        if frame_idx - last_frame < self.min_frame_gap:
            self.stats['rejected_timing'] += 1
            return False, 'timing'
        
        # Calculate effective score with diversity bonus
        effective_score = quality.total_score
        
        # Add diversity bonus if pose is underrepresented
        if len(gallery.captures) > 0:
            total_poses = sum(gallery.pose_distribution.values())
            pose_ratio = gallery.pose_distribution.get(quality.pose, 0) / max(1, total_poses)
            if pose_ratio < 0.3:  # Underrepresented pose
                effective_score += self.diversity_bonus
        
        # Check if this would make the cut
        if len(gallery.captures) < self.max_captures:
            # Gallery not full yet - accept if above minimum
            if effective_score >= self.min_quality:
                return True, 'gallery_not_full'
            else:
                self.stats['rejected_low_quality'] += 1
                return False, 'low_quality'
        else:
            # Gallery full - only accept if better than worst
            worst_score = min(c.quality_score for c in gallery.captures)
            if effective_score > worst_score:
                return True, 'better_than_worst'
            else:
                return False, 'not_better'
    
    def add_capture(
        self,
        track_id: int,
        frame_idx: int,
        image: np.ndarray,
        quality: QualityResult,
        save_immediately: bool = True
    ) -> Optional[str]:
        """
        Add a capture to person's gallery.
        
        Args:
            track_id: Person track ID
            frame_idx: Frame index
            image: Person crop image (BGR)
            quality: Quality assessment result
            save_immediately: Whether to save to disk immediately
            
        Returns:
            Saved filename or None
        """
        if track_id not in self.galleries:
            self.galleries[track_id] = PersonGallery(track_id=track_id)
        
        gallery = self.galleries[track_id]
        
        # Update last capture frame
        self.last_capture_frame[track_id] = frame_idx
        
        # Create capture info
        capture_info = CaptureInfo(
            frame_idx=frame_idx,
            quality_score=quality.total_score,
            pose=quality.pose,
            sharpness=quality.sharpness_score
        )
        
        # If gallery is full, remove worst
        if len(gallery.captures) >= self.max_captures:
            worst_idx = min(range(len(gallery.captures)),
                           key=lambda i: gallery.captures[i].quality_score)
            removed = gallery.captures.pop(worst_idx)
            # Delete old file if it was saved
            if removed.saved and removed.filename:
                old_path = self.output_dir / f"person_{track_id}" / removed.filename
                if old_path.exists():
                    old_path.unlink()
        
        # Add new capture
        gallery.add_capture(capture_info)
        self.stats['saved_captures'] += 1
        
        # Save to disk
        filename = None
        if save_immediately:
            filename = self._save_image(track_id, capture_info, image)
            capture_info.filename = filename
            capture_info.saved = True
        
        return filename
    
    def _save_image(
        self,
        track_id: int,
        capture: CaptureInfo,
        image: np.ndarray
    ) -> str:
        """Save image to disk."""
        # Create person directory
        person_dir = self.output_dir / f"person_{track_id}"
        person_dir.mkdir(exist_ok=True)
        
        # Generate filename with score
        filename = f"{capture.frame_idx:06d}_score{capture.quality_score:.1f}_{capture.pose}.jpg"
        filepath = person_dir / filename
        
        cv2.imwrite(str(filepath), image)
        
        return filename
    
    def get_gallery(self, track_id: int) -> Optional[PersonGallery]:
        """Get gallery for a specific person."""
        return self.galleries.get(track_id)
    
    def get_best_capture(self, track_id: int) -> Optional[CaptureInfo]:
        """Get best quality capture for a person."""
        gallery = self.galleries.get(track_id)
        if not gallery or not gallery.captures:
            return None
        return max(gallery.captures, key=lambda c: c.quality_score)
    
    def export_summary(self) -> str:
        """Export summary statistics to JSON."""
        summary = {
            'statistics': self.stats,
            'persons': {}
        }
        
        for track_id, gallery in self.galleries.items():
            summary['persons'][str(track_id)] = {
                'num_captures': len(gallery.captures),
                'best_score': gallery.best_score,
                'pose_distribution': gallery.pose_distribution,
                'captures': [
                    {
                        'frame': c.frame_idx,
                        'score': c.quality_score,
                        'pose': c.pose,
                        'filename': c.filename
                    }
                    for c in gallery.captures
                ]
            }
        
        # Save to file
        summary_path = self.output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print("GALLERY MANAGER SUMMARY")
        print("="*60)
        print(f"Total persons tracked: {self.stats['persons_seen']}")
        print(f"Total detections: {self.stats['total_captures']}")
        print(f"Saved captures: {self.stats['saved_captures']}")
        print(f"Rejected (low quality): {self.stats['rejected_low_quality']}")
        print(f"Rejected (timing): {self.stats['rejected_timing']}")
        print("-"*60)
        
        for track_id, gallery in sorted(self.galleries.items()):
            print(f"Person {track_id}: {len(gallery.captures)} captures, "
                  f"best score: {gallery.best_score:.1f}")


if __name__ == '__main__':
    # Test gallery manager
    manager = GalleryManager(output_dir='./test_gallery')
    
    # Simulate quality result
    from quality_scorer import QualityResult
    
    quality = QualityResult(
        total_score=75.0,
        sharpness_score=80.0,
        pose='front',
        pose_score=70.0,
        occlusion_score=75.0,
        is_acceptable=True
    )
    
    # Test
    should, reason = manager.should_capture(1, 0, quality)
    print(f"Should capture: {should}, reason: {reason}")
