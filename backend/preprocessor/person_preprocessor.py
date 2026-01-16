"""
Person Preprocessor - Quality-based person capture for ReID.

Orchestrates the complete preprocessing pipeline:
1. PeopleNet detection
2. IOU-based tracking
3. Quality assessment (sharpness, pose, occlusion)
4. Gallery management (top-K per person)

Outputs high-quality person crops organized by track ID.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import List, Optional

from preprocessor.peoplenet_detector import PeopleNetDetector
from preprocessor.quality_scorer import QualityScorer, QualityResult
from preprocessor.person_tracker import PersonTracker
from preprocessor.gallery_manager import GalleryManager


class PersonPreprocessor:
    """
    Preprocessing pipeline for collecting high-quality person captures.
    
    Designed for ReID preprocessing - filters out garbage captures
    (blurry, back-view, occluded) and keeps only the best shots per person.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = './gallery',
        device: str = 'cuda',
        detection_threshold: float = 0.4,
        max_captures_per_person: int = 5,
        min_quality_score: float = 30.0,
        min_frame_gap: int = 5,
        min_height: int = 128,  # Minimum pixel height for Person ReID (Model inputs are 256x128)
        quick_quality_mode: bool = False
    ):
        """
        Initialize preprocessor.
        
        Args:
            model_path: Path to PeopleNet ONNX model
            output_dir: Directory to save per-person galleries
            device: Inference device ('cuda' or 'cpu')
            detection_threshold: Person detection confidence threshold
            max_captures_per_person: Maximum images per person
            min_quality_score: Minimum quality to consider capture
            min_frame_gap: Minimum frames between captures of same person
            min_height: Minimum crop height in pixels (discard smaller)
            quick_quality_mode: Skip pose estimation for faster processing
        """
        self.output_dir = Path(output_dir)
        self.min_height = min_height
        self.quick_mode = quick_quality_mode
        
        # Initialize components
        print("[Preprocessor] Initializing PeopleNet detector...")
        self.detector = PeopleNetDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=detection_threshold,
            nms_threshold=0.5
        )
        
        print("[Preprocessor] Initializing quality scorer...")
        self.quality_scorer = QualityScorer(
            blur_threshold=50.0,
            min_pose_confidence=0.5,
            min_acceptable_score=min_quality_score
        )
        
        print("[Preprocessor] Initializing tracker...")
        self.tracker = PersonTracker(
            iou_threshold=0.3,
            max_age=30,
            min_hits=2
        )
        
        print("[Preprocessor] Initializing gallery manager...")
        self.gallery = GalleryManager(
            output_dir=output_dir,
            max_captures_per_person=max_captures_per_person,
            min_frame_gap=min_frame_gap,
            min_quality_for_save=min_quality_score
        )
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_total': 0,
            'quality_assessments': 0,
            'captures_accepted': 0,
            'processing_time_ms': 0.0
        }
        
        print("[Preprocessor] Ready!")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        persons_only: bool = True
    ) -> List[dict]:
        """
        Process a single frame.
        
        Args:
            frame: BGR frame
            frame_idx: Frame index
            persons_only: Only detect persons (no bags/faces)
            
        Returns:
            List of detection dicts with quality info
        """
        start_time = time.time()
        
        # 1. Detect persons
        classes = ['person'] if persons_only else None
        detections = self.detector.detect(frame, classes=classes)
        self.stats['detections_total'] += len(detections)
        
        if not detections:
            return []
        
        # 2. Track detections
        detections = self.tracker.update(detections, frame_idx)
        
        # 3. For each detection, assess quality and potentially capture
        results = []
        
        for det in detections:
            track_id = det.get('track_id', -1)
            if track_id < 0:
                continue
            
            # Extract crop
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            
            # Clip to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Filter small detections (too low resolution for ReID)
            crop_h = y2 - y1
            if crop_h < self.min_height:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            # 4. Quality assessment
            # Extract velocity if available
            velocity = det.get('velocity', (0.0, 0.0))
            
            quality = self.quality_scorer.score(
                crop, 
                bbox=(x1, y1, x2, y2),
                velocity=velocity,
                quick_mode=self.quick_mode
            )
            self.stats['quality_assessments'] += 1
            
            # Add quality info to detection
            det['quality'] = quality.to_dict()
            det['quality_score'] = quality.total_score
            
            # 5. Decide if we should capture this
            should_capture, reason = self.gallery.should_capture(
                track_id, frame_idx, quality
            )
            
            if should_capture:
                filename = self.gallery.add_capture(
                    track_id, frame_idx, crop, quality
                )
                det['captured'] = True
                det['capture_file'] = filename
                self.stats['captures_accepted'] += 1
            else:
                det['captured'] = False
                det['capture_reason'] = reason
            
            results.append(det)
        
        # Update stats
        elapsed = (time.time() - start_time) * 1000
        self.stats['processing_time_ms'] += elapsed
        self.stats['frames_processed'] += 1
        
        return results
    
    def process_video(
        self,
        video_path: str,
        show_preview: bool = True,
        max_frames: int = None,
        save_visualization: str = None
    ):
        """
        Process a video file.
        
        Args:
            video_path: Path to video file
            show_preview: Show live preview window
            max_frames: Maximum frames to process (None = all)
            save_visualization: Path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[Preprocessor] Processing video: {video_path}")
        print(f"[Preprocessor] Resolution: {width}x{height}, FPS: {fps}")
        print(f"[Preprocessor] Total frames: {total_frames}")
        
        # Video writer
        writer = None
        if save_visualization:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_visualization, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Process frame
                detections = self.process_frame(frame, frame_idx)
                
                # Visualize
                vis_frame = self._visualize(frame, detections)
                
                if writer:
                    writer.write(vis_frame)
                
                if show_preview:
                    cv2.imshow('Person Preprocessor', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[Preprocessor] User quit")
                        break
                
                frame_idx += 1
                
                # Progress
                if frame_idx % 100 == 0:
                    avg_time = self.stats['processing_time_ms'] / max(1, self.stats['frames_processed'])
                    print(f"[Preprocessor] Frame {frame_idx}/{total_frames}, "
                          f"avg time: {avg_time:.1f}ms, "
                          f"captures: {self.stats['captures_accepted']}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Export summary
        self._print_summary()
        summary_path = self.gallery.export_summary()
        print(f"[Preprocessor] Summary saved to: {summary_path}")
    
    def _visualize(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detections with quality info."""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            track_id = det.get('track_id', -1)
            quality_score = det.get('quality_score', 0)
            captured = det.get('captured', False)
            pose = det.get('quality', {}).get('pose', 'unknown')
            
            # Color based on quality
            if quality_score >= 60:
                color = (0, 255, 0)  # Green - good
            elif quality_score >= 40:
                color = (0, 255, 255)  # Yellow - acceptable
            else:
                color = (0, 0, 255)  # Red - poor
            
            # Draw box
            thickness = 3 if captured else 2
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"ID:{track_id} Q:{quality_score:.0f} {pose}"
            if captured:
                label += " [SAVED]"
            
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw stats
        stats_text = f"Persons: {len(self.gallery.galleries)} | Captures: {self.stats['captures_accepted']}"
        cv2.putText(result, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"Frames processed: {self.stats['frames_processed']}")
        print(f"Total detections: {self.stats['detections_total']}")
        print(f"Quality assessments: {self.stats['quality_assessments']}")
        print(f"Captures accepted: {self.stats['captures_accepted']}")
        print(f"Unique persons: {len(self.gallery.galleries)}")
        
        if self.stats['frames_processed'] > 0:
            avg_time = self.stats['processing_time_ms'] / self.stats['frames_processed']
            print(f"Avg processing time: {avg_time:.1f}ms/frame")
        
        self.gallery.print_summary()
    
    def close(self):
        """Release resources."""
        self.quality_scorer.close()


def main():
    """Run preprocessor from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Person Quality Preprocessor for ReID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video and save person galleries
  python person_preprocessor.py --model model.onnx --input video.mp4 --output ./gallery

  # Fast mode (skip pose estimation)
  python person_preprocessor.py --model model.onnx --input video.mp4 --quick

  # Save visualization video
  python person_preprocessor.py --model model.onnx --input video.mp4 --output ./gallery --save-video out.mp4
        """
    )
    
    parser.add_argument('--model', type=str, required=True, help='Path to PeopleNet ONNX model')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='./gallery', help='Output directory for galleries')
    parser.add_argument('--threshold', type=float, default=0.4, help='Detection threshold')
    parser.add_argument('--max-captures', type=int, default=5, help='Max captures per person')
    parser.add_argument('--min-quality', type=float, default=30.0, help='Minimum quality score')
    parser.add_argument('--min-height', type=int, default=128, help='Minimum crop height (pixels)')
    parser.add_argument('--frame-gap', type=int, default=5, help='Minimum frames between captures')
    parser.add_argument('--quick', action='store_true', help='Quick mode (skip pose estimation)')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    parser.add_argument('--max-frames', type=int, default=None, help='Maximum frames to process')
    parser.add_argument('--save-video', type=str, default=None, help='Save visualization video')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = PersonPreprocessor(
        model_path=args.model,
        output_dir=args.output,
        device=args.device,
        detection_threshold=args.threshold,
        max_captures_per_person=args.max_captures,
        min_quality_score=args.min_quality,
        min_frame_gap=args.frame_gap,
        min_height=args.min_height,
        quick_quality_mode=args.quick
    )
    
    try:
        # Process video
        preprocessor.process_video(
            video_path=args.input,
            show_preview=not args.no_preview,
            max_frames=args.max_frames,
            save_visualization=args.save_video
        )
    finally:
        preprocessor.close()


if __name__ == '__main__':
    main()
