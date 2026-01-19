"""
PeopleNet Detector - ONNX Runtime based inference for NVIDIA PeopleNet.

Based on correct post-processing from NVIDIA forum:
https://forums.developer.nvidia.com/t/run-peoplenet-with-tensorrt/128000/22

Supports detection of: person, bag, face
Input: RGB images resized to 960x544
Output: List of Detection dictionaries with bounding boxes and confidence scores
"""

import numpy as np
import cv2

# Preload CUDA DLLs before importing onnxruntime (fixes pip-installed CUDA packages on Windows)
try:
    import onnxruntime
    onnxruntime.preload_dlls(cuda=True, cudnn=True)
except (AttributeError, Exception):
    pass  # preload_dlls not available in older versions or no CUDA packages

import onnxruntime as ort
from typing import List, Tuple, Optional
from pathlib import Path

try:
    from preprocessor.postprocess_utils import (
        postprocess_detectnet_vectorized,
        nms,
        MODEL_W,
        MODEL_H
    )
except ImportError:
    # Fallback for direct execution
    from postprocess_utils import (
        postprocess_detectnet_vectorized,
        nms,
        MODEL_W,
        MODEL_H
    )


class PeopleNetDetector:
    """
    NVIDIA PeopleNet detector using ONNX Runtime.
    
    This detector wraps the PeopleNet ONNX model and provides an easy-to-use
    interface for detecting people, bags, and faces in images.
    """
    
    # Class definitions
    CLASSES = ['person', 'bag', 'face']
    
    # Color palette for visualization (BGR format)
    COLORS = {
        'person': (0, 255, 0),   # Green
        'bag': (255, 0, 0),      # Blue
        'face': (0, 0, 255)      # Red
    }
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.5
    ):
        """
        Initialize the PeopleNet detector.
        
        Args:
            model_path: Path to the ONNX model file
            device: Inference device ('cuda' or 'cpu')
            confidence_threshold: Minimum confidence for final detections
            nms_threshold: IOU threshold for NMS
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize ONNX Runtime session
        self._init_session()
        
        print(f"[PeopleNet] Loaded model: {self.model_path.name}")
        print(f"[PeopleNet] Device: {self.device}")
        print(f"[PeopleNet] Input shape: 3x{MODEL_H}x{MODEL_W}")
    
    def _init_session(self):
        """Initialize ONNX Runtime inference session with TensorRT optimization."""
        providers = []
        
        if self.device == 'cuda':
            available = ort.get_available_providers()
            
            # Try TensorRT first (highest performance)
            # Note: Provider name can be 'TensorrtExecutionProvider' or 'TensorRTExecutionProvider'
            has_trt = any('tensorrt' in p.lower() for p in available)
            trt_provider = next((p for p in available if 'tensorrt' in p.lower()), None)
            
            if has_trt and trt_provider:
                trt_options = {
                    'trt_fp16_enable': True,  # Enable FP16 for 2-3x speedup
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache/peoplenet',
                    'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                    'trt_max_partition_iterations': 1000,
                    'trt_min_subgraph_size': 1,
                }
                providers.append((trt_provider, trt_options))
                print(f"[PeopleNet] Using {trt_provider} with FP16")
            
            # Fallback to CUDA
            elif 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
                print("[PeopleNet] Using CUDA execution provider (TensorRT unavailable)")
            else:
                print("[PeopleNet] CUDA not available, falling back to CPU")
        
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        print(f"[PeopleNet] Input name: {self.input_name}")
        print(f"[PeopleNet] Output names: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Preprocess image for inference.
        
        Preprocessing steps (from NVIDIA forum):
        1. Convert BGR to RGB
        2. Resize to 960x544
        3. Scale to [0, 1] by dividing by 255
        4. Convert to NCHW format
        
        Args:
            image: Input image in BGR format (OpenCV default)
            
        Returns:
            Tuple of (preprocessed_tensor, original_height, original_width)
        """
        orig_h, orig_w = image.shape[:2]
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input dimensions
        img_resized = cv2.resize(
            img_rgb, 
            (MODEL_W, MODEL_H),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Scale to [0, 1] range
        img_scaled = img_resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format
        img_chw = np.transpose(img_scaled, (2, 0, 1))
        
        # Add batch dimension (NCHW)
        input_tensor = np.expand_dims(img_chw, axis=0)
        
        return input_tensor, orig_h, orig_w
    
    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on preprocessed input.
        
        Args:
            input_tensor: Preprocessed input tensor (1, 3, 544, 960)
            
        Returns:
            Tuple of (output_cov, output_bbox) tensors
        """
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # PeopleNet outputs: output_cov (coverage), output_bbox (bounding boxes)
        output_cov = None
        output_bbox = None
        
        # Log output shapes for debugging
        print(f"[PeopleNet] Model output shapes: {[o.shape for o in outputs]}")
        
        for output in outputs:
            if output.shape[1] == len(self.CLASSES):  # Coverage has num_classes channels
                output_cov = output
                print(f"[PeopleNet] Coverage tensor shape: {output.shape} (classes={len(self.CLASSES)})")
            elif output.shape[1] == len(self.CLASSES) * 4:  # Bbox has num_classes*4 channels
                output_bbox = output
                print(f"[PeopleNet] BBox tensor shape: {output.shape} (classes*4={len(self.CLASSES)*4})")
        
        if output_cov is None or output_bbox is None:
            raise ValueError(f"Unexpected output shapes: {[o.shape for o in outputs]}")
        
        return output_cov, output_bbox
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        classes: List[str] = None
    ) -> List[dict]:
        """
        Detect objects in an image.
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Optional override for confidence threshold
            classes: Optional list of class names to detect (default: all)
            
        Returns:
            List of detection dictionaries with keys:
            'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'class_name'
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Convert class names to indices
        if classes:
            analysis_classes = [self.CLASSES.index(c) for c in classes if c in self.CLASSES]
        else:
            analysis_classes = None
        
        # Preprocess
        input_tensor, orig_h, orig_w = self.preprocess(image)
        
        # Infer
        output_cov, output_bbox = self.infer(input_tensor)
        
        # Post-process using the correct NVIDIA forum formula
        detections = postprocess_detectnet_vectorized(
            output_bbox=output_bbox,
            output_cov=output_cov,
            num_classes=len(self.CLASSES),
            min_confidence=confidence_threshold,
            analysis_classes=analysis_classes,
            orig_width=orig_w,
            orig_height=orig_h
        )
        
        # Apply NMS
        detections = nms(detections, self.nms_threshold)
        
        # Add class names
        for det in detections:
            det['class_name'] = self.CLASSES[det['class_id']]
        
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence_threshold: float = None,
        classes: List[str] = None
    ) -> List[List[dict]]:
        """
        Detect objects in a batch of images (TensorRT optimized).
        
        Args:
            images: List of input images in BGR format
            confidence_threshold: Optional override for confidence threshold
            classes: Optional list of class names to detect (default: all)
            
        Returns:
            List of detection lists (one per image)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        # Convert class names to indices
        if classes:
            analysis_classes = [self.CLASSES.index(c) for c in classes if c in self.CLASSES]
        else:
            analysis_classes = None
        
        # Preprocess all images
        batch_tensors = []
        orig_sizes = []
        
        for image in images:
            input_tensor, orig_h, orig_w = self.preprocess(image)
            batch_tensors.append(input_tensor.squeeze(0))  # Remove batch dim, will stack later
            orig_sizes.append((orig_h, orig_w))
        
        # Stack into batch (N, C, H, W)
        batch_input = np.stack(batch_tensors, axis=0)
        
        # Run batched inference
        outputs = self.session.run(None, {self.input_name: batch_input})
        
        # Parse outputs
        output_cov = None
        output_bbox = None
        
        for output in outputs:
            if output.shape[1] == len(self.CLASSES):
                output_cov = output
            elif output.shape[1] == len(self.CLASSES) * 4:
                output_bbox = output
        
        if output_cov is None or output_bbox is None:
            raise ValueError(f"Unexpected output shapes: {[o.shape for o in outputs]}")
        
        # Post-process each image in batch
        all_detections = []
        
        for i, (orig_h, orig_w) in enumerate(orig_sizes):
            # Extract outputs for this image
            img_bbox = output_bbox[i:i+1]  # Keep batch dim
            img_cov = output_cov[i:i+1]
            
            # Post-process
            detections = postprocess_detectnet_vectorized(
                output_bbox=img_bbox,
                output_cov=img_cov,
                num_classes=len(self.CLASSES),
                min_confidence=confidence_threshold,
                analysis_classes=analysis_classes,
                orig_width=orig_w,
                orig_height=orig_h
            )
            
            # Apply NMS
            detections = nms(detections, self.nms_threshold)
            
            # Add class names
            for det in detections:
                det['class_name'] = self.CLASSES[det['class_id']]
            
            all_detections.append(detections)
        
        return all_detections
    
    def detect_persons(
        self,
        image: np.ndarray,
        confidence_threshold: float = None
    ) -> List[dict]:
        """Convenience method to detect only persons."""
        return self.detect(image, confidence_threshold, classes=['person'])
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[dict],
        show_labels: bool = True,
        show_confidence: bool = True,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detections on image.
        
        Args:
            image: Input image (will be copied)
            detections: List of detection dictionaries
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            thickness: Line thickness for bounding boxes
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for det in detections:
            class_name = det.get('class_name', self.CLASSES[det['class_id']])
            color = self.COLORS.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result.shape[1], x2)
            y2 = min(result.shape[0], y2)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(class_name)
                if show_confidence:
                    label_parts.append(f"{det['confidence']:.2f}")
                label = " ".join(label_parts)
                
                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    result,
                    (x1, y1 - text_h - 5),
                    (x1 + text_w + 5, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    label,
                    (x1 + 2, y1 - 3),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )
        
        return result
    
    def process_video(
        self,
        video_source,
        output_path: str = None,
        confidence_threshold: float = None,
        show_preview: bool = True,
        show_fps: bool = True,
        max_frames: int = None,
        classes: List[str] = None
    ) -> None:
        """
        Process video stream with live visualization.
        
        Args:
            video_source: Path to video file, camera index (0, 1, etc), or RTSP URL
            output_path: Optional path to save output video
            confidence_threshold: Confidence threshold (uses default if None)
            show_preview: Whether to show live preview window
            show_fps: Whether to display FPS on the video
            max_frames: Maximum number of frames to process (None = all)
            classes: List of class names to detect (None = all)
        """
        import time
        
        # Handle video source
        if isinstance(video_source, int):
            cap = cv2.VideoCapture(video_source)
            source_name = f"Camera {video_source}"
        else:
            cap = cv2.VideoCapture(video_source)
            source_name = str(video_source)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n[PeopleNet] Processing video: {source_name}")
        print(f"[PeopleNet] Resolution: {width}x{height}, FPS: {fps}")
        if total_frames > 0:
            print(f"[PeopleNet] Total frames: {total_frames}")
        
        # Initialize video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"[PeopleNet] Saving output to: {output_path}")
        
        frame_count = 0
        fps_display = 0
        fps_update_time = time.time()
        frame_times = []
        
        print("[PeopleNet] Press 'q' to quit, 'p' to pause")
        
        try:
            paused = False
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n[PeopleNet] End of video stream")
                        break
                    
                    if max_frames and frame_count >= max_frames:
                        print(f"\n[PeopleNet] Reached max frames: {max_frames}")
                        break
                    
                    # Measure inference time
                    start_time = time.time()
                    
                    # Detect
                    detections = self.detect(frame, confidence_threshold, classes)
                    
                    inference_time = (time.time() - start_time) * 1000
                    frame_times.append(inference_time)
                    
                    # Visualize
                    result = self.visualize(frame, detections)
                    
                    # Calculate and display FPS
                    if show_fps:
                        current_time = time.time()
                        if current_time - fps_update_time >= 0.5:  # Update FPS every 0.5 seconds
                            fps_display = 1000.0 / (sum(frame_times[-10:]) / len(frame_times[-10:]))
                            fps_update_time = current_time
                        
                        # Draw FPS and detection count
                        info_text = f"FPS: {fps_display:.1f} | Detections: {len(detections)}"
                        cv2.putText(result, info_text, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Write output
                    if writer:
                        writer.write(result)
                    
                    frame_count += 1
                else:
                    result = frame.copy() if 'frame' in locals() else np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(result, "PAUSED - Press 'p' to resume", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show preview
                if show_preview:
                    cv2.imshow('PeopleNet Live Detection', result)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[PeopleNet] User quit")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"[PeopleNet] {'Paused' if paused else 'Resumed'}")
                
                # Progress update
                if frame_count > 0 and frame_count % 100 == 0:
                    avg_time = sum(frame_times[-100:]) / min(len(frame_times), 100)
                    print(f"[PeopleNet] Processed {frame_count} frames, avg inference: {avg_time:.1f}ms")
        
        except KeyboardInterrupt:
            print("\n[PeopleNet] Interrupted by user")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Print summary
        if frame_times:
            avg_inference = sum(frame_times) / len(frame_times)
            print(f"\n[PeopleNet] Summary:")
            print(f"  - Processed frames: {frame_count}")
            print(f"  - Average inference time: {avg_inference:.1f}ms")
            print(f"  - Average FPS: {1000.0/avg_inference:.1f}")
            if output_path:
                print(f"  - Output saved to: {output_path}")


def main():
    """Run PeopleNet detector on image or video."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='NVIDIA PeopleNet ONNX Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python peoplenet_detector.py --model model.onnx --input image.png --output result.png

  # Process video file with live preview
  python peoplenet_detector.py --model model.onnx --input video.mp4 --video

  # Process webcam (camera 0)
  python peoplenet_detector.py --model model.onnx --input 0 --video

  # Save processed video
  python peoplenet_detector.py --model model.onnx --input video.mp4 --video --output out.mp4
        """
    )
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--input', type=str, required=True, help='Input image, video, or camera index')
    parser.add_argument('--output', type=str, default=None, help='Output path (image or video)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    parser.add_argument('--nms', type=float, default=0.5, help='NMS IOU threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--video', action='store_true', help='Process as video (enables live preview)')
    parser.add_argument('--persons-only', action='store_true', help='Detect only persons (no bags/faces)')
    args = parser.parse_args()
    
    # Initialize detector
    detector = PeopleNetDetector(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.threshold,
        nms_threshold=args.nms
    )
    
    # Determine classes to detect
    classes = ['person'] if args.persons_only else None
    
    if args.video:
        # Video mode
        # Check if input is a camera index
        try:
            video_source = int(args.input)
        except ValueError:
            video_source = args.input
        
        detector.process_video(
            video_source=video_source,
            output_path=args.output,
            show_preview=True,
            show_fps=True,
            classes=classes
        )
    else:
        # Image mode
        image = cv2.imread(args.input)
        if image is None:
            raise IOError(f"Cannot read image: {args.input}")
        
        print(f"\n[PeopleNet] Processing image: {args.input}")
        print(f"[PeopleNet] Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Detect
        detections = detector.detect(image, classes=classes)
        
        print(f"[PeopleNet] Found {len(detections)} detections:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.3f} @ "
                  f"({int(det['x1'])}, {int(det['y1'])}, {int(det['x2'])}, {int(det['y2'])})")
        
        # Visualize
        result = detector.visualize(image, detections)
        
        # Save or display
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"[PeopleNet] Output saved to: {args.output}")
        else:
            cv2.imshow('PeopleNet Detection', result)
            print("[PeopleNet] Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

