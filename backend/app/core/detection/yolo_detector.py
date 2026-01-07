"""
YOLOv8 Person Detection Module

Wraps the ultralytics YOLOv8 model for person detection.
Optimized for real-time surveillance applications.
Supports both PyTorch and ONNX Runtime (with CUDA EP) backends.
"""

from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import cv2
from loguru import logger

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. ONNX detector will not work.")

# Try to import ultralytics (for PyTorch backend)
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    torch = None
    ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed. PyTorch YOLOv8 detector will not work.")


@dataclass
class Detection:
    """Single detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0  # 0 = person in COCO
    
    @property
    def x1(self) -> float:
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Convert to (top-left-x, top-left-y, width, height) format."""
        return (self.x1, self.y1, self.width, self.height)
    
    def to_xyah(self) -> Tuple[float, float, float, float]:
        """Convert to (center-x, center-y, aspect-ratio, height) format."""
        cx, cy = self.center
        aspect_ratio = self.width / self.height if self.height > 0 else 1.0
        return (cx, cy, aspect_ratio, self.height)


class YOLOOnnxDetector:
    """
    ONNX Runtime based YOLOv8 detector with CUDA Execution Provider.
    
    Provides faster inference on NVIDIA GPUs compared to PyTorch backend.
    Requires ONNX model exported from ultralytics YOLOv8.
    """
    
    PERSON_CLASS_ID = 0  # COCO person class
    INPUT_SIZE = 640  # YOLOv8 default input size
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize ONNX YOLOv8 detector.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            confidence: Detection confidence threshold (0-1)
            iou_threshold: NMS IoU threshold (0-1)
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required for ONNX detector")
        
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Auto-detect device and set providers
        if device is None:
            # Check if CUDA EP is available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Configure providers based on device
        if device == 'cuda':
            self.providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'  # Fallback
            ]
        else:
            self.providers = ['CPUExecutionProvider']
        
        logger.info(f"Loading ONNX YOLO model from {model_path}")
        logger.info(f"Using providers: {[p if isinstance(p, str) else p[0] for p in self.providers]}")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] is not None:
            self.INPUT_SIZE = input_shape[2]
        
        # Warm up
        self._warmup()
        
        logger.info(f"ONNX YOLO detector initialized (confidence={confidence}, iou={iou_threshold}, device={device})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_input = np.zeros((1, 3, self.INPUT_SIZE, self.INPUT_SIZE), dtype=np.float32)
        _ = self.session.run(self.output_names, {self.input_name: dummy_input})
        logger.debug("ONNX model warmup complete")
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for YOLO inference.
        
        Returns:
            Tuple of (preprocessed tensor, scale ratio, padding offset)
        """
        h, w = frame.shape[:2]
        
        # Calculate scale to fit in INPUT_SIZE while maintaining aspect ratio
        scale = min(self.INPUT_SIZE / h, self.INPUT_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (letterbox)
        padded = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_h = (self.INPUT_SIZE - new_h) // 2
        pad_w = (self.INPUT_SIZE - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to float and normalize
        blob = padded.astype(np.float32) / 255.0
        
        # HWC to CHW
        blob = blob.transpose(2, 0, 1)
        
        # Add batch dimension
        blob = np.expand_dims(blob, axis=0)
        
        # Ensure contiguous
        blob = np.ascontiguousarray(blob)
        
        return blob, scale, (pad_w, pad_h)
    
    def _postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int],
    ) -> List[Detection]:
        """
        Postprocess YOLO outputs to detections.
        
        Args:
            outputs: Raw model outputs
            scale: Preprocessing scale ratio
            padding: Padding offset (pad_w, pad_h)
            original_shape: Original frame shape (h, w)
        """
        # YOLOv8 output shape: [1, 84, 8400] (for 80 classes)
        # Transpose to [8400, 84]
        predictions = outputs[0].T
        
        # Extract boxes and scores
        # First 4 values are cx, cy, w, h
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get person class scores only
        person_scores = scores[:, self.PERSON_CLASS_ID]
        
        # Filter by confidence
        mask = person_scores > self.confidence
        boxes = boxes[mask]
        person_scores = person_scores[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from center format to corner format
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Adjust for padding and scale
        pad_w, pad_h = padding
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # Clip to original image bounds
        orig_h, orig_w = original_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Stack for NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply NMS
        indices = self._nms(boxes_for_nms, person_scores, self.iou_threshold)
        
        # Create detections
        detections = []
        for i in indices:
            detections.append(Detection(
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                confidence=float(person_scores[i]),
                class_id=self.PERSON_CLASS_ID,
            ))
        
        return detections
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Non-maximum suppression using OpenCV."""
        # OpenCV NMS expects boxes as [x, y, w, h]
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        widths = x2 - x1
        heights = y2 - y1
        boxes_xywh = np.stack([x1, y1, widths, heights], axis=1).tolist()
        scores_list = scores.tolist()
        
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores_list,
            self.confidence,
            iou_threshold,
        )
        
        if len(indices) > 0:
            return indices.flatten().tolist()
        return []
    
    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            confidence: Override default confidence threshold
            
        Returns:
            List of Detection objects for detected persons
        """
        orig_conf = self.confidence
        if confidence is not None:
            self.confidence = confidence
        
        try:
            # Preprocess
            blob, scale, padding = self._preprocess(frame)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            
            # Postprocess
            detections = self._postprocess(
                outputs[0],
                scale,
                padding,
                frame.shape[:2],
            )
            
            return detections
        finally:
            self.confidence = orig_conf
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence: Optional[float] = None,
    ) -> List[List[Detection]]:
        """
        Detect persons in multiple frames.
        
        Note: For ONNX, we process frames sequentially as batch processing
        requires fixed input sizes.
        """
        return [self.detect(frame, confidence) for frame in frames]
    
    def crop_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        padding: float = 0.1,
    ) -> List[np.ndarray]:
        """Crop detected persons from frame."""
        h, w = frame.shape[:2]
        crops = []
        
        for det in detections:
            pad_w = det.width * padding
            pad_h = det.height * padding
            
            x1 = int(max(0, det.x1 - pad_w))
            y1 = int(max(0, det.y1 - pad_h))
            x2 = int(min(w, det.x2 + pad_w))
            y2 = int(min(h, det.y2 + pad_h))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops


class YOLODetector:
    """
    YOLOv8 based person detector using PyTorch/ultralytics.
    
    Uses ultralytics YOLOv8 for fast and accurate person detection.
    Only detects class 0 (person) from the COCO dataset.
    
    Attributes:
        model: The YOLOv8 model instance
        confidence: Detection confidence threshold
        iou_threshold: Non-max suppression IoU threshold
        device: Compute device ('cuda' or 'cpu')
    """
    
    PERSON_CLASS_ID = 0  # COCO person class
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to YOLOv8 weights file or model name
            confidence: Detection confidence threshold (0-1)
            iou_threshold: NMS IoU threshold (0-1)
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLOv8 detector")
        
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading YOLOv8 model from {model_path} on {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Warm up the model
        self._warmup()
        
        logger.info(f"YOLOv8 detector initialized (confidence={confidence}, iou={iou_threshold})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy_input,
            verbose=False,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
        )
        logger.debug("Model warmup complete")
    
    def detect(
        self,
        frame: np.ndarray,
        confidence: Optional[float] = None,
    ) -> List[Detection]:
        """
        Detect persons in a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, C) in BGR format
            confidence: Override default confidence threshold
            
        Returns:
            List of Detection objects for detected persons
        """
        conf = confidence if confidence is not None else self.confidence
        
        results = self.model.predict(
            frame,
            verbose=False,
            conf=conf,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf_score = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf_score,
                    class_id=class_id,
                ))
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence: Optional[float] = None,
    ) -> List[List[Detection]]:
        """
        Detect persons in multiple frames (batch inference).
        
        Args:
            frames: List of input frames
            confidence: Override default confidence threshold
            
        Returns:
            List of detection lists, one per frame
        """
        conf = confidence if confidence is not None else self.confidence
        
        results = self.model.predict(
            frames,
            verbose=False,
            conf=conf,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
        )
        
        all_detections = []
        for result in results:
            frame_detections = []
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf_score = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    frame_detections.append(Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf_score,
                        class_id=class_id,
                    ))
            all_detections.append(frame_detections)
        
        return all_detections
    
    def crop_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        padding: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Crop detected persons from frame.
        
        Args:
            frame: Original frame
            detections: List of detections
            padding: Padding ratio around bounding box
            
        Returns:
            List of cropped person images
        """
        h, w = frame.shape[:2]
        crops = []
        
        for det in detections:
            # Add padding
            pad_w = det.width * padding
            pad_h = det.height * padding
            
            x1 = int(max(0, det.x1 - pad_w))
            y1 = int(max(0, det.y1 - pad_h))
            x2 = int(min(w, det.x2 + pad_w))
            y2 = int(min(h, det.y2 + pad_h))
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        
        return crops


def export_yolo_to_onnx(
    pt_path: str = "yolov8n.pt",
    onnx_path: Optional[str] = None,
    opset: int = 17,
    simplify: bool = True,
) -> str:
    """
    Export YOLOv8 PyTorch model to ONNX format.
    
    Args:
        pt_path: Path to PyTorch model (.pt)
        onnx_path: Output ONNX path (default: same name with .onnx)
        opset: ONNX opset version
        simplify: Whether to simplify the model
    
    Returns:
        Path to exported ONNX model
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("ultralytics is required to export YOLO models")
    
    model = YOLO(pt_path)
    
    # Export with ultralytics built-in method
    export_path = model.export(
        format='onnx',
        opset=opset,
        simplify=simplify,
        dynamic=False,  # Fixed size for better optimization
        half=False,  # FP32 for compatibility
    )
    
    logger.info(f"Exported YOLO model to: {export_path}")
    return export_path


# Lazy singleton instance
_detector_instance: Optional[Union[YOLODetector, YOLOOnnxDetector]] = None


def get_detector(
    model_path: str = "yolov8n.pt",
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
    device: Optional[str] = None,
    use_onnx: bool = True,
) -> Union[YOLODetector, YOLOOnnxDetector]:
    """
    Get or create singleton detector instance.
    
    Automatically selects ONNX backend if:
    1. use_onnx is True
    2. Model path ends with .onnx
    3. ONNX Runtime is available
    
    Falls back to PyTorch backend otherwise.
    """
    global _detector_instance
    
    if _detector_instance is None:
        is_onnx_model = model_path.endswith('.onnx')
        
        # Prefer ONNX if available and requested
        if (use_onnx or is_onnx_model) and ONNX_AVAILABLE:
            if not is_onnx_model:
                # Try to find ONNX version of the model
                onnx_path = model_path.replace('.pt', '.onnx')
                import os
                if os.path.exists(onnx_path):
                    model_path = onnx_path
                    is_onnx_model = True
                else:
                    logger.warning(f"ONNX model not found at {onnx_path}, using PyTorch backend")
            
            if is_onnx_model:
                logger.info("Using ONNX Runtime detector with CUDA EP")
                _detector_instance = YOLOOnnxDetector(
                    model_path=model_path,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    device=device,
                )
                return _detector_instance
        
        # Fallback to PyTorch
        if ULTRALYTICS_AVAILABLE:
            logger.info("Using PyTorch/ultralytics detector")
            _detector_instance = YOLODetector(
                model_path=model_path,
                confidence=confidence,
                iou_threshold=iou_threshold,
                device=device,
            )
        else:
            raise ImportError("No YOLO backend available. Install ultralytics or onnxruntime-gpu.")
    
    return _detector_instance
