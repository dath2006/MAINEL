"""
YOLOv8 Person Detection Module

Wraps the ultralytics YOLOv8 model for person detection.
Optimized for real-time surveillance applications.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logger.warning("ultralytics not installed. YOLOv8 detector will not work.")


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


class YOLODetector:
    """
    YOLOv8 based person detector.
    
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
        if YOLO is None:
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


# Lazy singleton instance
_detector_instance: Optional[YOLODetector] = None


def get_detector(
    model_path: str = "yolov8n.pt",
    confidence: float = 0.5,
    iou_threshold: float = 0.45,
    device: Optional[str] = None,
) -> YOLODetector:
    """Get or create singleton detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLODetector(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            device=device,
        )
    return _detector_instance
