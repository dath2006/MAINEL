"""
InsightFace Feature Extractor

Face detection and embedding extraction using InsightFace with ONNX acceleration.
Provides 512-dimensional face embeddings for person re-identification.
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2
from loguru import logger

from app.config import settings

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not installed. Face recognition will not work.")


@dataclass
class FaceResult:
    """Face detection and embedding result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # 512-dim face embedding
    confidence: float  # Detection confidence
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get face center point."""
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


class InsightFaceExtractor:
    """
    InsightFace-based face detector and feature extractor.
    
    Uses Buffalo_L model for high accuracy face detection and recognition.
    Supports ONNX Runtime with CUDA Execution Provider for GPU acceleration.
    """
    
    EMBEDDING_DIM = 512
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = None,
        device: Optional[str] = None,
    ):
        """
        Initialize InsightFace extractor.
        
        Args:
            model_name: Model pack name (buffalo_l, buffalo_s, etc.)
            det_size: Detection input size
            det_thresh: Detection confidence threshold (from config if None)
            device: Device ('cuda' or 'cpu', None for auto)
        """
        # Use config value if not provided
        det_thresh = det_thresh if det_thresh is not None else settings.face_det_threshold
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface package is required for face extraction")
        
        # Determine providers
        if device is None:
            try:
                import onnxruntime as ort
                if 'CUDAExecutionProvider' in ort.get_available_providers():
                    device = 'cuda'
                else:
                    device = 'cpu'
            except:
                device = 'cpu'
        
        self.device = device
        
        # Set up providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        
        logger.info(f"Loading InsightFace model '{model_name}' with providers: {providers}")
        
        self.app = FaceAnalysis(
            name=model_name,
            root='~/.insightface',
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        
        self.det_thresh = det_thresh
        
        logger.info(f"InsightFace extractor initialized (device={device})")
    
    def detect_faces(self, image: np.ndarray) -> List[FaceResult]:
        """
        Detect faces and extract embeddings from image.
        
        Args:
            image: BGR image as numpy array
            
        Returns:
            List of FaceResult with bbox, embedding, confidence
        """
        # InsightFace expects BGR
        faces = self.app.get(image)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Clamp to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            results.append(FaceResult(
                bbox=(x1, y1, x2, y2),
                embedding=face.embedding,
                confidence=float(face.det_score),
                landmarks=face.kps if hasattr(face, 'kps') else None,
            ))
        
        return results
    
    def extract_from_person_crop(
        self,
        crop: np.ndarray,
        return_largest: bool = True,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
        """
        Extract face embedding from a person crop.
        
        Args:
            crop: Person crop (body bounding box region)
            return_largest: Return only the largest face if multiple detected
            
        Returns:
            Tuple of (face_embedding, face_bbox_in_crop, confidence) or (None, None, 0.0)
        """
        faces = self.detect_faces(crop)
        
        if not faces:
            return None, None, 0.0
        
        if return_largest:
            # Return largest face (most likely the main person)
            largest = max(faces, key=lambda f: f.area)
            return largest.embedding, largest.bbox, largest.confidence
        
        # Return first (highest confidence typically)
        return faces[0].embedding, faces[0].bbox, faces[0].confidence
    
    def extract_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]]:
        """
        Extract face embeddings from multiple images.
        
        Returns:
            List of (embedding, bbox, confidence) tuples
        """
        results = []
        for img in images:
            emb, bbox, conf = self.extract_from_person_crop(img)
            results.append((emb, bbox, conf))
        return results
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two face embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        return 0.0


class QualityScorer:
    """
    Image quality scorer for thumbnail selection.
    
    Scores images based on sharpness, face visibility, and body completeness.
    """
    
    def __init__(
        self,
        sharpness_weight: float = 0.3,
        face_weight: float = 0.4,
        size_weight: float = 0.3,
    ):
        self.sharpness_weight = sharpness_weight
        self.face_weight = face_weight
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
            face_bbox: Face bounding box if detected (x1, y1, x2, y2)
            face_confidence: Face detection confidence
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        h, w = image.shape[:2]
        
        # 1. Sharpness score (Laplacian variance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize: 100 is decent, 500+ is very sharp
        sharpness = min(1.0, laplacian_var / 200.0)
        
        # 2. Face visibility score
        if face_bbox is not None and face_confidence > 0:
            fx1, fy1, fx2, fy2 = face_bbox
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            
            # Face size relative to image (ideal: face is ~20-40% of height)
            face_ratio = face_h / h
            if face_ratio < 0.1:
                face_size_score = face_ratio / 0.1 * 0.5
            elif face_ratio < 0.5:
                face_size_score = 1.0
            else:
                face_size_score = 0.8
            
            # Combine with confidence
            face_score = face_confidence * face_size_score
        else:
            face_score = 0.0
        
        # 3. Image size score (larger is better for thumbnails)
        # Normalize: 128x256 is baseline, larger is better up to 256x512
        size_score = min(1.0, (h * w) / (256 * 512))
        
        # Combine scores
        total = (
            self.sharpness_weight * sharpness +
            self.face_weight * face_score +
            self.size_weight * size_score
        )
        
        return min(1.0, max(0.0, total))


def create_fused_embedding(
    body_embedding: np.ndarray,
    face_embedding: Optional[np.ndarray],
    face_weight: float = None,
    body_weight: float = None,
    face_quality: float = 1.0,
) -> np.ndarray:
    """
    Fuse face and body embeddings with Gated Dynamic Fusion.
    
    Uses quality-tier gating to prevent noise from low-quality faces
    polluting the embedding. Ensures proper L2 normalization at
    every step (pre-fusion AND post-fusion) for correct cosine similarity.
    
    Quality Tiers (configurable via settings):
    - HIGH (>face_quality_high_tier): Trust face heavily
    - MEDIUM (face_quality_low_tier to high_tier): Trust body more
    - LOW (<face_quality_low_tier): Ignore face completely (body only)
    
    Args:
        body_embedding: 512-dim body embedding from OSNet
        face_embedding: 512-dim face embedding from InsightFace (or None)
        face_weight: Base weight for face (default from config)
        body_weight: Base weight for body (default from config)
        face_quality: Quality score of the face (0.0 - 1.0)
        
    Returns:
        Fused 512-dim embedding (L2-normalized)
    """
    # === STEP 1: Pre-normalize body embedding (CRITICAL) ===
    body_norm = np.linalg.norm(body_embedding)
    if body_norm > 0:
        body_embedding = body_embedding / body_norm
    
    # === STEP 2: Gated Fusion based on Quality Tiers ===
    low_tier = settings.face_quality_low_tier
    high_tier = settings.face_quality_high_tier
    
    # TIER 3: LOW quality (<low_tier) or no face -> Body Only
    if face_embedding is None or face_quality < low_tier:
        # No face or garbage face - use body only
        return body_embedding  # Already normalized
    
    # Pre-normalize face embedding
    face_norm = np.linalg.norm(face_embedding)
    if face_norm > 0:
        face_embedding = face_embedding / face_norm
    
    # TIER 1: HIGH quality (>high_tier) -> Trust face heavily
    if face_quality > high_tier:
        high_face_weight = settings.face_high_tier_weight
        fused = high_face_weight * face_embedding + (1 - high_face_weight) * body_embedding
    
    # TIER 2: MEDIUM quality (low_tier to high_tier) -> Trust body more
    else:
        medium_face_weight = settings.face_medium_tier_weight
        fused = medium_face_weight * face_embedding + (1 - medium_face_weight) * body_embedding
    
    # === STEP 3: Post-normalize the fused result (CRITICAL) ===
    # This step is often forgotten and breaks cosine similarity!
    fused_norm = np.linalg.norm(fused)
    if fused_norm > 0:
        fused = fused / fused_norm
    
    return fused


# Singleton instance
_face_extractor: Optional[InsightFaceExtractor] = None


def get_face_extractor(
    model_name: str = "buffalo_l",
    device: Optional[str] = None,
) -> InsightFaceExtractor:
    """Get or create singleton face extractor."""
    global _face_extractor
    
    if _face_extractor is None:
        _face_extractor = InsightFaceExtractor(
            model_name=model_name,
            device=device,
        )
    
    return _face_extractor


# Quality scorer singleton
_quality_scorer: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    """Get or create singleton quality scorer."""
    global _quality_scorer
    
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    
    return _quality_scorer
