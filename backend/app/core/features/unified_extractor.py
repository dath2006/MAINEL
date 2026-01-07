import numpy as np
import torch
import cv2
from typing import List, Optional
from pathlib import Path
from insightface.app import FaceAnalysis
# Correct BoxMOT imports
from boxmot.reid.backbones.osnet import osnet_x1_0
from loguru import logger

# Import ONNX OSNet extractor
try:
    from app.core.features.osnet_onnx import ONNXOSNetExtractor, get_onnx_osnet_extractor
    ONNX_OSNET_AVAILABLE = True
except ImportError:
    ONNX_OSNET_AVAILABLE = False

class UnifiedFeatureExtractor:
    def __init__(
        self,
        model_name: str = 'osnet_x1_0',
        device: str = 'cuda',
        face_det_name: str = 'buffalo_l',
        verbose: bool = False,
        use_onnx: bool = True,
        onnx_model_path: str = 'model_weights/osnet_x1_0.onnx',
    ):
        self.device = device
        self.verbose = verbose
        self.use_onnx = use_onnx
        self.onnx_extractor = None
        
        # 1. Initialize Body ReID - Try ONNX first, fallback to PyTorch
        if use_onnx and ONNX_OSNET_AVAILABLE:
            self.onnx_extractor = get_onnx_osnet_extractor(
                model_path=onnx_model_path,
                device=device,
                use_tensorrt=False,  # Disabled - TensorRT DLLs not in PATH, use CUDA EP
            )
            if self.onnx_extractor:
                logger.info("Using ONNX Runtime for OSNet body ReID (optimized)")
            else:
                logger.info("ONNX model not found, falling back to PyTorch OSNet")
        
        # PyTorch fallback (or primary if ONNX disabled)
        if self.onnx_extractor is None:
            try:
                 # Load OSNet model directly
                 self.body_model = osnet_x1_0(
                    num_classes=1000, # Default
                    loss='softmax',
                    pretrained=True
                )
                 self.body_model.eval()
                 self.body_model.to(device)
                 logger.info("Using PyTorch for OSNet body ReID")
            except Exception as e:
                # Fallback or error handling if imports fail (BoxMOT structure might vary)
                print(f"Error loading BoxMOT model: {e}")
                raise e
        else:
            self.body_model = None  # Not needed when using ONNX

        # 2. Initialize Face ReID (InsightFace) - Already uses ONNX Runtime
        self.face_app = FaceAnalysis(name=face_det_name, providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        
        # Define embedding sizes
        self.body_dim = 512 # osnet_x1_0
        self.face_dim = 512 # insightface buffalo_l arcface
        self.total_dim = self.body_dim + self.face_dim


    def _preprocess_body(self, crops: List[np.ndarray]) -> torch.Tensor:
        # Standard ReID transforms (Resize, Normalize, ToTensor)
        # Simplified for brevity; usually entails 256x128 resize
        processed = []
        for crop in crops:
            img = cv2.resize(crop, (128, 256))
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1) # HWC -> CHW
            # Normalize (ImageNet mean/std commonly used)
            img = (img - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / \
                  np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            processed.append(img)
        
        tensor = torch.tensor(np.stack(processed)).float().to(self.device)
        return tensor

    def extract(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Extract combined embeddings for detections.
        detections: (N, 4) [x1, y1, x2, y2]
        Returns: (N, total_dim)
        """
        embeddings, _ = self.extract_with_faces(image, detections)
        return embeddings
    
    def extract_with_faces(self, image: np.ndarray, detections: np.ndarray) -> tuple:
        """
        Extract combined embeddings AND face bounding box info for detections.
        
        Args:
            image: Full frame (H, W, C)
            detections: (N, 4) [x1, y1, x2, y2] body bounding boxes
            
        Returns:
            tuple: (embeddings, face_info_list)
                - embeddings: (N, total_dim) combined body+face embeddings
                - face_info_list: List of dicts with 'bbox' in absolute frame coords, or None if no face
        """
        if len(detections) == 0:
            return np.empty((0, self.total_dim)), []
            
        crops = []
        face_feats = []
        face_info_list = []  # Store face detection info
        
        h, w, _ = image.shape
        
        for det in detections:
            body_x1, body_y1, body_x2, body_y2 = map(int, det[:4])
            body_x1 = max(0, body_x1)
            body_y1 = max(0, body_y1)
            body_x2 = min(w, body_x2)
            body_y2 = min(h, body_y2)
            
            # Crop body
            crop = image[body_y1:body_y2, body_x1:body_x2]
            if crop.size == 0:
                # Handle edge case of empty crop
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8)) 
                face_feats.append(np.zeros(self.face_dim))
                face_info_list.append(None)
                continue
                
            crops.append(crop)
            
            # Extract Face
            # Run face detection on the body crop for speed and direct association
            try:
                faces = self.face_app.get(crop)
                if len(faces) > 0:
                    # Pick largest face
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    face_feats.append(best_face.normed_embedding)
                    
                    # Convert face bbox from crop-relative to absolute frame coords
                    # InsightFace returns bbox as [x1, y1, x2, y2]
                    fx1, fy1, fx2, fy2 = best_face.bbox
                    abs_fx1 = body_x1 + int(fx1)
                    abs_fy1 = body_y1 + int(fy1)
                    abs_fx2 = body_x1 + int(fx2)
                    abs_fy2 = body_y1 + int(fy2)
                    
                    face_info_list.append({
                        'bbox': [abs_fx1, abs_fy1, abs_fx2, abs_fy2],
                        'det_score': float(best_face.det_score) if hasattr(best_face, 'det_score') else 0.0
                    })
                else:
                    face_feats.append(np.zeros(self.face_dim))
                    face_info_list.append(None)
            except Exception as e:
                face_feats.append(np.zeros(self.face_dim))
                face_info_list.append(None)
                logger.debug(f"Face extraction error: {e}")

        # Body Embeddings
        if not crops:
             return np.empty((0, self.total_dim)), []

        # Use ONNX extractor if available, else PyTorch
        if self.onnx_extractor is not None:
            body_feats = self.onnx_extractor.extract_batch(crops, normalize=True)
        else:
            batch_tensor = self._preprocess_body(crops)
            with torch.no_grad():
                body_feats = self.body_model(batch_tensor)
                # Normalize
                body_feats = torch.nn.functional.normalize(body_feats, p=2, dim=1)
                body_feats = body_feats.cpu().numpy()
            
        # Combine body + face embeddings
        face_feats = np.stack(face_feats)
        combined = np.concatenate([body_feats, face_feats], axis=1)
        
        # Log embedding merge for debugging
        if self.verbose:
            logger.info(f"UnifiedExtractor: Merged body({self.body_dim}) + face({self.face_dim}) -> {combined.shape[1]}-dim embedding")
        
        # Normalize combined for cosine similarity
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        combined = combined / norms
        
        return combined, face_info_list
