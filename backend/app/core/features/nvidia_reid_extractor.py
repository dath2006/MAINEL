"""
NVIDIA TAO ReID Feature Extractor

Uses NVIDIA TAO Toolkit's ResNet50 ReIdentificationNet (ONNX) for person re-identification.
Produces 256-dimensional embeddings.

Model: resnet50_market1501_aicity156.onnx
Input: 256x128 (NCHW, RGB, ImageNet normalized)
"""

import os
from typing import List, Optional, Tuple, Union
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
    logger.warning("onnxruntime not installed. NVIDIA ReID extractor will not work.")


class NvidiaReIDExtractor:
    """
    NVIDIA TAO ReID feature extractor using ONNX Runtime.
    
    Optimized for GPU inference with CUDA Execution Provider.
    Outputs 256-dimensional L2-normalized embeddings.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 256
    
    # ImageNet normalization values
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        gpu_mem_limit: int = 4,  # GB
    ):
        """
        Initialize NVIDIA ReID extractor.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            device: Compute device ('cuda', 'cpu', or None for auto)
            gpu_mem_limit: GPU memory limit in GB (for CUDA EP)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required for NVIDIA ReID extractor")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"NVIDIA ReID model not found at: {model_path}")
            
        # Auto-detect device
        if device is None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                device = 'cuda'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Configure providers
        if device == 'cuda':
            self.providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': gpu_mem_limit * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'
            ]
        else:
            self.providers = ['CPUExecutionProvider']
        
        logger.info(f"Loading NVIDIA TAO ReID model from {model_path}")
        logger.info(f"Using providers: {[p if isinstance(p, str) else p[0] for p in self.providers]}")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers
            )
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            raise
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Get input shape
        self.input_shape = self.session.get_inputs()[0].shape
        logger.debug(f"Model input shape: {self.input_shape}")
        
        self.dist_metric = 'cosine'
        
        # Warm up
        self._warmup()
        
        logger.info(f"NVIDIA ReID extractor initialized (embedding_dim={self.EMBEDDING_DIM}, device={device})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_input = np.zeros(
            (1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]), 
            dtype=np.float32
        )
        try:
            _ = self.session.run(self.output_names, {self.input_name: dummy_input})
            logger.debug("NVIDIA ReID warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Steps:
        1. Resize to 256x128 (height x width)
        2. Convert BGR to RGB
        3. Normalize to [0, 1]
        4. Apply ImageNet normalization (mean/std)
        5. Convert to CHW format
        """
        # Resize to input size (height, width) -> cv2 uses (width, height)
        # Note: Model expects (256, 128)
        image = cv2.resize(image, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.MEAN) / self.STD
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        return image.astype(np.float32)
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract feature embedding from a single person image.
        
        Args:
            image: Person crop as numpy array (H, W, C) in BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array of shape (256,)
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image input")

        # Preprocess
        tensor = self._preprocess(image)
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension
        tensor = np.ascontiguousarray(tensor)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        features = outputs[0].flatten()
        
        # L2 normalize
        if normalize:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract features from multiple person images.
        
        Args:
            images: List of person crops
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for inference
            
        Returns:
            Features as 2D numpy array of shape (N, 256)
        """
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM))
        
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            valid_images = [img for img in batch_images if img is not None and img.size > 0]
            if not valid_images:
                continue
                
            tensors = [self._preprocess(img) for img in valid_images]
            batch = np.stack(tensors, axis=0)
            batch = np.ascontiguousarray(batch)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: batch})
            features = outputs[0]
            
            # Ensure 2D shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # L2 normalize
            if normalize:
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                features = features / norms
            
            all_features.append(features)
        
        if not all_features:
             return np.empty((0, self.EMBEDDING_DIM))
             
        return np.vstack(all_features)
    
    def compute_distance(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = 'cosine',
    ) -> np.ndarray:
        """Compute distance matrix between query and gallery features."""
        if metric == 'cosine':
            return 1 - np.dot(query_features, gallery_features.T)
        elif metric == 'euclidean':
            m = query_features.shape[0]
            n = gallery_features.shape[0]
            dist = np.zeros((m, n))
            for i in range(m):
                dist[i] = np.sum((gallery_features - query_features[i]) ** 2, axis=1)
            return np.sqrt(dist)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and gallery features."""
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        return np.dot(query_features, gallery_features.T)


# Singleton instance
_nvidia_extractor = None

def get_nvidia_extractor(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> NvidiaReIDExtractor:
    """Get or create singleton NVIDIA ReID extractor."""
    global _nvidia_extractor
    
    if _nvidia_extractor is not None:
        return _nvidia_extractor
        
    if not model_path:
        # Fallback to default relative path
        model_path = "model_weights/resnet50_market1501_aicity156.onnx"
    
    _nvidia_extractor = NvidiaReIDExtractor(
        model_path=model_path,
        device=device,
    )
    return _nvidia_extractor
