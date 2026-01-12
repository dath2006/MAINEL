"""
FastReID Feature Extractor

SOTA person re-identification using FastReID with SBS(R50-ibn) backbone.
Produces 2048-dimensional embeddings for cross-camera matching.

Reference: https://github.com/JDAI-CV/fast-reid
Model: SBS(R50-ibn) - 95.7% Rank-1, 89.3% mAP on Market-1501
"""

import os
import sys
from typing import List, Optional
import numpy as np
import cv2
from loguru import logger

# Add fastreid_lib to path
FASTREID_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fastreid_lib')
if FASTREID_PATH not in sys.path:
    sys.path.insert(0, os.path.abspath(FASTREID_PATH))

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. FastReID extractor will not work.")

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. ONNX FastReID extractor will not work.")

# Try to import FastReID components
FASTREID_AVAILABLE = False
try:
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer
    FASTREID_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FastReID not available: {e}")


class FastReIDExtractor:
    """
    PyTorch-based FastReID feature extractor.
    
    Uses SBS(R50-ibn) model for SOTA person re-identification.
    Outputs 2048-dimensional L2-normalized embeddings.
    """
    
    EMBEDDING_DIM = 2048
    
    # NOTE: Input size and normalization are read from config file
    # Default fallback values (will be overwritten from config):
    DEFAULT_INPUT_SIZE = (384, 128)  # (height, width) from Market1501 config
    
    # FastReID uses 0-255 range normalization, not 0-1
    
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: Optional[str] = None,
    ):
        """
        Initialize FastReID extractor.
        
        Args:
            config_path: Path to FastReID config YAML
            weights_path: Path to pretrained weights (.pth)
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        if not FASTREID_AVAILABLE:
            raise ImportError("FastReID is required. Please check fastreid_lib installation.")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FastReID extractor.")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading FastReID model from {weights_path}")
        logger.info(f"Config: {config_path}")
        
        # Build config and get input size / normalization params
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.BACKBONE.PRETRAIN = False  # We're loading full weights
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = self.device
        
        # Extract preprocessing params from config
        self.input_size = tuple(cfg.INPUT.SIZE_TEST)  # (height, width)
        self.pixel_mean = np.array(cfg.MODEL.PIXEL_MEAN, dtype=np.float32)
        self.pixel_std = np.array(cfg.MODEL.PIXEL_STD, dtype=np.float32)
        
        logger.info(f"Input size from config: {self.input_size}")
        logger.info(f"PIXEL_MEAN: {self.pixel_mean}, PIXEL_STD: {self.pixel_std}")
        
        # Build and load model
        self.model = build_model(cfg)
        self.model.eval()
        
        Checkpointer(self.model).load(weights_path)
        
        self.cfg = cfg
        self.dist_metric = 'cosine'
        
        # Warm up
        self._warmup()
        
        logger.info(f"FastReID extractor initialized (embedding_dim={self.EMBEDDING_DIM}, device={self.device})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        h, w = self.input_size
        dummy_input = torch.zeros(1, 3, h, w).to(self.device)
        with torch.no_grad():
            _ = self.model({"images": dummy_input})
        logger.debug("FastReID warmup complete")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.
        
        NOTE: FastReID model normalizes internally using pixel_mean/pixel_std
        registered as buffers. We should NOT normalize here - just resize and 
        convert to CHW format with values in 0-255 range.
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to input size (height, width) -> cv2 uses (width, height)
        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Convert to float32 (keep 0-255 range for model's internal normalization)
        image = image.astype(np.float32)
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        return image
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract feature embedding from a single person image.
        
        Args:
            image: Person crop as numpy array (H, W, C) in BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array of shape (2048,)
        """
        # Preprocess
        tensor = self._preprocess(image)
        tensor = torch.as_tensor(tensor[None]).to(self.device)
        
        # Run inference
        with torch.no_grad():
            features = self.model({"images": tensor})
        
        features = features.cpu().numpy().flatten()
        
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
        batch_size: int = 16,  # Conservative for RTX 4050 6GB
    ) -> np.ndarray:
        """
        Extract features from multiple person images.
        
        Args:
            images: List of person crops
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for inference
            
        Returns:
            Features as 2D numpy array of shape (N, 2048)
        """
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM))
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Preprocess batch
                tensors = [self._preprocess(img) for img in batch_images]
                batch = torch.as_tensor(np.stack(tensors, axis=0)).to(self.device)
                
                # Run inference
                features = self.model({"images": batch})
                features = features.cpu().numpy()
                
                # L2 normalize
                if normalize:
                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    features = features / norms
                
                all_features.append(features)
        
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


class FastReIDOnnxExtractor:
    """
    ONNX Runtime-based FastReID feature extractor with CUDA EP.
    
    Optimized for production inference. Requires pre-exported ONNX model.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 2048
    
    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        onnx_path: str,
        device: Optional[str] = None,
        gpu_mem_limit: int = 4,  # GB, safe for RTX 4050 6GB
    ):
        """
        Initialize ONNX FastReID extractor.
        
        Args:
            onnx_path: Path to ONNX model file
            device: Compute device ('cuda', 'cpu', or None for auto)
            gpu_mem_limit: GPU memory limit in GB
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required for ONNX extractor")
        
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
        
        logger.info(f"Loading ONNX FastReID model from {onnx_path}")
        logger.info(f"Using providers: {[p if isinstance(p, str) else p[0] for p in self.providers]}")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=self.providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        self.dist_metric = 'cosine'
        
        # Warm up
        self._warmup()
        
        logger.info(f"ONNX FastReID extractor initialized (embedding_dim={self.EMBEDDING_DIM}, device={device})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_input = np.zeros((1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]), dtype=np.float32)
        _ = self.session.run(self.output_names, {self.input_name: dummy_input})
        logger.debug("ONNX FastReID warmup complete")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]), interpolation=cv2.INTER_CUBIC)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.MEAN) / self.STD
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        return image.astype(np.float32)
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Extract feature embedding from a single person image."""
        # Preprocess
        tensor = self._preprocess(image)
        tensor = np.expand_dims(tensor, axis=0)
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
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract features from multiple person images."""
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM))
        
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            tensors = [self._preprocess(img) for img in batch_images]
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


# Default paths - using BoT R50-ibn which has better generalization than SBS
DEFAULT_CONFIG = "fastreid_lib/configs/Market1501/bagtricks_R50-ibn.yml"
DEFAULT_WEIGHTS = "model_weights/market_bot_R50-ibn.pth"
DEFAULT_ONNX = "model_weights/fastreid_bot_R50-ibn.onnx"

# Singleton instance
_fastreid_extractor = None


def get_fastreid_extractor(
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    device: Optional[str] = None,
    use_onnx: bool = False,
):
    """
    Get or create singleton FastReID extractor.
    
    Prefers ONNX backend if use_onnx=True and model exists.
    Falls back to PyTorch backend.
    """
    global _fastreid_extractor
    
    if _fastreid_extractor is not None:
        return _fastreid_extractor
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Check for ONNX model
    if use_onnx:
        onnx_path = onnx_path or os.path.join(base_dir, DEFAULT_ONNX)
        if os.path.exists(onnx_path) and ONNX_AVAILABLE:
            logger.info("Using ONNX FastReID extractor")
            _fastreid_extractor = FastReIDOnnxExtractor(
                onnx_path=onnx_path,
                device=device,
            )
            return _fastreid_extractor
    
    # Fall back to PyTorch
    if FASTREID_AVAILABLE:
        config_path = config_path or os.path.join(base_dir, DEFAULT_CONFIG)
        weights_path = weights_path or os.path.join(base_dir, DEFAULT_WEIGHTS)
        
        if os.path.exists(weights_path):
            logger.info("Using PyTorch FastReID extractor")
            _fastreid_extractor = FastReIDExtractor(
                config_path=config_path,
                weights_path=weights_path,
                device=device,
            )
            return _fastreid_extractor
    
    raise RuntimeError(
        "No FastReID model available. Please ensure:\n"
        f"1. PyTorch weights exist at: {weights_path}\n"
        f"2. Config exists at: {config_path}\n"
        "Or provide ONNX model path."
    )
