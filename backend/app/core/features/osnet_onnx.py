"""
ONNX-based OSNet Feature Extractor

Uses ONNX Runtime with TensorRT/CUDA execution providers for fast inference.
Falls back to CPU execution if GPU is not available.
"""

import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path
from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. ONNX-based OSNet will not work.")


class ONNXOSNetExtractor:
    """
    ONNX Runtime based OSNet feature extractor.
    
    Uses TensorRT or CUDA execution provider for GPU acceleration.
    Produces 512-dimensional L2-normalized embeddings.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512
    
    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_tensorrt: bool = False,  # Disabled by default - requires DLLs in PATH
    ):
        """
        Initialize ONNX OSNet extractor.
        
        Args:
            model_path: Path to ONNX model file
            device: 'cuda' or 'cpu'
            use_tensorrt: Try TensorRT provider (requires TensorRT DLLs in PATH)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNX OSNet extractor")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Configure execution providers
        providers = self._get_providers(device, use_tensorrt)
        
        logger.info(f"Loading ONNX OSNet from {model_path}")
        logger.info(f"Available providers: {ort.get_available_providers()}")
        logger.info(f"Requested providers: {[p if isinstance(p, str) else p[0] for p in providers]}")
        
        # Create inference session with fallback handling
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Try to create session, handle TensorRT failures gracefully
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            # TensorRT might fail if DLLs not in PATH - fallback to CUDA
            logger.warning(f"Failed with requested providers: {e}")
            logger.info("Retrying with CUDA-only providers...")
            
            cuda_providers = self._get_providers(device, use_tensorrt=False)
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=cuda_providers,
            )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Log which provider is actually being used
        active_provider = self.session.get_providers()[0]
        logger.info(f"ONNX OSNet initialized with provider: {active_provider}")
        
        # Warn if fell back to CPU
        if active_provider == "CPUExecutionProvider" and device == "cuda":
            logger.warning("⚠️ ONNX OSNet running on CPU! Check CUDA installation.")
    
    def _get_providers(self, device: str, use_tensorrt: bool) -> List:
        """Get execution providers based on device preference."""
        providers = []
        
        if device == "cuda":
            # TensorRT EP (optional - requires TensorRT DLLs in PATH)
            if use_tensorrt and "TensorrtExecutionProvider" in ort.get_available_providers():
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2GB
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "model_weights/trt_cache",
                    }
                ))
            
            # CUDA EP (reliable, works out of the box)
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append((
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB limit
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                    }
                ))
        
        # Always add CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def _preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess images for ONNX inference.
        
        Args:
            images: List of BGR images (H, W, C)
            
        Returns:
            Preprocessed batch as (N, 3, H, W) float32 array
        """
        batch = []
        
        for img in images:
            # Resize to input size
            img = cv2.resize(img, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
            
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # HWC to CHW
            img = img.transpose(2, 0, 1)
            
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            batch.append(img)
        
        batch = np.stack(batch, axis=0)
        
        # Apply ImageNet normalization
        batch = (batch - self.MEAN) / self.STD
        
        return batch.astype(np.float32)
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract feature embedding from a single image.
        
        Args:
            image: Person crop as numpy array (H, W, C) in BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array (512,)
        """
        return self.extract_batch([image], normalize=normalize)[0]
    
    def extract_batch(
        self,
        images: List[np.ndarray],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract features from multiple images.
        
        Args:
            images: List of person crops (BGR)
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Features as 2D numpy array (N, 512)
        """
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        
        # Preprocess
        batch = self._preprocess(images)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch}
        )
        
        features = outputs[0]
        
        # L2 normalize
        if normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            features = features / norms
        
        return features.astype(np.float32)
    
    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and gallery features."""
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        return np.dot(query_features, gallery_features.T)


def get_onnx_osnet_extractor(
    model_path: str = "model_weights/osnet_x1_0.onnx",
    device: str = "cuda",
    use_tensorrt: bool = True,
) -> Optional[ONNXOSNetExtractor]:
    """
    Get ONNX OSNet extractor if model exists.
    
    Returns None if model file doesn't exist or ONNX runtime unavailable.
    """
    if not ONNX_AVAILABLE:
        return None
    
    model_path = Path(model_path)
    if not model_path.exists():
        logger.debug(f"ONNX model not found at {model_path}")
        return None
    
    try:
        return ONNXOSNetExtractor(
            model_path=str(model_path),
            device=device,
            use_tensorrt=use_tensorrt,
        )
    except Exception as e:
        logger.error(f"Failed to load ONNX OSNet: {e}")
        return None
