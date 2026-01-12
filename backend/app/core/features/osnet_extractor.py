"""
OSNet Feature Extractor

Uses TorchReID's OSNet model for person re-identification feature extraction.
Produces 512-dimensional embeddings for visual matching.
Supports both PyTorch and ONNX Runtime (with CUDA EP) backends.

Falls back to ResNet18 if torchreid is not installed.
"""

from typing import List, Optional
import numpy as np
import cv2
from PIL import Image
from loguru import logger

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not installed. ONNX extractor will not work.")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    transforms = None
    models = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. PyTorch extractors will not work.")

# Try to import torchreid
try:
    import torchreid
    from torchreid.data.transforms import build_transforms
    TORCHREID_AVAILABLE = True
except ImportError:
    torchreid = None
    build_transforms = None
    TORCHREID_AVAILABLE = False
    logger.warning("torchreid not installed. Using fallback for feature extraction.")
    logger.info("To install torchreid: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")


class OSNetOnnxExtractor:
    """
    ONNX Runtime based OSNet feature extractor with CUDA Execution Provider.
    
    Provides faster inference on NVIDIA GPUs compared to PyTorch backend.
    Requires ONNX model exported from torchreid OSNet.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512
    
    # ImageNet normalization values
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        """
        Initialize ONNX OSNet extractor.
        
        Args:
            model_path: Path to ONNX model file (.onnx)
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is required for ONNX extractor")
        
        # Auto-detect device and set providers
        if device is None:
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
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'  # Fallback
            ]
        else:
            self.providers = ['CPUExecutionProvider']
        
        logger.info(f"Loading ONNX OSNet model from {model_path}")
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
        
        self.dist_metric = 'cosine'
        
        # Warm up
        self._warmup()
        
        logger.info(f"ONNX OSNet extractor initialized (embedding_dim={self.EMBEDDING_DIM}, device={device})")
    
    def _warmup(self):
        """Warm up the model with a dummy inference."""
        dummy_input = np.zeros((1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]), dtype=np.float32)
        _ = self.session.run(self.output_names, {self.input_name: dummy_input})
        logger.debug("ONNX OSNet warmup complete")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to input size (height, width)
        image = cv2.resize(image, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        
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
            image: Person crop as numpy array (H, W, C) in RGB or BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array of shape (512,)
        """
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
            Features as 2D numpy array of shape (N, 512)
        """
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


class OSNetExtractor:
    """
    OSNet-based feature extractor for person re-identification.
    
    Uses Omni-Scale Network (OSNet) architecture which captures
    multi-scale features for robust person representation.
    
    Outputs:
        512-dimensional L2-normalized feature vectors
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512
    
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained: bool = True,
    ):
        """
        Initialize OSNet feature extractor.
        
        Args:
            model_name: OSNet variant ('osnet_x1_0', 'osnet_x0_75', etc.)
            model_path: Path to custom weights (optional)
            device: Compute device ('cuda', 'cpu', or None for auto)
            pretrained: Use pretrained weights if no custom path provided
        """
        if not TORCHREID_AVAILABLE:
            raise ImportError("torchreid package is required for OSNet extractor")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading OSNet model '{model_name}' on {self.device}")
        
        # Build model
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,  # Not used for feature extraction
            loss='softmax',
            pretrained=pretrained,
            use_gpu=(self.device == 'cuda'),
        )
        
        # Load custom weights if provided
        if model_path:
            logger.info(f"Loading custom weights from {model_path}")
            torchreid.utils.load_pretrained_weights(self.model, model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Build transforms (test-time transforms only)
        _, self.transform = build_transforms(
            height=self.INPUT_SIZE[0],
            width=self.INPUT_SIZE[1],
            random_erase=False,
            color_jitter=False,
            color_aug=False,
        )
        
        self.dist_metric = 'cosine'
        logger.info(f"OSNet extractor initialized (embedding_dim={self.EMBEDDING_DIM})")
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract feature embedding from a single person image.
        
        Args:
            image: Person crop as numpy array (H, W, C) in RGB or BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array of shape (512,)
        """
        with torch.no_grad():
            # Preprocess
            tensor = self._preprocess(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.model(tensor)
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
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract features from multiple person images.
        
        Args:
            images: List of person crops
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for inference
            
        Returns:
            Features as 2D numpy array of shape (N, 512)
        """
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM))
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Preprocess batch
                tensors = [self._preprocess(img) for img in batch_images]
                batch = torch.stack(tensors).to(self.device)
                
                # Extract features
                features = self.model(batch)
                features = features.cpu().numpy()
                
                # L2 normalize
                if normalize:
                    norms = np.linalg.norm(features, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)  # Avoid division by zero
                    features = features / norms
                
                all_features.append(features)
        
        return np.vstack(all_features)
    
    def compute_distance(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
        metric: str = 'cosine',
    ) -> np.ndarray:
        """
        Compute distance matrix between query and gallery features.
        
        Args:
            query_features: Query embeddings of shape (M, 512)
            gallery_features: Gallery embeddings of shape (N, 512)
            metric: Distance metric ('cosine' or 'euclidean')
            
        Returns:
            Distance matrix of shape (M, N)
        """
        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarity = np.dot(query_features, gallery_features.T)
            return 1 - similarity
        elif metric == 'euclidean':
            # Squared Euclidean distance
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
        """
        Compute cosine similarity between query and gallery features.
        
        Args:
            query_features: Query embeddings of shape (M, 512) or (512,)
            gallery_features: Gallery embeddings of shape (N, 512)
            
        Returns:
            Similarity matrix of shape (M, N) or (N,)
        """
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        return np.dot(query_features, gallery_features.T)


class ResNet18Extractor:
    """
    Fallback feature extractor using ResNet18.
    
    Used when torchreid is not installed. Provides 512-dim embeddings
    using a pretrained ResNet18 backbone.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512
    
    def __init__(self, device: Optional[str] = None):
        """Initialize ResNet18 feature extractor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResNet18 extractor")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading ResNet18 fallback extractor on {self.device}")
        
        # Load pretrained ResNet18 and modify for ReID
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove classification layer, keep up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection layer to get 512-dim output
        self.projection = nn.Linear(512, self.EMBEDDING_DIM)
        
        self.backbone.to(self.device)
        self.projection.to(self.device)
        self.backbone.eval()
        self.projection.eval()
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        logger.info(f"ResNet18 extractor initialized (embedding_dim={self.EMBEDDING_DIM})")
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return self.transform(image)
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Extract feature embedding from a single image."""
        with torch.no_grad():
            tensor = self._preprocess(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            features = self.backbone(tensor)
            features = features.view(features.size(0), -1)
            features = self.projection(features)
            features = features.cpu().numpy().flatten()
            
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
        """Extract features from multiple images."""
        if len(images) == 0:
            return np.empty((0, self.EMBEDDING_DIM))
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                tensors = [self._preprocess(img) for img in batch_images]
                batch = torch.stack(tensors).to(self.device)
                
                features = self.backbone(batch)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
                features = features.cpu().numpy()
                
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
        """Compute distance matrix."""
        if metric == 'cosine':
            return 1 - np.dot(query_features, gallery_features.T)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity."""
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        return np.dot(query_features, gallery_features.T)


def export_osnet_to_onnx(
    onnx_path: str = "model_weights/osnet_x1_0.onnx",
    model_name: str = "osnet_x1_0",
    opset: int = 17,
) -> str:
    """
    Export OSNet PyTorch model to ONNX format.
    
    Args:
        onnx_path: Output ONNX path
        model_name: OSNet variant name
        opset: ONNX opset version
    
    Returns:
        Path to exported ONNX model
    """
    if not TORCHREID_AVAILABLE or not TORCH_AVAILABLE:
        raise ImportError("torchreid and torch are required to export OSNet")
    
    import os
    os.makedirs(os.path.dirname(onnx_path) if os.path.dirname(onnx_path) else ".", exist_ok=True)
    
    # Build model
    model = torchreid.models.build_model(
        name=model_name,
        num_classes=1,
        loss='softmax',
        pretrained=True,
        use_gpu=False,  # Export on CPU
    )
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=256, width=128)
    dummy_input = torch.randn(1, 3, 256, 128)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    
    logger.info(f"Exported OSNet model to: {onnx_path}")
    return onnx_path


# Lazy singleton instance
_extractor_instance = None


def get_extractor(
    model_name: str = "osnet_x1_0",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_onnx: bool = True,
    use_fastreid: bool = False,  # Disabled by default
    use_nvidia: bool = True,   # Default to NVIDIA model
):
    """
    Get or create singleton extractor instance.
    
    Priority order:
    1. NVIDIA TAO (if use_nvidia=True and model exists)
    2. FastReID (if use_fastreid=True and available)
    3. ONNX OSNet (if use_onnx=True and available)
    4. PyTorch OSNet
    5. ResNet18 fallback
    """
    global _extractor_instance
    
    if _extractor_instance is not None:
        return _extractor_instance
    
    import os
    
    # Priority 1: NVIDIA TAO ReID (New Standard)
    if use_nvidia:
        try:
            from app.core.features.nvidia_reid_extractor import get_nvidia_extractor, NvidiaReIDExtractor
            # We can instantiate directly since we have the class import
            # Check default path from settings if not provided
            if not model_path:
                 # Standard path for this project
                 model_path = "model_weights/resnet50_market1501_aicity156.onnx"
            
            if os.path.exists(model_path):
                logger.info("Using NVIDIA TAO ReID extractor")
                _extractor_instance = NvidiaReIDExtractor(
                    model_path=model_path,
                    device=device,
                )
                return _extractor_instance
        except Exception as e:
            logger.warning(f"NVIDIA ReID not available, falling back: {e}")

    # Priority 2: FastReID
        try:
            from app.core.features.fastreid_extractor import get_fastreid_extractor, FASTREID_AVAILABLE
            if FASTREID_AVAILABLE:
                logger.info("Attempting to load FastReID extractor (SOTA accuracy)")
                _extractor_instance = get_fastreid_extractor(device=device, use_onnx=use_onnx)
                return _extractor_instance
        except Exception as e:
            logger.warning(f"FastReID not available, falling back to OSNet: {e}")
    
    # Fall back to OSNet
    is_onnx_model = model_path is not None and model_path.endswith('.onnx')
    
    # Prefer ONNX if available and requested
    if (use_onnx or is_onnx_model) and ONNX_AVAILABLE:
        if not is_onnx_model and model_path:
            # Try to find ONNX version of the model
            onnx_path = model_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
            if os.path.exists(onnx_path):
                model_path = onnx_path
                is_onnx_model = True
        elif not is_onnx_model:
            # Check default ONNX path
            default_onnx = f"model_weights/{model_name}.onnx"
            if os.path.exists(default_onnx):
                model_path = default_onnx
                is_onnx_model = True
        
        if is_onnx_model and model_path:
            logger.info("Using ONNX Runtime OSNet with CUDA EP")
            _extractor_instance = OSNetOnnxExtractor(
                model_path=model_path,
                device=device,
            )
            return _extractor_instance
    
    # Fallback to PyTorch OSNet
    if TORCHREID_AVAILABLE:
        logger.info("Using PyTorch/torchreid OSNet extractor")
        _extractor_instance = OSNetExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )
    elif TORCH_AVAILABLE:
        # Final fallback to ResNet18
        logger.warning("Using ResNet18 fallback (torchreid not available)")
        _extractor_instance = ResNet18Extractor(device=device)
    else:
        raise ImportError("No feature extraction backend available. Install torchreid or onnxruntime-gpu.")
    
    return _extractor_instance

