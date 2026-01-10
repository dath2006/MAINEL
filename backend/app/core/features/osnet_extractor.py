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


class MultiScaleOSNetExtractor:
    """
    Multi-Scale OSNet feature extractor for robust person re-identification.
    
    Based on the paper's recommendation to pool features from multiple OSNet
    stages to capture both "coarse" (color, shape) and "fine" (texture, logos)
    details. This reduces false positives when two people wear similar clothes.
    
    OSNet Architecture:
    - conv1: Initial features
    - conv2: Low-level features (64 channels)
    - conv3: Mid-level features (256 channels)  
    - conv4: Mid-high features (384 channels)
    - conv5: High-level features (512 channels)
    
    We extract from conv3, conv4, conv5 and concatenate:
    - conv3: Captures edges, textures (fine details like logos)
    - conv4: Captures parts and accessories
    - conv5: Captures overall appearance (color, shape)
    
    Total: 256 + 384 + 512 = 1152 dims -> projected to 512 dims
    """
    
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512  # Output dimension (projected from multi-scale concat)
    
    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        pretrained: bool = True,
    ):
        """
        Initialize Multi-Scale OSNet feature extractor.
        
        Args:
            model_name: OSNet variant ('osnet_x1_0', 'osnet_x0_75', etc.)
            model_path: Path to custom weights (optional)
            device: Compute device ('cuda', 'cpu', or None for auto)
            pretrained: Use pretrained weights if no custom path provided
        """
        if not TORCHREID_AVAILABLE:
            raise ImportError("torchreid package is required for Multi-Scale OSNet extractor")
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading Multi-Scale OSNet model '{model_name}' on {self.device}")
        
        # Build model
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
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
        
        # Hook containers for intermediate features
        self._features = {}
        
        # Register hooks to capture intermediate layer outputs
        # OSNet structure: conv1, conv2, conv3, conv4, conv5
        self._register_hooks()
        
        # Projection layer: 1152 (256+384+512) -> 512
        # Using simple linear projection to maintain compatibility
        self.projection = nn.Sequential(
            nn.Linear(1152, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(768, self.EMBEDDING_DIM),
        ).to(self.device)
        self.projection.eval()
        
        # Build transforms
        _, self.transform = build_transforms(
            height=self.INPUT_SIZE[0],
            width=self.INPUT_SIZE[1],
            random_erase=False,
            color_jitter=False,
            color_aug=False,
        )
        
        self.dist_metric = 'cosine'
        logger.info(f"Multi-Scale OSNet extractor initialized (1152 -> {self.EMBEDDING_DIM} dim)")
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        def get_hook(name):
            def hook(module, input, output):
                self._features[name] = output
            return hook
        
        # Access OSNet layers - structure is: conv1, conv2, conv3, conv4, conv5
        # Note: Exact layer names depend on torchreid implementation
        try:
            # Try to hook into OSNet conv layers
            if hasattr(self.model, 'conv3'):
                self.model.conv3.register_forward_hook(get_hook('conv3'))
            if hasattr(self.model, 'conv4'):
                self.model.conv4.register_forward_hook(get_hook('conv4'))
            if hasattr(self.model, 'conv5'):
                self.model.conv5.register_forward_hook(get_hook('conv5'))
        except Exception as e:
            logger.warning(f"Could not register all hooks: {e}")
    
    def _pool_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Global Average Pooling for feature map."""
        return nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)).view(feature_map.size(0), -1)
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        tensor = self.transform(image)
        return tensor
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract multi-scale feature embedding from a single person image.
        
        Args:
            image: Person crop as numpy array (H, W, C) in RGB or BGR
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Feature embedding as 1D numpy array of shape (512,)
        """
        with torch.no_grad():
            # Clear previous features
            self._features.clear()
            
            # Preprocess and run forward pass
            tensor = self._preprocess(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Forward pass (hooks will capture intermediate features)
            _ = self.model(tensor)
            
            # Check if we got multi-scale features
            if 'conv3' in self._features and 'conv4' in self._features and 'conv5' in self._features:
                # Pool each feature map
                f3 = self._pool_features(self._features['conv3'])  # 256-dim
                f4 = self._pool_features(self._features['conv4'])  # 384-dim
                f5 = self._pool_features(self._features['conv5'])  # 512-dim
                
                # Concatenate
                multi_feat = torch.cat([f3, f4, f5], dim=1)  # 1152-dim
                
                # Project to 512-dim
                features = self.projection(multi_feat)
            else:
                # Fallback to standard single-scale extraction
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
        Extract multi-scale features from multiple person images.
        
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
                
                # Clear and run forward
                self._features.clear()
                _ = self.model(batch)
                
                # Extract multi-scale features
                if 'conv3' in self._features and 'conv4' in self._features and 'conv5' in self._features:
                    f3 = self._pool_features(self._features['conv3'])
                    f4 = self._pool_features(self._features['conv4'])
                    f5 = self._pool_features(self._features['conv5'])
                    multi_feat = torch.cat([f3, f4, f5], dim=1)
                    features = self.projection(multi_feat)
                else:
                    features = self.model(batch)
                
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


class ResNet50Extractor:
    """
    Strong feature extractor using ResNet-50 backbone.
    
    Provides higher-quality appearance features compared to the tiny CNN
    used in standard DeepSORT. The paper recommends using ResNet-50 for
    the tracker's appearance model to prevent "micro-swaps" before they
    pollute the ReID gallery.
    
    Key advantages:
    - 2048-dim internal features (much richer than standard DeepSORT)
    - Projected to 512-dim for compatibility with OSNet
    - Better captures fine-grained details (clothing texture, accessories)
    - Reduces ID swaps during occlusions
    
    Trade-off: ~10ms slower per inference than tiny CNN, but much more accurate.
    """
    
    INPUT_SIZE = (256, 128)  # (height, width) - standard ReID size
    EMBEDDING_DIM = 512  # Output dimension (projected from 2048)
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ResNet-50 feature extractor.
        
        Args:
            device: Compute device ('cuda', 'cpu', or None for auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResNet50 extractor")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading ResNet-50 strong backbone on {self.device}")
        
        # Load pretrained ResNet-50 (much stronger than ResNet-18)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove classification layer, keep up to avgpool
        # ResNet-50 outputs 2048-dim features before FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer: 2048 -> 512 for OSNet compatibility
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, self.EMBEDDING_DIM),
        )
        
        self.backbone.to(self.device)
        self.projection.to(self.device)
        self.backbone.eval()
        self.projection.eval()
        
        # Standard ImageNet transforms with ReID-specific sizing
        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.dist_metric = 'cosine'
        logger.info(f"ResNet-50 extractor initialized (2048 -> {self.EMBEDDING_DIM} dim)")
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return self.transform(image)
    
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
            tensor = self._preprocess(image)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Extract backbone features (2048-dim)
            features = self.backbone(tensor)
            features = features.view(features.size(0), -1)
            
            # Project to 512-dim
            features = self.projection(features)
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
                tensors = [self._preprocess(img) for img in batch_images]
                batch = torch.stack(tensors).to(self.device)
                
                # Extract and project
                features = self.backbone(batch)
                features = features.view(features.size(0), -1)
                features = self.projection(features)
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
    backbone: str = "osnet",
    multi_scale: bool = True,
):
    """
    Get or create singleton extractor instance.
    
    Supports multiple backbones:
    - 'osnet': Default OSNet for ReID (fast, good accuracy)
    - 'osnet_multiscale': Multi-scale OSNet (more robust to clothing)
    - 'resnet50': Strong ResNet-50 backbone (slower, better for tracker)
    - 'resnet18': Lightweight fallback
    
    For DeepSORT tracker, 'resnet50' is recommended for better appearance matching.
    For main ReID, 'osnet' with multi_scale=True is the best choice.
    
    Args:
        model_name: OSNet variant name (if using osnet backbone)
        model_path: Path to model weights
        device: Compute device
        use_onnx: Use ONNX backend if available
        backbone: 'osnet', 'resnet50', or 'resnet18'
        multi_scale: If True and using OSNet, use MultiScaleOSNetExtractor
    
    Returns:
        Feature extractor instance
    """
    global _extractor_instance
    
    if _extractor_instance is None:
        import os
        
        # ResNet-50 explicit selection (for DeepSORT tracker)
        if backbone == "resnet50":
            if TORCH_AVAILABLE:
                logger.info("Using ResNet-50 strong backbone (recommended for DeepSORT)")
                _extractor_instance = ResNet50Extractor(device=device)
                return _extractor_instance
            else:
                logger.warning("PyTorch not available for ResNet-50, falling back to OSNet")
        
        # ResNet-18 explicit selection
        if backbone == "resnet18":
            if TORCH_AVAILABLE:
                logger.info("Using ResNet-18 lightweight backbone")
                _extractor_instance = ResNet18Extractor(device=device)
                return _extractor_instance
            else:
                logger.warning("PyTorch not available for ResNet-18, falling back to OSNet")
        
        # OSNet path (default)
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
        
        # Fallback to PyTorch OSNet (with multi-scale option)
        if TORCHREID_AVAILABLE:
            if multi_scale:
                logger.info("Using Multi-Scale OSNet extractor (pyramid pooling for robust features)")
                _extractor_instance = MultiScaleOSNetExtractor(
                    model_name=model_name,
                    model_path=model_path,
                    device=device,
                )
            else:
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


# Separate extractor instance for DeepSORT tracker (can use different backbone)
_tracker_extractor_instance = None


def get_tracker_extractor(
    device: Optional[str] = None,
    backbone: str = "resnet50",
):
    """
    Get or create singleton extractor for DeepSORT tracker.
    
    This is separate from the main ReID extractor to allow using
    a stronger backbone specifically for tracking.
    
    Args:
        device: Compute device
        backbone: 'resnet50' (recommended), 'osnet', or 'resnet18'
    
    Returns:
        Feature extractor instance for tracker
    """
    global _tracker_extractor_instance
    
    if _tracker_extractor_instance is None:
        if backbone == "resnet50" and TORCH_AVAILABLE:
            logger.info("Using ResNet-50 for DeepSORT tracker (strong backbone)")
            _tracker_extractor_instance = ResNet50Extractor(device=device)
        elif backbone == "resnet18" and TORCH_AVAILABLE:
            logger.info("Using ResNet-18 for DeepSORT tracker")
            _tracker_extractor_instance = ResNet18Extractor(device=device)
        elif TORCHREID_AVAILABLE:
            logger.info("Using OSNet for DeepSORT tracker")
            _tracker_extractor_instance = OSNetExtractor(device=device)
        elif TORCH_AVAILABLE:
            logger.info("Using ResNet-18 for DeepSORT tracker (fallback)")
            _tracker_extractor_instance = ResNet18Extractor(device=device)
        else:
            raise ImportError("No feature extraction backend available for tracker.")
    
    return _tracker_extractor_instance

