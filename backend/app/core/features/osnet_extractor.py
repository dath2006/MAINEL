"""
OSNet Feature Extractor

Uses TorchReID's OSNet model for person re-identification feature extraction.
Produces 512-dimensional embeddings for visual matching.

Falls back to ResNet18 if torchreid is not installed.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from loguru import logger

# Try to import torchreid
try:
    import torchreid
    from torchreid.data.transforms import build_transforms
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    logger.warning("torchreid not installed. Using ResNet18 fallback for feature extraction.")
    logger.info("To install torchreid: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")


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


# Lazy singleton instance
_extractor_instance = None


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


def get_extractor(
    model_name: str = "osnet_x1_0",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Get or create singleton extractor instance.
    
    Uses OSNet if torchreid is available, otherwise falls back to ResNet18.
    """
    global _extractor_instance
    if _extractor_instance is None:
        if TORCHREID_AVAILABLE:
            _extractor_instance = OSNetExtractor(
                model_name=model_name,
                model_path=model_path,
                device=device,
            )
        else:
            logger.warning("Using ResNet18 fallback (torchreid not available)")
            _extractor_instance = ResNet18Extractor(device=device)
    return _extractor_instance

