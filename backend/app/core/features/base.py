"""Base feature extractor interface."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    EMBEDDING_DIM: int
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image."""
        pass
    
    @abstractmethod
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract features from multiple images."""
        pass
