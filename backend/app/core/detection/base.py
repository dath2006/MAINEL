"""Base detection interface."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from app.schemas.track import Detection


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a single frame."""
        pass
    
    @abstractmethod
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames."""
        pass
