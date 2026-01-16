"""Base tracking interface."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from app.schemas.track import Detection


class BaseTracker(ABC):
    """Abstract base class for multi-object trackers."""
    
    @abstractmethod
    def predict(self):
        """Propagate track states forward."""
        pass
    
    @abstractmethod
    def update(self, detections: List[Detection], features: np.ndarray = None):
        """Update tracks with new detections."""
        pass
