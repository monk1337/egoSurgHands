"""Abstract base for hand pose models with a uniform batched predict() interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image as PILImage


@dataclass
class BatchPrediction:
    """Uniform output format for a batch of hand predictions."""
    # Keypoint predictions in OpenPose joint order
    joints_3d: np.ndarray                 # (K, 21, 3) meters, root-centered recommended
    vertices_3d: Optional[np.ndarray] = None   # (K, 778, 3) full MANO mesh (optional)
    joints_2d_px: Optional[np.ndarray] = None  # (K, 21, 2) image pixel coords (optional)
    # Which input indices these outputs correspond to (bbox-valid, non-degenerate)
    valid_idx: List[int] = field(default_factory=list)


class HandModel(ABC):
    """
    Uniform interface for zero-shot hand pose models.

    Subclasses must implement ``predict_batch`` which takes a list of PIL images,
    bboxes, hand sides and returns a BatchPrediction.

    Predictions should be in OpenPose joint order (wrist, thumb×4, index×4, ...).
    """

    name: str = "base"

    @abstractmethod
    def predict_batch(
        self,
        images: List[PILImage.Image],
        bboxes: Sequence,
        hand_sides: Sequence[str],
    ) -> BatchPrediction:
        """Run model on a batch. Returns BatchPrediction (may have K ≤ B)."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"
