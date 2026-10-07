"""The contract between the API and any model implementation.

The API and service layers depend only on this Protocol, never on PyTorch. That keeps
them testable with a fake detector and lets the model be swapped (e.g. for Effort or an
ONNX runtime) without touching any endpoint code.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    weights_sha256: str
    device: str
    input_size: int


class Detector(Protocol):
    @property
    def info(self) -> DetectorInfo: ...

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """Score a preprocessed batch.

        Args:
            batch: float32 array of shape (N, 3, H, W), already normalized.
        Returns:
            float array of shape (N,) with P(fake) for each image.
        """
        ...
