"""Business logic: turn uploaded image bytes into labelled predictions.

Knows nothing about HTTP or PyTorch: it receives bytes, uses the preprocessing module and
whatever Detector it is given, and applies the decision threshold.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.core.errors import InvalidImageError
from app.ml.base import Detector
from app.ml.preprocessing import preprocess


@dataclass(frozen=True)
class ImageInput:
    filename: str | None
    data: bytes


@dataclass(frozen=True)
class ScoredImage:
    filename: str | None
    p_fake: float
    is_fake: bool

    @property
    def p_real(self) -> float:
        return 1.0 - self.p_fake


@dataclass(frozen=True)
class PredictionResult:
    images: list[ScoredImage]
    inference_ms: float


class PredictionService:
    def __init__(self, detector: Detector, *, threshold: float) -> None:
        self._detector = detector
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(self, inputs: Sequence[ImageInput]) -> PredictionResult:
        """Synchronous and CPU-bound: call it from a worker thread, not the event loop."""
        size = self._detector.info.input_size
        arrays = []
        for item in inputs:
            try:
                arrays.append(preprocess(item.data, size))
            except InvalidImageError as exc:
                # Say which file failed, so batch clients can fix the right one.
                raise InvalidImageError(
                    exc.message, details={**exc.details, "filename": item.filename}
                ) from exc

        started = time.perf_counter()
        probs = self._detector.predict(np.stack(arrays))
        elapsed_ms = (time.perf_counter() - started) * 1000

        images = [
            ScoredImage(filename=item.filename, p_fake=float(p), is_fake=bool(p >= self._threshold))
            for item, p in zip(inputs, probs, strict=True)
        ]
        return PredictionResult(images=images, inference_ms=round(elapsed_ms, 2))
