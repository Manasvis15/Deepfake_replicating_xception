"""Business logic: turn uploaded image bytes into labelled predictions.

Knows nothing about HTTP or PyTorch: it receives bytes, uses the preprocessing module and
whatever Detector it is given, and applies the request's decision policy.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.core.errors import InvalidImageError
from app.core.policy import DecisionPolicy, Label
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
    label: Label

    @property
    def p_real(self) -> float:
        return 1.0 - self.p_fake


@dataclass(frozen=True)
class PredictionResult:
    images: list[ScoredImage]
    inference_ms: float


class PredictionService:
    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    def predict(self, inputs: Sequence[ImageInput], policy: DecisionPolicy) -> PredictionResult:
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

        batch = np.stack(arrays)
        started = time.perf_counter()
        probs = self._score_with_tta(batch) if policy.tta else self._detector.predict(batch)
        elapsed_ms = (time.perf_counter() - started) * 1000

        images = [
            ScoredImage(filename=item.filename, p_fake=float(p), label=policy.classify(float(p)))
            for item, p in zip(inputs, probs, strict=True)
        ]
        return PredictionResult(images=images, inference_ms=round(elapsed_ms, 2))

    def _score_with_tta(self, batch: np.ndarray) -> np.ndarray:
        """Average each image's score with its horizontal mirror, in one forward pass.

        Faces are roughly left-right symmetric, so a mirrored face is still a valid input;
        averaging the two smooths out scores that depend on the exact pixel layout.
        """
        # ascontiguousarray: the flipped view has negative strides, which torch rejects.
        mirrored = np.ascontiguousarray(batch[..., ::-1])
        probs = self._detector.predict(np.concatenate([batch, mirrored]))
        n = len(batch)
        averaged: np.ndarray = (probs[:n] + probs[n:]) / 2
        return averaged
