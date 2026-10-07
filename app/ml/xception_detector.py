"""PyTorch implementation of the Detector interface for DeepfakeBench's xception_best.pth."""

import hashlib
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

from app.core.errors import ModelLoadError
from app.ml.base import DetectorInfo
from app.ml.xception import Xception

logger = logging.getLogger(__name__)

# The checkpoint is XceptionDetector's state_dict ("backbone." prefix); DDP adds "module.".
_KEY_PREFIXES = ("module.", "backbone.")
FAKE_CLASS_INDEX = 1


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _strip_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    clean = {}
    for key, value in state.items():
        for prefix in _KEY_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        clean[key] = value
    return clean


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("DFD_DEVICE=cuda but no CUDA device is available.")
    return torch.device(device)


class XceptionDetector:
    """Loads the weights once and serves thread-safe, bounded-concurrency inference."""

    def __init__(
        self,
        weights_path: Path,
        *,
        device: str = "auto",
        input_size: int = 256,
        batch_size: int = 16,
        max_concurrency: int = 2,
    ) -> None:
        if not weights_path.is_file():
            raise ModelLoadError(f"Weights file not found: {weights_path}")

        started = time.perf_counter()
        self._device = _resolve_device(device)
        self._batch_size = batch_size
        # Caps simultaneous forward passes so concurrent requests can't exhaust memory.
        self._slots = threading.BoundedSemaphore(max_concurrency)

        try:
            # weights_only=True refuses to unpickle arbitrary objects (safe loading).
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ModelLoadError(f"Could not read checkpoint: {exc}") from exc
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model = Xception()
        try:
            model.load_state_dict(_strip_prefixes(state), strict=True)
        except RuntimeError as exc:
            raise ModelLoadError(f"Checkpoint does not match the architecture: {exc}") from exc
        self._model = model.to(self._device).eval()

        self._info = DetectorInfo(
            name="xception-deepfakebench",
            weights_sha256=_sha256(weights_path),
            device=str(self._device),
            input_size=input_size,
        )
        logger.info(
            "model loaded",
            extra={
                "device": self._info.device,
                "tensors": len(state),
                "sha256": self._info.weights_sha256[:12],
                "load_s": round(time.perf_counter() - started, 2),
            },
        )

    @property
    def info(self) -> DetectorInfo:
        return self._info

    def warmup(self) -> None:
        """One dummy pass so the first real request doesn't pay one-time setup costs."""
        size = self._info.input_size
        self.predict(np.zeros((1, 3, size, size), dtype=np.float32))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {batch.shape}")
        scores = []
        with self._slots, torch.inference_mode():
            for start in range(0, len(batch), self._batch_size):
                chunk = torch.from_numpy(batch[start : start + self._batch_size]).to(self._device)
                logits, _ = self._model(chunk)
                scores.append(torch.softmax(logits, dim=1)[:, FAKE_CLASS_INDEX].cpu())
        return torch.cat(scores).numpy()
