"""Image decoding and preprocessing, matching DeepfakeBench's Xception training setup.

Pure NumPy/OpenCV (no torch), so it is cheap to unit-test.
"""

import cv2
import numpy as np

from app.core.errors import InvalidImageError

MEAN = 0.5
STD = 0.5
MIN_SIDE = 32  # anything smaller cannot be a usable face crop


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw upload bytes into a BGR uint8 array of shape (H, W, 3)."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image: np.ndarray | None = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # gray/alpha -> 3-ch
    if image is None:
        raise InvalidImageError("File could not be decoded as an image.")
    h, w = image.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InvalidImageError(
            f"Image is too small ({w}x{h}); minimum side is {MIN_SIDE}px.",
            details={"width": w, "height": h},
        )
    return image


def to_model_input(bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> normalized float32 (3, size, size) in [-1, 1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    chw: np.ndarray = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return (chw - MEAN) / STD


def preprocess(data: bytes, size: int) -> np.ndarray:
    return to_model_input(decode_image(data), size)
