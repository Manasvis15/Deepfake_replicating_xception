import cv2
import numpy as np
import pytest

from app.core.errors import InvalidImageError
from app.ml.preprocessing import decode_image, preprocess, to_model_input
from tests.conftest import make_png

# --- preprocessing (no torch needed) ----------------------------------------------------


def test_preprocess_shape_dtype_and_range() -> None:
    x = preprocess(make_png(size=100), size=256)
    assert x.shape == (3, 256, 256)
    assert x.dtype == np.float32
    assert -1.0 <= x.min() <= x.max() <= 1.0


def test_normalization_maps_black_and_white_to_minus_one_and_one() -> None:
    black = to_model_input(np.zeros((64, 64, 3), np.uint8), 64)
    white = to_model_input(np.full((64, 64, 3), 255, np.uint8), 64)
    assert np.allclose(black, -1.0) and np.allclose(white, 1.0)


def test_bgr_is_converted_to_rgb() -> None:
    bgr = np.zeros((64, 64, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in OpenCV's BGR order
    x = to_model_input(bgr, 64)
    assert np.allclose(x[2], 1.0) and np.allclose(x[0], -1.0)  # blue lands in channel 2


def test_grayscale_is_decoded_as_three_channels() -> None:
    ok, buf = cv2.imencode(".png", np.full((64, 64), 100, np.uint8))
    assert ok and decode_image(buf.tobytes()).shape == (64, 64, 3)


def test_garbage_bytes_raise() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"definitely not an image")
