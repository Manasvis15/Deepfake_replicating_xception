from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.ml.base import DetectorInfo


class FakeDetector:
    """Stands in for the real model: deterministic, instant, no torch needed.

    Scores an image by its mean brightness, so tests can choose the label by
    choosing the pixel values (bright -> fake, dark -> real).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(name="fake", weights_sha256="0" * 64, device="cpu", input_size=64)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(batch.shape)
        return ((batch.mean(axis=(1, 2, 3)) + 1) / 2).clip(0, 1)  # [-1,1] -> [0,1]


def make_png(value: int = 128, size: int = 64) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((size, size, 3), value, dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        log_json=False,
        load_model_on_startup=False,
        max_upload_bytes=50_000,
        max_batch_files=3,
        input_size=64,
    )


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def client(settings: Settings, detector: FakeDetector) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings, detector=detector)) as c:
        yield c


@pytest.fixture
def client_no_model(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as c:
        yield c
