"""The per-request decision filters: threshold, mode, uncertain_margin, tta."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.policy import DecisionPolicy
from app.services.prediction import ImageInput, PredictionService
from tests.conftest import FakeDetector, make_png

PNG = "image/png"
URL = "/api/v1/predict"


def post(client: TestClient, pixel: int = 128, **params: object):  # type: ignore[no-untyped-def]
    return client.post(URL, params=params, files={"file": ("f.png", make_png(pixel), PNG)})


# FakeDetector scores by brightness: pixel 128 gives p_fake ~= 0.502.


@pytest.mark.parametrize(("threshold", "label"), [(0.3, "fake"), (0.7, "real")])
def test_custom_threshold_changes_label(client: TestClient, threshold: float, label: str) -> None:
    r = post(client, threshold=threshold)
    assert r.status_code == 200, r.text
    assert r.json()["prediction"]["label"] == label
    assert r.json()["decision"]["threshold"] == threshold


def test_threshold_does_not_change_the_score(client: TestClient) -> None:
    low = post(client, threshold=0.1).json()["prediction"]["p_fake"]
    high = post(client, threshold=0.9).json()["prediction"]["p_fake"]
    assert low == high


@pytest.mark.parametrize(
    ("mode", "threshold", "label"),
    [("high_recall", 0.25, "fake"), ("balanced", 0.45, "fake"), ("high_precision", 0.8, "real")],
)
def test_modes_use_configured_thresholds(
    client: TestClient, mode: str, threshold: float, label: str
) -> None:
    body = post(client, mode=mode).json()
    assert body["decision"]["mode"] == mode
    assert body["decision"]["threshold"] == threshold
    assert body["prediction"]["label"] == label


def test_threshold_and_mode_together_is_rejected(client: TestClient) -> None:
    r = post(client, threshold=0.5, mode="balanced")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.parametrize(
    "params",
    [{"threshold": 1.5}, {"threshold": -0.1}, {"mode": "aggressive"}, {"uncertain_margin": 0.6}],
)
def test_out_of_range_parameters_are_rejected(client: TestClient, params: dict) -> None:
    r = post(client, **params)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_uncertain_margin_labels_close_calls(client: TestClient) -> None:
    assert post(client, uncertain_margin=0.1).json()["prediction"]["label"] == "uncertain"
    assert post(client, 250, uncertain_margin=0.1).json()["prediction"]["label"] == "fake"
    assert post(client, 5, uncertain_margin=0.1).json()["prediction"]["label"] == "real"


def test_filters_apply_to_batch(client: TestClient) -> None:
    files = [("files", (f"{i}.png", make_png(v), PNG)) for i, v in enumerate([5, 128, 250])]
    r = client.post("/api/v1/predict/batch", params={"uncertain_margin": 0.1}, files=files)
    assert [p["label"] for p in r.json()["predictions"]] == ["real", "uncertain", "fake"]


def test_tta_scores_image_and_mirror_in_one_pass(
    client: TestClient, detector: FakeDetector
) -> None:
    r = post(client, tta="true")
    assert r.json()["decision"]["tta"] is True
    assert detector.calls == [(2, 3, 64, 64)]  # original + mirror, one forward pass


class LeftHalfDetector(FakeDetector):
    """Scores only the left half of the image, so an image and its mirror score differently."""

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.calls.append(batch.shape)
        width = batch.shape[-1]
        return (batch[..., : width // 2].mean(axis=(1, 2, 3)) + 1) / 2


def test_tta_averages_original_and_mirror() -> None:
    # Left half black, right half white. Seen as-is the left half is black (p = 0);
    # mirrored it is white (p = 1). TTA must return their average.
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, 32:] = 255
    item = ImageInput(filename="half.png", data=cv2.imencode(".png", img)[1].tobytes())
    service = PredictionService(LeftHalfDetector())

    plain = service.predict([item], DecisionPolicy(threshold=0.5)).images[0].p_fake
    tta = service.predict([item], DecisionPolicy(threshold=0.5, tta=True)).images[0].p_fake
    assert plain == pytest.approx(0.0, abs=1e-6)
    assert tta == pytest.approx(0.5, abs=1e-6)
