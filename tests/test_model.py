"""Model tests. Skipped automatically when torch is not installed."""

from pathlib import Path

import numpy as np
import pytest

from app.core.errors import ModelLoadError

torch = pytest.importorskip("torch")


def test_architecture_output_shapes() -> None:
    from app.ml.xception import Xception

    model = Xception().eval()
    with torch.inference_mode():
        logits, feat = model(torch.zeros(2, 3, 256, 256))
    assert logits.shape == (2, 2)
    assert feat.shape == (2, 2048, 8, 8)


def test_architecture_has_same_parameter_count_as_checkpoint() -> None:
    from app.ml.xception import Xception

    # xception_best.pth contains 283 tensors (incl. BatchNorm buffers).
    assert len(Xception().state_dict()) == 283


def test_detector_loads_prefixed_checkpoint(tmp_path: Path) -> None:
    from app.ml.xception import Xception
    from app.ml.xception_detector import XceptionDetector

    # Simulate DeepfakeBench's checkpoint layout: every key under "backbone.".
    path = tmp_path / "ckpt.pth"
    torch.save({f"backbone.{k}": v for k, v in Xception().state_dict().items()}, path)

    detector = XceptionDetector(path, device="cpu", input_size=64)
    scores = detector.predict(np.zeros((3, 3, 64, 64), np.float32))
    assert scores.shape == (3,)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert len(detector.info.weights_sha256) == 64


def test_detector_rejects_mismatched_checkpoint(tmp_path: Path) -> None:
    from app.ml.xception_detector import XceptionDetector

    path = tmp_path / "bad.pth"
    torch.save({"backbone.not_a_layer": torch.zeros(1)}, path)
    with pytest.raises(ModelLoadError):
        XceptionDetector(path, device="cpu")


def test_detector_missing_file(tmp_path: Path) -> None:
    from app.ml.xception_detector import XceptionDetector

    with pytest.raises(ModelLoadError):
        XceptionDetector(tmp_path / "nope.pth")
