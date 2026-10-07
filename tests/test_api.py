import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import Settings
from app.main import create_app
from tests.conftest import FakeDetector, make_png

PNG = "image/png"


# --- health -----------------------------------------------------------------------------


def test_liveness(client: TestClient) -> None:
    r = client.get("/health/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_when_model_loaded(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model_loaded": True}


def test_not_ready_without_model(client_no_model: TestClient) -> None:
    r = client_no_model.get("/health/ready")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False


# --- single prediction ------------------------------------------------------------------


@pytest.mark.parametrize(("pixel", "label"), [(250, "fake"), (5, "real")])
def test_predict_labels(client: TestClient, pixel: int, label: str) -> None:
    r = client.post("/api/v1/predict", files={"file": ("face.png", make_png(pixel), PNG)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"]["label"] == label
    assert body["prediction"]["filename"] == "face.png"
    assert body["prediction"]["p_fake"] + body["prediction"]["p_real"] == pytest.approx(1.0)
    assert body["model"]["threshold"] == 0.5
    assert len(body["model"]["weights_sha256"]) == 64


def test_predict_resizes_to_model_input(client: TestClient, detector: FakeDetector) -> None:
    client.post("/api/v1/predict", files={"file": ("f.png", make_png(size=200), PNG)})
    assert detector.calls == [(1, 3, 64, 64)]


def test_invalid_image_bytes(client: TestClient) -> None:
    r = client.post("/api/v1/predict", files={"file": ("x.png", b"not an image", PNG)})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_image"
    assert err["details"]["filename"] == "x.png"


def test_empty_file(client: TestClient) -> None:
    r = client.post("/api/v1/predict", files={"file": ("e.png", b"", PNG)})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_image"


def test_too_small_image(client: TestClient) -> None:
    r = client.post("/api/v1/predict", files={"file": ("s.png", make_png(size=8), PNG)})
    assert r.status_code == 422
    assert "too small" in r.json()["error"]["message"]


def test_unsupported_media_type(client: TestClient) -> None:
    r = client.post("/api/v1/predict", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_payload_too_large(client: TestClient) -> None:
    r = client.post("/api/v1/predict", files={"file": ("big.png", b"0" * 60_000, PNG)})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "payload_too_large"


def test_missing_file_field(client: TestClient) -> None:
    r = client.post("/api/v1/predict")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_model_not_ready_returns_503(client_no_model: TestClient) -> None:
    r = client_no_model.post("/api/v1/predict", files={"file": ("f.png", make_png(), PNG)})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_not_ready"


# --- batch prediction -------------------------------------------------------------------


def test_batch_runs_one_forward_pass(client: TestClient, detector: FakeDetector) -> None:
    files = [("files", (f"{i}.png", make_png(v), PNG)) for i, v in enumerate([5, 250])]
    r = client.post("/api/v1/predict/batch", files=files)
    assert r.status_code == 200, r.text
    labels = [p["label"] for p in r.json()["predictions"]]
    assert labels == ["real", "fake"]
    assert detector.calls == [(2, 3, 64, 64)]


def test_batch_too_many_files(client: TestClient) -> None:
    files = [("files", (f"{i}.png", make_png(), PNG)) for i in range(4)]
    r = client.post("/api/v1/predict/batch", files=files)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "too_many_files"


def test_batch_reports_which_file_is_bad(client: TestClient) -> None:
    files = [("files", ("good.png", make_png(), PNG)), ("files", ("bad.png", b"xx", PNG))]
    r = client.post("/api/v1/predict/batch", files=files)
    assert r.status_code == 422
    assert r.json()["error"]["details"]["filename"] == "bad.png"


# --- model info, request IDs, auth -----------------------------------------------------


def test_model_info(client: TestClient) -> None:
    r = client.get("/api/v1/model")
    assert r.status_code == 200
    assert r.json()["name"] == "fake"


def test_request_id_generated_and_echoed(client: TestClient) -> None:
    assert len(client.get("/health/live").headers["X-Request-ID"]) == 32
    r = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_error_body_carries_request_id(client: TestClient) -> None:
    r = client.post(
        "/api/v1/predict",
        files={"file": ("x.png", b"bad", PNG)},
        headers={"X-Request-ID": "trace-1"},
    )
    assert r.json()["error"]["request_id"] == "trace-1"


@pytest.fixture
def secured_client(settings: Settings, detector: FakeDetector):
    secured = settings.model_copy(update={"api_key": SecretStr("s3cret")})
    with TestClient(create_app(settings=secured, detector=detector)) as c:
        yield c


def test_api_key_required_when_configured(secured_client: TestClient) -> None:
    files = {"file": ("f.png", make_png(), PNG)}
    assert secured_client.post("/api/v1/predict", files=files).status_code == 401
    ok = secured_client.post("/api/v1/predict", files=files, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200
    # Health probes stay open so orchestrators can reach them.
    assert secured_client.get("/health/live").status_code == 200


class CrashingDetector(FakeDetector):
    def predict(self, batch):  # type: ignore[no-untyped-def]
        raise RuntimeError("CUDA out of memory: secret internal detail")


def test_unexpected_error_returns_500_without_leaking(settings: Settings) -> None:
    app = create_app(settings=settings, detector=CrashingDetector())
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/api/v1/predict", files={"file": ("f.png", make_png(), PNG)})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text
