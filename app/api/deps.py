"""Dependency injection. Endpoints declare what they need; tests can override any of it."""

import secrets
from typing import Annotated

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.errors import ModelNotReadyError, UnauthorizedError
from app.ml.base import Detector
from app.services.prediction import PredictionService

SettingsDep = Annotated[Settings, Depends(get_settings)]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    settings: SettingsDep, provided: Annotated[str | None, Security(_api_key_header)]
) -> None:
    """No-op when DFD_API_KEY is unset; otherwise require a matching X-API-Key header."""
    if settings.api_key is None:
        return
    expected = settings.api_key.get_secret_value()
    # Constant-time comparison avoids leaking the key through response timing.
    if provided is None or not secrets.compare_digest(provided, expected):
        raise UnauthorizedError("Missing or invalid API key.")


def get_detector(request: Request) -> Detector:
    detector: Detector | None = getattr(request.app.state, "detector", None)
    if detector is None:
        raise ModelNotReadyError("The model is not loaded yet.")
    return detector


def get_prediction_service(
    detector: Annotated[Detector, Depends(get_detector)], settings: SettingsDep
) -> PredictionService:
    return PredictionService(detector, threshold=settings.threshold)


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
DetectorDep = Annotated[Detector, Depends(get_detector)]
