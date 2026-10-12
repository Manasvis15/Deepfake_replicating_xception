"""Dependency injection. Endpoints declare what they need; tests can override any of it."""

import secrets
from typing import Annotated

from fastapi import Depends, Query, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.errors import InvalidParameterError, ModelNotReadyError, UnauthorizedError
from app.core.policy import DecisionPolicy, Mode
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
    detector: Annotated[Detector, Depends(get_detector)],
) -> PredictionService:
    return PredictionService(detector)


def get_decision_policy(
    settings: SettingsDep,
    threshold: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, description="P(fake) at or above this is labelled fake."),
    ] = None,
    mode: Annotated[
        Mode | None,
        Query(description="Named operating point. Use instead of `threshold`, not with it."),
    ] = None,
    uncertain_margin: Annotated[
        float,
        Query(
            ge=0.0,
            le=0.5,
            description="Scores within +/- this of the threshold are labelled `uncertain`.",
        ),
    ] = 0.0,
    tta: Annotated[
        bool, Query(description="Average with the mirrored image (about 2x slower).")
    ] = False,
) -> DecisionPolicy:
    """Turn the query parameters into one validated DecisionPolicy."""
    if threshold is not None and mode is not None:
        raise InvalidParameterError(
            "Set either `threshold` or `mode`, not both.",
            details={"threshold": threshold, "mode": mode.value},
        )
    if mode is not None:
        chosen = settings.mode_thresholds[mode.value]
    elif threshold is not None:
        chosen = threshold
    else:
        chosen = settings.threshold
    return DecisionPolicy(threshold=chosen, mode=mode, uncertain_margin=uncertain_margin, tta=tta)


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
DecisionPolicyDep = Annotated[DecisionPolicy, Depends(get_decision_policy)]
DetectorDep = Annotated[Detector, Depends(get_detector)]
