"""Liveness and readiness probes (e.g. for Docker or Kubernetes)."""

from fastapi import APIRouter, Request, Response, status

from app.api.deps import SettingsDep
from app.schemas.prediction import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
def live(settings: SettingsDep) -> HealthResponse:
    """The process is up. Never touches the model, so it stays fast and cheap."""
    return HealthResponse(status="ok", version=settings.app_version)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
def ready(request: Request, response: Response) -> ReadinessResponse:
    """The service can take traffic: the model is loaded. 503 otherwise."""
    loaded = getattr(request.app.state, "detector", None) is not None
    if not loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if loaded else "not_ready", model_loaded=loaded)
