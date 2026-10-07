"""Application factory.

Run:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import verify_api_key
from app.api.errors import register_exception_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routes import health, predict
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.ml.base import Detector

logger = logging.getLogger(__name__)


def _load_detector(settings: Settings) -> Detector:
    # Imported here so torch is only needed when a real model is loaded
    # (the API, its tests and the docs work without it).
    from app.ml.xception_detector import XceptionDetector

    detector = XceptionDetector(
        settings.weights_path,
        device=settings.device,
        input_size=settings.input_size,
        batch_size=settings.batch_size,
        max_concurrency=settings.max_concurrent_inferences,
    )
    detector.warmup()
    return detector


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """Build the app. ``settings`` and ``detector`` can be injected (tests do this)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load once per process at startup, never per request. Failing here crashes the
        # process on purpose: a container that can't load its model should not serve.
        if detector is not None:
            app.state.detector = detector
        elif settings.load_model_on_startup:
            app.state.detector = _load_detector(settings)
        else:
            app.state.detector = None
        logger.info(
            "startup complete",
            extra={
                "environment": settings.environment,
                "model_loaded": app.state.detector is not None,
            },
        )
        yield
        app.state.detector = None
        logger.info("shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Frame-level deepfake detection on face crops (DeepfakeBench Xception).",
        lifespan=lifespan,
        # Keep interactive docs out of production unless deliberately re-enabled.
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None,
    )
    # Settings dependency must resolve to the same object the app was built with.
    app.dependency_overrides[get_settings] = lambda: settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
    return app
