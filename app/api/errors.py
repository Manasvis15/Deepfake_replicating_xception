"""Maps every error to the same JSON shape: {"error": {code, message, details, request_id}}."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core import errors as e
from app.schemas.prediction import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)

# The only place domain errors meet HTTP status codes.
STATUS_BY_ERROR: dict[type[e.AppError], int] = {
    # Plain integers: Starlette has renamed some status constants between versions.
    e.InvalidImageError: 422,
    e.TooManyFilesError: 422,
    e.InvalidParameterError: 422,
    e.UnsupportedMediaTypeError: 415,
    e.PayloadTooLargeError: 413,
    e.UnauthorizedError: 401,
    e.ModelNotReadyError: 503,
}


def _respond(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "-")
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details or {}, request_id=request_id)
    )
    return JSONResponse(
        status_code=status_code, content=body.model_dump(), headers={"X-Request-ID": request_id}
    )


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, e.AppError)
    status_code = STATUS_BY_ERROR.get(type(exc), 500)
    log = logger.warning if status_code < 500 else logger.error
    log("request failed", extra={"code": exc.code, "status": status_code})
    return _respond(request, status_code, exc.code, exc.message, exc.details)


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in exc.errors()]
    return _respond(
        request, 422, "validation_error", "Request validation failed.", {"errors": errors}
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full traceback server-side, but never leak internals to the client.
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("unhandled error", extra={"request_id": request_id})
    return _respond(request, 500, "internal_error", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(e.AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
