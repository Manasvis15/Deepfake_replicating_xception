"""Domain errors. They know nothing about HTTP; the API layer maps them to status codes."""

from typing import Any


class AppError(Exception):
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidImageError(AppError):
    code = "invalid_image"


class UnsupportedMediaTypeError(AppError):
    code = "unsupported_media_type"


class PayloadTooLargeError(AppError):
    code = "payload_too_large"


class TooManyFilesError(AppError):
    code = "too_many_files"


class ModelNotReadyError(AppError):
    code = "model_not_ready"


class ModelLoadError(AppError):
    code = "model_load_error"


class InvalidParameterError(AppError):
    code = "invalid_parameter"


class UnauthorizedError(AppError):
    code = "unauthorized"
