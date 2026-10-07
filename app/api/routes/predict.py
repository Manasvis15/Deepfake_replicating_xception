"""Prediction endpoints. Thin by design: validate the HTTP input, delegate to the service,
shape the response. Inference runs in a worker thread so the event loop stays free."""

from typing import Annotated, Any

from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import DetectorDep, PredictionServiceDep, SettingsDep
from app.core.config import Settings
from app.core.errors import (
    InvalidImageError,
    PayloadTooLargeError,
    TooManyFilesError,
    UnsupportedMediaTypeError,
)
from app.ml.base import Detector
from app.schemas.prediction import (
    BatchPredictResponse,
    ErrorResponse,
    Label,
    ModelInfo,
    Prediction,
    PredictResponse,
)
from app.services.prediction import ImageInput, ScoredImage

router = APIRouter(tags=["prediction"])

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 413, 415, 422, 503)
}


async def _read_upload(file: UploadFile, settings: Settings) -> ImageInput:
    content_type = (file.content_type or "").lower()
    if content_type not in settings.allowed_content_types:
        raise UnsupportedMediaTypeError(
            f"Unsupported content type '{content_type or 'unknown'}'.",
            details={"filename": file.filename, "allowed": sorted(settings.allowed_content_types)},
        )
    # Read at most limit+1 bytes: enough to detect an oversized file without
    # pulling an arbitrarily large upload into memory.
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise PayloadTooLargeError(
            f"File exceeds {settings.max_upload_bytes} bytes.",
            details={"filename": file.filename, "max_bytes": settings.max_upload_bytes},
        )
    if not data:
        raise InvalidImageError("File is empty.", details={"filename": file.filename})
    return ImageInput(filename=file.filename, data=data)


def _to_schema(image: ScoredImage) -> Prediction:
    return Prediction(
        filename=image.filename,
        label=Label.FAKE if image.is_fake else Label.REAL,
        p_fake=round(image.p_fake, 6),
        p_real=round(image.p_real, 6),
    )


def _model_info(detector: Detector, threshold: float) -> ModelInfo:
    info = detector.info
    return ModelInfo(
        name=info.name,
        weights_sha256=info.weights_sha256,
        device=info.device,
        input_size=info.input_size,
        threshold=threshold,
    )


@router.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
async def predict(
    file: Annotated[UploadFile, File(description="A face-crop image (JPEG/PNG/WebP/BMP).")],
    service: PredictionServiceDep,
    detector: DetectorDep,
    settings: SettingsDep,
) -> PredictResponse:
    """Score one face image. Returns P(fake), P(real) and the thresholded label."""
    item = await _read_upload(file, settings)
    result = await run_in_threadpool(service.predict, [item])
    return PredictResponse(
        prediction=_to_schema(result.images[0]),
        model=_model_info(detector, service.threshold),
        inference_ms=result.inference_ms,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse, responses=ERROR_RESPONSES)
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="Several face-crop images.")],
    service: PredictionServiceDep,
    detector: DetectorDep,
    settings: SettingsDep,
) -> BatchPredictResponse:
    """Score several face images in one forward pass (all-or-nothing validation)."""
    if len(files) > settings.max_batch_files:
        raise TooManyFilesError(
            f"At most {settings.max_batch_files} files per request.",
            details={"received": len(files), "max": settings.max_batch_files},
        )
    items = [await _read_upload(f, settings) for f in files]
    result = await run_in_threadpool(service.predict, items)
    return BatchPredictResponse(
        predictions=[_to_schema(img) for img in result.images],
        model=_model_info(detector, service.threshold),
        inference_ms=result.inference_ms,
    )


@router.get("/model", response_model=ModelInfo, responses=ERROR_RESPONSES)
def model_info(detector: DetectorDep, settings: SettingsDep) -> ModelInfo:
    """Which model and weights are serving, on what device, with what threshold."""
    return _model_info(detector, settings.threshold)
