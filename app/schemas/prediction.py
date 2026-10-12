"""Public API contract. These models define (and document, via OpenAPI) every response."""

from typing import Any

from pydantic import BaseModel, Field

from app.core.policy import Label, Mode


class ModelInfo(BaseModel):
    name: str
    weights_sha256: str = Field(description="Identifies exactly which weights produced a result.")
    device: str
    input_size: int


class ModelDetails(ModelInfo):
    default_threshold: float = Field(description="Used when a request sets no threshold/mode.")
    mode_thresholds: dict[Mode, float] = Field(description="Threshold behind each `mode`.")


class DecisionApplied(BaseModel):
    """Echoes how the label was decided, so a result can always be interpreted."""

    threshold: float
    mode: Mode | None
    uncertain_margin: float
    tta: bool


class Prediction(BaseModel):
    filename: str | None
    label: Label
    p_fake: float = Field(ge=0, le=1, description="Probability the face is manipulated.")
    p_real: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    prediction: Prediction
    decision: DecisionApplied
    model: ModelInfo
    inference_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    decision: DecisionApplied
    model: ModelInfo
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
