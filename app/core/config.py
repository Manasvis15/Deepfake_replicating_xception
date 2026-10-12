"""Application settings, loaded from environment variables (prefix ``DFD_``) or a .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.policy import Mode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DFD_", env_file=".env", extra="ignore")

    # Service
    app_name: str = "Deepfake Detection API"
    app_version: str = "0.1.0"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    cors_origins: list[str] = Field(default_factory=list)
    api_key: SecretStr | None = None  # if set, every /api request needs X-API-Key

    # Model
    weights_path: Path = Path("weights/xception_best.pth")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    input_size: int = Field(256, gt=0)
    threshold: float = Field(0.5, ge=0.0, le=1.0)  # used when a request sets neither
    # Thresholds behind each `mode`. Choose them from your own ROC curve: "balanced" is
    # the best threshold from our evaluation; the other two are starting points.
    mode_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"balanced": 0.45, "high_recall": 0.25, "high_precision": 0.8}
    )
    batch_size: int = Field(16, gt=0)
    max_concurrent_inferences: int = Field(2, gt=0)
    load_model_on_startup: bool = True

    # Upload limits
    max_upload_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_batch_files: int = Field(16, gt=0)
    allowed_content_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    )

    @field_validator("mode_thresholds")
    @classmethod
    def _check_mode_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        expected = {m.value for m in Mode}
        if set(value) != expected:
            raise ValueError(f"mode_thresholds must define exactly {sorted(expected)}")
        if not all(0.0 <= t <= 1.0 for t in value.values()):
            raise ValueError("mode thresholds must be between 0 and 1")
        return value


@lru_cache
def get_settings() -> Settings:
    """Cached so settings are parsed once per process."""
    return Settings()
