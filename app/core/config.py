"""Application settings, loaded from environment variables (prefix ``DFD_``) or a .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    batch_size: int = Field(16, gt=0)
    max_concurrent_inferences: int = Field(2, gt=0)
    load_model_on_startup: bool = True

    # Upload limits
    max_upload_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_batch_files: int = Field(16, gt=0)
    allowed_content_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    )


@lru_cache
def get_settings() -> Settings:
    """Cached so settings are parsed once per process."""
    return Settings()
