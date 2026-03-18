"""
Core configuration module for the Durian Classification API.

Uses pydantic-settings to load and validate environment variables
from a .env file with strong type safety and sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        APP_NAME: Display name for the API service.
        APP_VERSION: Semantic version string.
        DEBUG: Toggle debug mode (verbose errors, auto-reload).
        LOG_LEVEL: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        MODEL_PATH: Relative or absolute path to the ONNX model file.
        CLASS_NAMES: Comma-separated list of durian variety class labels.
        IMAGE_SIZE: Target dimension for image preprocessing (square).
        ALLOWED_EXTENSIONS: Comma-separated list of accepted image file extensions.
        MAX_FILE_SIZE_MB: Maximum upload file size in megabytes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "Durian Classification API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- Model Configuration ---
    MODEL_PATH: str = "models/weights/efficientnet_b0.onnx"
    CLASS_NAMES: str = "D101,D13,D197,D198,D2,D200,D24,D88"

    # --- Image Processing ---
    IMAGE_SIZE: int = 224

    # --- File Upload Constraints ---
    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,webp"
    MAX_FILE_SIZE_MB: int = 10

    # --- Validators ---
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL '{v}'. Must be one of: {valid_levels}"
            )
        return upper

    @field_validator("IMAGE_SIZE")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        """Ensure image size is a positive integer within reasonable bounds."""
        if not 32 <= v <= 1024:
            raise ValueError(
                f"IMAGE_SIZE must be between 32 and 1024, got {v}"
            )
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        """Ensure max file size is a positive integer."""
        if v <= 0:
            raise ValueError(
                f"MAX_FILE_SIZE_MB must be positive, got {v}"
            )
        return v

    # --- Computed Properties ---
    @property
    def class_names_list(self) -> List[str]:
        """Return class names as a list of strings."""
        return [name.strip() for name in self.CLASS_NAMES.split(",")]

    @property
    def num_classes(self) -> int:
        """Return the number of classification classes."""
        return len(self.class_names_list)

    @property
    def image_size_tuple(self) -> Tuple[int, int]:
        """Return image size as a (width, height) tuple."""
        return (self.IMAGE_SIZE, self.IMAGE_SIZE)

    @property
    def allowed_extensions_set(self) -> set:
        """Return allowed extensions as a set of lowercase strings."""
        return {ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")}

    @property
    def max_file_size_bytes(self) -> int:
        """Return maximum file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def model_abs_path(self) -> Path:
        """Return the absolute path to the ONNX model file."""
        path = Path(self.MODEL_PATH)
        if path.is_absolute():
            return path
        return Path.cwd() / path


@lru_cache()
def get_settings() -> Settings:
    """Create and cache a Settings instance (singleton pattern).

    Returns:
        Settings: The validated application settings.
    """
    return Settings()


# Module-level convenience alias
settings: Settings = get_settings()
