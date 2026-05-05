# core/__init__.py

from core.config import settings, get_settings
from core.logger import logger, get_logger, setup_logging
from core.exceptions import (
    DurianServiceException,
    ModelNotLoadedException,
    ModelLoadException,
    InvalidImageException,
    UnsupportedFileTypeException,
    FileTooLargeException,
    ImageProcessingException,
    InferenceException,
)
from core.varieties import (
    VarietyInfo,
    VARIETY_MAP,
    get_variety_info,
    get_display_name,
)
from core.file_validator import validate_upload, check_magic_bytes, MAGIC_BYTES

__all__ = [
    # Config
    "settings",
    "get_settings",
    # Logger
    "logger",
    "get_logger",
    "setup_logging",
    # Exceptions
    "DurianServiceException",
    "ModelNotLoadedException",
    "ModelLoadException",
    "InvalidImageException",
    "UnsupportedFileTypeException",
    "FileTooLargeException",
    "ImageProcessingException",
    "InferenceException",
    # Varieties
    "VarietyInfo",
    "VARIETY_MAP",
    "get_variety_info",
    "get_display_name",
    # File validation
    "validate_upload",
    "check_magic_bytes",
    "MAGIC_BYTES",
]