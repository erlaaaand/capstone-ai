"""
Core package for the Durian Classification API.

Re-exports key components for convenient imports:
    from core import settings, logger, get_logger
    from core import DurianServiceException, ModelNotLoadedException, ...
"""

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

__all__ = [
    # Config
    "settings",
    "get_settings",
    # Logging
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
]
