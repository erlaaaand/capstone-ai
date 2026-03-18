"""
Custom exception hierarchy for the Durian Classification API.

All service-level exceptions inherit from DurianServiceException and carry
an HTTP status code + human-readable detail message, enabling clean
translation to FastAPI HTTPException responses at the API boundary.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, status


# ============================================================
# Base Exception
# ============================================================

class DurianServiceException(Exception):
    """Base exception for all Durian ML Service errors.

    Attributes:
        status_code: HTTP status code to return to the client.
        detail: Human-readable error description.
        headers: Optional HTTP headers to include in the response.
    """

    def __init__(
        self,
        detail: str = "An internal service error occurred.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.headers = headers
        super().__init__(self.detail)

    def to_http_exception(self) -> HTTPException:
        """Convert this exception to a FastAPI HTTPException.

        Returns:
            HTTPException: Ready-to-raise FastAPI exception.
        """
        return HTTPException(
            status_code=self.status_code,
            detail=self.detail,
            headers=self.headers,
        )


# ============================================================
# Model Exceptions
# ============================================================

class ModelNotLoadedException(DurianServiceException):
    """Raised when the ONNX model is not loaded or unavailable."""

    def __init__(
        self,
        detail: str = "ML model is not loaded. The service is not ready for inference.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ModelLoadException(DurianServiceException):
    """Raised when the ONNX model fails to load from disk."""

    def __init__(
        self,
        detail: str = "Failed to load the ML model.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# ============================================================
# Image / File Validation Exceptions
# ============================================================

class InvalidImageException(DurianServiceException):
    """Raised when the uploaded file is not a valid image."""

    def __init__(
        self,
        detail: str = "The uploaded file is not a valid image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class UnsupportedFileTypeException(DurianServiceException):
    """Raised when the uploaded file has an unsupported extension."""

    def __init__(
        self,
        detail: str = "Unsupported file type. Please upload a JPG, JPEG, PNG, or WebP image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )


class FileTooLargeException(DurianServiceException):
    """Raised when the uploaded file exceeds the maximum allowed size."""

    def __init__(
        self,
        detail: str = "File size exceeds the maximum allowed limit.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


# ============================================================
# Processing & Inference Exceptions
# ============================================================

class ImageProcessingException(DurianServiceException):
    """Raised when image preprocessing (resize, normalize) fails."""

    def __init__(
        self,
        detail: str = "An error occurred while processing the image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class InferenceException(DurianServiceException):
    """Raised when the ONNX inference session encounters an error."""

    def __init__(
        self,
        detail: str = "An error occurred during model inference.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
