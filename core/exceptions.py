from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class DurianServiceException(Exception):

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
        return HTTPException(
            status_code=self.status_code,
            detail=self.detail,
            headers=self.headers,
        )


class ModelNotLoadedException(DurianServiceException):

    def __init__(
        self,
        detail: str = "ML model is not loaded. The service is not ready for inference.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ModelLoadException(DurianServiceException):

    def __init__(
        self,
        detail: str = "Failed to load the ML model.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class InvalidImageException(DurianServiceException):

    def __init__(
        self,
        detail: str = "The uploaded file is not a valid image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class UnsupportedFileTypeException(DurianServiceException):

    def __init__(
        self,
        detail: str = "Unsupported file type. Please upload a JPG, JPEG, PNG, or WebP image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )


class FileTooLargeException(DurianServiceException):

    def __init__(
        self,
        detail: str = "File size exceeds the maximum allowed limit.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


class ImageProcessingException(DurianServiceException):

    def __init__(
        self,
        detail: str = "An error occurred while processing the image.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class InferenceException(DurianServiceException):

    def __init__(
        self,
        detail: str = "An error occurred during model inference.",
    ) -> None:
        super().__init__(
            detail=detail,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
