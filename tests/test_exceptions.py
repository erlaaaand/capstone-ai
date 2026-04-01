"""
Tests untuk core/exceptions.py — DurianServiceException dan semua turunannya.
"""
import pytest
from fastapi import HTTPException, status

from core.exceptions import (
    DurianServiceException,
    FileTooLargeException,
    ImageProcessingException,
    InferenceException,
    InvalidImageException,
    ModelLoadException,
    ModelNotLoadedException,
    UnsupportedFileTypeException,
)


class TestDurianServiceException:
    def test_default_status_code_500(self):
        exc = DurianServiceException()
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_custom_detail_message(self):
        exc = DurianServiceException(detail="Test error message")
        assert exc.detail == "Test error message"

    def test_custom_status_code(self):
        exc = DurianServiceException(status_code=418)
        assert exc.status_code == 418

    def test_to_http_exception_returns_http_exception(self):
        exc  = DurianServiceException(detail="something went wrong", status_code=500)
        http = exc.to_http_exception()
        assert isinstance(http, HTTPException)

    def test_to_http_exception_status_code_matches(self):
        exc  = DurianServiceException(status_code=422)
        http = exc.to_http_exception()
        assert http.status_code == 422

    def test_to_http_exception_detail_matches(self):
        exc  = DurianServiceException(detail="test detail")
        http = exc.to_http_exception()
        assert http.detail == "test detail"

    def test_to_http_exception_with_custom_headers(self):
        headers = {"X-Custom": "value"}
        exc     = DurianServiceException(headers=headers)
        http    = exc.to_http_exception()
        assert http.headers == headers

    def test_is_exception_subclass(self):
        exc = DurianServiceException()
        assert isinstance(exc, Exception)

    def test_str_representation(self):
        exc = DurianServiceException(detail="custom message")
        assert "custom message" in str(exc)


class TestModelNotLoadedException:
    def test_status_code_is_503(self):
        exc = ModelNotLoadedException()
        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_default_message_not_empty(self):
        exc = ModelNotLoadedException()
        assert exc.detail != ""

    def test_custom_detail(self):
        exc = ModelNotLoadedException(detail="Model sedang loading")
        assert "Model sedang loading" in exc.detail

    def test_is_durian_service_exception(self):
        exc = ModelNotLoadedException()
        assert isinstance(exc, DurianServiceException)


class TestModelLoadException:
    def test_status_code_is_500(self):
        exc = ModelLoadException()
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_default_message_not_empty(self):
        exc = ModelLoadException()
        assert exc.detail != ""

    def test_is_durian_service_exception(self):
        assert isinstance(ModelLoadException(), DurianServiceException)


class TestInvalidImageException:
    def test_status_code_is_400(self):
        exc = InvalidImageException()
        assert exc.status_code == status.HTTP_400_BAD_REQUEST

    def test_custom_detail(self):
        exc = InvalidImageException(detail="File kosong.")
        assert exc.detail == "File kosong."

    def test_is_durian_service_exception(self):
        assert isinstance(InvalidImageException(), DurianServiceException)


class TestUnsupportedFileTypeException:
    def test_status_code_is_415(self):
        exc = UnsupportedFileTypeException()
        assert exc.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

    def test_custom_detail(self):
        exc = UnsupportedFileTypeException(detail="Tipe 'bmp' tidak didukung.")
        assert "bmp" in exc.detail

    def test_is_durian_service_exception(self):
        assert isinstance(UnsupportedFileTypeException(), DurianServiceException)


class TestFileTooLargeException:
    def test_status_code_is_413(self):
        exc = FileTooLargeException()
        assert exc.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_custom_detail(self):
        exc = FileTooLargeException(detail="Maksimum 10MB.")
        assert "10MB" in exc.detail

    def test_is_durian_service_exception(self):
        assert isinstance(FileTooLargeException(), DurianServiceException)


class TestImageProcessingException:
    def test_status_code_is_422(self):
        exc = ImageProcessingException()
        assert exc.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_is_durian_service_exception(self):
        assert isinstance(ImageProcessingException(), DurianServiceException)


class TestInferenceException:
    def test_status_code_is_500(self):
        exc = InferenceException()
        assert exc.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_is_durian_service_exception(self):
        assert isinstance(InferenceException(), DurianServiceException)


class TestExceptionHierarchy:
    """Memastikan semua exception bisa di-catch dengan base class."""

    ALL_EXCEPTIONS = [
        ModelNotLoadedException,
        ModelLoadException,
        InvalidImageException,
        UnsupportedFileTypeException,
        FileTooLargeException,
        ImageProcessingException,
        InferenceException,
    ]

    def test_all_catchable_as_durian_service_exception(self):
        for ExcClass in self.ALL_EXCEPTIONS:
            try:
                raise ExcClass()
            except DurianServiceException:
                pass  # expected
            except Exception as e:
                pytest.fail(f"{ExcClass.__name__} tidak ter-catch sebagai DurianServiceException: {e}")

    def test_all_have_non_empty_default_detail(self):
        for ExcClass in self.ALL_EXCEPTIONS:
            exc = ExcClass()
            assert exc.detail, f"{ExcClass.__name__} harus punya default detail"

    def test_all_to_http_exception_gives_valid_status(self):
        for ExcClass in self.ALL_EXCEPTIONS:
            exc  = ExcClass()
            http = exc.to_http_exception()
            assert 400 <= http.status_code < 600, (
                f"{ExcClass.__name__} menghasilkan status {http.status_code}"
            )
