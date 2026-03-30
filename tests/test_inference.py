import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from core.exceptions import ImageProcessingException, InvalidImageException, InferenceException
from services.image_processor import ImageProcessor


@pytest.fixture
def rgb_png_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(200, 50, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def rgb_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (300, 300), color=(30, 150, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def rgba_png_bytes() -> bytes:
    img = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def grayscale_png_bytes() -> bytes:
    img = Image.new("L", (150, 150), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_png_bytes() -> bytes:
    img = Image.new("RGB", (10, 10), color=(100, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def b64_png(rgb_png_bytes) -> str:
    return base64.b64encode(rgb_png_bytes).decode("utf-8")


@pytest.fixture
def b64_png_with_prefix(b64_png) -> str:
    return f"data:image/png;base64,{b64_png}"


class TestImageProcessorOutput:
    def test_output_shape_from_bytes(self, rgb_png_bytes):
        tensor = ImageProcessor.process(rgb_png_bytes)
        assert tensor.shape == (1, 224, 224, 3)

    def test_output_dtype_float32(self, rgb_png_bytes):
        tensor = ImageProcessor.process(rgb_png_bytes)
        assert tensor.dtype == np.float32

    def test_output_shape_from_base64(self, b64_png):
        tensor = ImageProcessor.process(b64_png)
        assert tensor.shape == (1, 224, 224, 3)

    def test_output_shape_jpeg(self, rgb_jpeg_bytes):
        tensor = ImageProcessor.process(rgb_jpeg_bytes)
        assert tensor.shape == (1, 224, 224, 3)
        assert tensor.dtype == np.float32

    def test_small_image_resized_correctly(self, small_png_bytes):
        tensor = ImageProcessor.process(small_png_bytes)
        assert tensor.shape == (1, 224, 224, 3)


class TestImageProcessorPixelRange:
    def test_pixel_range_is_0_to_255(self, rgb_png_bytes):
        tensor = ImageProcessor.process(rgb_png_bytes)
        assert tensor.min() >= 0.0,   f"Min pixel {tensor.min()} < 0"
        assert tensor.max() <= 255.0, f"Max pixel {tensor.max()} > 255"

    def test_pixel_not_rescaled_to_0_1(self, rgb_jpeg_bytes):
        tensor = ImageProcessor.process(rgb_jpeg_bytes)
        assert tensor.max() > 1.0, (
            f"Pixel tampaknya di-rescale ke [0,1] (max={tensor.max():.4f}). "
            "Ini SALAH — EfficientNetB0 handle normalisasi internal."
        )

    def test_pixel_values_are_finite(self, rgb_png_bytes):
        tensor = ImageProcessor.process(rgb_png_bytes)
        assert np.all(np.isfinite(tensor)), "Ada nilai NaN atau Inf di tensor!"


class TestImageProcessorModeConversion:
    def test_rgba_converted_to_rgb(self, rgba_png_bytes):
        tensor = ImageProcessor.process(rgba_png_bytes)
        assert tensor.shape == (1, 224, 224, 3), "Channel harus 3 setelah konversi RGBA→RGB"

    def test_grayscale_converted_to_rgb(self, grayscale_png_bytes):
        tensor = ImageProcessor.process(grayscale_png_bytes)
        assert tensor.shape == (1, 224, 224, 3), "Channel harus 3 setelah konversi L→RGB"

    def test_output_has_3_channels_always(self, rgba_png_bytes):
        tensor = ImageProcessor.process(rgba_png_bytes)
        assert tensor.shape[-1] == 3


class TestImageProcessorBase64:
    def test_base64_without_prefix(self, b64_png):
        tensor = ImageProcessor.process(b64_png)
        assert tensor.shape == (1, 224, 224, 3)

    def test_base64_with_data_uri_prefix(self, b64_png_with_prefix):
        tensor = ImageProcessor.process(b64_png_with_prefix)
        assert tensor.shape == (1, 224, 224, 3)

    def test_base64_with_missing_padding(self, rgb_png_bytes):
        b64 = base64.b64encode(rgb_png_bytes).decode("utf-8").rstrip("=")
        tensor = ImageProcessor.process(b64)
        assert tensor.shape == (1, 224, 224, 3)


class TestImageProcessorErrors:
    def test_invalid_bytes_raises_exception(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(b"this is not an image file at all!!!")

    def test_empty_bytes_raises_exception(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(b"")

    def test_invalid_base64_raises_exception(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process("!!@#$%^&*()_INVALID_BASE64")

    def test_wrong_type_raises_exception(self):
        with pytest.raises((ImageProcessingException, TypeError, Exception)):
            ImageProcessor.process(12345)  # type: ignore

    def test_truncated_image_raises_exception(self):
        img = Image.new("RGB", (100, 100), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        truncated = buf.getvalue()[:50]
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(truncated)


class TestInferenceService:

    CLASS_NAMES = ["D101", "D13", "D197", "D198", "D2", "D200", "D24", "D88"]
    NUM_CLASSES = 8

    def _make_valid_tensor(self) -> np.ndarray:
        return np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)

    def _make_mock_session(self, output_probs: np.ndarray) -> MagicMock:
        session = MagicMock()
        session.run.return_value = [output_probs.reshape(1, -1)]
        return session

    def _make_mock_loader(self, session: MagicMock) -> MagicMock:
        loader = MagicMock()
        loader.session     = session
        loader.input_name  = "image_input"
        loader.output_name = "predictions"
        return loader

    def test_predict_returns_correct_top_class(self):
        from services.inference_service import InferenceService

        probs = np.array([0.02, 0.03, 0.85, 0.03, 0.02, 0.02, 0.02, 0.01], dtype=np.float32)
        session = self._make_mock_session(probs)
        loader  = self._make_mock_loader(session)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            response = InferenceService.predict(self._make_valid_tensor())

        assert response.prediction.class_name == "D197"
        assert response.success is True

    def test_predict_confidence_scores_contain_all_8_classes(self):
        from services.inference_service import InferenceService

        probs   = np.ones(8, dtype=np.float32) / 8.0
        session = self._make_mock_session(probs)
        loader  = self._make_mock_loader(session)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            response = InferenceService.predict(self._make_valid_tensor())

        assert len(response.confidence_scores) == 8
        for cls in self.CLASS_NAMES:
            assert cls in response.confidence_scores

    def test_predict_no_double_softmax(self):
        from services.inference_service import InferenceService

        probs = np.array([0.90, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01], dtype=np.float32)
        session = self._make_mock_session(probs)
        loader  = self._make_mock_loader(session)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            response = InferenceService.predict(self._make_valid_tensor())

        top_conf = response.prediction.confidence_score
        assert top_conf > 0.5, (
            f"Top confidence {top_conf:.4f} terlalu rendah. "
            "Kemungkinan ada double softmax!"
        )

    def test_predict_class_mismatch_raises_exception(self):
        from services.inference_service import InferenceService

        wrong_probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        session     = self._make_mock_session(wrong_probs)
        loader      = self._make_mock_loader(session)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            with pytest.raises(InferenceException) as exc_info:
                InferenceService.predict(self._make_valid_tensor())

        assert "sinkron" in exc_info.value.detail.lower() or "mismatch" in exc_info.value.detail.lower()

    def test_predict_invalid_input_shape_raises_exception(self):
        from services.inference_service import InferenceService

        probs   = np.ones(8, dtype=np.float32) / 8.0
        session = self._make_mock_session(probs)
        loader  = self._make_mock_loader(session)

        wrong_tensor = np.zeros((1, 128, 128, 3), dtype=np.float32)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            with pytest.raises(InferenceException):
                InferenceService.predict(wrong_tensor)

    def test_predict_logits_output_auto_softmax(self):
        from services.inference_service import InferenceService

        logits  = np.array([3.5, 0.1, 0.2, -0.5, 0.0, 0.1, -0.2, 0.3], dtype=np.float32)
        session = self._make_mock_session(logits)
        loader  = self._make_mock_loader(session)

        with patch("services.inference_service.get_model_loader", return_value=loader), \
             patch("services.inference_service.settings") as mock_settings:
            mock_settings.class_names_list = self.CLASS_NAMES
            mock_settings.APP_VERSION      = "1.0.0"

            response = InferenceService.predict(self._make_valid_tensor())

        total = sum(response.confidence_scores.values())
        assert abs(total - 1.0) < 0.01, f"Sum={total:.4f} setelah auto-softmax"
        assert response.prediction.class_name == "D101"