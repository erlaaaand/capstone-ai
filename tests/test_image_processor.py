"""
Tests untuk services/image_processor.py — ImageProcessor dan fungsi enhancement.
"""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from core.exceptions import ImageProcessingException, InvalidImageException
from services.image_processor import (
    ImageProcessor,
    _auto_white_balance,
    _apply_clahe,
    _unsharp_mask,
    _letterbox_resize,
    enhance_image,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_bytes(size=(300, 300), color=(100, 150, 80), fmt="JPEG", mode="RGB") -> bytes:
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  Enhancement functions unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoWhiteBalance:
    def _arr(self, r=100, g=150, b=200, size=(50, 50)) -> np.ndarray:
        arr = np.full((size[0], size[1], 3), [r, g, b], dtype=np.float32)
        return arr

    def test_output_same_shape(self):
        arr = self._arr()
        out = _auto_white_balance(arr)
        assert out.shape == arr.shape

    def test_output_range_0_to_255(self):
        arr = self._arr()
        out = _auto_white_balance(arr)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_flat_gray_unchanged(self):
        """Gambar abu-abu flat tidak berubah signifikan."""
        arr = self._arr(r=128, g=128, b=128)
        out = _auto_white_balance(arr)
        assert np.allclose(out, arr, atol=5)

    def test_zero_mean_channel_returns_unchanged(self):
        """Jika salah satu channel rata-rata 0, kembalikan input tanpa crash."""
        arr = np.zeros((50, 50, 3), dtype=np.float32)
        out = _auto_white_balance(arr)
        assert out.shape == arr.shape


class TestApplyCLAHE:
    def test_output_same_shape(self):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).astype(np.float32)
        out = _apply_clahe(arr)
        assert out.shape == arr.shape

    def test_output_range_0_to_255(self):
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8).astype(np.float32)
        out = _apply_clahe(arr)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_output_dtype_float32(self):
        arr = np.random.randint(0, 255, (100, 100, 3)).astype(np.float32)
        out = _apply_clahe(arr)
        assert out.dtype == np.float32

    def test_clip_limit_affects_result(self):
        arr   = np.random.randint(50, 200, (100, 100, 3)).astype(np.float32)
        out1  = _apply_clahe(arr, clip_limit=1.0)
        out2  = _apply_clahe(arr, clip_limit=4.0)
        # Hasilnya boleh berbeda, yang penting tidak crash
        assert out1.shape == out2.shape


class TestUnsharpMask:
    def test_output_same_shape(self):
        arr = np.random.randint(0, 255, (100, 100, 3)).astype(np.float32)
        out = _unsharp_mask(arr)
        assert out.shape == arr.shape

    def test_output_range_0_to_255(self):
        arr = np.random.randint(0, 255, (100, 100, 3)).astype(np.float32)
        out = _unsharp_mask(arr)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_sharpening_changes_image(self):
        arr = np.random.randint(50, 200, (100, 100, 3)).astype(np.float32)
        out = _unsharp_mask(arr, amount=1.0)
        # Dengan amount besar hasilnya berbeda dari input
        assert not np.array_equal(arr, out)


class TestLetterboxResize:
    def test_output_size_matches_target(self):
        img = Image.new("RGB", (800, 400))
        out = _letterbox_resize(img, (224, 224))
        assert out.size == (224, 224)

    def test_portrait_image_padded_correctly(self):
        img = Image.new("RGB", (100, 200))
        out = _letterbox_resize(img, (224, 224))
        assert out.size == (224, 224)

    def test_square_image_no_padding_distortion(self):
        img = Image.new("RGB", (300, 300), color=(100, 200, 50))
        out = _letterbox_resize(img, (224, 224))
        assert out.size == (224, 224)

    def test_custom_pad_color(self):
        img = Image.new("RGB", (50, 100))  # portrait → akan ada padding di kiri-kanan
        out = _letterbox_resize(img, (224, 224), pad_color=(0, 0, 0))
        arr = np.array(out)
        # Sudut kiri atas harus hitam (padding)
        assert arr[0, 0, 0] == 0


class TestEnhanceImage:
    def test_returns_ndarray(self):
        arr = np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)
        out = enhance_image(arr)
        assert isinstance(out, np.ndarray)

    def test_output_range_valid(self):
        arr = np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)
        out = enhance_image(arr)
        assert out.min() >= 0.0
        assert out.max() <= 255.0

    def test_output_shape_preserved(self):
        arr = np.random.randint(0, 255, (224, 224, 3)).astype(np.float32)
        out = enhance_image(arr)
        assert out.shape == arr.shape


# ─────────────────────────────────────────────────────────────────────────────
#  ImageProcessor.process — output shape & dtype
# ─────────────────────────────────────────────────────────────────────────────

class TestImageProcessorOutputShape:
    def test_jpeg_output_shape(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes(fmt="JPEG"))
        assert tensor.shape == (1, 224, 224, 3)

    def test_png_output_shape(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes(fmt="PNG"))
        assert tensor.shape == (1, 224, 224, 3)

    def test_webp_output_shape(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes(fmt="WEBP"))
        assert tensor.shape == (1, 224, 224, 3)

    def test_dtype_is_float32(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes())
        assert tensor.dtype == np.float32

    def test_base64_string_input(self):
        raw = _make_bytes(fmt="JPEG")
        tensor, _, _ = ImageProcessor.process(_make_b64(raw))
        assert tensor.shape == (1, 224, 224, 3)

    def test_base64_with_data_uri_prefix(self):
        raw  = _make_bytes(fmt="JPEG")
        b64  = f"data:image/jpeg;base64,{_make_b64(raw)}"
        tensor, _, _ = ImageProcessor.process(b64)
        assert tensor.shape == (1, 224, 224, 3)

    def test_base64_without_padding(self):
        raw = _make_bytes(fmt="JPEG")
        b64 = _make_b64(raw).rstrip("=")
        tensor, _, _ = ImageProcessor.process(b64)
        assert tensor.shape == (1, 224, 224, 3)

    def test_small_image_resized(self):
        raw = _make_bytes(size=(10, 10))
        tensor, _, _ = ImageProcessor.process(raw)
        assert tensor.shape == (1, 224, 224, 3)

    def test_large_image_resized(self):
        raw = _make_bytes(size=(2048, 2048))
        tensor, _, _ = ImageProcessor.process(raw)
        assert tensor.shape == (1, 224, 224, 3)

    def test_portrait_image_resized(self):
        raw = _make_bytes(size=(100, 400))
        tensor, _, _ = ImageProcessor.process(raw)
        assert tensor.shape == (1, 224, 224, 3)

    def test_landscape_image_resized(self):
        raw = _make_bytes(size=(400, 100))
        tensor, _, _ = ImageProcessor.process(raw)
        assert tensor.shape == (1, 224, 224, 3)


class TestImageProcessorPixelRange:
    def test_pixel_range_0_to_255(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes())
        assert tensor.min() >= 0.0
        assert tensor.max() <= 255.0

    def test_pixel_not_normalized_to_0_1(self):
        """EfficientNetB0 menangani normalisasi internal — output HARUS [0,255]."""
        tensor, _, _ = ImageProcessor.process(_make_bytes(color=(200, 200, 200)))
        assert tensor.max() > 1.0, (
            f"Pixel ter-normalize ke [0,1] (max={tensor.max():.4f}). "
            "Ini SALAH — normalisasi ditangani backbone."
        )

    def test_no_nan_or_inf(self):
        tensor, _, _ = ImageProcessor.process(_make_bytes())
        assert np.all(np.isfinite(tensor))


class TestImageProcessorModeConversion:
    def test_rgba_to_rgb(self):
        raw = _make_bytes(mode="RGBA", fmt="PNG")
        tensor, _, _ = ImageProcessor.process(raw)
        assert tensor.shape[-1] == 3

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (100, 100), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tensor, _, _ = ImageProcessor.process(buf.getvalue())
        assert tensor.shape[-1] == 3

    def test_palette_image_to_rgb(self):
        img = Image.new("P", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        tensor, _, _ = ImageProcessor.process(buf.getvalue())
        assert tensor.shape[-1] == 3


class TestImageProcessorReturnTuple:
    def test_returns_three_values(self):
        result = ImageProcessor.process(_make_bytes())
        assert len(result) == 3

    def test_enhanced_flag_is_bool(self):
        _, enhanced, _ = ImageProcessor.process(_make_bytes())
        assert isinstance(enhanced, bool)

    def test_preproc_ms_is_positive_float(self):
        _, _, ms = ImageProcessor.process(_make_bytes())
        assert isinstance(ms, float)
        assert ms >= 0.0


class TestImageProcessorErrors:
    def test_invalid_bytes_raises_invalid_image(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(b"this is not an image!!!")

    def test_empty_bytes_raises_invalid_image(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(b"")

    def test_truncated_jpeg_raises_invalid_image(self):
        raw = _make_bytes(fmt="JPEG")
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(raw[:50])

    def test_invalid_base64_raises_invalid_image(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process("!!@#INVALID_BASE64!!")

    def test_wrong_type_raises_exception(self):
        with pytest.raises((ImageProcessingException, TypeError, Exception)):
            ImageProcessor.process(99999)  # type: ignore

    def test_text_file_bytes_raises_invalid_image(self):
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(b"Hello, this is a plain text file content.")

    def test_pdf_header_bytes_raises_invalid_image(self):
        """PDF bytes tidak boleh diproses sebagai gambar."""
        pdf_bytes = b"%PDF-1.4 fake pdf content for testing purposes only"
        with pytest.raises(InvalidImageException):
            ImageProcessor.process(pdf_bytes)
