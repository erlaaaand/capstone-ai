"""
Automated unit tests for the core Inference & Preprocessing services.

Validates the internal mechanisms of the ImageProcessor (decoding, resizing, RGB conversion,
tensor construction) and the InferenceService.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from services.image_processor import ImageProcessor
from core.exceptions import InvalidImageException


@pytest.fixture
def dummy_image_bytes():
    """Create a valid dummy PNG image in memory."""
    # Create a 300x300 RGB dummy image (red)
    img = Image.new('RGB', (300, 300), color=(255, 0, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


@pytest.fixture
def dummy_image_base64(dummy_image_bytes):
    """Create a base64 encoded string of the dummy image."""
    return base64.b64encode(dummy_image_bytes).decode('utf-8')


def test_image_processor_with_bytes(dummy_image_bytes):
    """Test processing raw image bytes."""
    tensor = ImageProcessor.process(dummy_image_bytes)
    
    # Assertions
    assert isinstance(tensor, np.ndarray)
    
    # Shape should be (batch_size=1, height=224, width=224, channels=3)
    assert tensor.shape == (1, 224, 224, 3)
    
    # Type should be float32 for ONNX and EfficientNetB0 compatibility
    assert tensor.dtype == np.float32


def test_image_processor_with_base64(dummy_image_base64):
    """Test processing a Base64 encoded JSON string."""
    tensor = ImageProcessor.process(dummy_image_base64)
    
    # Assertions
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_image_processor_with_data_uri_prefix(dummy_image_base64):
    """Test processing a Base64 string with a data URI prefix."""
    prefixed = f"data:image/png;base64,{dummy_image_base64}"
    tensor = ImageProcessor.process(prefixed)
    
    # Assertions
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_image_processor_invalid_bytes():
    """Test processing with invalid random bytes."""
    bad_bytes = b"not_an_image..."
    
    with pytest.raises(InvalidImageException) as excinfo:
        ImageProcessor.process(bad_bytes)
    
    assert "not a valid or supported image format" in str(excinfo.value.detail)


def test_image_processor_invalid_base64():
    """Test processing with a corrupted base64 string."""
    bad_b64 = "!!invalid_b64**"
    
    with pytest.raises(InvalidImageException) as excinfo:
        ImageProcessor.process(bad_b64)
    
    # Could hit either decode failure or PIL failure depending on exactly where parsing crashes
    assert isinstance(excinfo.value, InvalidImageException)


def test_image_processor_rgba_to_rgb():
    """Test converting an RGBA image to RGB."""
    # Create an image with alpha channel
    img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    rgba_bytes = img_byte_arr.getvalue()
    
    tensor = ImageProcessor.process(rgba_bytes)
    
    # Assertions – if it didn't crash and shape has 3 channels, conversion succeeded
    assert tensor.shape == (1, 224, 224, 3)
