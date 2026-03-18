"""
Image processing service for the Durian Classification API.

Handles decoding, resizing, and normalizing images (from bytes or Base64)
to match the exact input requirements of the EfficientNetB0 ONNX model.
"""

import base64
import io
from typing import Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from core.config import settings
from core.exceptions import ImageProcessingException, InvalidImageException
from core.logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """Service class for processing durian images for inference.

    Prepares images for EfficientNetB0:
    1. Decodes from raw bytes or Base64.
    2. Converts to RGB (removing alpha/transparency if present).
    3. Resizes to the target dimensions defined in settings (e.g., 224x224).
    4. Converts to a numpy array of type float32.
    5. Normalizes pixel values (model-specific; typically 0-255 or 0-1 depending on training).
       *Note: Standard EfficientNet in Keras expects unscaled 0-255 inputs if it includes
       a built-in preprocessing layer, or standardized inputs otherwise. We assume
       standard ImageNet-style normalization here, or basic scaling, but will provide
       the Keras application's expected format.*
    6. Expands dimensions to create a batch of 1 (1, 224, 224, 3).
    """

    @staticmethod
    def _decode_image_bytes(image_data: bytes) -> Image.Image:
        """Decode raw bytes into a PIL Image.

        Args:
            image_data: Raw image bytes.

        Returns:
            A PIL Image object.

        Raises:
            InvalidImageException: If the bytes cannot be decoded into a valid image.
        """
        try:
            return Image.open(io.BytesIO(image_data))
        except UnidentifiedImageError as e:
            logger.error(f"Failed to identify image from bytes: {str(e)}")
            raise InvalidImageException(detail="Uploaded file is not a valid or supported image format.") from e
        except Exception as e:
            logger.error(f"Unexpected error decoding image bytes: {str(e)}")
            raise InvalidImageException(detail="Could not read the uploaded image data.") from e

    @staticmethod
    def _decode_base64_image(base64_str: str) -> Image.Image:
        """Decode a Base64 string into a PIL Image.

        Args:
            base64_str: Base64 encoded image string (without data URI prefix).

        Returns:
            A PIL Image object.

        Raises:
            InvalidImageException: If the Base64 string is invalid or not an image.
        """
        try:
            # Padding check/fix not strictly needed for standard b64decode but safe
            missing_padding = len(base64_str) % 4
            if missing_padding:
                base64_str += '=' * (4 - missing_padding)
            
            image_data = base64.b64decode(base64_str)
            return ImageProcessor._decode_image_bytes(image_data)
        except base64.binascii.Error as e:
            logger.error(f"Invalid Base64 encoding: {str(e)}")
            raise InvalidImageException(detail="Provided string is not valid Base64.") from e
        except InvalidImageException:
            raise
        except Exception as e:
            logger.error(f"Error decoding Base64 string to image: {str(e)}")
            raise InvalidImageException(detail="Failed to decode Base64 image.") from e

    @staticmethod
    def process(image_input: Union[bytes, str]) -> np.ndarray:
        """Process an image (bytes or Base64) into an inference-ready tensor.

        Args:
            image_input: Raw image bytes or a Base64 encoded string.

        Returns:
            A numpy array of shape (1, height, width, channels) of type float32.

        Raises:
            InvalidImageException: If decoding fails.
            ImageProcessingException: If resizing or numpy conversion fails.
        """
        logger.debug("Starting image processing.")
        
        # 1. Decode
        if isinstance(image_input, bytes):
            image = ImageProcessor._decode_image_bytes(image_input)
        elif isinstance(image_input, str):
            image = ImageProcessor._decode_base64_image(image_input)
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")

        try:
            # 2. Convert to RGB
            if image.mode != "RGB":
                logger.debug(f"Converting image from {image.mode} to RGB.")
                image = image.convert("RGB")

            # 3. Resize (using LANCZOS for high quality downsampling)
            target_size = settings.image_size_tuple
            logger.debug(f"Resizing image to {target_size}.")
            image = image.resize(target_size, Image.Resampling.LANCZOS)

            # 4. Convert to Numpy array
            img_array = np.array(image, dtype=np.float32)

            # 5. Normalize (EfficientNetB0 Keras implementation specifically expects inputs 
            #    to NOT be scaled to [0,1] or ImageNet Standardized if using the built-in
            #    preprocessing. However, commonly ONNX exports expect ImageNet preprocessing 
            #    or [0, 1] scaling depending on how the export was done.
            #    Assuming standard Keras tf.keras.applications.efficientnet.preprocess_input
            #    which actually leaves inputs as 0-255 because EfficientNet has a built-in
            #    normalization layer. We will leave it as 0-255 float32.)
            #    *If the model was trained differently, this normalization step needs adjustment.*
            pass # img_array remains 0.0 - 255.0

            # 6. Expand dimensions to (1, 224, 224, 3)
            tensor = np.expand_dims(img_array, axis=0)
            
            logger.debug(f"Image processed successfully. Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            return tensor

        except Exception as e:
            logger.error(f"Failed during image preprocessing: {str(e)}")
            raise ImageProcessingException(detail="Failed to preprocess the image for inference.") from e
