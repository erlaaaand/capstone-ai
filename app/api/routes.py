"""
Main predict route configuration.

Exposes a POST endpoint at /api/v1/predict for image classification.
Supports both direct multipart/form-data file uploads and
application/json Base64 encoded payloads.
"""

from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi import Body

from app.core_dependencies import verify_api_key
from core.config import settings
from core.exceptions import (
    DurianServiceException,
    FileTooLargeException,
    InvalidImageException,
    UnsupportedFileTypeException,
)
from core.logger import get_logger
from schemas.request import PredictionRequestBase64
from schemas.response import PredictionResponse
from services.image_processor import ImageProcessor
from services.inference_service import InferenceService

logger = get_logger(__name__)

# Router setup with API Key dependency
router = APIRouter(
    dependencies=[Depends(verify_api_key)],
    tags=["Prediction"],
)


def _validate_upload_file(file: UploadFile) -> bytes:
    """Validate and extract bytes from an uploaded file.

    Checks file extension and size constraints defined in settings.

    Args:
        file: The FastAPI UploadFile object.

    Returns:
        The raw file bytes.

    Raises:
        UnsupportedFileTypeException: If the extension is invalid.
        FileTooLargeException: If the file size exceeds the limit.
        InvalidImageException: If the file cannot be read.
    """
    # 1. Validate Extension
    filename = file.filename or ""
    if "." not in filename:
        ext = ""
    else:
        ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in settings.allowed_extensions_set:
        error_msg = f"File type '{ext}' is not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
        logger.warning(error_msg)
        raise UnsupportedFileTypeException(detail=error_msg)

    # 2. Validate Size & Read Bytes
    try:
        # Read the entire file content into memory
        file_bytes = file.file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file: {str(e)}")
        raise InvalidImageException(detail="Could not read the uploaded file.") from e
        
    file_size_bytes = len(file_bytes)
    
    if file_size_bytes > settings.max_file_size_bytes:
        error_msg = (
            f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB "
            f"({settings.max_file_size_bytes} bytes), received {file_size_bytes} bytes."
        )
        logger.warning(error_msg)
        raise FileTooLargeException(detail=error_msg)
        
    if file_size_bytes == 0:
        logger.warning("Received empty file.")
        raise InvalidImageException(detail="The uploaded file is empty.")

    return file_bytes


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a Durian Image",
    description="Requires `X-API-Key` header. Upload an image (`multipart/form-data`) "
                "or send a Base64 JSON payload to identify the variety.",
)
async def predict_durian(
    file: Optional[UploadFile] = File(
        default=None, 
        description="Image file upload (multipart/form-data)."
    ),
    payload: Optional[PredictionRequestBase64] = Body(default=None),
) -> PredictionResponse:
    """Endpoint for durian classification.

    Accepts an image via `file` (UploadFile) OR `payload` (Base64 string).
    Passes the input to the ImageProcessor, then the InferenceService.

    Args:
        file: Multipart file upload.
        payload: JSON body with base64 encoded image.

    Returns:
        PredictionResponse with top class and fully mapped confidence scores.

    Raises:
        HTTPException: Custom DurianServiceExceptions map to specific HTTP codes.
    """
    # 1. Ensure exactly one input method is provided
    if file and payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide ONLY a file upload OR a JSON payload, not both.",
        )
    if not file and not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing image data. Upload a file or provide a base64 payload.",
        )

    logger.info("Received prediction request.")

    try:
        # 2. Extract RAW Input (bytes or Base64 string)
        raw_input = None
        if file:
            logger.debug(f"Processing UploadFile: {file.filename}")
            raw_input = _validate_upload_file(file)
        elif payload:
            logger.debug("Processing Base64 payload.")
            # Note: filename validation for Base64 (if provided)
            if payload.filename:
                # We reuse the logic by faking an UploadFile-like struct, or just duplicate extension check
                ext = payload.filename.rsplit(".", 1)[-1].lower() if "." in payload.filename else ""
                if ext and ext not in settings.allowed_extensions_set:
                    raise UnsupportedFileTypeException(
                        detail=f"File type '{ext}' is not supported. Allowed: {settings.ALLOWED_EXTENSIONS}"
                    )
            raw_input = payload.image_base64
            
        if raw_input is None:
            raise InvalidImageException(detail="Failed to extract image data.")

        # 3. Process Image (Resize, RGB, Numpy Normalize)
        # Process is synchronous but fast (CPU bound). In extremely high-load scenarios,
        # we might offload this to a asyncio.to_thread pool.
        image_tensor = ImageProcessor.process(raw_input)

        # 4. Run Inference
        response = InferenceService.predict(image_tensor)
        
        return response

    except DurianServiceException as e:
        # Handled custom exceptions directly map to HTTP codes
        raise e.to_http_exception()
    except Exception as e:
        # Unhandled, unexpected server errors
        logger.error(f"Unhandled exception during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred processing the prediction.",
        )
