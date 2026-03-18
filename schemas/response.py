"""
Response schemas for the Durian Classification API.

Defines Pydantic models for structuring prediction results returned
to the client, including classification details and metadata.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Individual prediction result for a single class.

    Attributes:
        class_name: The predicted durian variety label.
        confidence_score: Prediction probability (0.0 to 1.0).
    """

    class_name: str = Field(
        ...,
        description="Predicted durian variety name.",
        json_schema_extra={"example": "Musang_King"},
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence probability between 0 and 1.",
        json_schema_extra={"example": 0.9542},
    )


class PredictionResponse(BaseModel):
    """Successful prediction response schema.

    Attributes:
        success: Whether the prediction completed successfully.
        prediction: The top predicted class and its confidence.
        confidence_scores: Full probability distribution across all classes.
        inference_time_ms: Time taken for model inference in milliseconds.
        model_version: Version string of the loaded model.
    """

    success: bool = Field(
        default=True,
        description="Whether the prediction was successful.",
    )
    prediction: PredictionResult = Field(
        ...,
        description="Top-1 predicted class with confidence score.",
    )
    confidence_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for all classes.",
        json_schema_extra={
            "example": {
                "Musang_King": 0.9542,
                "Duri_Hitam": 0.0231,
                "Sultan": 0.0156,
                "Golden_Phoenix": 0.0071,
            }
        },
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Inference time in milliseconds.",
        json_schema_extra={"example": 45.32},
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Version identifier of the loaded model.",
        json_schema_extra={"example": "1.0.0"},
    )


class HealthResponse(BaseModel):
    """Health check response schema.

    Attributes:
        status: Service health status string.
        model_loaded: Whether the ONNX model is loaded and ready.
        app_name: Name of the application.
        version: Application version string.
    """

    status: str = Field(
        ...,
        description="Service health status.",
        json_schema_extra={"example": "healthy"},
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded and ready for inference.",
    )
    app_name: str = Field(
        ...,
        description="Application name.",
        json_schema_extra={"example": "Durian Classification API"},
    )
    version: str = Field(
        ...,
        description="Application version.",
        json_schema_extra={"example": "1.0.0"},
    )


class ErrorResponse(BaseModel):
    """Standard error response schema.

    Attributes:
        success: Always False for error responses.
        error: Human-readable error type/name.
        detail: Detailed error description.
    """

    success: bool = Field(
        default=False,
        description="Always False for error responses.",
    )
    error: str = Field(
        ...,
        description="Error type or name.",
        json_schema_extra={"example": "InvalidImageException"},
    )
    detail: str = Field(
        ...,
        description="Detailed error description.",
        json_schema_extra={"example": "The uploaded file is not a valid image."},
    )
