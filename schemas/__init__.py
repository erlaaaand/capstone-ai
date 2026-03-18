"""
Schemas package for the Durian Classification API.

Re-exports all Pydantic models for convenient imports:
    from schemas import PredictionResponse, PredictionRequestBase64, ...
"""

from schemas.request import PredictionRequestBase64
from schemas.response import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    PredictionResult,
)

__all__ = [
    # Request
    "PredictionRequestBase64",
    # Response
    "PredictionResult",
    "PredictionResponse",
    "HealthResponse",
    "ErrorResponse",
]
