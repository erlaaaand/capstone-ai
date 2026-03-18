"""
Health check route configuration.

Provides a simple GET endpoint to verify the service is running
and the ML model is successfully loaded into memory.
"""

from fastapi import APIRouter

from core.config import settings
from models.model_loader import get_model_loader
from schemas.response import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Verify the API is running and the ML model is loaded.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check application health and model loading status.

    Returns:
        HealthResponse containing status, app name, version, and model_loaded flag.
    """
    model_loader = get_model_loader()
    
    return HealthResponse(
        status="healthy" if model_loader.is_loaded else "degraded",
        model_loaded=model_loader.is_loaded,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
    )
