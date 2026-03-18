"""
API routes package definition.

Re-exports the endpoints for the FastAPI application router inclusion.
"""

from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.routes import router as predict_router

# Central API Router
api_router = APIRouter(prefix="/api/v1")

# Include the endpoints
api_router.include_router(predict_router)
api_router.include_router(health_router)
