"""
Main entrypoint for the Durian Classification API.

Configures the FastAPI application instance, sets up CORS middleware,
registers API routers, and manages the application lifespan
(loading/unloading the ONNX ML model).
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import api_router
from core.config import settings
from core.exceptions import DurianServiceException
from core.logger import get_logger
from models.model_loader import get_model_loader

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events.

    Startup: Loads the ONNX model into memory (Singleton) so it's ready
             for inference immediately upon the first request.
    Shutdown: Unloads the model and frees resources gracefully.

    Args:
        app: The FastAPI application instance.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # --- Startup ---
    try:
        model_loader = get_model_loader()
        model_loader.load_model()
        logger.info("Lifespan: ML Model loaded successfully. Service ready.")
    except Exception as e:
        logger.critical(f"Lifespan: Failed to load ML model during startup: {str(e)}")
        # We don't necessarily raise here; the /health endpoint will show degraded status,
        # and /predict will return 503 ModelNotLoadedException automatically.
        # Alternatively, raise e if you want the app to crash on failure.

    yield  # The application runs while here

    # --- Shutdown ---
    logger.info("Shutting down the service...")
    try:
        model_loader = get_model_loader()
        model_loader.unload_model()
        logger.info("Lifespan: ML Model unloaded automatically.")
    except Exception as e:
        logger.error(f"Lifespan: Error unloading ML model: {str(e)}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Computer Vision API for classifying premium Durian varieties using EfficientNetB0.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Middleware ---
    # Allow Cross-Origin Resource Sharing (CORS) for external frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Adjust in production securely
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # --- Exception Handlers ---
    @app.exception_handler(DurianServiceException)
    async def custom_exception_handler(request, exc: DurianServiceException):
        """Map custom DurianServiceExceptions globally to standard JSONResponses."""
        logger.warning(f"Handled Exception ({exc.status_code}): {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.__class__.__name__,
                "detail": exc.detail,
            },
            headers=exc.headers,
        )

    # --- Routers ---
    app.include_router(api_router)

    return app


# Application Entrypoint
app = create_app()
