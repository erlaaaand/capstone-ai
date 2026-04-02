from contextlib import asynccontextmanager
from typing import AsyncGenerator, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api import api_router
from core.config import settings
from core.exceptions import DurianServiceException
from core.logger import get_logger
from core.middleware import (
    PayloadSizeLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from core.security import get_key_manager
from models.model_loader import get_model_loader

from services.clip_service import CLIPService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=" * 60)
    logger.info(f"  Memulai {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)

    logger.info("[Startup] Memuat API keys...")
    try:
        key_manager = get_key_manager()
        key_manager.load_keys()
    except Exception as e:
        logger.critical(f"[Startup] Gagal load API keys: {str(e)}")

    logger.info("[Startup] Memuat ONNX model...")
    try:
        model_loader = get_model_loader()
        model_loader.load_model()
        logger.info("[Startup] ONNX model siap menerima request.")
    except Exception as e:
        logger.critical(
            f"[Startup] Gagal load model: {str(e)}. "
            "Service berjalan tapi /predict akan return 503."
        )

    logger.info("[Startup] Memuat model CLIP untuk validasi gambar...")
    try:
        clip_ready = CLIPService.warmup()
        if clip_ready:
            logger.info("[Startup] CLIP model siap.")
        else:
            logger.warning(
                "[Startup] CLIP model gagal dimuat. "
                "Validasi zero-shot dinonaktifkan — semua gambar akan diizinkan."
            )
    except Exception as e:
        logger.error(f"[Startup] Error saat warmup CLIP: {str(e)}")

    logger.info(
        f"[Startup] Config: classes={settings.num_classes} "
        f"| img_size={settings.IMAGE_SIZE} "
        f"| enhancement={settings.ENABLE_ENHANCEMENT} "
        f"| debug={settings.DEBUG}"
    )
    logger.info("[Startup] Service READY.")
    logger.info("=" * 60)

    yield

    logger.info("[Shutdown] Memulai graceful shutdown...")
    try:
        get_model_loader().unload_model()
        logger.info("[Shutdown] ONNX model di-unload.")
    except Exception as e:
        logger.error(f"[Shutdown] Error unload model: {str(e)}")
    logger.info("[Shutdown] Selesai.")


def custom_openapi(app: FastAPI):
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = (
            "## Durian Variety Classification API\n\n"
            "API enterprise untuk klasifikasi varietas durian menggunakan "
            "EfficientNetB0 deep learning model.\n\n"
            "### Autentikasi\n"
            "Semua endpoint `/api/v1/predict` dan `/api/v1/health` memerlukan "
            "API key yang valid.\n\n"
            "**Header:** `X-API-Key: dk_live_your_key_here`\n\n"
            "**Alternatif:** `Authorization: Bearer dk_live_your_key_here`\n\n"
            "### Rate Limiting\n"
            "| Tier | Limit |\n"
            "|------|-------|\n"
            "| Free | 60 req/menit |\n"
            "| Standard | 300 req/menit |\n"
            "| Premium | 1000 req/menit |\n"
            "| Unlimited | Tidak terbatas |\n\n"
            "Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, "
            "`X-RateLimit-Reset`"
        ),
        routes      = app.routes,
        contact     = {
            "name":  "Durian API Support",
            "email": "api-support@example.com",
        },
    )

    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in":   "header",
            "name": "X-API-Key",
            "description": (
                "API key dalam format `dk_live_xxx` (production) "
                "atau `dk_test_xxx` (testing). "
                "Dapatkan key dari administrator."
            ),
        },
        "BearerAuth": {
            "type":         "http",
            "scheme":       "bearer",
            "bearerFormat": "ApiKey",
            "description":  "Bearer token — masukkan API key setelah 'Bearer '.",
        },
    }

    schema["security"] = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    app.openapi_schema = schema
    return schema


def create_app() -> FastAPI:
    app = FastAPI(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = "Enterprise Durian Classification API",
        lifespan    = lifespan,
        docs_url    = "/docs"    if settings.DEBUG else None,
        redoc_url   = "/redoc"   if settings.DEBUG else None,
        openapi_url = "/openapi.json" if settings.DEBUG else None,
    )

    app.add_middleware(
        PayloadSizeLimitMiddleware,
        max_bytes = settings.max_file_size_bytes + (1024 * 1024),
    )

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    allowed_hosts = settings.ALLOWED_HOSTS if hasattr(settings, "ALLOWED_HOSTS") else ["*"]
    if allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.CORS_ORIGINS,
        allow_credentials = False,
        allow_methods     = ["POST", "GET", "OPTIONS"],
        allow_headers     = ["X-API-Key", "Authorization", "Content-Type", "Accept"],
        expose_headers    = [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-API-Version",
        ],
        max_age           = 600,
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.add_middleware(SecurityHeadersMiddleware)

    @app.exception_handler(DurianServiceException)
    async def service_exception_handler(request, exc: DurianServiceException):
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code = exc.status_code,
            content     = {
                "success":    False,
                "error":      exc.__class__.__name__,
                "detail":     exc.detail,
                "request_id": request_id,
            },
            headers     = exc.headers or {},
        )

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code = 404,
            content     = {
                "success":    False,
                "error":      "NotFound",
                "detail":     f"Endpoint '{request.url.path}' tidak ditemukan.",
                "request_id": request_id,
            },
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request, exc):
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code = 405,
            content     = {
                "success":    False,
                "error":      "MethodNotAllowed",
                "detail":     f"Method '{request.method}' tidak diizinkan di '{request.url.path}'.",
                "request_id": request_id,
            },
        )

    @app.get(
        "/",
        include_in_schema = False,
        summary           = "API Info",
    )
    async def root():
        return {
            "name":         settings.APP_NAME,
            "version":      settings.APP_VERSION,
            "status":       "operational",
            "docs":         "/docs" if settings.DEBUG else "disabled (production)",
            "health":       "/api/v1/health",
            "predict":      "/api/v1/predict",
            "auth_required": True,
            "auth_header":  "X-API-Key",
        }

    app.include_router(api_router)

    app.openapi = lambda: custom_openapi(app)

    return app


app = create_app()
