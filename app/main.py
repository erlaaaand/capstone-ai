from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
from core.rate_limiter import get_rate_limiter
from core.security import get_key_manager
from models.model_loader import get_model_loader
from services.clip_service import CLIPService
from agents.market_intelligence.scheduler import get_scheduler

logger = get_logger(__name__)


def _load_api_keys() -> None:
    get_key_manager().load_keys()


def _load_onnx_model() -> None:
    get_model_loader().load_model()
    logger.info("[Startup] ONNX model siap menerima request.")


def _load_clip_model() -> None:
    ready = CLIPService.warmup()
    if ready:
        logger.info("[Startup] CLIP model siap.")
    else:
        logger.warning(
            "[Startup] CLIP model gagal dimuat. "
            "Validasi zero-shot dinonaktifkan — semua gambar akan diizinkan."
        )


async def _start_rate_limiter() -> None:
    await get_rate_limiter().start_cleanup_task()
    logger.info("[Startup] Rate limiter cleanup task aktif.")


async def _start_scheduler() -> None:
    await get_scheduler().start()
    logger.info("[Startup] Market Intelligence Agent scheduler aktif.")


async def _stop_scheduler() -> None:
    await get_scheduler().stop()


async def _stop_rate_limiter() -> None:
    await get_rate_limiter().stop_cleanup_task()


def _safe_startup(label: str, fn) -> None:
    logger.info(f"[Startup] {label}...")
    try:
        fn()
    except Exception as e:
        logger.critical(f"[Startup] Gagal — {label}: {e}", exc_info=True)


async def _safe_startup_async(label: str, fn) -> None:
    logger.info(f"[Startup] {label}...")
    try:
        await fn()
    except Exception as e:
        logger.error(
            f"[Startup] Gagal — {label}: {e}. "
            "Service tetap berjalan tanpa fitur ini.",
            exc_info=True,
        )


def _safe_shutdown(label: str, fn) -> None:
    try:
        fn()
        logger.info(f"[Shutdown] {label} dihentikan.")
    except Exception as e:
        logger.error(f"[Shutdown] Error saat stop {label}: {e}", exc_info=True)


async def _safe_shutdown_async(label: str, fn) -> None:
    try:
        await fn()
        logger.info(f"[Shutdown] {label} dihentikan.")
    except Exception as e:
        logger.error(f"[Shutdown] Error saat stop {label}: {e}", exc_info=True)


async def _run_startup() -> None:
    logger.info("=" * 60)
    logger.info(f"  Memulai {settings.APP_NAME} v{settings.APP_VERSION}")
    if sys.platform == "win32":
        logger.info("  Platform: Windows — WindowsProactorEventLoopPolicy aktif.")
    logger.info("=" * 60)

    _safe_startup("Memuat API keys",   _load_api_keys)
    _safe_startup("Memuat ONNX model", _load_onnx_model)
    _safe_startup("Memuat CLIP model", _load_clip_model)
    await _safe_startup_async("Memulai rate limiter cleanup task", _start_rate_limiter)
    await _safe_startup_async("Memulai Market Intelligence scheduler", _start_scheduler)

    logger.info(
        f"[Startup] Config: classes={settings.num_classes} "
        f"| img_size={settings.IMAGE_SIZE} "
        f"| enhancement={settings.ENABLE_ENHANCEMENT} "
        f"| debug={settings.DEBUG}"
    )
    logger.info("[Startup] Service READY.")
    logger.info("=" * 60)


async def _run_shutdown() -> None:
    logger.info("[Shutdown] Memulai graceful shutdown...")
    await _safe_shutdown_async("Market Intelligence scheduler", _stop_scheduler)
    await _safe_shutdown_async("Rate limiter cleanup task", _stop_rate_limiter)
    _safe_shutdown("ONNX model", get_model_loader().unload_model)
    logger.info("[Shutdown] Selesai.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await _run_startup()
    yield
    await _run_shutdown()


def _build_openapi_schema(app: FastAPI) -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = (
            "## Durian Authenticity & Market Intelligence API\n\n"
            "API untuk dua fitur utama aplikasi konsumen:\n\n"
            "### 1. Klasifikasi Varietas Durian\n"
            "Validasi keaslian varietas durian dari gambar menggunakan "
            "EfficientNetB0 deep learning model.\n\n"
            "**Endpoint:** `POST /api/v1/predict`\n\n"
            "### 2. Informasi Harga Pasar\n"
            "Data harga durian premium (IDR/kg) dari marketplace Indonesia, "
            "diperbarui otomatis setiap hari oleh Market Intelligence Agent.\n\n"
            "**Endpoint:** `GET /api/v1/market/prices`\n\n"
            "**Endpoint:** `GET /api/v1/market/report`\n\n"
            "**Endpoint:** `GET /api/v1/market/diagnostics` *(Admin)*\n\n"
            "### Autentikasi\n"
            "Semua endpoint (kecuali `/ping`) memerlukan API key yang valid.\n\n"
            "**Header:** `X-API-Key: dk_live_your_key_here`\n\n"
            "**Alternatif:** `Authorization: Bearer dk_live_your_key_here`\n\n"
            "### Rate Limiting\n"
            "| Tier | Limit |\n"
            "|------|-------|\n"
            "| Free | 60 req/menit |\n"
            "| Standard | 300 req/menit |\n"
            "| Premium | 1000 req/menit |\n"
            "| Unlimited | Tidak terbatas |\n"
        ),
        routes  = app.routes,
        contact = {
            "name":  settings.API_SUPPORT_NAME,
            "email": settings.API_SUPPORT_EMAIL,
        },
    )

    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type":        "apiKey",
            "in":          "header",
            "name":        "X-API-Key",
            "description": "API key format `dk_live_xxx` (prod) atau `dk_test_xxx` (test).",
        },
        "BearerAuth": {
            "type":         "http",
            "scheme":       "bearer",
            "bearerFormat": "ApiKey",
            "description":  "Bearer token — masukkan API key setelah 'Bearer '.",
        },
    }
    schema["security"]    = [{"ApiKeyAuth": []}, {"BearerAuth": []}]
    app.openapi_schema    = schema
    return schema


def create_app() -> FastAPI:
    app = FastAPI(
        title       = settings.APP_NAME,
        version     = settings.APP_VERSION,
        description = "Durian Authenticity & Market Intelligence API",
        lifespan    = lifespan,
        docs_url    = "/docs"         if settings.DEBUG else None,
        redoc_url   = "/redoc"        if settings.DEBUG else None,
        openapi_url = "/openapi.json" if settings.DEBUG else None,
    )

    app.add_middleware(
        PayloadSizeLimitMiddleware,
        max_bytes=settings.max_file_size_bytes + (1024 * 1024),
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    allowed_hosts = settings.ALLOWED_HOSTS
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
            "X-RateLimit-Limit", "X-RateLimit-Remaining",
            "X-RateLimit-Reset", "X-API-Version",
        ],
        max_age = 600,
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)

    @app.exception_handler(DurianServiceException)
    async def _service_exc(request, exc: DurianServiceException):
        return JSONResponse(
            status_code = exc.status_code,
            content     = {
                "success":    False,
                "error":      exc.__class__.__name__,
                "detail":     exc.detail,
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
            headers = exc.headers or {},
        )

    @app.exception_handler(404)
    async def _not_found(request, exc):
        return JSONResponse(
            status_code = 404,
            content     = {
                "success":    False,
                "error":      "NotFound",
                "detail":     f"Endpoint '{request.url.path}' tidak ditemukan.",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    @app.exception_handler(405)
    async def _method_not_allowed(request, exc):
        return JSONResponse(
            status_code = 405,
            content     = {
                "success":    False,
                "error":      "MethodNotAllowed",
                "detail":     f"Method '{request.method}' tidak diizinkan di '{request.url.path}'.",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name":          settings.APP_NAME,
            "version":       settings.APP_VERSION,
            "status":        "operational",
            "docs":          "/docs" if settings.DEBUG else "disabled (production)",
            "endpoints": {
                "predict":            "POST /api/v1/predict",
                "market_prices":      "GET /api/v1/market/prices",
                "market_report":      "GET /api/v1/market/report",
                "market_diagnostics": "GET /api/v1/market/diagnostics",
                "health":             "GET /api/v1/health",
                "ping":               "GET /api/v1/ping",
            },
            "auth_required": True,
            "auth_header":   "X-API-Key",
        }

    app.include_router(api_router)
    app.openapi = lambda: _build_openapi_schema(app)

    return app


app = create_app()