import os
import time

import psutil
from fastapi import APIRouter, Depends

from app.core_dependencies import AuthResult, verify_api_key
from core.config import settings
from core.rate_limiter import get_rate_limiter
from models.model_loader import get_model_loader
from schemas.response import HealthResponse

router = APIRouter()
_startup_time = time.time()


@router.get(
    "/ping",
    summary     = "Liveness Check (Public)",
    description = "Endpoint publik tanpa autentikasi. Untuk load balancer health check.",
    tags        = ["System"],
    include_in_schema = True,
)
async def ping():
    return {
        "status":  "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Detailed Health Check (Protected)",
    description    = "Status detail service termasuk model, memory, dan rate limiter. Memerlukan API key.",
    tags           = ["System"],
)
async def health_check(
    auth: AuthResult = Depends(verify_api_key),
) -> HealthResponse:
    loader     = get_model_loader()
    is_loaded  = loader.is_loaded
    uptime_sec = int(time.time() - _startup_time)

    try:
        process    = psutil.Process(os.getpid())
        mem_mb     = process.memory_info().rss / (1024 * 1024)
        cpu_pct    = process.cpu_percent(interval=0.1)
    except Exception:
        mem_mb  = 0.0
        cpu_pct = 0.0

    rl_stats = get_rate_limiter().get_stats()

    return HealthResponse(
        status      = "healthy" if is_loaded else "degraded",
        model_loaded = is_loaded,
        app_name    = settings.APP_NAME,
        version     = settings.APP_VERSION,
        uptime_seconds     = uptime_sec,
        memory_usage_mb    = round(mem_mb, 1),
        cpu_percent        = round(cpu_pct, 1),
        rate_limiter_stats = rl_stats,
        config_summary     = {
            "num_classes":        settings.num_classes,
            "image_size":         settings.IMAGE_SIZE,
            "enhancement":        settings.ENABLE_ENHANCEMENT,
            "max_file_size_mb":   settings.MAX_FILE_SIZE_MB,
        },
    )