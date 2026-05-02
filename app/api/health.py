# app/api/health.py

from __future__ import annotations

import os
import time

import psutil
from fastapi import APIRouter, Depends

from app.core_dependencies import AuthResult, verify_api_key, require_scope
from core.config import settings, reload_settings
from core.rate_limiter import get_rate_limiter
from core.security import KeyScope, get_key_manager
from models.model_loader import get_model_loader
from schemas.response import HealthResponse
from agents.market_intelligence.store import get_market_store
from agents.market_intelligence.scheduler import get_scheduler

router = APIRouter()
_startup_time = time.time()


@router.get(
    "/ping",
    summary           = "Liveness Check (Public)",
    description       = "Endpoint publik tanpa autentikasi. Untuk load balancer health check.",
    tags              = ["System"],
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
    description    = (
        "Status detail service termasuk model, memory, rate limiter, "
        "dan status data Market Intelligence.\n\n"
        "**Memerlukan API key.**"
    ),
    tags = ["System"],
)
async def health_check(
    auth: AuthResult = Depends(verify_api_key),
) -> HealthResponse:
    loader     = get_model_loader()
    is_loaded  = loader.is_loaded
    uptime_sec = int(time.time() - _startup_time)

    try:
        process = psutil.Process(os.getpid())
        mem_mb  = process.memory_info().rss / (1024 * 1024)
        cpu_pct = process.cpu_percent(interval=0.1)
    except Exception:
        mem_mb  = 0.0
        cpu_pct = 0.0

    rl_stats = get_rate_limiter().get_stats()

    store            = get_market_store()
    market_available = await store.has_data()
    market_stale     = await store.is_stale()

    scheduler         = get_scheduler()
    next_run_time     = scheduler.get_next_run_time() if scheduler.is_running else None

    return HealthResponse(
        status                 = "healthy" if is_loaded else "degraded",
        model_loaded           = is_loaded,
        app_name               = settings.APP_NAME,
        version                = settings.APP_VERSION,
        uptime_seconds         = uptime_sec,
        memory_usage_mb        = round(mem_mb, 1),
        cpu_percent            = round(cpu_pct, 1),
        rate_limiter_stats     = rl_stats,
        config_summary         = {
            "num_classes":      settings.num_classes,
            "image_size":       settings.IMAGE_SIZE,
            "enhancement":      settings.ENABLE_ENHANCEMENT,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        },
        market_data_available  = market_available,
        market_data_stale      = market_stale,
        market_next_run        = next_run_time,
    )


@router.post(
    "/admin/reload-keys",
    summary     = "Hot-Reload API Keys (Admin Only)",
    description = (
        "Reload API keys dari environment variables tanpa restart server.\n\n"
        "**Memerlukan scope `admin`.**"
    ),
    tags = ["Admin"],
)
async def reload_api_keys(
    auth: AuthResult = Depends(require_scope(KeyScope.ADMIN)),
):
    manager = get_key_manager()
    try:
        manager.load_keys()
        key_count      = manager.loaded_key_count()
        fresh_settings = reload_settings()
        return {
            "success":            True,
            "message":            "API keys dan settings berhasil di-reload.",
            "key_count":          key_count,
            "reloaded_by":        auth.key_prefix,
            "app_version":        fresh_settings.APP_VERSION,
            "settings_refreshed": True,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Gagal reload API keys: {str(e)}",
        }