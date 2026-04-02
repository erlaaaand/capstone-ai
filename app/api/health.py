"""
app/api/health.py — PERBAIKAN:

  FIX #5 ADDED : Endpoint POST /admin/reload-keys untuk hot-reload API keys
                 tanpa perlu restart server. Memungkinkan zero-downtime key
                 rotation yang sudah didokumentasikan di README tapi sebelumnya
                 tetap butuh restart server.

  Endpoint baru : POST /api/v1/admin/reload-keys  (scope: admin)
"""

import os
import time

import psutil
from fastapi import APIRouter, Depends

from app.core_dependencies import AuthResult, verify_api_key, require_scope
from core.config import settings
from core.rate_limiter import get_rate_limiter
from core.security import KeyScope, get_key_manager
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


# ── FIX #5: Hot-reload API keys ────────────────────────────────────────────────
@router.post(
    "/admin/reload-keys",
    summary     = "Hot-Reload API Keys (Admin Only)",
    description = (
        "Reload API keys dari environment variables tanpa restart server.\n\n"
        "Gunakan untuk **zero-downtime key rotation**:\n"
        "1. Tambahkan key baru ke `.env` atau environment variables\n"
        "2. Panggil endpoint ini\n"
        "3. Set key lama sebagai `API_KEY_N_DEPRECATED=true`\n"
        "4. Panggil endpoint ini lagi\n"
        "5. Hapus key lama setelah semua client update\n\n"
        "**Memerlukan scope `admin`.**"
    ),
    tags        = ["Admin"],
)
async def reload_api_keys(
    auth: AuthResult = Depends(require_scope(KeyScope.ADMIN)),
):
    """
    Reload API keys dari environment variables secara hot.

    SEBELUM FIX #5: Rotasi key SELALU butuh restart server meskipun
                    README menjanjikan zero-downtime rotation.

    SESUDAH FIX #5: Key baru aktif dalam hitungan detik tanpa downtime.
    """
    manager = get_key_manager()

    # Reload keys dari env (dotenv akan di-load ulang di dalam load_keys())
    try:
        manager.load_keys()
        key_count = len(manager._keys)
        return {
            "success":    True,
            "message":    f"API keys berhasil di-reload.",
            "key_count":  key_count,
            "reloaded_by": auth.key_prefix,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Gagal reload API keys: {str(e)}",
        }
