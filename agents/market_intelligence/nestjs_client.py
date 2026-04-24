# agents/market_intelligence/nestjs_client.py
"""
HTTP client untuk mengirim MarketReportPayload ke NestJS backend.

Fitur:
  - Retry dengan exponential backoff
  - HMAC signature header untuk validasi internal-service
  - Logging setiap attempt
  - Tidak memblokir event loop FastAPI
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from typing import Optional

import httpx

from core.logger import get_logger
from agents.market_intelligence.config import NESTJS_CONFIG, NestJSClientConfig
from agents.market_intelligence.schemas import MarketReportPayload

logger = get_logger("agent.nestjs_client")


# ---------------------------------------------------------------------------
# HMAC Signature (optional security — skip jika NESTJS_INTERNAL_API_KEY kosong)
# ---------------------------------------------------------------------------

def _build_signature(payload_bytes: bytes, secret: str) -> str:
    """
    Buat HMAC-SHA256 signature dari payload untuk verifikasi di NestJS.
    Format: `sha256=<hex_digest>`
    """
    if not secret:
        return ""
    sig = hmac.new(
        secret.encode("utf-8"),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={sig}"


# ---------------------------------------------------------------------------
# Core sender
# ---------------------------------------------------------------------------

async def send_report(
    payload: MarketReportPayload,
    config: Optional[NestJSClientConfig] = None,
) -> bool:
    """
    Kirim MarketReportPayload ke NestJS via HTTP POST.

    Kembalikan True jika berhasil (2xx), False jika semua retry gagal.
    """
    if config is None:
        config = NESTJS_CONFIG

    url = f"{config.base_url.rstrip('/')}{config.endpoint}"

    # Serialize payload
    payload_dict = payload.model_dump(mode="json")
    payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

    # Build headers
    headers = {
        "Content-Type":    "application/json; charset=utf-8",
        "X-Agent-Version": payload.agent_version,
        "X-Run-ID":        payload.run_id,
    }

    if config.api_key:
        headers["X-Internal-API-Key"] = config.api_key

    signature = _build_signature(payload_bytes, config.api_key)
    if signature:
        headers["X-Signature"] = signature

    logger.info(
        f"[NestJS] Mengirim laporan ke {url} "
        f"(run_id={payload.run_id}, entries={payload.entry_count}, "
        f"status={payload.status.value})"
    )

    for attempt in range(1, config.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
                response = await client.post(
                    url,
                    content=payload_bytes,
                    headers=headers,
                )

            if response.is_success:
                logger.info(
                    f"[NestJS] Berhasil kirim laporan "
                    f"(status={response.status_code}, attempt={attempt})."
                )
                return True

            # 4xx — jangan retry (client error, bukan transient)
            if 400 <= response.status_code < 500:
                logger.error(
                    f"[NestJS] Client error {response.status_code}: "
                    f"{response.text[:300]}. Tidak di-retry."
                )
                return False

            # 5xx — retry
            logger.warning(
                f"[NestJS] Server error {response.status_code} "
                f"(attempt={attempt}/{config.max_retries}). "
                f"Response: {response.text[:200]}"
            )

        except httpx.ConnectError:
            logger.error(
                f"[NestJS] Tidak bisa terhubung ke {url} (attempt={attempt}). "
                "Pastikan NestJS service berjalan."
            )
        except httpx.TimeoutException:
            logger.warning(
                f"[NestJS] Timeout setelah {config.timeout_sec}s "
                f"(attempt={attempt}/{config.max_retries})."
            )
        except Exception as e:
            logger.error(
                f"[NestJS] Error tidak terduga (attempt={attempt}): {e}",
                exc_info=True,
            )

        if attempt < config.max_retries:
            backoff = config.retry_backoff_sec * (2 ** (attempt - 1))
            logger.info(f"[NestJS] Retry dalam {backoff:.1f}s...")
            await asyncio.sleep(backoff)

    logger.error(
        f"[NestJS] Semua {config.max_retries} attempt gagal untuk run_id={payload.run_id}."
    )
    return False