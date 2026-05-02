# agents/market_intelligence/nestjs_client.py

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Optional

import httpx

from core.logger import get_logger
from agents.market_intelligence.config import NESTJS_CONFIG, NestJSClientConfig
from agents.market_intelligence.schemas import MarketReportPayload

logger = get_logger("agent.nestjs_client")


def _build_signature(payload_bytes: bytes, secret: str) -> str:
    if not secret:
        return ""
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


async def send_report(
    payload: MarketReportPayload,
    config:  Optional[NestJSClientConfig] = None,
) -> bool:

    if config is None:
        config = NESTJS_CONFIG

    url           = f"{config.base_url.rstrip('/')}{config.endpoint}"
    payload_bytes = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")

    headers: dict = {
        "Content-Type":    "application/json; charset=utf-8",
        "X-Agent-Version": payload.agent_version,
        "X-Run-ID":        payload.run_id,
    }
    if config.api_key:
        headers["X-Internal-API-Key"] = config.api_key
    sig = _build_signature(payload_bytes, config.api_key)
    if sig:
        headers["X-Signature"] = sig

    logger.info(
        f"[NestJS] Mengirim laporan ke {url} "
        f"(run_id={payload.run_id}, entries={payload.entry_count}, status={payload.status.value})"
    )

    for attempt in range(1, config.max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=config.timeout_sec) as client:
                resp = await client.post(url, content=payload_bytes, headers=headers)

            if resp.is_success:
                logger.info(f"[NestJS] Berhasil (status={resp.status_code}, attempt={attempt}).")
                return True

            if 400 <= resp.status_code < 500:
                logger.error(
                    f"[NestJS] Client error {resp.status_code}: {resp.text[:300]}. Tidak di-retry."
                )
                return False

            logger.warning(
                f"[NestJS] Server error {resp.status_code} (attempt={attempt}/{config.max_retries})."
            )

        except httpx.ConnectError:
            logger.error(f"[NestJS] Tidak bisa terhubung ke {url} (attempt={attempt}).")
        except httpx.TimeoutException:
            logger.warning(f"[NestJS] Timeout {config.timeout_sec}s (attempt={attempt}).")
        except Exception as e:
            logger.error(f"[NestJS] Error tak terduga (attempt={attempt}): {e}", exc_info=True)

        if attempt < config.max_retries:
            backoff = config.retry_backoff_sec * (2 ** (attempt - 1))
            logger.info(f"[NestJS] Retry dalam {backoff:.1f}s...")
            await asyncio.sleep(backoff)

    logger.error(
        f"[NestJS] Semua {config.max_retries} attempt gagal untuk run_id={payload.run_id}."
    )
    return False