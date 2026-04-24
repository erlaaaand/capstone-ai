# agents/market_intelligence/task.py
"""
Orchestrator Market Intelligence Agent.

Tanggung jawab tunggal: merantai scraper → llm_analyzer → nestjs_client
dengan manajemen:
  - Run Guard (asyncio.Lock) — cegah eksekusi paralel
  - Timeout global (asyncio.wait_for)
  - Aggregasi status dan error ke MarketReportPayload
  - Tidak tahu tentang scheduler — bisa dipanggil manual untuk testing
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from core.logger import get_logger
from agents.market_intelligence.config import SCHEDULER_CONFIG
from agents.market_intelligence.schemas import (
    AgentRunStatus,
    MarketReportPayload,
)
from agents.market_intelligence import scraper, llm_analyzer, nestjs_client

logger = get_logger("agent.task")


# ---------------------------------------------------------------------------
# Run Guard — satu asyncio.Lock yang shared selama lifetime proses
# ---------------------------------------------------------------------------

_run_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Core task
# ---------------------------------------------------------------------------

async def _run_pipeline() -> MarketReportPayload:
    """
    Pipeline inti (tanpa guard / timeout).
    Dipanggil oleh run_once() yang sudah wrap dengan guard dan timeout.
    """
    run_id     = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)
    logger.info(f"[Task] === Market Intelligence Agent START | run_id={run_id} ===")

    # ------------------------------------------------------------------
    # TAHAP 1: Scraping
    # ------------------------------------------------------------------
    logger.info("[Task] Tahap 1/3: Scraping target...")
    try:
        scraped_pages = await scraper.scrape_all_targets()
    except Exception as e:
        logger.critical(f"[Task] Scraper melempar exception tak terduga: {e}", exc_info=True)
        return MarketReportPayload(
            run_id         = run_id,
            run_started_at = started_at,
            status         = AgentRunStatus.SCRAPER_ERROR,
            error_details  = f"Scraper fatal: {str(e)[:500]}",
        )

    sources_scraped = sum(1 for p in scraped_pages if p.success)
    sources_failed  = sum(1 for p in scraped_pages if not p.success)

    if sources_scraped == 0:
        logger.error("[Task] Tidak ada satu pun halaman berhasil di-scrape.")
        return MarketReportPayload(
            run_id          = run_id,
            run_started_at  = started_at,
            status          = AgentRunStatus.SCRAPER_ERROR,
            sources_scraped = 0,
            sources_failed  = sources_failed,
            error_details   = "Zero successful scrape results.",
        )

    logger.info(
        f"[Task] Scraping selesai: {sources_scraped} berhasil, {sources_failed} gagal."
    )

    # ------------------------------------------------------------------
    # TAHAP 2: LLM Analysis
    # ------------------------------------------------------------------
    logger.info("[Task] Tahap 2/3: Analisis LLM...")
    try:
        entries, llm_parse_errors = await llm_analyzer.analyze_pages(scraped_pages)
    except Exception as e:
        logger.error(f"[Task] LLM analyzer error: {e}", exc_info=True)
        return MarketReportPayload(
            run_id          = run_id,
            run_started_at  = started_at,
            status          = AgentRunStatus.LLM_ERROR,
            sources_scraped = sources_scraped,
            sources_failed  = sources_failed,
            error_details   = f"LLM analyzer fatal: {str(e)[:500]}",
        )

    if not entries:
        logger.warning("[Task] LLM tidak menghasilkan entry harga yang valid.")
        status = AgentRunStatus.NO_DATA
    elif sources_failed > 0:
        status = AgentRunStatus.PARTIAL
    else:
        status = AgentRunStatus.SUCCESS

    payload = MarketReportPayload(
        run_id           = run_id,
        run_started_at   = started_at,
        status           = status,
        entries          = entries,
        sources_scraped  = sources_scraped,
        sources_failed   = sources_failed,
        llm_parse_errors = llm_parse_errors,
    )

    logger.info(
        f"[Task] LLM selesai: {len(entries)} entry | "
        f"status={status.value} | parse_errors={llm_parse_errors}"
    )

    # ------------------------------------------------------------------
    # TAHAP 3: Kirim ke NestJS
    # ------------------------------------------------------------------
    logger.info("[Task] Tahap 3/3: Mengirim laporan ke NestJS...")
    try:
        send_ok = await nestjs_client.send_report(payload)
        if not send_ok:
            logger.error("[Task] Gagal mengirim laporan ke NestJS (semua retry habis).")
            # Tetap lanjut — data sudah ada, hanya delivery yang gagal
    except Exception as e:
        logger.error(f"[Task] NestJS client error: {e}", exc_info=True)

    logger.info(
        f"[Task] === Market Intelligence Agent SELESAI | "
        f"run_id={run_id} | entries={len(entries)} | status={status.value} ==="
    )
    return payload


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_once() -> Optional[MarketReportPayload]:
    """
    Jalankan pipeline satu kali dengan proteksi:
      1. Run Guard — tolak jika masih ada run yang berjalan
      2. Global Timeout — paksa berhenti jika melebihi batas waktu

    Dipanggil oleh scheduler, atau bisa dipanggil manual untuk testing:
        asyncio.run(run_once())
    """
    if _run_lock.locked():
        logger.warning(
            "[Task] Run sebelumnya masih berjalan — run ini di-skip. "
            "Pertimbangkan menambah interval scheduler jika ini sering terjadi."
        )
        return None

    async with _run_lock:
        try:
            logger.info(
                f"[Task] Memulai run dengan timeout={SCHEDULER_CONFIG.max_run_duration_sec}s"
            )
            result = await asyncio.wait_for(
                _run_pipeline(),
                timeout=float(SCHEDULER_CONFIG.max_run_duration_sec),
            )
            return result
        except asyncio.TimeoutError:
            logger.error(
                f"[Task] Run dibatalkan paksa — melebihi timeout "
                f"{SCHEDULER_CONFIG.max_run_duration_sec}s."
            )
            return None
        except Exception as e:
            logger.critical(
                f"[Task] Error tidak tertangani di run_once(): {e}",
                exc_info=True,
            )
            return None