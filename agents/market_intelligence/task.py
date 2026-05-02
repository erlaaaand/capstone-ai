# agents/market_intelligence/task.py

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from core.logger import get_logger
from agents.market_intelligence.config import SCHEDULER_CONFIG
from agents.market_intelligence.schemas import (
    AgentRunStatus,
    MarketPriceEntry,
    MarketReportPayload,
)
from agents.market_intelligence import scraper, llm_analyzer, nestjs_client

logger = get_logger("agent.task")

_run_lock = asyncio.Lock()

def _apply_python_whole_fruit_gate(
    raw_entries: List[MarketPriceEntry],
    run_id:      str,
) -> Tuple[List[MarketPriceEntry], int]:
   
    clean_entries: List[MarketPriceEntry] = []
    discarded_here: int = 0

    for entry in raw_entries:
        if not entry.is_whole_fruit:
            logger.error(
                f"[Task][DoubleValidation] PERINGATAN: Entry dengan is_whole_fruit=False "
                f"LOLOS dari lapisan LLM Analyzer! "
                f"run_id={run_id} | "
                f"variety={entry.variety_code.value} | "
                f"alias='{entry.variety_alias}' | "
                f"weight_ref='{entry.weight_reference}' | "
                f"notes='{entry.notes}'. "
                f"Entry DIBUANG di lapisan Python. [DATA LEAKAGE PREVENTED - PYTHON LAYER]"
            )
            discarded_here += 1
            continue

        clean_entries.append(entry)

    if discarded_here > 0:
        logger.warning(
            f"[Task][DoubleValidation] {discarded_here} entry dibuang di lapisan Python. "
            "Periksa kualitas LLM output dan system prompt."
        )
    else:
        logger.info(
            f"[Task][DoubleValidation] Semua {len(clean_entries)} entry lolos validasi Python. "
            "Tidak ada data leakage terdeteksi di lapisan ini."
        )

    return clean_entries, discarded_here

async def _run_pipeline() -> MarketReportPayload:

    run_id     = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    logger.info(
        f"[Task] ══════════════════════════════════════════════════════════\n"
        f"[Task]  Market Intelligence Agent START | run_id={run_id}\n"
        f"[Task] ══════════════════════════════════════════════════════════"
    )

    logger.info("[Task] ── Tahap 1/4: Network Intercept Scraping ─────────────")
    try:
        scraped_pages = await scraper.scrape_all_targets()
    except Exception as exc:
        logger.critical(
            f"[Task] Scraper melempar exception tidak tertangani: {exc}",
            exc_info=True,
        )
        return MarketReportPayload(
            run_id         = run_id,
            run_started_at = started_at,
            status         = AgentRunStatus.SCRAPER_ERROR,
            error_details  = f"Scraper fatal: {str(exc)[:500]}",
        )

    sources_scraped = sum(1 for p in scraped_pages if p.success)
    sources_failed  = sum(1 for p in scraped_pages if not p.success)

    if sources_scraped == 0:
        logger.error(
            "[Task] Tidak ada satu pun halaman berhasil di-intercept. "
            "Periksa api_url_pattern di config.py dan koneksi internet."
        )
        return MarketReportPayload(
            run_id          = run_id,
            run_started_at  = started_at,
            status          = AgentRunStatus.SCRAPER_ERROR,
            sources_scraped = 0,
            sources_failed  = sources_failed,
            error_details   = "Zero successful network intercept results.",
        )

    logger.info(
        f"[Task] Scraping selesai: "
        f"{sources_scraped} berhasil | {sources_failed} gagal."
    )

    logger.info("[Task] ── Tahap 2/4: LLM Analysis (Lapisan 1 Anti-Leakage) ──")
    try:
        raw_entries, llm_parse_errors, discarded_by_llm = await llm_analyzer.analyze_pages(
            scraped_pages
        )
    except Exception as exc:
        logger.error(
            f"[Task] LLM Analyzer melempar exception tidak tertangani: {exc}",
            exc_info=True,
        )
        return MarketReportPayload(
            run_id           = run_id,
            run_started_at   = started_at,
            status           = AgentRunStatus.LLM_ERROR,
            sources_scraped  = sources_scraped,
            sources_failed   = sources_failed,
            error_details    = f"LLM Analyzer fatal: {str(exc)[:500]}",
        )

    logger.info(
        f"[Task] LLM selesai: "
        f"{len(raw_entries)} entry dari LLM | "
        f"{discarded_by_llm} dibuang di lapisan LLM | "
        f"{llm_parse_errors} parse error."
    )

    logger.info("[Task] ── Tahap 3/4: Double Validation (Lapisan 2 Python) ────")

    clean_entries, discarded_by_python = _apply_python_whole_fruit_gate(
        raw_entries, run_id
    )

    total_discarded = discarded_by_llm + discarded_by_python

    logger.info(
        f"[Task] Double Validation selesai:\n"
        f"          Entry lolos LLM      : {len(raw_entries)}\n"
        f"          Dibuang lapisan LLM  : {discarded_by_llm}\n"
        f"          Dibuang lapisan Python: {discarded_by_python}\n"
        f"          Entry FINAL valid    : {len(clean_entries)}\n"
        f"          Total entry dibuang : {total_discarded} (pencegahan data leakage)"
    )

    if not clean_entries:
        logger.warning(
            "[Task] Tidak ada entry harga valid setelah double validation. "
            "Kemungkinan semua produk yang ditemukan adalah varian kupas/frozen/olahan."
        )
        status = AgentRunStatus.NO_DATA
    elif sources_failed > 0:
        status = AgentRunStatus.PARTIAL
    else:
        status = AgentRunStatus.SUCCESS

    payload = MarketReportPayload(
        run_id            = run_id,
        run_started_at    = started_at,
        status            = status,
        entries           = clean_entries,
        sources_scraped   = sources_scraped,
        sources_failed    = sources_failed,
        llm_parse_errors  = llm_parse_errors,
        entries_discarded = total_discarded,
    )

    logger.info("[Task] ── Tahap 4/4: Pengiriman ke NestJS Backend ────────────")
    try:
        send_ok = await nestjs_client.send_report(payload)
        if not send_ok:
            logger.error(
                "[Task] Gagal mengirim laporan ke NestJS (semua retry habis). "
                "Data tersimpan di log untuk recovery manual."
            )
    except Exception as exc:
        logger.error(
            f"[Task] NestJS client melempar exception tidak tertangani: {exc}",
            exc_info=True,
        )

    logger.info(
        f"[Task] ══════════════════════════════════════════════════════════\n"
        f"[Task]  Market Intelligence Agent SELESAI\n"
        f"[Task]  run_id={run_id} | entries={len(clean_entries)} | "
        f"status={status.value} | discarded={total_discarded}\n"
        f"[Task] ══════════════════════════════════════════════════════════"
    )
    return payload


async def run_once() -> Optional[MarketReportPayload]:

    if _run_lock.locked():
        logger.warning(
            "[Task] Run sebelumnya MASIH BERJALAN — run ini di-skip. "
            "Jika ini sering terjadi, pertimbangkan mengurangi jumlah target "
            "atau menambah interval scheduler."
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
                f"[Task] Run dibatalkan paksa — melebihi global timeout "
                f"({SCHEDULER_CONFIG.max_run_duration_sec}s). "
                "Pertimbangkan mengurangi jumlah target atau menaikkan timeout."
            )
            return None
        except Exception as exc:
            logger.critical(
                f"[Task] Error TIDAK TERTANGANI di run_once(): {exc}",
                exc_info=True,
            )
            return None