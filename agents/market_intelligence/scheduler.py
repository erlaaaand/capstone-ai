# agents/market_intelligence/scheduler.py
"""
APScheduler setup untuk Market Intelligence Agent.

Desain:
  - Gunakan AsyncIOScheduler agar berjalan di event loop yang sama dengan FastAPI
    TANPA memblokir — task.run_once() sendiri yang melakukan to_thread isolation
    untuk operasi berat (Playwright).
  - CronTrigger untuk eksekusi di jam off-peak (default 02:30 WIB).
  - Satu instance scheduler (singleton) per proses.
  - start() / stop() dipanggil dari lifespan context manager di main.py.
"""

from __future__ import annotations

from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

from core.logger import get_logger
from agents.market_intelligence.config import SCHEDULER_CONFIG
from agents.market_intelligence.task import run_once

logger = get_logger("agent.scheduler")

_JOB_ID = "market_intelligence_agent"


# ---------------------------------------------------------------------------
# APScheduler event listeners
# ---------------------------------------------------------------------------

def _on_job_executed(event: JobExecutionEvent) -> None:
    if event.job_id != _JOB_ID:
        return
    retval = event.retval
    if retval is None:
        logger.warning(f"[Scheduler] Job '{_JOB_ID}' selesai — run di-skip atau timeout.")
    else:
        logger.info(
            f"[Scheduler] Job '{_JOB_ID}' selesai — "
            f"status={retval.status.value}, entries={retval.entry_count}"
        )


def _on_job_error(event: JobExecutionEvent) -> None:
    if event.job_id != _JOB_ID:
        return
    logger.error(
        f"[Scheduler] Job '{_JOB_ID}' melempar exception: {event.exception}",
        exc_info=event.traceback,
    )


# ---------------------------------------------------------------------------
# Singleton scheduler
# ---------------------------------------------------------------------------

class MarketIntelligenceScheduler:
    """
    Wrapper tipis di atas APScheduler AsyncIOScheduler.
    Satu instance per proses — dikelola dari lifespan main.py.
    """

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None

    def _build_scheduler(self) -> AsyncIOScheduler:
        scheduler = AsyncIOScheduler(
            timezone=SCHEDULER_CONFIG.timezone,
            job_defaults={
                "coalesce":       True,   # Jalankan sekali jika tertinggal beberapa jadwal
                "max_instances":  1,      # Tidak boleh ada dua instance bersamaan
                "misfire_grace_time": 600,  # Toleransi 10 menit keterlambatan trigger
            },
        )
        scheduler.add_listener(_on_job_executed, EVENT_JOB_EXECUTED)
        scheduler.add_listener(_on_job_error,    EVENT_JOB_ERROR)
        return scheduler

    async def start(self) -> None:
        if SCHEDULER_CONFIG.disabled:
            logger.warning(
                "[Scheduler] MI_AGENT_DISABLED=true — "
                "Market Intelligence Agent TIDAK dijadwalkan."
            )
            return

        self._scheduler = self._build_scheduler()

        trigger = CronTrigger(
            hour     = SCHEDULER_CONFIG.cron_hour,
            minute   = SCHEDULER_CONFIG.cron_minute,
            timezone = SCHEDULER_CONFIG.timezone,
        )

        self._scheduler.add_job(
            func    = run_once,
            trigger = trigger,
            id      = _JOB_ID,
            name    = "Market Intelligence Agent",
        )

        self._scheduler.start()

        next_run = self._scheduler.get_job(_JOB_ID).next_run_time
        logger.info(
            f"[Scheduler] Market Intelligence Agent terjadwal. "
            f"Cron: {SCHEDULER_CONFIG.cron_hour:02d}:{SCHEDULER_CONFIG.cron_minute:02d} UTC. "
            f"Eksekusi berikutnya: {next_run}"
        )

    async def stop(self) -> None:
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[Scheduler] Market Intelligence Scheduler dihentikan.")

    async def trigger_now(self) -> None:
        """
        Jalankan agent sekarang (untuk debugging/manual trigger).
        Bisa dipanggil dari endpoint admin jika diperlukan.
        """
        logger.info("[Scheduler] Manual trigger: menjalankan agent sekarang...")
        await run_once()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler_instance: Optional[MarketIntelligenceScheduler] = None


def get_scheduler() -> MarketIntelligenceScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = MarketIntelligenceScheduler()
    return _scheduler_instance