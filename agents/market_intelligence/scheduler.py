# agents/market_intelligence/scheduler.py

from __future__ import annotations

import asyncio
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, JobExecutionEvent

from core.logger import get_logger
from agents.market_intelligence.config import SCHEDULER_CONFIG
from agents.market_intelligence.task import run_once

logger = get_logger("agent.scheduler")

_JOB_ID = "market_intelligence_agent"


def _on_job_executed(event: JobExecutionEvent) -> None:
    if event.job_id != _JOB_ID:
        return

    retval = event.retval
    if retval is None:
        logger.warning(
            f"[Scheduler] Job '{_JOB_ID}' selesai — run di-skip (ada run aktif) atau timeout."
        )
    else:
        logger.info(
            f"[Scheduler] Job '{_JOB_ID}' selesai — "
            f"status={retval.status.value} | "
            f"entries={retval.entry_count} | "
            f"duration={retval.run_duration_sec:.1f}s"
        )


def _on_job_error(event: JobExecutionEvent) -> None:
    if event.job_id != _JOB_ID:
        return
    logger.error(
        f"[Scheduler] Job '{_JOB_ID}' melempar exception: {event.exception}",
        exc_info=event.traceback,
    )


class MarketIntelligenceScheduler:

    def __init__(self) -> None:
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._started:   bool = False

    def _build_scheduler(self) -> AsyncIOScheduler:
        scheduler = AsyncIOScheduler(
            timezone=SCHEDULER_CONFIG.timezone,
            job_defaults={
                "coalesce":           True,   # Tidak jalankan run yang terlewat
                "max_instances":      1,       # Hanya satu instance berjalan sekaligus
                "misfire_grace_time": 600,     # Toleransi 10 menit keterlambatan start
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

        if self._started and self._scheduler is not None and self._scheduler.running:
            logger.warning("[Scheduler] Scheduler sudah berjalan, skip start().")
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
        self._started = True

        next_run = self._scheduler.get_job(_JOB_ID).next_run_time
        logger.info(
            f"[Scheduler] Market Intelligence Agent terjadwal. "
            f"Cron: {SCHEDULER_CONFIG.cron_hour:02d}:{SCHEDULER_CONFIG.cron_minute:02d} "
            f"{SCHEDULER_CONFIG.timezone}. "
            f"Eksekusi berikutnya: {next_run}"
        )

    async def stop(self) -> None:
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("[Scheduler] Market Intelligence Scheduler dihentikan.")

    async def trigger_now(self) -> None:
        """Jalankan agent sekarang secara manual (fire-and-forget)."""
        logger.info("[Scheduler] Manual trigger: menjalankan agent sekarang...")
        asyncio.create_task(run_once(), name="manual_market_run_scheduler")

    @property
    def is_running(self) -> bool:
        return self._scheduler is not None and self._scheduler.running

    def get_next_run_time(self) -> Optional[str]:
        """Kembalikan string waktu run berikutnya, atau None jika tidak dijadwalkan."""
        if not self.is_running:
            return None
        job = self._scheduler.get_job(_JOB_ID)
        if job is None or job.next_run_time is None:
            return None
        return job.next_run_time.isoformat()


# ──────────────────────────────────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────────────────────────────────

_scheduler_instance: Optional[MarketIntelligenceScheduler] = None


def get_scheduler() -> MarketIntelligenceScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = MarketIntelligenceScheduler()
    return _scheduler_instance