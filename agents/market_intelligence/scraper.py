# agents/market_intelligence/scraper.py
"""
Async Playwright scraper dengan:
  - Fallback selector (coba satu per satu)
  - Retry per halaman dengan exponential backoff
  - Circuit breaker per target (nonaktif setelah N kali gagal berturut-turut)
  - Stealth mode dasar (user-agent, viewport realistis)
  - asyncio.to_thread isolation — TIDAK memblokir FastAPI event loop
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger
from agents.market_intelligence.config import (
    CIRCUIT_BREAKER_COOLDOWN_SEC,
    CIRCUIT_BREAKER_THRESHOLD,
    ScrapingTarget,
    SCRAPING_TARGETS,
)
from agents.market_intelligence.schemas import ScrapedPage

logger = get_logger("agent.scraper")


# ---------------------------------------------------------------------------
# Circuit Breaker State (in-process, per target name)
# ---------------------------------------------------------------------------

class _CircuitBreakerState:
    """Menyimpan state circuit breaker untuk semua target."""

    def __init__(self) -> None:
        # {target_name: consecutive_failures}
        self._failures:    Dict[str, int]   = {}
        # {target_name: trip_timestamp}
        self._tripped_at:  Dict[str, float] = {}

    def is_open(self, target_name: str) -> bool:
        """Kembalikan True jika circuit OPEN (target sedang diblokir)."""
        tripped = self._tripped_at.get(target_name)
        if tripped is None:
            return False
        if time.time() - tripped < CIRCUIT_BREAKER_COOLDOWN_SEC:
            return True
        # Cooldown selesai — reset
        self.reset(target_name)
        return False

    def record_failure(self, target_name: str) -> None:
        self._failures[target_name] = self._failures.get(target_name, 0) + 1
        if self._failures[target_name] >= CIRCUIT_BREAKER_THRESHOLD:
            if target_name not in self._tripped_at:
                self._tripped_at[target_name] = time.time()
                logger.warning(
                    f"[CircuitBreaker] OPEN untuk '{target_name}' "
                    f"setelah {self._failures[target_name]} kegagalan. "
                    f"Cooldown {CIRCUIT_BREAKER_COOLDOWN_SEC}s."
                )

    def record_success(self, target_name: str) -> None:
        self._failures.pop(target_name, None)
        self._tripped_at.pop(target_name, None)

    def reset(self, target_name: str) -> None:
        self._failures.pop(target_name, None)
        self._tripped_at.pop(target_name, None)
        logger.info(f"[CircuitBreaker] RESET untuk '{target_name}' (cooldown selesai).")


_circuit_breaker = _CircuitBreakerState()


# ---------------------------------------------------------------------------
# Core scraping logic (sync — dijalankan via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _scrape_single_target_sync(target: ScrapingTarget) -> ScrapedPage:
    """
    Sinkron — WAJIB dijalankan via asyncio.to_thread() agar tidak
    memblokir uvicorn event loop.

    Playwright sendiri memiliki async API, tapi saat kita wrap di to_thread
    kita menggunakan sync_playwright untuk menghindari nested event loop
    (Playwright async membutuhkan event loop sendiri yang tidak kompatibel
    dengan event loop uvicorn yang sedang berjalan).
    """
    # Import di dalam fungsi — library berat, lazy load
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

    logger.info(f"[Scraper] Memulai scraping: '{target.name}' → {target.url}")

    last_error: Optional[str] = None

    for attempt in range(1, target.max_retries + 2):  # +1 untuk attempt pertama
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1366, "height": 768},
                    locale="id-ID",
                    timezone_id="Asia/Jakarta",
                    java_script_enabled=True,
                )
                page = context.new_page()

                # Blokir resource yang tidak perlu (percepat load)
                page.route(
                    "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,otf}",
                    lambda route: route.abort(),
                )
                page.route(
                    "**/{analytics,tracking,gtm,ads,pixel}**",
                    lambda route: route.abort(),
                )

                page.goto(
                    target.url,
                    wait_until=target.wait_until,
                    timeout=target.page_timeout_ms,
                )

                # Coba selector satu per satu (fallback strategy)
                raw_text = _extract_text_with_fallback(page, target)

                browser.close()

                if not raw_text.strip():
                    raise ValueError("Semua selector tidak menemukan teks yang berarti.")

                logger.info(
                    f"[Scraper] Berhasil scrape '{target.name}' "
                    f"({len(raw_text)} chars, attempt={attempt})"
                )
                return ScrapedPage(
                    source_name=target.name,
                    source_url=target.url,
                    raw_text=raw_text,
                )

        except PlaywrightTimeout as e:
            last_error = f"Timeout setelah {target.page_timeout_ms}ms: {str(e)[:200]}"
            logger.warning(f"[Scraper] Timeout '{target.name}' attempt={attempt}: {last_error}")
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:300]}"
            logger.warning(f"[Scraper] Error '{target.name}' attempt={attempt}: {last_error}")

        if attempt <= target.max_retries:
            backoff = 2 ** attempt
            logger.info(f"[Scraper] Retry '{target.name}' dalam {backoff}s...")
            time.sleep(backoff)

    logger.error(
        f"[Scraper] Semua attempt gagal untuk '{target.name}'. "
        f"Error terakhir: {last_error}"
    )
    return ScrapedPage(
        source_name=target.name,
        source_url=target.url,
        raw_text="",
        success=False,
        error_message=last_error,
    )


def _extract_text_with_fallback(page, target: ScrapingTarget) -> str:
    """
    Coba setiap CSS selector secara berurutan.
    Gabungkan semua teks yang ditemukan dari elemen-elemen yang cocok.
    """
    from playwright.sync_api import Page

    for selector in target.content_selectors:
        try:
            elements = page.query_selector_all(selector)
            if elements:
                texts = [el.inner_text() for el in elements if el.inner_text().strip()]
                combined = "\n\n".join(texts)
                if len(combined) > 100:  # Minimal 100 char untuk dianggap valid
                    logger.debug(
                        f"[Scraper] Selector '{selector}' berhasil "
                        f"({len(elements)} elemen, {len(combined)} chars)."
                    )
                    return combined
                logger.debug(f"[Scraper] Selector '{selector}' ada elemen tapi teks terlalu pendek.")
        except Exception as e:
            logger.debug(f"[Scraper] Selector '{selector}' error: {e}")
            continue

    # Fallback terakhir: ambil semua <p> dan teks body
    logger.warning(
        f"[Scraper] Semua selector gagal untuk '{target.name}'. "
        "Fallback ke body text."
    )
    try:
        body_text = page.evaluate("() => document.body.innerText") or ""
        return body_text[:20_000]  # Cap 20k char dari body
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def scrape_all_targets(
    targets: Optional[List[ScrapingTarget]] = None,
) -> List[ScrapedPage]:
    """
    Scrape semua target secara SEQUENTIAL (bukan concurrent).

    Alasan sequential bukan concurrent:
    - Menghindari IP ban dari marketplace (rate limiting)
    - Chromium instance baru per target — parallel = N×RAM Chromium
    - Task ini berjalan jam 02:00 — latency tidak kritis

    Setiap target dijalankan via asyncio.to_thread() agar tidak
    memblokir event loop FastAPI.
    """
    if targets is None:
        targets = SCRAPING_TARGETS

    results: List[ScrapedPage] = []

    for target in targets:
        if _circuit_breaker.is_open(target.name):
            logger.warning(
                f"[Scraper] SKIP '{target.name}' — circuit breaker OPEN."
            )
            results.append(ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_text="",
                success=False,
                error_message="Circuit breaker open — target dinonaktifkan sementara.",
            ))
            continue

        # Jalankan sync Playwright di thread terpisah
        page_result = await asyncio.to_thread(_scrape_single_target_sync, target)

        if page_result.success:
            _circuit_breaker.record_success(target.name)
        else:
            _circuit_breaker.record_failure(target.name)

        results.append(page_result)

        # Jeda antar target untuk menghindari rate limiting
        await asyncio.sleep(3.0)

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        f"[Scraper] Selesai. {succeeded}/{len(results)} target berhasil."
    )
    return results