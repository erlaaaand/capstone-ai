# agents/market_intelligence/scraper.py

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from core.logger import get_logger
from agents.market_intelligence.config import (
    CIRCUIT_BREAKER_CONFIG,
    ScrapingTarget,
    SCRAPING_TARGETS,
)
from agents.market_intelligence.schemas import ScrapedPage

logger = get_logger("agent.scraper")

_INTER_TARGET_DELAY_SEC: float = 3.0

_USER_AGENTS = [
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
]

_BLOCKED_RESOURCE_PATTERNS = [
    "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,otf,eot}",
    "**/{gtm,analytics,tracking,doubleclick,pixel,clarity,hotjar,mixpanel}**",
    "**/ads/**",
    "**/advertisement/**",
]


class _CircuitBreakerState:

    def __init__(self) -> None:
        self._lock:       threading.Lock        = threading.Lock()
        self._failures:   Dict[str, int]        = {}
        self._tripped_at: Dict[str, float]      = {}

    def is_open(self, target_name: str) -> bool:
        with self._lock:
            tripped = self._tripped_at.get(target_name)
            if tripped is None:
                return False
            if time.time() - tripped < CIRCUIT_BREAKER_CONFIG.cooldown_sec:
                return True
            # Cooldown selesai — reset otomatis
            self._failures.pop(target_name, None)
            self._tripped_at.pop(target_name, None)
            logger.info(f"[CircuitBreaker] RESET untuk '{target_name}' (cooldown selesai).")
            return False

    def record_failure(self, target_name: str) -> None:
        with self._lock:
            self._failures[target_name] = self._failures.get(target_name, 0) + 1
            if (
                self._failures[target_name] >= CIRCUIT_BREAKER_CONFIG.threshold
                and target_name not in self._tripped_at
            ):
                self._tripped_at[target_name] = time.time()
                logger.warning(
                    f"[CircuitBreaker] OPEN untuk '{target_name}' "
                    f"setelah {self._failures[target_name]} kegagalan berturut-turut. "
                    f"Cooldown {CIRCUIT_BREAKER_CONFIG.cooldown_sec}s."
                )

    def record_success(self, target_name: str) -> None:
        with self._lock:
            self._failures.pop(target_name, None)
            self._tripped_at.pop(target_name, None)

    def get_state_snapshot(self) -> Dict[str, dict]:
        """Kembalikan snapshot state untuk diagnostik / logging."""
        with self._lock:
            return {
                name: {
                    "failures":   self._failures.get(name, 0),
                    "tripped_at": self._tripped_at.get(name),
                    "is_open":    name in self._tripped_at,
                }
                for name in set(self._failures) | set(self._tripped_at)
            }


_circuit_breaker = _CircuitBreakerState()


def _url_matches_pattern(url: str, pattern: str) -> bool:
    """Cek apakah URL cocok dengan pola glob Playwright-style."""
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*", ".+")
    escaped = escaped.replace(r"\*", "[^/]+")
    try:
        return bool(re.search(escaped, url, re.IGNORECASE))
    except re.error:
        # Fallback ke substring match jika regex gagal
        stripped = pattern.replace("*", "").replace("/", "")
        return stripped.lower() in url.lower()


def _select_user_agent(attempt: int) -> str:
    """Pilih User-Agent berdasarkan nomor attempt untuk variasi minimal."""
    return _USER_AGENTS[attempt % len(_USER_AGENTS)]


def _scrape_single_target_sync(target: ScrapingTarget) -> ScrapedPage:

    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

    logger.info(
        f"[Scraper] Memulai network intercept: '{target.name}' → {target.url}"
    )

    last_error: Optional[str] = None
    max_attempts = target.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        collected_jsons: List[str] = []
        # Lock untuk thread-safe append dari response handler
        collected_lock  = threading.Lock()

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
                        "--disable-extensions",
                        "--disable-plugins",
                    ],
                )
                context = browser.new_context(
                    user_agent=_select_user_agent(attempt - 1),
                    viewport={"width": 1366, "height": 768},
                    locale="id-ID",
                    timezone_id="Asia/Jakarta",
                    java_script_enabled=True,
                    extra_http_headers={
                        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                    },
                )
                page = context.new_page()

                # Blokir resource statis dan tracking
                for pattern in _BLOCKED_RESOURCE_PATTERNS:
                    page.route(pattern, lambda route: route.abort())

                def _response_handler(response) -> None:
                    with collected_lock:
                        if len(collected_jsons) >= target.max_responses:
                            return

                    resp_url = response.url
                    if not _url_matches_pattern(resp_url, target.api_url_pattern):
                        return

                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        logger.debug(
                            f"[Scraper] URL cocok tapi Content-Type bukan JSON "
                            f"({content_type}): {resp_url[:100]}"
                        )
                        return

                    if not (200 <= response.status < 300):
                        logger.debug(
                            f"[Scraper] URL cocok tapi status {response.status}: "
                            f"{resp_url[:100]}"
                        )
                        return

                    try:
                        body_bytes = response.body()

                        if len(body_bytes) < target.min_response_bytes:
                            logger.debug(
                                f"[Scraper] Response terlalu kecil "
                                f"({len(body_bytes)} bytes < {target.min_response_bytes}): "
                                f"{resp_url[:100]}"
                            )
                            return

                        body_str = body_bytes.decode("utf-8", errors="replace")
                        json.loads(body_str)  # Validasi JSON sebelum disimpan

                        with collected_lock:
                            if len(collected_jsons) < target.max_responses:
                                collected_jsons.append(body_str)
                                logger.info(
                                    f"[Scraper] ✓ JSON intercepted dari '{target.name}': "
                                    f"URL={resp_url[:80]}... | "
                                    f"size={len(body_bytes)} bytes | "
                                    f"total_collected={len(collected_jsons)}"
                                )

                    except json.JSONDecodeError:
                        logger.debug(
                            f"[Scraper] Response bukan JSON valid: {resp_url[:100]}"
                        )
                    except Exception as exc:
                        logger.debug(
                            f"[Scraper] Gagal baca body response ({resp_url[:80]}): {exc}"
                        )

                page.on("response", _response_handler)
                page.goto(
                    target.url,
                    wait_until=target.wait_until,
                    timeout=target.page_timeout_ms,
                )

                if target.xhr_settle_ms > 0:
                    logger.debug(
                        f"[Scraper] Menunggu {target.xhr_settle_ms}ms "
                        f"untuk XHR settle di '{target.name}'..."
                    )
                    page.wait_for_timeout(target.xhr_settle_ms)

                browser.close()

            if not collected_jsons:
                raise ValueError(
                    f"Tidak ada JSON response yang di-intercept. "
                    f"Pola '{target.api_url_pattern}' mungkin tidak cocok "
                    f"dengan URL di '{target.url}'."
                )

            raw_json = _merge_json_responses(collected_jsons, target.name)

            logger.info(
                f"[Scraper] Berhasil intercept '{target.name}': "
                f"{len(collected_jsons)} response | "
                f"{len(raw_json)} chars gabungan | "
                f"attempt={attempt}/{max_attempts}"
            )
            return ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_json=raw_json,
            )

        except PlaywrightTimeout as exc:
            last_error = (
                f"Playwright timeout setelah {target.page_timeout_ms}ms: "
                f"{str(exc)[:200]}"
            )
            logger.warning(
                f"[Scraper] Timeout '{target.name}' attempt={attempt}/{max_attempts}: "
                f"{last_error}"
            )

        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                f"[Scraper] Intercept gagal '{target.name}' attempt={attempt}/{max_attempts}: "
                f"{last_error}"
            )

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning(
                f"[Scraper] Error '{target.name}' attempt={attempt}/{max_attempts}: "
                f"{last_error}",
                exc_info=True,
            )

        if attempt < max_attempts:
            backoff = min(2 ** attempt, 30)  # Cap backoff di 30 detik
            logger.info(
                f"[Scraper] Retry '{target.name}' dalam {backoff}s "
                f"(attempt {attempt}/{max_attempts})..."
            )
            time.sleep(backoff)

    logger.error(
        f"[Scraper] Semua {max_attempts} attempt gagal untuk '{target.name}'. "
        f"Error terakhir: {last_error}"
    )
    return ScrapedPage(
        source_name=target.name,
        source_url=target.url,
        raw_json="",
        success=False,
        error_message=last_error or "Unknown error after all attempts.",
    )


def _merge_json_responses(responses: List[str], source_name: str) -> str:
    """Gabungkan beberapa JSON response menjadi satu struktur."""
    if len(responses) == 1:
        return responses[0]

    merged_parts = []
    for i, resp_str in enumerate(responses):
        try:
            parsed = json.loads(resp_str)
            merged_parts.append(parsed)
        except json.JSONDecodeError:
            logger.warning(
                f"[Scraper] Gagal parse JSON response #{i} dari '{source_name}' "
                "saat merge. Di-skip."
            )

    if not merged_parts:
        logger.warning(
            f"[Scraper] Semua response gagal di-parse untuk '{source_name}'. "
            "Fallback ke response pertama (raw string)."
        )
        return responses[0]

    merged = json.dumps(
        {"intercepted_responses": merged_parts},
        ensure_ascii=False,
    )
    logger.debug(
        f"[Scraper] Merged {len(merged_parts)} JSON responses dari "
        f"'{source_name}': {len(merged)} chars total."
    )
    return merged


async def scrape_all_targets(
    targets: Optional[List[ScrapingTarget]] = None,
) -> List[ScrapedPage]:
    """
    Scrape semua target secara sequential (untuk menghindari deteksi bot)
    dengan jeda antar-target.
    """
    if targets is None:
        targets = SCRAPING_TARGETS

    if not targets:
        logger.warning("[Scraper] Tidak ada scraping target yang dikonfigurasi.")
        return []

    results: List[ScrapedPage] = []

    for i, target in enumerate(targets):
        if _circuit_breaker.is_open(target.name):
            logger.warning(
                f"[Scraper] SKIP '{target.name}' — circuit breaker OPEN."
            )
            results.append(ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_json="",
                success=False,
                error_message="Circuit breaker open — target dinonaktifkan sementara.",
            ))
            continue

        page_result = await asyncio.to_thread(
            _scrape_single_target_sync, target
        )

        if page_result.success:
            _circuit_breaker.record_success(target.name)
        else:
            _circuit_breaker.record_failure(target.name)

        results.append(page_result)

        # Jeda antar-target kecuali target terakhir
        if i < len(targets) - 1:
            logger.debug(
                f"[Scraper] Jeda {_INTER_TARGET_DELAY_SEC}s sebelum target berikutnya..."
            )
            await asyncio.sleep(_INTER_TARGET_DELAY_SEC)

    succeeded = sum(1 for r in results if r.success)
    failed    = len(results) - succeeded
    logger.info(
        f"[Scraper] Intercept selesai. "
        f"{succeeded}/{len(results)} target berhasil | {failed} gagal."
    )

    cb_state = _circuit_breaker.get_state_snapshot()
    if cb_state:
        logger.debug(f"[Scraper] Circuit breaker state: {cb_state}")

    return results