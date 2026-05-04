from __future__ import annotations

import asyncio
import sys
import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
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

_BLOCKED_RESOURCE_TYPES = {"image", "font", "media"}

_BLOCKED_URL_FRAGMENTS = {
    "gtm", "analytics", "tracking", "doubleclick",
    "pixel", "clarity", "hotjar", "mixpanel", "/ads/", "/advertisement/",
}


class _CircuitBreakerState:

    def __init__(self) -> None:
        self._lock:       threading.Lock   = threading.Lock()
        self._failures:   Dict[str, int]   = {}
        self._tripped_at: Dict[str, float] = {}

    def is_open(self, target_name: str) -> bool:
        with self._lock:
            tripped = self._tripped_at.get(target_name)
            if tripped is None:
                return False
            if time.time() - tripped < CIRCUIT_BREAKER_CONFIG.cooldown_sec:
                return True
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
    escaped = re.escape(pattern)
    escaped = escaped.replace(r"\*\*", ".+")
    escaped = escaped.replace(r"\*", "[^/]+")
    try:
        return bool(re.search(escaped, url, re.IGNORECASE))
    except re.error:
        stripped = pattern.replace("*", "").replace("/", "")
        return stripped.lower() in url.lower()


def _is_blocked_url(url: str) -> bool:
    url_lower = url.lower()
    return any(fragment in url_lower for fragment in _BLOCKED_URL_FRAGMENTS)


def _select_user_agent(attempt: int) -> str:
    return _USER_AGENTS[attempt % len(_USER_AGENTS)]


def _merge_json_responses(responses: List[str], source_name: str) -> str:
    if len(responses) == 1:
        return responses[0]

    merged_parts = []
    for i, resp_str in enumerate(responses):
        try:
            parsed = json.loads(resp_str)
            merged_parts.append(parsed)
        except json.JSONDecodeError:
            logger.warning(
                f"[Scraper] Gagal parse JSON response #{i} dari '{source_name}' saat merge. Di-skip."
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


async def _scrape_single_target(target: ScrapingTarget) -> ScrapedPage:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

    logger.info(f"[Scraper] Memulai network intercept: '{target.name}' → {target.url}")

    last_error: Optional[str] = None
    max_attempts = target.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        collected_responses: List[dict] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
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

                context = await browser.new_context(
                    user_agent=_select_user_agent(attempt - 1),
                    viewport={"width": 1366, "height": 768},
                    locale="id-ID",
                    timezone_id="Asia/Jakarta",
                    java_script_enabled=True,
                    extra_http_headers={
                        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                    },
                )

                page = await context.new_page()

                async def _route_handler(route):
                    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
                        await route.abort()
                        return
                    if _is_blocked_url(route.request.url):
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", _route_handler)

                def _on_response(response) -> None:
                    if len(collected_responses) >= target.max_responses:
                        return
                    if not _url_matches_pattern(response.url, target.api_url_pattern):
                        return
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        return
                    if not (200 <= response.status < 300):
                        return
                    collected_responses.append({"url": response.url, "status": response.status})

                page.on("response", _on_response)

                await page.goto(
                    target.url,
                    wait_until=target.wait_until,
                    timeout=target.page_timeout_ms,
                )

                if target.xhr_settle_ms > 0:
                    logger.debug(
                        f"[Scraper] Menunggu {target.xhr_settle_ms}ms untuk XHR settle di '{target.name}'..."
                    )
                    await page.wait_for_timeout(target.xhr_settle_ms)

                await browser.close()

            if not collected_responses:
                raise ValueError(
                    f"Tidak ada JSON response yang di-intercept. "
                    f"Pola '{target.api_url_pattern}' mungkin tidak cocok dengan URL di '{target.url}'."
                )

            logger.info(
                f"[Scraper] Berhasil intercept '{target.name}': "
                f"{len(collected_responses)} response | attempt={attempt}/{max_attempts}"
            )

            placeholder_json = json.dumps(
                {"intercepted_urls": [r["url"] for r in collected_responses]},
                ensure_ascii=False,
            )

            return ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_json=placeholder_json,
            )

        except PlaywrightTimeout as exc:
            last_error = (
                f"Playwright timeout setelah {target.page_timeout_ms}ms: {str(exc)[:200]}"
            )
            logger.warning(
                f"[Scraper] Timeout '{target.name}' attempt={attempt}/{max_attempts}: {last_error}"
            )

        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                f"[Scraper] Intercept gagal '{target.name}' attempt={attempt}/{max_attempts}: {last_error}"
            )

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning(
                f"[Scraper] Error '{target.name}' attempt={attempt}/{max_attempts}: {last_error}",
                exc_info=True,
            )

        if attempt < max_attempts:
            backoff = min(2 ** attempt, 30)
            logger.info(
                f"[Scraper] Retry '{target.name}' dalam {backoff}s (attempt {attempt}/{max_attempts})..."
            )
            await asyncio.sleep(backoff)

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


async def _scrape_with_body_capture(target: ScrapingTarget) -> ScrapedPage:
    logger.info(f"[Scraper] Memulai body-capture intercept: '{target.name}' → {target.url}")
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    last_error: Optional[str] = None
    max_attempts = target.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        collected_jsons: List[str] = []

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
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

                context = await browser.new_context(
                    user_agent=_select_user_agent(attempt - 1),
                    viewport={"width": 1366, "height": 768},
                    locale="id-ID",
                    timezone_id="Asia/Jakarta",
                    java_script_enabled=True,
                    extra_http_headers={
                        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
                    },
                )

                page = await context.new_page()

                async def _route_handler(route):
                    if route.request.resource_type in _BLOCKED_RESOURCE_TYPES:
                        await route.abort()
                        return
                    if _is_blocked_url(route.request.url):
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", _route_handler)

                intercepted_bodies: List[str] = []

                async def _handle_response(response) -> None:
                    if len(intercepted_bodies) >= target.max_responses:
                        return
                    if not _url_matches_pattern(response.url, target.api_url_pattern):
                        return
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        logger.debug(
                            f"[Scraper] URL cocok tapi Content-Type bukan JSON ({content_type}): {response.url[:100]}"
                        )
                        return
                    if not (200 <= response.status < 300):
                        logger.debug(
                            f"[Scraper] URL cocok tapi status {response.status}: {response.url[:100]}"
                        )
                        return
                    try:
                        body_bytes = await response.body()
                        if len(body_bytes) < target.min_response_bytes:
                            logger.debug(
                                f"[Scraper] Response terlalu kecil ({len(body_bytes)} bytes "
                                f"< {target.min_response_bytes}): {response.url[:100]}"
                            )
                            return
                        body_str = body_bytes.decode("utf-8", errors="replace")
                        json.loads(body_str)
                        if len(intercepted_bodies) < target.max_responses:
                            intercepted_bodies.append(body_str)
                            logger.info(
                                f"[Scraper] ✓ JSON intercepted dari '{target.name}': "
                                f"URL={response.url[:80]}... | "
                                f"size={len(body_bytes)} bytes | "
                                f"total_collected={len(intercepted_bodies)}"
                            )
                    except json.JSONDecodeError:
                        logger.debug(f"[Scraper] Response bukan JSON valid: {response.url[:100]}")
                    except Exception as exc:
                        logger.debug(f"[Scraper] Gagal baca body response ({response.url[:80]}): {exc}")

                page.on("response", _handle_response)

                await page.goto(
                    target.url,
                    wait_until=target.wait_until,
                    timeout=target.page_timeout_ms,
                )

                if target.xhr_settle_ms > 0:
                    logger.debug(
                        f"[Scraper] Menunggu {target.xhr_settle_ms}ms untuk XHR settle di '{target.name}'..."
                    )
                    await page.wait_for_timeout(target.xhr_settle_ms)

                collected_jsons = list(intercepted_bodies)

                await browser.close()

            if not collected_jsons:
                raise ValueError(
                    f"Tidak ada JSON response yang di-intercept. "
                    f"Pola '{target.api_url_pattern}' mungkin tidak cocok dengan URL di '{target.url}'."
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
                f"Playwright timeout setelah {target.page_timeout_ms}ms: {str(exc)[:200]}"
            )
            logger.warning(
                f"[Scraper] Timeout '{target.name}' attempt={attempt}/{max_attempts}: {last_error}"
            )

        except ValueError as exc:
            last_error = str(exc)
            logger.warning(
                f"[Scraper] Intercept gagal '{target.name}' attempt={attempt}/{max_attempts}: {last_error}"
            )

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning(
                f"[Scraper] Error '{target.name}' attempt={attempt}/{max_attempts}: {last_error}",
                exc_info=True,
            )

        if attempt < max_attempts:
            backoff = min(2 ** attempt, 30)
            logger.info(
                f"[Scraper] Retry '{target.name}' dalam {backoff}s (attempt {attempt}/{max_attempts})..."
            )
            await asyncio.sleep(backoff)

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


async def scrape_all_targets(
    targets: Optional[List[ScrapingTarget]] = None,
) -> List[ScrapedPage]:
    if targets is None:
        targets = SCRAPING_TARGETS

    if not targets:
        logger.warning("[Scraper] Tidak ada scraping target yang dikonfigurasi.")
        return []

    results: List[ScrapedPage] = []

    for i, target in enumerate(targets):
        if _circuit_breaker.is_open(target.name):
            logger.warning(f"[Scraper] SKIP '{target.name}' — circuit breaker OPEN.")
            results.append(ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_json="",
                success=False,
                error_message="Circuit breaker open — target dinonaktifkan sementara.",
            ))
            continue

        page_result = await _scrape_with_body_capture(target)

        if page_result.success:
            _circuit_breaker.record_success(target.name)
        else:
            _circuit_breaker.record_failure(target.name)

        results.append(page_result)

        if i < len(targets) - 1:
            logger.debug(f"[Scraper] Jeda {_INTER_TARGET_DELAY_SEC}s sebelum target berikutnya...")
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