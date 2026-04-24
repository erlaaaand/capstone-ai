# agents/market_intelligence/scraper.py
"""
Async Playwright scraper dengan teknik Network Interception.

Changelog v2 (Major Refactor — Anti DOM Fragility):
  - Strategi scraping DIUBAH TOTAL: dari CSS Selector DOM extraction
    menjadi Network Response Interception.
  - Scraper tidak lagi membaca HTML/teks dari DOM. Sebagai gantinya,
    Playwright "mendengarkan" semua network response dan menangkap
    body JSON dari XHR/Fetch yang URL-nya cocok dengan `api_url_pattern`.
  - Hasilnya disimpan dalam `ScrapedPage.raw_json` (bukan `raw_text`).
  - Keuntungan: jauh lebih tahan terhadap perubahan UI/CSS karena
    API endpoint e-commerce jarang berubah dibanding class HTML.
  - Circuit Breaker, asyncio.to_thread isolation, exponential backoff,
    dan stealth mode dipertahankan dari versi sebelumnya.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

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
# Circuit Breaker State
# ---------------------------------------------------------------------------

class _CircuitBreakerState:
    """Menyimpan state circuit breaker untuk semua target (in-process)."""

    def __init__(self) -> None:
        self._failures:   Dict[str, int]   = {}
        self._tripped_at: Dict[str, float] = {}

    def is_open(self, target_name: str) -> bool:
        """True jika circuit OPEN (target sedang diblokir sementara)."""
        tripped = self._tripped_at.get(target_name)
        if tripped is None:
            return False
        if time.time() - tripped < CIRCUIT_BREAKER_COOLDOWN_SEC:
            return True
        self.reset(target_name)
        return False

    def record_failure(self, target_name: str) -> None:
        self._failures[target_name] = self._failures.get(target_name, 0) + 1
        if self._failures[target_name] >= CIRCUIT_BREAKER_THRESHOLD:
            if target_name not in self._tripped_at:
                self._tripped_at[target_name] = time.time()
                logger.warning(
                    f"[CircuitBreaker] OPEN untuk '{target_name}' "
                    f"setelah {self._failures[target_name]} kegagalan berturut-turut. "
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
# Helper: URL Pattern Matching
# ---------------------------------------------------------------------------

def _url_matches_pattern(url: str, pattern: str) -> bool:
    """
    Cocokkan URL dengan pola glob-style atau substring.

    Mendukung dua format:
    1. Glob-style: "**/api/v4/search/**" → dikonversi ke regex
       `**` = cocokkan apa saja (termasuk `/`)
       `*`  = cocokkan apa saja kecuali `/`
    2. Substring biasa: "*search*" → cek apakah ada di URL
    """
    # Konversi glob ke regex
    # Escape karakter regex kecuali * yang kita tangani sendiri
    escaped = re.escape(pattern)
    # \*\* → .+ (satu atau lebih karakter, termasuk /)
    escaped = escaped.replace(r"\*\*", ".+")
    # \*   → [^/]+ (satu atau lebih karakter, tidak termasuk /)
    escaped = escaped.replace(r"\*", "[^/]+")

    try:
        return bool(re.search(escaped, url, re.IGNORECASE))
    except re.error:
        # Fallback ke substring match jika regex invalid
        stripped = pattern.replace("*", "").replace("/", "")
        return stripped.lower() in url.lower()


# ---------------------------------------------------------------------------
# Core Network Interception Logic (sync — dijalankan via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _scrape_single_target_sync(target: ScrapingTarget) -> ScrapedPage:
    """
    Navigasi ke halaman pencarian e-commerce dan intercept JSON API response.

    Teknik Network Interception:
    1. Daftarkan handler `page.on("response", ...)` sebelum navigasi.
    2. Handler dipanggil untuk SETIAP response saat halaman dimuat.
    3. Hanya response yang URL-nya cocok dengan `target.api_url_pattern` DAN
       Content-Type JSON yang dikumpulkan.
    4. Setelah navigasi, tunggu `xhr_settle_ms` ms untuk XHR lazy-load.
    5. Gabungkan semua JSON yang terkumpul menjadi satu JSON array string.

    WAJIB dijalankan via asyncio.to_thread() — menggunakan sync_playwright
    untuk menghindari konflik nested event loop dengan uvicorn.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

    logger.info(
        f"[Scraper] Memulai network intercept: '{target.name}' → {target.url}"
    )

    last_error: Optional[str] = None

    for attempt in range(1, target.max_retries + 2):
        collected_jsons: List[str] = []

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

                # ── Blokir resource yang tidak perlu (percepat load) ──────────
                page.route(
                    "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,otf}",
                    lambda route: route.abort(),
                )
                # Blokir analytics/tracking untuk mengurangi noise response
                page.route(
                    "**/{gtm,analytics,tracking,doubleclick,pixel,clarity}**",
                    lambda route: route.abort(),
                )

                # ── Daftarkan Network Intercept Handler ───────────────────────
                def _response_handler(response) -> None:
                    """
                    Callback dipanggil Playwright untuk setiap network response.
                    Mengumpulkan JSON dari response yang cocok dengan pola target.
                    """
                    # Stop jika sudah cukup response (hindari memory bloat)
                    if len(collected_jsons) >= target.max_responses:
                        return

                    resp_url = response.url

                    # Cek apakah URL cocok dengan pola target
                    if not _url_matches_pattern(resp_url, target.api_url_pattern):
                        return

                    # Cek Content-Type harus JSON
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        logger.debug(
                            f"[Scraper] URL cocok tapi Content-Type bukan JSON "
                            f"({content_type}): {resp_url[:100]}"
                        )
                        return

                    # Cek status code harus sukses
                    if response.status < 200 or response.status >= 300:
                        logger.debug(
                            f"[Scraper] URL cocok tapi status {response.status}: "
                            f"{resp_url[:100]}"
                        )
                        return

                    try:
                        body_bytes = response.body()

                        # Cek ukuran minimum untuk menghindari JSON config kecil
                        if len(body_bytes) < target.min_response_bytes:
                            logger.debug(
                                f"[Scraper] Response terlalu kecil "
                                f"({len(body_bytes)} bytes < {target.min_response_bytes}): "
                                f"{resp_url[:100]}"
                            )
                            return

                        body_str = body_bytes.decode("utf-8", errors="replace")

                        # Validasi bahwa body adalah JSON valid
                        json.loads(body_str)

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
                            f"[Scraper] Gagal baca body response "
                            f"({resp_url[:80]}): {exc}"
                        )

                # Pasang handler SEBELUM navigasi
                page.on("response", _response_handler)

                # ── Navigasi ke halaman pencarian ─────────────────────────────
                page.goto(
                    target.url,
                    wait_until=target.wait_until,
                    timeout=target.page_timeout_ms,
                )

                # ── Tunggu XHR lazy-load selesai ──────────────────────────────
                if target.xhr_settle_ms > 0:
                    logger.debug(
                        f"[Scraper] Menunggu {target.xhr_settle_ms}ms "
                        f"untuk XHR settle di '{target.name}'..."
                    )
                    page.wait_for_timeout(target.xhr_settle_ms)

                browser.close()

            # ── Evaluasi hasil intercept ───────────────────────────────────────
            if not collected_jsons:
                raise ValueError(
                    f"Tidak ada JSON response yang di-intercept. "
                    f"Pola '{target.api_url_pattern}' mungkin tidak cocok "
                    f"dengan endpoint e-commerce saat ini."
                )

            # Gabungkan semua JSON yang terkumpul menjadi satu representasi
            raw_json = _merge_json_responses(collected_jsons, target.name)

            logger.info(
                f"[Scraper] Berhasil intercept '{target.name}': "
                f"{len(collected_jsons)} response | "
                f"{len(raw_json)} chars gabungan | "
                f"attempt={attempt}"
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
                f"[Scraper] Timeout '{target.name}' attempt={attempt}: {last_error}"
            )
        except ValueError as exc:
            # ValueError dari validasi kita sendiri (tidak ada JSON terintercept)
            last_error = str(exc)
            logger.warning(
                f"[Scraper] Intercept gagal '{target.name}' attempt={attempt}: {last_error}"
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            logger.warning(
                f"[Scraper] Error '{target.name}' attempt={attempt}: {last_error}"
            )

        if attempt <= target.max_retries:
            backoff = 2 ** attempt
            logger.info(
                f"[Scraper] Retry '{target.name}' dalam {backoff}s "
                f"(attempt {attempt}/{target.max_retries})..."
            )
            time.sleep(backoff)

    logger.error(
        f"[Scraper] Semua attempt gagal untuk '{target.name}'. "
        f"Error terakhir: {last_error}"
    )
    return ScrapedPage(
        source_name=target.name,
        source_url=target.url,
        raw_json="",
        success=False,
        error_message=last_error,
    )


def _merge_json_responses(responses: List[str], source_name: str) -> str:
    """
    Gabungkan satu atau lebih JSON response menjadi satu string JSON.

    Strategi penggabungan:
    - Jika hanya satu response: kembalikan langsung.
    - Jika beberapa response: parse masing-masing dan gabungkan dalam
      wrapper object {"sources": [...]} agar LLM bisa membedakan asal data.

    Ini lebih baik dari konkatenasi string mentah karena tetap menghasilkan
    JSON valid yang bisa di-parse ulang.
    """
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
        return responses[0]  # Fallback ke raw pertama

    merged = json.dumps(
        {"intercepted_responses": merged_parts},
        ensure_ascii=False,
    )

    logger.debug(
        f"[Scraper] Merged {len(merged_parts)} JSON responses dari "
        f"'{source_name}': {len(merged)} chars total."
    )
    return merged


# ---------------------------------------------------------------------------
# Public Async Interface
# ---------------------------------------------------------------------------

async def scrape_all_targets(
    targets: Optional[List[ScrapingTarget]] = None,
) -> List[ScrapedPage]:
    """
    Scrape semua target secara SEQUENTIAL.

    Alasan sequential (sama seperti v1):
    - Menghindari IP ban / rate limiting dari marketplace
    - Setiap target meluncurkan instance Chromium baru — parallel = N×RAM
    - Task berjalan jam 02:00 — latency tidak kritis

    Setiap target dijalankan via asyncio.to_thread() agar sync_playwright
    tidak memblokir uvicorn event loop.
    """
    if targets is None:
        targets = SCRAPING_TARGETS

    results: List[ScrapedPage] = []

    for target in targets:
        # ── Cek Circuit Breaker ───────────────────────────────────────────────
        if _circuit_breaker.is_open(target.name):
            logger.warning(
                f"[Scraper] SKIP '{target.name}' — circuit breaker OPEN. "
                "Target dinonaktifkan sementara karena terlalu banyak kegagalan."
            )
            results.append(ScrapedPage(
                source_name=target.name,
                source_url=target.url,
                raw_json="",
                success=False,
                error_message="Circuit breaker open — target dinonaktifkan sementara.",
            ))
            continue

        # ── Jalankan scraper di thread terpisah ───────────────────────────────
        page_result = await asyncio.to_thread(
            _scrape_single_target_sync, target
        )

        if page_result.success:
            _circuit_breaker.record_success(target.name)
        else:
            _circuit_breaker.record_failure(target.name)

        results.append(page_result)

        # Jeda antar target untuk menghindari rate limiting
        await asyncio.sleep(3.0)

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        f"[Scraper] Intercept selesai. "
        f"{succeeded}/{len(results)} target berhasil."
    )
    return results