# agents/market_intelligence/config.py
"""
Konfigurasi terpusat Market Intelligence Agent.

Changelog v2:
  - ScrapingTarget: `content_selectors` diganti `api_url_pattern` (regex/glob).
    Ini adalah perubahan arsitektur inti: dari DOM scraping ke Network Interception.
  - SCRAPING_TARGETS diperbarui dengan pola URL XHR/Fetch API e-commerce.
  - OllamaConfig: `max_input_chars` dinaikkan (JSON intercept lebih ringkas dari HTML).
  - Semua nilai konfigurasi WAJIB diset via environment variable — tidak hardcode.

Changelog v2.1 (Bug Fix):
  - [FIX BUG-04/11] NestJSClientConfig.endpoint dikoreksi:
    '/api/market-intelligence/ingest' → '/api/v1/ai-integration/market-report'
    Sesuai NestJS: @Controller('ai-integration') + @Post('market-report')
    dengan global prefix 'api/v1' di main.ts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# ScrapingTarget — konfigurasi satu sumber data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScrapingTarget:
    """
    Satu sumber data yang akan di-intercept network-nya oleh Playwright.

    Perubahan dari v1:
    - `content_selectors` (List[str]) DIHAPUS — tidak lagi melakukan DOM scraping.
    - `api_url_pattern` (str) DITAMBAHKAN — pola glob/regex untuk mencocokkan
      URL XHR atau Fetch request yang berisi data produk JSON dari e-commerce.

    Cara kerja baru:
      Playwright menavigasi ke `url` (halaman pencarian), lalu memantau semua
      network response. Response yang URL-nya cocok dengan `api_url_pattern`
      akan di-intercept dan body JSON-nya disimpan sebagai raw data.
      Ini jauh lebih stabil dibanding CSS selector karena API endpoint
      e-commerce berubah jauh lebih jarang daripada class HTML/CSS-nya.

    Panduan `api_url_pattern`:
      - Gunakan glob-style: "**/api/v1/search**", "**pdp/get_product**"
      - Gunakan substring: "*search*", "*product*", "*catalog*"
      - Makin spesifik makin baik untuk menghindari false-positive.
    """
    name:             str
    url:              str
    # Pola glob/substring untuk mencocokkan URL XHR/Fetch response target
    api_url_pattern:  str = "**/api/**"
    # Timeout Playwright per halaman (milliseconds)
    page_timeout_ms:  int = 35_000
    # Jumlah retry jika halaman gagal dimuat
    max_retries:      int = 2
    # Strategi tunggu: "networkidle" untuk SPA (lebih lama tapi aman),
    # "domcontentloaded" untuk halaman server-side render
    wait_until:       str = "networkidle"
    # Timeout tambahan (ms) untuk menunggu XHR setelah page load
    # Berguna untuk SPA yang melakukan lazy loading setelah initial render
    xhr_settle_ms:    int = 3_000
    # Jumlah maksimum JSON response yang dikumpulkan per halaman
    # Mencegah pengumpulan resource kecil yang tidak relevan (icon, analytics, dll.)
    max_responses:    int = 5
    # Ukuran minimum body response (bytes) agar tidak mengambil response kecil
    # seperti tracking pixel atau config JSON berukuran kecil
    min_response_bytes: int = 500


# ---------------------------------------------------------------------------
# Target E-commerce
#
# CATATAN IMPLEMENTASI:
# URL menggunakan kata kunci "utuh" atau "whole" untuk membantu e-commerce
# menampilkan hasil yang relevan. Namun tetap ada kemungkinan produk kupas
# muncul, sehingga lapisan filter LLM dan Python di task.py tetap krusial.
#
# `api_url_pattern` disesuaikan dengan pola XHR/Fetch yang digunakan
# masing-masing platform berdasarkan observasi network tab DevTools.
# Pola ini HARUS diverifikasi ulang secara berkala karena platform dapat
# mengubah endpoint API mereka.
# ---------------------------------------------------------------------------

SCRAPING_TARGETS: list[ScrapingTarget] = [
    # ── Tokopedia: Durian Musang King Utuh ──────────────────────────────────
    ScrapingTarget(
        name            = "Tokopedia - Durian Musang King Utuh",
        url             = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+musang+king+utuh+berkulit&sort=5"
        ),
        # Tokopedia menggunakan GraphQL via endpoint /graphql/
        # Response mengandung data produk dalam structure data.searchProduct.data
        api_url_pattern = "**/graphql/**",
        wait_until      = "networkidle",
        page_timeout_ms = 45_000,
        xhr_settle_ms   = 4_000,
        max_responses   = 3,
        min_response_bytes = 2_000,
    ),

    # ── Tokopedia: Durian Duri Hitam Utuh ───────────────────────────────────
    ScrapingTarget(
        name            = "Tokopedia - Durian Duri Hitam Utuh",
        url             = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+duri+hitam+ochee+utuh+berkulit&sort=5"
        ),
        api_url_pattern = "**/graphql/**",
        wait_until      = "networkidle",
        page_timeout_ms = 45_000,
        xhr_settle_ms   = 4_000,
        max_responses   = 3,
        min_response_bytes = 2_000,
    ),

    # ── Tokopedia: Durian Sultan & Golden Bun ────────────────────────────────
    ScrapingTarget(
        name            = "Tokopedia - Durian Sultan D24 Golden Bun D13 Utuh",
        url             = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+sultan+golden+bun+utuh+berkulit&sort=5"
        ),
        api_url_pattern = "**/graphql/**",
        wait_until      = "networkidle",
        page_timeout_ms = 45_000,
        xhr_settle_ms   = 4_000,
        max_responses   = 3,
        min_response_bytes = 2_000,
    ),

    # ── Shopee: Durian Premium Utuh ─────────────────────────────────────────
    ScrapingTarget(
        name            = "Shopee - Durian Premium Utuh",
        url             = (
            "https://shopee.co.id/search"
            "?keyword=durian+musang+king+duri+hitam+utuh+berkulit&sortBy=sales"
        ),
        # Shopee menggunakan endpoint /api/v4/ untuk product search
        api_url_pattern = "**/api/v4/search/**",
        wait_until      = "networkidle",
        page_timeout_ms = 45_000,
        xhr_settle_ms   = 5_000,
        max_responses   = 3,
        min_response_bytes = 2_000,
    ),

    # ── Shopee: Durian Golden Bun & Sultan Utuh ─────────────────────────────
    ScrapingTarget(
        name            = "Shopee - Durian Golden Bun Sultan Utuh",
        url             = (
            "https://shopee.co.id/search"
            "?keyword=durian+golden+bun+sultan+d24+utuh+berkulit&sortBy=sales"
        ),
        api_url_pattern = "**/api/v4/search/**",
        wait_until      = "networkidle",
        page_timeout_ms = 45_000,
        xhr_settle_ms   = 5_000,
        max_responses   = 3,
        min_response_bytes = 2_000,
    ),
]


# ---------------------------------------------------------------------------
# LLM Configuration (Ollama)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    base_url:    str   = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model:       str   = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    timeout_sec: float = 120.0
    # JSON intercept jauh lebih ringkas dari HTML — naikkan sedikit untuk
    # mengakomodasi output CoT yang lebih panjang di field "notes"
    max_input_chars:    int = 10_000
    # Temperatur rendah untuk output terstruktur yang konsisten
    temperature:        float = 0.05
    top_p:              float = 0.9
    # Retry jika LLM tidak mengembalikan JSON valid
    max_parse_retries:  int   = 2


OLLAMA_CONFIG = OllamaConfig()


# ---------------------------------------------------------------------------
# NestJS Client Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestJSClientConfig:
    base_url:          str   = field(
        default_factory=lambda: os.getenv("NESTJS_BASE_URL", "http://localhost:3000")
    )
    # [FIX BUG-04] Endpoint dikoreksi sesuai NestJS routing aktual:
    #   @Controller('ai-integration') + @Post('market-report')
    #   + global prefix 'api/v1' (set di main.ts: app.setGlobalPrefix('api/v1'))
    #   = /api/v1/ai-integration/market-report
    #
    # SEBELUMNYA (salah): "/api/market-intelligence/ingest"
    #   → route tidak pernah ada di NestJS → selalu 404
    endpoint:          str   = "/api/v1/ai-integration/market-report"
    api_key:           str   = field(
        default_factory=lambda: os.getenv("NESTJS_INTERNAL_API_KEY", "")
    )
    timeout_sec:       float = 15.0
    max_retries:       int   = 3
    retry_backoff_sec: float = 2.0


NESTJS_CONFIG = NestJSClientConfig()


# ---------------------------------------------------------------------------
# Scheduler Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    # Default: setiap hari pukul 02:30 WIB (= 19:30 UTC sehari sebelumnya)
    cron_hour:   int  = int(os.getenv("MI_AGENT_CRON_HOUR",   "19"))
    cron_minute: int  = int(os.getenv("MI_AGENT_CRON_MINUTE", "30"))
    timezone:    str  = "UTC"
    disabled:    bool = os.getenv("MI_AGENT_DISABLED", "false").lower() == "true"
    max_run_duration_sec: int = int(os.getenv("MI_AGENT_MAX_DURATION_SEC", "3600"))


SCHEDULER_CONFIG = SchedulerConfig()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

CIRCUIT_BREAKER_THRESHOLD:    int = int(os.getenv("MI_CIRCUIT_BREAKER_THRESHOLD",    "3"))
CIRCUIT_BREAKER_COOLDOWN_SEC: int = int(os.getenv("MI_CIRCUIT_BREAKER_COOLDOWN_SEC", "86400"))