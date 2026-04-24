# agents/market_intelligence/config.py
"""
Konfigurasi terpusat Market Intelligence Agent.

Semua URL target, CSS selector, LLM setting, dan konstanta agen
WAJIB didefinisikan di sini — bukan di-hardcode di scraper atau analyzer.
Ubah nilai via environment variable (.env) atau langsung di dataclass ini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Selector Strategy: gunakan list untuk fallback jika selector utama berubah.
# Scraper akan mencoba satu per satu dari kiri ke kanan.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScrapingTarget:
    """Satu sumber data yang akan di-scrape."""
    name:              str
    url:               str
    # CSS selectors untuk container teks utama (dicoba berurutan)
    content_selectors: List[str] = field(default_factory=list)
    # Timeout Playwright per halaman (milliseconds)
    page_timeout_ms:   int = 30_000
    # Jumlah retry jika halaman gagal dimuat
    max_retries:       int = 2
    # Tunggu network idle sebelum scraping (True untuk SPA/JS-heavy)
    wait_until:        str = "domcontentloaded"


# ---------------------------------------------------------------------------
# Target URLs
# Sesuaikan url dan selector berdasarkan situs marketplace/forum yang relevan.
# Contoh di bawah adalah placeholder struktural yang valid secara kode.
# ---------------------------------------------------------------------------

SCRAPING_TARGETS: List[ScrapingTarget] = [
    ScrapingTarget(
        name="Tokopedia - Durian Musang King",
        url=(
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+musang+king+D197&sort=5"
        ),
        content_selectors=[
            "[data-testid='divSRPContentProducts']",  # selector primer
            ".css-54k5sq",                             # fallback v1
            "[class*='product-list']",                 # fallback v2 (wildcard)
        ],
        wait_until="networkidle",
        page_timeout_ms=45_000,
    ),
    ScrapingTarget(
        name="Tokopedia - Durian Duri Hitam",
        url=(
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+duri+hitam+D200+ochee&sort=5"
        ),
        content_selectors=[
            "[data-testid='divSRPContentProducts']",
            ".css-54k5sq",
            "[class*='product-list']",
        ],
        wait_until="networkidle",
        page_timeout_ms=45_000,
    ),
    ScrapingTarget(
        name="Shopee - Durian Premium",
        url="https://shopee.co.id/search?keyword=durian+musang+king+duri+hitam&sortBy=sales",
        content_selectors=[
            ".shopee-search-item-result__items",
            "[data-sqe='item']",
            ".col-xs-2-4",
        ],
        wait_until="networkidle",
        page_timeout_ms=45_000,
    ),
    ScrapingTarget(
        name="Forum Durian Indonesia",
        url="https://www.kaskus.co.id/thread/harga-durian-premium-musang-king",
        content_selectors=[
            ".post-message",
            "[class*='postbody']",
            "article",
        ],
        wait_until="domcontentloaded",
        page_timeout_ms=20_000,
    ),
]


# ---------------------------------------------------------------------------
# LLM Configuration (Ollama)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaConfig:
    base_url:       str   = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    model:          str   = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    # Timeout per request LLM (detik)
    timeout_sec:    float = 120.0
    # Sampling — gunakan temperature rendah untuk output terstruktur
    temperature:    float = 0.1
    top_p:          float = 0.9
    # Panjang konteks maksimum yang dikirim ke LLM (karakter)
    max_input_chars: int  = 8_000
    # Retry jika LLM tidak mengembalikan JSON valid
    max_parse_retries: int = 2


OLLAMA_CONFIG = OllamaConfig()


# ---------------------------------------------------------------------------
# NestJS Client Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NestJSClientConfig:
    base_url:       str   = field(default_factory=lambda: os.getenv("NESTJS_BASE_URL", "http://localhost:3000"))
    endpoint:       str   = "/api/market-intelligence/ingest"
    api_key:        str   = field(default_factory=lambda: os.getenv("NESTJS_INTERNAL_API_KEY", ""))
    timeout_sec:    float = 15.0
    max_retries:    int   = 3
    # Backoff awal sebelum retry (detik)
    retry_backoff_sec: float = 2.0


NESTJS_CONFIG = NestJSClientConfig()


# ---------------------------------------------------------------------------
# Scheduler Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    # Waktu eksekusi (WIB = UTC+7)
    # Default: setiap hari pukul 02:30 WIB (= 19:30 UTC sehari sebelumnya)
    cron_hour:   int = int(os.getenv("MI_AGENT_CRON_HOUR", "19"))   # UTC
    cron_minute: int = int(os.getenv("MI_AGENT_CRON_MINUTE", "30"))
    # Timezone scheduler
    timezone:    str = "UTC"
    # Nonaktifkan agent sepenuhnya (untuk testing)
    disabled:    bool = os.getenv("MI_AGENT_DISABLED", "false").lower() == "true"
    # Maksimum waktu total satu siklus run (detik) sebelum dipaksa stop
    max_run_duration_sec: int = int(os.getenv("MI_AGENT_MAX_DURATION_SEC", "3600"))


SCHEDULER_CONFIG = SchedulerConfig()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

# Berapa kali scraping satu target boleh gagal berturut-turut
# sebelum dinonaktifkan sementara
CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("MI_CIRCUIT_BREAKER_THRESHOLD", "3"))

# Berapa lama (detik) target dinonaktifkan setelah circuit breaker trip
CIRCUIT_BREAKER_COOLDOWN_SEC: int = int(os.getenv("MI_CIRCUIT_BREAKER_COOLDOWN_SEC", "86400"))  # 24 jam