# agents/market_intelligence/config.py

from __future__ import annotations

import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ScrapingTarget:
    name:               str
    url:                str
    api_url_pattern:    str = "**/api/**"
    page_timeout_ms:    int = 35_000
    max_retries:        int = 2
    wait_until:         str = "networkidle"
    xhr_settle_ms:      int = 3_000
    max_responses:      int = 5
    min_response_bytes: int = 500


SCRAPING_TARGETS: list[ScrapingTarget] = [
    ScrapingTarget(
        name               = "Tokopedia - Durian Musang King Utuh",
        url                = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+musang+king+utuh+berkulit&sort=5"
        ),
        api_url_pattern    = "**/graphql/**",
        wait_until         = "networkidle",
        page_timeout_ms    = 45_000,
        xhr_settle_ms      = 4_000,
        max_responses      = 3,
        min_response_bytes = 2_000,
    ),
    ScrapingTarget(
        name               = "Tokopedia - Durian Duri Hitam Utuh",
        url                = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+duri+hitam+ochee+utuh+berkulit&sort=5"
        ),
        api_url_pattern    = "**/graphql/**",
        wait_until         = "networkidle",
        page_timeout_ms    = 45_000,
        xhr_settle_ms      = 4_000,
        max_responses      = 3,
        min_response_bytes = 2_000,
    ),
    ScrapingTarget(
        name               = "Tokopedia - Durian Sultan D24 Golden Bun D13 Utuh",
        url                = (
            "https://www.tokopedia.com/search"
            "?st=product&q=durian+sultan+golden+bun+utuh+berkulit&sort=5"
        ),
        api_url_pattern    = "**/graphql/**",
        wait_until         = "networkidle",
        page_timeout_ms    = 45_000,
        xhr_settle_ms      = 4_000,
        max_responses      = 3,
        min_response_bytes = 2_000,
    ),
    ScrapingTarget(
        name               = "Shopee - Durian Premium Utuh",
        url                = (
            "https://shopee.co.id/search"
            "?keyword=durian+musang+king+duri+hitam+utuh+berkulit&sortBy=sales"
        ),
        api_url_pattern    = "**/api/v4/search/**",
        wait_until         = "networkidle",
        page_timeout_ms    = 45_000,
        xhr_settle_ms      = 5_000,
        max_responses      = 3,
        min_response_bytes = 2_000,
    ),
    ScrapingTarget(
        name               = "Shopee - Durian Golden Bun Sultan Utuh",
        url                = (
            "https://shopee.co.id/search"
            "?keyword=durian+golden+bun+sultan+d24+utuh+berkulit&sortBy=sales"
        ),
        api_url_pattern    = "**/api/v4/search/**",
        wait_until         = "networkidle",
        page_timeout_ms    = 45_000,
        xhr_settle_ms      = 5_000,
        max_responses      = 3,
        min_response_bytes = 2_000,
    ),
]

@dataclass(frozen=True)
class OllamaConfig:
    base_url:        str   = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model:           str   = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    timeout_sec:     float = 120.0
    max_input_chars: int   = 10_000
    temperature:     float = 0.05
    top_p:           float = 0.9
    max_parse_retries: int = 2


OLLAMA_CONFIG = OllamaConfig()

@dataclass(frozen=True)
class SchedulerConfig:
    cron_hour:            int  = int(os.getenv("MI_AGENT_CRON_HOUR",   "19"))
    cron_minute:          int  = int(os.getenv("MI_AGENT_CRON_MINUTE", "30"))
    timezone:             str  = "UTC"
    disabled:             bool = os.getenv("MI_AGENT_DISABLED", "false").lower() == "true"
    max_run_duration_sec: int  = int(os.getenv("MI_AGENT_MAX_DURATION_SEC", "3600"))


SCHEDULER_CONFIG = SchedulerConfig()

@dataclass(frozen=True)
class CircuitBreakerConfig:
    threshold:    int = int(os.getenv("MI_CIRCUIT_BREAKER_THRESHOLD",    "3"))
    cooldown_sec: int = int(os.getenv("MI_CIRCUIT_BREAKER_COOLDOWN_SEC", "86400"))


CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()
