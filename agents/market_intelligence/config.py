# agents/market_intelligence/config.py

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final


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

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ScrapingTarget.name tidak boleh kosong.")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(f"ScrapingTarget.url tidak valid: {self.url}")
        if self.page_timeout_ms <= 0:
            raise ValueError("page_timeout_ms harus > 0.")
        if self.max_retries < 0:
            raise ValueError("max_retries harus >= 0.")
        if self.xhr_settle_ms < 0:
            raise ValueError("xhr_settle_ms harus >= 0.")
        if self.max_responses <= 0:
            raise ValueError("max_responses harus > 0.")
        if self.min_response_bytes < 0:
            raise ValueError("min_response_bytes harus >= 0.")


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


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "")
    try:
        return int(raw) if raw.strip() else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


@dataclass(frozen=True)
class OllamaConfig:
    base_url:          str   = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    model:             str   = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    )
    timeout_sec:       float = field(
        default_factory=lambda: _env_float("OLLAMA_TIMEOUT_SEC", 120.0)
    )
    max_input_chars:   int   = field(
        default_factory=lambda: _env_int("OLLAMA_MAX_INPUT_CHARS", 10_000)
    )
    temperature:       float = field(
        default_factory=lambda: _env_float("OLLAMA_TEMPERATURE", 0.05)
    )
    top_p:             float = field(
        default_factory=lambda: _env_float("OLLAMA_TOP_P", 0.9)
    )
    max_parse_retries: int   = field(
        default_factory=lambda: _env_int("OLLAMA_MAX_PARSE_RETRIES", 2)
    )
    num_predict:       int   = field(
        default_factory=lambda: _env_int("OLLAMA_NUM_PREDICT", 4096)
    )

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"OLLAMA_BASE_URL tidak valid: {self.base_url}")
        if not self.model.strip():
            raise ValueError("OLLAMA_MODEL tidak boleh kosong.")
        if self.timeout_sec <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SEC harus > 0.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("OLLAMA_TEMPERATURE harus antara 0.0 dan 2.0.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("OLLAMA_TOP_P harus antara 0.0 (exclusive) dan 1.0.")
        if self.max_parse_retries < 0:
            raise ValueError("OLLAMA_MAX_PARSE_RETRIES harus >= 0.")
        if self.num_predict <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT harus > 0.")


OLLAMA_CONFIG = OllamaConfig()


@dataclass(frozen=True)
class SchedulerConfig:
    cron_hour:            int  = field(
        default_factory=lambda: _env_int("MI_AGENT_CRON_HOUR", 19)
    )
    cron_minute:          int  = field(
        default_factory=lambda: _env_int("MI_AGENT_CRON_MINUTE", 30)
    )
    timezone:             str  = field(
        default_factory=lambda: os.getenv("MI_AGENT_TIMEZONE", "UTC")
    )
    disabled:             bool = field(
        default_factory=lambda: _env_bool("MI_AGENT_DISABLED", False)
    )
    max_run_duration_sec: int  = field(
        default_factory=lambda: _env_int("MI_AGENT_MAX_DURATION_SEC", 3600)
    )

    def __post_init__(self) -> None:
        if not 0 <= self.cron_hour <= 23:
            raise ValueError(f"MI_AGENT_CRON_HOUR harus 0–23, dapat {self.cron_hour}.")
        if not 0 <= self.cron_minute <= 59:
            raise ValueError(f"MI_AGENT_CRON_MINUTE harus 0–59, dapat {self.cron_minute}.")
        if self.max_run_duration_sec <= 0:
            raise ValueError("MI_AGENT_MAX_DURATION_SEC harus > 0.")


SCHEDULER_CONFIG = SchedulerConfig()


@dataclass(frozen=True)
class CircuitBreakerConfig:
    threshold:    int = field(
        default_factory=lambda: _env_int("MI_CIRCUIT_BREAKER_THRESHOLD", 3)
    )
    cooldown_sec: int = field(
        default_factory=lambda: _env_int("MI_CIRCUIT_BREAKER_COOLDOWN_SEC", 86_400)
    )

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError("MI_CIRCUIT_BREAKER_THRESHOLD harus >= 1.")
        if self.cooldown_sec <= 0:
            raise ValueError("MI_CIRCUIT_BREAKER_COOLDOWN_SEC harus > 0.")


CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig()