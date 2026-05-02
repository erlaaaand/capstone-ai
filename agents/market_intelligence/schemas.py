# agents/market_intelligence/schemas.py

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DurianVariety(str, Enum):
    GOLDEN_BUN  = "D13"   # Golden Bun (Johor)
    MUSANG_KING = "D197"  # Musang King / Mao Shan Wang / Raja Kunyit (Kelantan)
    DATO_NINA   = "D2"    # Dato Nina (Melaka)
    DURI_HITAM  = "D200"  # Duri Hitam / Black Thorn / Ochee (Penang)
    SULTAN      = "D24"   # Sultan / Bukit Merah (Perak / Selangor)


class AgentRunStatus(str, Enum):
    SUCCESS       = "success"
    PARTIAL       = "partial"        # Ada target gagal, tapi ada data masuk
    SCRAPER_ERROR = "scraper_error"
    LLM_ERROR     = "llm_error"
    NO_DATA       = "no_data"


class ScrapedPage(BaseModel):
    source_name:   str      = Field(..., description="Nama sumber (dari ScrapingTarget.name).")
    source_url:    str      = Field(..., description="URL halaman yang di-navigate Playwright.")
    raw_json:      str      = Field(
        default="",
        description=(
            "JSON mentah yang di-intercept dari XHR/Fetch API e-commerce. "
            "Berupa JSON string tunggal atau array JSON yang di-join."
        ),
    )
    scraped_at:    datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    success:       bool     = Field(default=True)
    error_message: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_raw_json_on_success(self) -> "ScrapedPage":
        if self.success and not self.raw_json.strip():
            raise ValueError("raw_json tidak boleh kosong jika success=True.")
        return self

    @model_validator(mode="after")
    def validate_error_message_on_failure(self) -> "ScrapedPage":
        if not self.success and not self.error_message:
            raise ValueError("error_message harus diisi jika success=False.")
        return self


class MarketPriceEntry(BaseModel):
    variety_code:     DurianVariety = Field(
        ...,
        description="Kode DOA varietas. D13 / D197 / D2 / D200 / D24.",
    )
    variety_alias:    str = Field(
        ...,
        description="Alias yang ditemukan di JSON sumber (cth: 'musang king', 'ochee').",
        max_length=100,
    )
    is_whole_fruit:   bool = Field(
        ...,
        description=(
            "TRUE jika dan hanya jika produk adalah DURIAN UTUH DENGAN KULIT. "
            "Gatekeeper utama anti data leakage."
        ),
    )
    weight_reference: str = Field(
        ...,
        description=(
            "Referensi berat ASLI dari listing penjual sebelum normalisasi, "
            "cth: 'per buah 2-3 kg', 'per kg'."
        ),
        max_length=200,
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description=(
            "Chain-of-Thought LLM: catatan matematis normalisasi harga ke per-Kg. "
            "None jika harga sudah dalam satuan per-Kg dari sumber."
        ),
    )

    price_per_kg_min:   Optional[float] = Field(default=None, ge=0)
    price_per_kg_max:   Optional[float] = Field(default=None, ge=0)
    price_per_kg_avg:   Optional[float] = Field(default=None, ge=0)
    price_per_unit_min: Optional[float] = Field(default=None, ge=0)
    price_per_unit_max: Optional[float] = Field(default=None, ge=0)

    location_hint:    Optional[str] = Field(default=None, max_length=200)
    seller_type:      Optional[str] = Field(default=None, max_length=100)
    confidence:       float         = Field(default=0.5, ge=0.0, le=1.0)
    raw_text_snippet: Optional[str] = Field(default=None, max_length=500)

    @field_validator("variety_alias", mode="before")
    @classmethod
    def strip_variety_alias(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @field_validator("weight_reference", mode="before")
    @classmethod
    def strip_weight_reference(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def at_least_one_price_present(self) -> "MarketPriceEntry":
        prices = [
            self.price_per_kg_min,
            self.price_per_kg_max,
            self.price_per_kg_avg,
            self.price_per_unit_min,
            self.price_per_unit_max,
        ]
        if not any(p is not None for p in prices):
            raise ValueError("Setidaknya satu field harga harus terisi.")
        return self

    @model_validator(mode="after")
    def max_not_less_than_min(self) -> "MarketPriceEntry":
        if (
            self.price_per_kg_min is not None
            and self.price_per_kg_max is not None
            and self.price_per_kg_max < self.price_per_kg_min
        ):
            raise ValueError("price_per_kg_max tidak boleh lebih kecil dari price_per_kg_min.")
        if (
            self.price_per_unit_min is not None
            and self.price_per_unit_max is not None
            and self.price_per_unit_max < self.price_per_unit_min
        ):
            raise ValueError("price_per_unit_max tidak boleh lebih kecil dari price_per_unit_min.")
        return self

    @model_validator(mode="after")
    def avg_within_range(self) -> "MarketPriceEntry":
        """Pastikan avg berada di antara min dan max jika ketiganya tersedia."""
        if (
            self.price_per_kg_avg is not None
            and self.price_per_kg_min is not None
            and self.price_per_kg_avg < self.price_per_kg_min
        ):
            raise ValueError("price_per_kg_avg tidak boleh lebih kecil dari price_per_kg_min.")
        if (
            self.price_per_kg_avg is not None
            and self.price_per_kg_max is not None
            and self.price_per_kg_avg > self.price_per_kg_max
        ):
            raise ValueError("price_per_kg_avg tidak boleh lebih besar dari price_per_kg_max.")
        return self


class MarketReportPayload(BaseModel):
    agent_version:   str            = Field(..., description="Versi service saat run (dari settings.APP_VERSION).")
    run_id:          str            = Field(..., description="UUID unik per run agen.")
    run_started_at:  datetime       = Field(..., description="Waktu mulai run agen.")
    run_ended_at:    datetime       = Field(default_factory=lambda: datetime.now(timezone.utc))
    status:          AgentRunStatus = Field(...)
    entries:         List[MarketPriceEntry] = Field(
        default_factory=list,
        description=(
            "Semua entry harga yang LOLOS validasi ganda: "
            "LLM validation + Python is_whole_fruit gate."
        ),
    )
    sources_scraped:   int = Field(default=0, ge=0)
    sources_failed:    int = Field(default=0, ge=0)
    llm_parse_errors:  int = Field(default=0, ge=0)
    entries_discarded: int = Field(
        default=0,
        ge=0,
        description="Entry yang dibuang karena is_whole_fruit=False.",
    )
    error_details: Optional[str] = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_run_times(self) -> "MarketReportPayload":
        if self.run_ended_at < self.run_started_at:
            raise ValueError("run_ended_at tidak boleh lebih awal dari run_started_at.")
        return self

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def run_duration_sec(self) -> float:
        return (self.run_ended_at - self.run_started_at).total_seconds()