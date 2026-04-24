# agents/market_intelligence/schemas.py
"""
Pydantic models untuk Market Intelligence Agent.

Tiga layer schema:
  1. ScrapedPage      — output mentah dari scraper.py
  2. MarketPriceEntry — satu entri harga yang berhasil di-extract LLM
  3. MarketReportPayload — payload final yang dikirim ke NestJS
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enum: Kode varietas yang agent ini tangani
# ---------------------------------------------------------------------------

class DurianVariety(str, Enum):
    """
    Hanya varietas premium yang relevan untuk market intelligence.
    D197 dikenal secara komersial sebagai Musang King / Mao Shan Wang.
    D200 dikenal sebagai Duri Hitam / Ochee / Black Thorn.
    """
    MUSANG_KING = "D197"
    DURI_HITAM  = "D200"


# ---------------------------------------------------------------------------
# Layer 1: Raw scraping result
# ---------------------------------------------------------------------------

class ScrapedPage(BaseModel):
    """Hasil mentah Playwright sebelum diproses LLM."""

    source_name:    str       = Field(..., description="Nama sumber (dari ScrapingTarget.name).")
    source_url:     str       = Field(..., description="URL halaman yang di-scrape.")
    raw_text:       str       = Field(..., description="Teks mentah yang di-extract dari DOM.")
    scraped_at:     datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
    success:        bool      = Field(default=True)
    error_message:  Optional[str] = Field(default=None)

    @field_validator("raw_text")
    @classmethod
    def text_not_empty_on_success(cls, v: str, info) -> str:
        # Validasi hanya saat scraping berhasil
        if info.data.get("success", True) and not v.strip():
            raise ValueError("raw_text tidak boleh kosong jika success=True.")
        return v


# ---------------------------------------------------------------------------
# Layer 2: Satu entry harga hasil analisis LLM
# ---------------------------------------------------------------------------

class MarketPriceEntry(BaseModel):
    """
    Satu data point harga yang berhasil di-ekstrak dari teks.
    Semua field harga dalam satuan IDR (Rupiah).
    """

    variety_code:       DurianVariety = Field(
        ...,
        description="Kode DOA varietas. D197=Musang King, D200=Duri Hitam.",
    )
    variety_alias:      str = Field(
        ...,
        description="Alias yang ditemukan di teks (cth: 'musang king', 'ochee', 'black thorn').",
        max_length=100,
    )
    price_per_kg_min:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga minimum per kg (IDR). None jika tidak ada range.",
    )
    price_per_kg_max:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga maksimum per kg (IDR). None jika tidak ada range.",
    )
    price_per_kg_avg:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga rata-rata / titik tunggal per kg (IDR).",
    )
    price_per_unit_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga minimum per buah (IDR).",
    )
    price_per_unit_max: Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga maksimum per buah (IDR).",
    )
    location_hint:      Optional[str] = Field(
        default=None,
        max_length=200,
        description="Lokasi yang disebut di teks (cth: 'Jakarta', 'Medan', 'online').",
    )
    seller_type:        Optional[str] = Field(
        default=None,
        max_length=100,
        description="Jenis penjual yang teridentifikasi (cth: 'kebun', 'reseller', 'importir').",
    )
    confidence:         float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Kepercayaan LLM terhadap keakuratan entry ini (0–1).",
    )
    raw_text_snippet:   Optional[str] = Field(
        default=None,
        max_length=500,
        description="Potongan teks sumber yang menjadi dasar ekstraksi ini.",
    )

    @model_validator(mode="after")
    def at_least_one_price_present(self) -> MarketPriceEntry:
        prices = [
            self.price_per_kg_min,
            self.price_per_kg_max,
            self.price_per_kg_avg,
            self.price_per_unit_min,
            self.price_per_unit_max,
        ]
        if not any(p is not None for p in prices):
            raise ValueError(
                "Setidaknya satu field harga harus terisi "
                "(price_per_kg_min/max/avg atau price_per_unit_min/max)."
            )
        return self

    @model_validator(mode="after")
    def max_not_less_than_min(self) -> MarketPriceEntry:
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


# ---------------------------------------------------------------------------
# Layer 3: Final payload ke NestJS
# ---------------------------------------------------------------------------

class AgentRunStatus(str, Enum):
    SUCCESS          = "success"
    PARTIAL          = "partial"    # Ada target yang gagal, tapi ada data masuk
    SCRAPER_ERROR    = "scraper_error"
    LLM_ERROR        = "llm_error"
    NO_DATA          = "no_data"


class MarketReportPayload(BaseModel):
    """
    Payload lengkap yang dikirim ke NestJS endpoint `/api/market-intelligence/ingest`.
    """

    agent_version:  str             = Field(default="1.0.0")
    run_id:         str             = Field(..., description="UUID unik per run agen.")
    run_started_at: datetime        = Field(..., description="Waktu mulai run agen.")
    run_ended_at:   datetime        = Field(default_factory=lambda: datetime.now(timezone.utc))
    status:         AgentRunStatus  = Field(...)
    entries:        List[MarketPriceEntry] = Field(
        default_factory=list,
        description="Semua entry harga yang berhasil di-ekstrak.",
    )
    sources_scraped:      int = Field(default=0, description="Jumlah URL yang berhasil di-scrape.")
    sources_failed:       int = Field(default=0, description="Jumlah URL yang gagal di-scrape.")
    llm_parse_errors:     int = Field(default=0, description="Jumlah error parsing output LLM.")
    error_details:        Optional[str] = Field(default=None, description="Pesan error ringkas jika ada.")

    @property
    def entry_count(self) -> int:
        return len(self.entries)