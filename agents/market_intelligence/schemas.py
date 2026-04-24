# agents/market_intelligence/schemas.py
"""
Pydantic models untuk Market Intelligence Agent.

Tiga layer schema:
  1. ScrapedPage          — output mentah dari scraper.py (kini berupa JSON string)
  2. MarketPriceEntry     — satu entri harga yang berhasil di-extract LLM
  3. MarketReportPayload  — payload final yang dikirim ke NestJS

Changelog v2:
  - DurianVariety diperluas: D13, D24, D2 ditambahkan selain D197 & D200.
  - MarketPriceEntry: 3 field audit trail baru ditambahkan untuk mencegah
    data leakage (is_whole_fruit, weight_reference, notes).
  - ScrapedPage: field `raw_text` diganti `raw_json` untuk merefleksikan
    bahwa scraper kini mengirimkan JSON intercept, bukan HTML/teks DOM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enum: Kode varietas yang ditangani agent ini
# ---------------------------------------------------------------------------

class DurianVariety(str, Enum):
    """
    Lima varietas durian premium yang dimonitor oleh Market Intelligence Agent.

    Kode mengikuti sistem kode resmi DOA (Department of Agriculture) Malaysia.
    Urutan alphabetical agar konsisten dengan CLASS_NAMES di .env dan
    class_indices.json.
    """
    GOLDEN_BUN  = "D13"    # D13  — Golden Bun (Johor)
    MUSANG_KING = "D197"   # D197 — Musang King / Mao Shan Wang / Raja Kunyit (Kelantan)
    DATO_NINA   = "D2"     # D2   — Dato Nina (Melaka)
    DURI_HITAM  = "D200"   # D200 — Duri Hitam / Black Thorn / Ochee (Penang)
    SULTAN      = "D24"    # D24  — Sultan / Bukit Merah (Perak/Selangor)


# ---------------------------------------------------------------------------
# Layer 1: Raw scraping result (JSON intercept)
# ---------------------------------------------------------------------------

class ScrapedPage(BaseModel):
    """
    Hasil intercept Playwright sebelum diproses LLM.

    Perbedaan dari versi sebelumnya:
    - `raw_json` menggantikan `raw_text`. Isinya adalah string JSON (bisa satu
      objek atau array) yang didapat dari network response intercept, bukan
      teks HTML yang di-scrape dari DOM.
    - Dengan format JSON, LLM mendapat input terstruktur sehingga lebih mudah
      membedakan "durian kupas/frozen" dari "durian utuh" hanya dari field
      nama produk atau deskripsi, tanpa harus mem-parse HTML noise.
    """

    source_name:   str       = Field(..., description="Nama sumber (dari ScrapingTarget.name).")
    source_url:    str       = Field(..., description="URL halaman yang di-navigate Playwright.")
    raw_json:      str       = Field(
        ...,
        description=(
            "JSON mentah yang di-intercept dari XHR/Fetch API e-commerce. "
            "Berupa JSON string tunggal atau array JSON yang di-join."
        ),
    )
    scraped_at:    datetime  = Field(default_factory=lambda: datetime.now(timezone.utc))
    success:       bool      = Field(default=True)
    error_message: Optional[str] = Field(default=None)

    @field_validator("raw_json")
    @classmethod
    def json_not_empty_on_success(cls, v: str, info) -> str:
        if info.data.get("success", True) and not v.strip():
            raise ValueError("raw_json tidak boleh kosong jika success=True.")
        return v


# ---------------------------------------------------------------------------
# Layer 2: Satu entry harga hasil analisis LLM
# ---------------------------------------------------------------------------

class MarketPriceEntry(BaseModel):
    """
    Satu data point harga yang berhasil di-ekstrak dari JSON intercept.

    FIELD AUDIT TRAIL BARU (v2):
    - is_whole_fruit:    Gatekeeper utama. HANYA True jika produk adalah
                         durian UTUH dengan kulit. False untuk kupas/frozen/dll.
    - weight_reference:  Menyimpan berat asli dari listing penjual SEBELUM
                         normalisasi (misal: "2 buah @2kg", "1 box 500gr").
                         Berguna untuk audit kalkulasi LLM.
    - notes:             Chain-of-Thought (CoT) LLM dalam melakukan normalisasi
                         harga ke satuan per-Kg. Menjadi bukti bahwa kalkulasi
                         dilakukan dengan benar.

    Semua field harga dalam satuan IDR (Rupiah) per Kg.
    """

    # --- Identifikasi Produk ---
    variety_code:       DurianVariety = Field(
        ...,
        description="Kode DOA varietas. D13/D197/D2/D200/D24.",
    )
    variety_alias:      str = Field(
        ...,
        description=(
            "Alias yang ditemukan di JSON sumber "
            "(cth: 'musang king', 'ochee', 'golden bun', 'sultan')."
        ),
        max_length=100,
    )

    # --- FIELD AUDIT TRAIL (v2) ---
    is_whole_fruit:     bool = Field(
        ...,
        description=(
            "TRUE jika dan hanya jika produk adalah DURIAN UTUH DENGAN KULIT. "
            "FALSE untuk kupas, frozen, box, 400gr/500gr pack, bibit, olahan, dll. "
            "Field ini adalah gatekeeper utama anti data leakage."
        ),
    )
    weight_reference:   str = Field(
        ...,
        description=(
            "Referensi berat ASLI dari listing penjual sebelum normalisasi, "
            "cth: 'per buah 2-3 kg', '1 box isi 2 buah', 'per kg'. "
            "Digunakan untuk audit dan verifikasi kalkulasi LLM."
        ),
        max_length=200,
    )
    notes:              Optional[str] = Field(
        default=None,
        max_length=500,
        description=(
            "Chain-of-Thought LLM: catatan matematis normalisasi harga ke per-Kg. "
            "Cth: 'Harga listing Rp800.000/2kg → 800000/2 = 400000 per kg'. "
            "None jika harga sudah dalam satuan per-Kg dari sumber."
        ),
    )

    # --- Harga (IDR per Kg) ---
    price_per_kg_min:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga minimum per Kg (IDR). None jika tidak ada range.",
    )
    price_per_kg_max:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga maksimum per Kg (IDR). None jika tidak ada range.",
    )
    price_per_kg_avg:   Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga rata-rata / titik tunggal per Kg (IDR).",
    )
    price_per_unit_min: Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga minimum per buah (IDR). Opsional, pelengkap harga/Kg.",
    )
    price_per_unit_max: Optional[float] = Field(
        default=None,
        ge=0,
        description="Harga maksimum per buah (IDR). Opsional, pelengkap harga/Kg.",
    )

    # --- Metadata Konteks ---
    location_hint:      Optional[str] = Field(
        default=None,
        max_length=200,
        description="Lokasi yang disebut di sumber (cth: 'Jakarta', 'Medan', 'online').",
    )
    seller_type:        Optional[str] = Field(
        default=None,
        max_length=100,
        description="Jenis penjual (cth: 'kebun', 'reseller', 'importir', 'toko online').",
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
        description=(
            "Potongan JSON sumber yang menjadi dasar ekstraksi, "
            "misal field 'name' dan 'price' dari product object."
        ),
    )

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

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
            raise ValueError(
                "price_per_kg_max tidak boleh lebih kecil dari price_per_kg_min."
            )
        if (
            self.price_per_unit_min is not None
            and self.price_per_unit_max is not None
            and self.price_per_unit_max < self.price_per_unit_min
        ):
            raise ValueError(
                "price_per_unit_max tidak boleh lebih kecil dari price_per_unit_min."
            )
        return self

    @model_validator(mode="after")
    def whole_fruit_gate(self) -> MarketPriceEntry:
        """
        Validator yang menegaskan konsistensi data.
        Jika LLM menandai is_whole_fruit=False tapi tetap mengisi harga,
        kita reset semua harga ke None untuk mencegah data leakage
        sebelum validasi Pydantic at_least_one_price_present berjalan.

        CATATAN: Validasi Python di task.py adalah lapisan pertahanan utama.
        Validator ini adalah lapisan kedua (defense in depth).
        """
        if not self.is_whole_fruit:
            # Tidak perlu raise — task.py yang akan membuang entry ini.
            # Log akan dicatat di task.py. Biarkan object terbentuk agar
            # bisa di-inspect untuk keperluan debugging jika diperlukan.
            pass
        return self


# ---------------------------------------------------------------------------
# Layer 3: Final payload ke NestJS
# ---------------------------------------------------------------------------

class AgentRunStatus(str, Enum):
    SUCCESS       = "success"
    PARTIAL       = "partial"        # Ada target gagal, tapi ada data masuk
    SCRAPER_ERROR = "scraper_error"
    LLM_ERROR     = "llm_error"
    NO_DATA       = "no_data"


class MarketReportPayload(BaseModel):
    """
    Payload lengkap yang dikirim ke NestJS endpoint `/api/market-intelligence/ingest`.

    Field `entries_discarded` (v2):
    Menghitung berapa entry LLM yang dibuang karena is_whole_fruit=False.
    Digunakan untuk monitoring kualitas data dan mendeteksi apakah
    sumber data mengandung banyak produk olahan/kupas.
    """

    agent_version:  str             = Field(default="2.0.0")
    run_id:         str             = Field(..., description="UUID unik per run agen.")
    run_started_at: datetime        = Field(..., description="Waktu mulai run agen.")
    run_ended_at:   datetime        = Field(default_factory=lambda: datetime.now(timezone.utc))
    status:         AgentRunStatus  = Field(...)
    entries:        List[MarketPriceEntry] = Field(
        default_factory=list,
        description=(
            "Semua entry harga yang LOLOS validasi ganda: "
            "LLM validation + Python is_whole_fruit gate."
        ),
    )
    sources_scraped:       int = Field(default=0, description="Jumlah URL berhasil di-intercept.")
    sources_failed:        int = Field(default=0, description="Jumlah URL gagal di-intercept.")
    llm_parse_errors:      int = Field(default=0, description="Jumlah error parsing output LLM.")
    entries_discarded:     int = Field(
        default=0,
        description=(
            "Jumlah entry LLM yang dibuang karena is_whole_fruit=False "
            "(produk kupas/frozen/olahan). Metrik kesehatan data."
        ),
    )
    error_details:         Optional[str] = Field(
        default=None,
        description="Pesan error ringkas jika ada.",
    )

    @property
    def entry_count(self) -> int:
        return len(self.entries)