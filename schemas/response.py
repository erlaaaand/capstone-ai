# schemas/response.py

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    variety_code:     str   = Field(..., description="Kode resmi DOA Malaysia.", examples=["D200"])
    variety_name:     str   = Field(..., description="Nama populer varietas.", examples=["Black Thorn"])
    local_name:       str   = Field(..., description="Nama lokal / semua alias.", examples=["D200 / Ochee / Duri Hitam"])
    origin:           str   = Field(..., description="Asal daerah.", examples=["Malaysia (Penang)"])
    description:      str   = Field(..., description="Deskripsi rasa dan karakteristik.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Skor kepercayaan 0–1.", examples=[0.9231])


class VarietyScore(BaseModel):
    variety_code:     str   = Field(..., examples=["D200"])
    confidence_score: float = Field(..., ge=0.0, le=1.0, examples=[0.9231])


class MarketContextResponse(BaseModel):
    variety_code:  str      = Field(..., description="Kode varietas yang sama dengan prediksi.")
    price_min_idr: int      = Field(..., ge=0, description="Harga minimum per kg (IDR), dari data pasar terbaru.")
    price_max_idr: int      = Field(..., ge=0, description="Harga maksimum per kg (IDR).")
    price_avg_idr: int      = Field(..., ge=0, description="Harga rata-rata per kg (IDR).")
    sample_count:  int      = Field(..., ge=0, description="Jumlah listing yang menjadi basis harga.")
    scraped_at:    datetime = Field(..., description="Waktu data harga diambil.")
    data_is_stale: bool     = Field(
        default=False,
        description="True jika data lebih dari 25 jam. Gunakan sebagai sinyal di UI.",
    )


class PredictionResponse(BaseModel):
    success:    bool             = Field(default=True)
    prediction: PredictionResult = Field(..., description="Prediksi varietas teratas.")

    all_varieties: List[VarietyScore] = Field(
        ...,
        description="Semua varietas diurutkan descending berdasarkan confidence.",
    )
    confidence_scores: Dict[str, float] = Field(
        ...,
        description="Map variety_code → confidence_score untuk semua varietas.",
    )

    image_enhanced:        bool  = Field(default=False)
    inference_time_ms:     float = Field(..., ge=0.0)
    preprocessing_time_ms: float = Field(default=0.0, ge=0.0)
    model_version:         Optional[str] = Field(default=None)
    request_id:            Optional[str] = Field(default=None)

    market_context: Optional[MarketContextResponse] = Field(
        default=None,
        description=(
            "Ringkasan harga pasar terkini untuk varietas yang terdeteksi. "
            "None jika belum ada data dari Market Intelligence Agent."
        ),
    )


class HealthResponse(BaseModel):
    status:       str  = Field(..., examples=["healthy"])
    model_loaded: bool = Field(...)
    app_name:     str  = Field(...)
    version:      str  = Field(...)
    uptime_seconds:     Optional[int]            = Field(default=None)
    memory_usage_mb:    Optional[float]          = Field(default=None)
    cpu_percent:        Optional[float]          = Field(default=None)
    rate_limiter_stats: Optional[Dict[str, Any]] = Field(default=None)
    config_summary:     Optional[Dict[str, Any]] = Field(default=None)

    market_data_available: bool     = Field(
        default=False,
        description="True jika MarketDataStore memiliki data dari minimal satu run agent.",
    )
    market_data_stale:     bool     = Field(
        default=True,
        description="True jika data market lebih dari 25 jam atau belum ada.",
    )
    market_next_run:       Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp jadwal run agent berikutnya, atau None jika tidak aktif.",
    )


class VarietyPriceSummary(BaseModel):
    variety_code:  str      = Field(..., examples=["D197"])
    variety_name:  str      = Field(..., examples=["Musang King"])
    price_min_idr: int      = Field(..., ge=0, description="IDR per kg")
    price_max_idr: int      = Field(..., ge=0, description="IDR per kg")
    price_avg_idr: int      = Field(..., ge=0, description="IDR per kg")
    sample_count:  int      = Field(..., ge=0, description="Jumlah listing sumber")
    scraped_at:    datetime = Field(...)


class MarketPricesResponse(BaseModel):
    success:    bool                      = Field(default=True)
    data_fresh: bool                      = Field(
        ...,
        description="False jika data lebih dari 25 jam — tampilkan warning di UI.",
    )
    scraped_at:    Optional[datetime]         = Field(default=None, description="Waktu data terakhir diambil.")
    prices:        List[VarietyPriceSummary]  = Field(default_factory=list)
    variety_count: int                        = Field(default=0, ge=0, description="Jumlah varietas dengan data harga.")


class MarketReportResponse(BaseModel):
    success:           bool                     = Field(default=True)
    run_id:            Optional[str]            = Field(default=None)
    agent_version:     Optional[str]            = Field(default=None)
    status:            Optional[str]            = Field(default=None, description="AgentRunStatus value.")
    run_started_at:    Optional[datetime]       = Field(default=None)
    run_ended_at:      Optional[datetime]       = Field(default=None)
    sources_scraped:   int                      = Field(default=0, ge=0)
    sources_failed:    int                      = Field(default=0, ge=0)
    entry_count:       int                      = Field(default=0, ge=0)
    entries_discarded: int                      = Field(default=0, ge=0)
    llm_parse_errors:  int                      = Field(default=0, ge=0)
    data_fresh:        bool                     = Field(default=False)
    error_details:     Optional[str]            = Field(default=None)
    entries: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Daftar lengkap MarketPriceEntry. Hanya ada jika ?include_entries=true.",
    )


class ErrorResponse(BaseModel):
    success:    bool          = Field(default=False)
    error:      str           = Field(...)
    detail:     str           = Field(...)
    request_id: Optional[str] = Field(default=None)