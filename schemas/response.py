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


class VarietyPriceSummary(BaseModel):
    variety_code:  str      = Field(..., examples=["D197"])
    variety_name:  str      = Field(..., examples=["Musang King"])
    price_min_idr: int      = Field(..., ge=0, description="IDR per kg")
    price_max_idr: int      = Field(..., ge=0, description="IDR per kg")
    price_avg_idr: int      = Field(..., ge=0, description="IDR per kg")
    sample_count:  int      = Field(..., ge=0, description="Jumlah listing sumber")
    scraped_at:    datetime = Field(...)


class ErrorResponse(BaseModel):
    success:    bool          = Field(default=False)
    error:      str           = Field(...)
    detail:     str           = Field(...)
    request_id: Optional[str] = Field(default=None)