from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    variety_code:     str   = Field(..., description="Kode resmi DOA Malaysia.", json_schema_extra={"example": "D200"})
    variety_name:     str   = Field(..., description="Nama populer varietas.", json_schema_extra={"example": "Musang King"})
    local_name:       str   = Field(..., description="Nama lokal / alias.", json_schema_extra={"example": "D200 / Musang King / Raja Kunyit"})
    origin:           str   = Field(..., description="Asal daerah.", json_schema_extra={"example": "Malaysia (Kelantan / Gua Musang)"})
    description:      str   = Field(..., description="Deskripsi rasa dan karakteristik.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Skor kepercayaan 0–1.", json_schema_extra={"example": 0.9231})


class VarietyScore(BaseModel):
    variety_code:     str   = Field(..., json_schema_extra={"example": "D200"})
    variety_name:     str   = Field(..., json_schema_extra={"example": "Musang King"})
    confidence_score: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.9231})


class PredictionResponse(BaseModel):
    success:               bool                  = Field(default=True)
    prediction:            PredictionResult       = Field(..., description="Prediksi varietas teratas.")
    all_varieties:         List[VarietyScore]     = Field(..., description="Semua varietas diurutkan descending.")
    confidence_scores:     Dict[str, float]       = Field(..., description="Skor per nama varietas.")
    image_enhanced:        bool                   = Field(default=False, description="Apakah enhancement diterapkan.")
    inference_time_ms:     float                  = Field(..., ge=0.0, description="Waktu inferensi ONNX (ms).")
    preprocessing_time_ms: float                  = Field(default=0.0, ge=0.0, description="Waktu preprocessing (ms).")
    model_version:         Optional[str]          = Field(default=None, description="Versi model.")
    request_id:            Optional[str]          = Field(default=None, description="ID request untuk tracing.")


class HealthResponse(BaseModel):
    status:       str  = Field(..., description="'healthy' atau 'degraded'.", json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., description="Model ter-load dan siap.")
    app_name:     str  = Field(...)
    version:      str  = Field(...)
    uptime_seconds:     Optional[int]              = Field(default=None, description="Uptime service dalam detik.")
    memory_usage_mb:    Optional[float]            = Field(default=None, description="Penggunaan RAM dalam MB.")
    cpu_percent:        Optional[float]            = Field(default=None, description="CPU usage %.")
    rate_limiter_stats: Optional[Dict[str, Any]]   = Field(default=None, description="Statistik rate limiter.")
    config_summary:     Optional[Dict[str, Any]]   = Field(default=None, description="Ringkasan konfigurasi aktif.")


class ErrorResponse(BaseModel):
    success:    bool         = Field(default=False)
    error:      str          = Field(...)
    detail:     str          = Field(...)
    request_id: Optional[str] = Field(default=None)