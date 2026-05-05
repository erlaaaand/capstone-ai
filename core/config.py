# core/config.py

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Re-export dari varieties.py agar kode lama yang import dari sini tetap berjalan ──────
from core.varieties import (  # noqa: F401  (re-export intentional)
    VARIETY_MAP,
    VarietyInfo,
    get_display_name,
    get_variety_info,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = True,
        extra             = "ignore",
    )

    # ── Application ───────────────────────────────────────────────────────
    APP_NAME:    str  = "Durian Classification API"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False
    LOG_LEVEL:   str  = "INFO"

    # ── Model ─────────────────────────────────────────────────────────────
    MODEL_PATH:  str = "models/weights/efficientnet_b0.onnx"
    CLASS_NAMES: str = "D13,D197,D2,D200,D24"
    IMAGE_SIZE:  int = 224

    # ── Image Processing ──────────────────────────────────────────────────
    ENABLE_ENHANCEMENT:   bool  = True
    ENABLE_CLAHE:         bool  = True
    ENABLE_SHARPENING:    bool  = True
    ENABLE_WHITE_BALANCE: bool  = True
    CLAHE_CLIP_LIMIT:     float = 2.0

    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,webp"
    MAX_FILE_SIZE_MB:   int = 10

    # ── CLIP / Zero-Shot Validation ───────────────────────────────────────
    CLIP_MODEL_ID:             str   = "openai/clip-vit-base-patch32"
    CLIP_REVISION_HASH:        str   = ""
    CLIP_NON_DURIAN_THRESHOLD: float = 0.40

    # ── Security ──────────────────────────────────────────────────────────
    PBKDF2_ITERATIONS: int = 600_000

    # ── Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    BURST_LIMIT_PER_SECOND:    int = 20

    # ── OpenAPI Contact ───────────────────────────────────────────────────
    API_SUPPORT_EMAIL: str = "api-support@yourdomain.com"
    API_SUPPORT_NAME:  str = "Durian API Support"

    # ── Network ───────────────────────────────────────────────────────────
    CORS_ORIGINS_STR:  str = "http://localhost:3000,http://localhost:8080"
    ALLOWED_HOSTS_STR: str = "*"

    API_KEY_REQUIRED: bool = True

    # ── Validators ────────────────────────────────────────────────────────

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL '{v}' tidak valid. Pilih: {sorted(valid)}")
        return upper

    @field_validator("IMAGE_SIZE")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        if not 32 <= v <= 1024:
            raise ValueError(f"IMAGE_SIZE={v} tidak valid. Harus antara 32–1024.")
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE_MB harus > 0.")
        if v > 100:
            raise ValueError("MAX_FILE_SIZE_MB tidak boleh lebih dari 100MB.")
        return v

    @field_validator("PBKDF2_ITERATIONS")
    @classmethod
    def validate_pbkdf2_iterations(cls, v: int) -> int:
        if v < 260_000:
            raise ValueError(
                f"PBKDF2_ITERATIONS={v} terlalu rendah. "
                "Minimum 260.000 (rekomendasi OWASP 2023: 600.000)."
            )
        return v

    @field_validator("CLIP_NON_DURIAN_THRESHOLD")
    @classmethod
    def validate_clip_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(
                "CLIP_NON_DURIAN_THRESHOLD harus antara 0.0 dan 1.0 (exclusive)."
            )
        return v

    @field_validator("CLAHE_CLIP_LIMIT")
    @classmethod
    def validate_clahe_clip_limit(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("CLAHE_CLIP_LIMIT harus > 0.")
        return v

    @field_validator("RATE_LIMIT_WINDOW_SECONDS")
    @classmethod
    def validate_rate_limit_window(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS harus > 0.")
        return v

    @field_validator("BURST_LIMIT_PER_SECOND")
    @classmethod
    def validate_burst_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("BURST_LIMIT_PER_SECOND harus > 0.")
        return v

    # ── Computed Properties ───────────────────────────────────────────────

    @property
    def class_names_list(self) -> List[str]:
        return [n.strip() for n in self.CLASS_NAMES.split(",") if n.strip()]

    @property
    def num_classes(self) -> int:
        return len(self.class_names_list)

    @property
    def image_size_tuple(self) -> Tuple[int, int]:
        return (self.IMAGE_SIZE, self.IMAGE_SIZE)

    @property
    def allowed_extensions_set(self) -> set:
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",") if e.strip()}

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def model_abs_path(self) -> Path:
        p = Path(self.MODEL_PATH)
        return p if p.is_absolute() else Path.cwd() / p

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS_STR.split(",") if o.strip()]

    @property
    def ALLOWED_HOSTS(self) -> List[str]:
        return [h.strip() for h in self.ALLOWED_HOSTS_STR.split(",") if h.strip()]

    @property
    def variety_map(self) -> Dict[str, VarietyInfo]:
        return VARIETY_MAP

    @property
    def clip_revision(self) -> Optional[str]:
        rev = self.CLIP_REVISION_HASH.strip()
        return rev if rev else None


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


settings: Settings = get_settings()