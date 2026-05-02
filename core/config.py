from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VarietyInfo(NamedTuple):
    display_name: str
    local_name:   str
    origin:       str
    description:  str


VARIETY_MAP: Dict[str, VarietyInfo] = {
    "D2": VarietyInfo(
        display_name="Dato Nina",
        local_name="D2 / Dato Nina",
        origin="Malaysia (Melaka)",
        description=(
            "Durian dengan bentuk buah agak bulat. Daging buahnya berwarna tembaga "
            "atau kuning kecokelatan dengan kombinasi rasa manis dan sedikit pahit."
        ),
    ),
    "D13": VarietyInfo(
        display_name="Golden Bun",
        local_name="D13 / Golden Bun",
        origin="Malaysia (Johor)",
        description=(
            "Memiliki daging berwarna oranye kemerahan yang pekat. Rasanya manis, "
            "sangat wangi, dan bijinya besar. Ciri luarnya cenderung membulat dengan duri tebal."
        ),
    ),
    "D24": VarietyInfo(
        display_name="Sultan",
        local_name="D24 / Sultan / Bukit Merah",
        origin="Malaysia (Perak / Selangor)",
        description=(
            "Varietas legendaris dengan daging kuning pucat hingga krem. Rasa pahit-manis "
            "yang kaya. Ciri fisik luarnya memiliki duri yang cukup tajam dan rapat "
            "dengan bentuk cenderung oval."
        ),
    ),
    "D197": VarietyInfo(
        display_name="Musang King",
        local_name="D197 / Musang King / Raja Kunyit / Mao Shan Wang",
        origin="Malaysia (Kelantan)",
        description=(
            "Raja durian Malaysia dengan daging kuning-emas tebal. Rasa kaya manis-pahit "
            "yang kompleks. Ciri khas luarnya memiliki pola bintang (star-shape) botak "
            "di bagian bawah dan duri berbentuk piramida."
        ),
    ),
    "D200": VarietyInfo(
        display_name="Black Thorn",
        local_name="D200 / Ochee / Duri Hitam / Black Thorn",
        origin="Malaysia (Penang)",
        description=(
            "Durian super premium dengan daging oranye kemerahan dan rasa manis-pahit "
            "yang sangat pekat. Ciri khas luarnya bentuknya membulat dengan garis lekukan "
            "di bagian bawah dan ujung duri berwarna kehitaman."
        ),
    ),
}

_UNKNOWN_VARIETY = VarietyInfo(
    display_name="Varietas Tidak Dikenal",
    local_name="Unknown",
    origin="Tidak diketahui",
    description="Kode varietas tidak ditemukan dalam database.",
)


def get_variety_info(code: str) -> VarietyInfo:
    return VARIETY_MAP.get(code.strip(), _UNKNOWN_VARIETY)


def get_display_name(code: str) -> str:
    info = VARIETY_MAP.get(code.strip())
    return info.display_name if info else code


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = True,
        extra             = "ignore",
    )

    APP_NAME:    str  = "Durian Classification API"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False
    LOG_LEVEL:   str  = "INFO"

    # [FIX BUG-09] MODEL_PATH default diperbarui ke model v10 (EfficientNetV2-S).
    # Default ini sebagai fallback jika .env tidak menyetel MODEL_PATH.
    # Nilai aktual selalu dibaca dari .env — pastikan .env sudah diperbarui.
    MODEL_PATH: str = "models/weights/efficientnet_b0.onnx"

    # CLASS_NAMES: urutan WAJIB alphabetical sesuai folder training (indeks 0–4).
    # D101 sudah dihapus — tidak ada dalam training data model v10.
    CLASS_NAMES: str = "D13,D197,D2,D200,D24"

    # [FIX BUG-08] IMAGE_SIZE default diubah 640 → 480.
    # Model v10 (EfficientNetV2-S) menggunakan input tensor 480×480.
    # Default 640 sebelumnya tidak sinkron dengan .env.example dan akan
    # menyebabkan ONNX shape mismatch error jika .env tidak diset eksplisit.
    IMAGE_SIZE: int = 480

    ENABLE_ENHANCEMENT:   bool  = True
    ENABLE_CLAHE:         bool  = True
    ENABLE_SHARPENING:    bool  = True
    ENABLE_WHITE_BALANCE: bool  = True
    CLAHE_CLIP_LIMIT:     float = 2.0

    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,webp"
    MAX_FILE_SIZE_MB:   int = 10

    CORS_ORIGINS_STR:  str = "http://localhost:3000,http://localhost:8080"
    ALLOWED_HOSTS_STR: str = "*"

    API_KEY_REQUIRED: bool = True

    # --- Validators ---

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL '{v}' tidak valid. Pilih: {valid}")
        return upper

    @field_validator("IMAGE_SIZE")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        # [FIX BUG-08] Validator tetap 32–1024 untuk fleksibilitas.
        # Nilai yang valid untuk production: 224 (model v0), 480 (model v10).
        if not 32 <= v <= 1024:
            raise ValueError(
                f"IMAGE_SIZE={v} tidak valid. Harus antara 32–1024. "
                "Untuk model v10 gunakan IMAGE_SIZE=480."
            )
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE_MB harus > 0.")
        return v

    # --- Computed properties ---

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
        return {e.strip().lower() for e in self.ALLOWED_EXTENSIONS.split(",")}

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


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


settings: Settings = get_settings()