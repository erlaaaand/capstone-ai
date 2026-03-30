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
        display_name="Chanee",
        local_name="D2 / Chanee",
        origin="Thailand / Malaysia Utara",
        description="Durian berukuran besar dengan daging kuning pucat. Rasa manis ringan dengan sedikit pahit, tekstur creamy.",
    ),
    "D13": VarietyInfo(
        display_name="Kuk San",
        local_name="D13 / Kuk San",
        origin="Malaysia Barat",
        description="Daging berwarna kuning dengan rasa manis-pahit yang seimbang. Biji relatif kecil sehingga proporsi daging lebih banyak.",
    ),
    "D197": VarietyInfo(
        display_name="Golden Phoenix",
        local_name="D197 / Jin Feng / Golden Phoenix",
        origin="Malaysia",
        description="Varietas premium dengan daging kuning cerah hampir keemasan. Rasa manis-pahit yang intens dan tekstur sangat lembut.",
    ),
    "D198": VarietyInfo(
        display_name="Red Prawn",
        local_name="D198 / Udang Merah / Red Prawn",
        origin="Malaysia (Balik Pulau, Penang)",
        description="Ciri khas daging berwarna merah-orange yang unik. Rasa manis kuat dengan tekstur lembut seperti krim.",
    ),
    "D200": VarietyInfo(
        display_name="Musang King",
        local_name="D200 / Musang King / Raja Kunyit / Mao Shan Wang",
        origin="Malaysia (Kelantan / Gua Musang)",
        description="Raja durian Malaysia dengan daging kuning-emas yang tebal. Rasa kaya manis-pahit yang kompleks, tekstur creamy-padat.",
    ),
    "D24": VarietyInfo(
        display_name="Sultan",
        local_name="D24 / Sultan",
        origin="Malaysia",
        description="Varietas legendaris dengan daging kuning-cream yang lebat. Rasa pahit-manis yang kaya dan kompleks, aroma sangat kuat.",
    ),
    "D88": VarietyInfo(
        display_name="Tekka",
        local_name="D88 / Tekka",
        origin="Malaysia (Johor)",
        description="Durian dengan daging kuning dan rasa manis yang dominan. Tekstur lembut dengan sedikit hint pahit di akhir.",
    ),
    "D101": VarietyInfo(
        display_name="Nyuk Kun",
        local_name="D101 / Nyuk Kun",
        origin="Malaysia (Penang)",
        description="Varietas Penang dengan daging kuning-orange tebal. Rasa sangat manis dengan tekstur lembut dan creamy. Biji kecil.",
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

    LOG_LEVEL: str = "INFO"

    MODEL_PATH:  str = "models/weights/efficientnet_b0.onnx"
    CLASS_NAMES: str = "D101,D13,D197,D198,D2,D200,D24,D88"

    IMAGE_SIZE: int = 224

    ENABLE_ENHANCEMENT:   bool  = True
    ENABLE_CLAHE:         bool  = True
    ENABLE_SHARPENING:    bool  = True
    ENABLE_WHITE_BALANCE: bool  = True
    CLAHE_CLIP_LIMIT:     float = 2.0

    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,webp"
    MAX_FILE_SIZE_MB:   int = 10

    CORS_ORIGINS_STR: str = "http://localhost:3000,http://localhost:8080"

    ALLOWED_HOSTS_STR: str = "*"

    API_KEY_REQUIRED: bool = True

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL '{v}' tidak valid.")
        return upper

    @field_validator("IMAGE_SIZE")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        if not 32 <= v <= 1024:
            raise ValueError(f"IMAGE_SIZE harus 32–1024.")
        return v

    @field_validator("MAX_FILE_SIZE_MB")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_FILE_SIZE_MB harus positif.")
        return v

    @property
    def class_names_list(self) -> List[str]:
        return [n.strip() for n in self.CLASS_NAMES.split(",")]

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


settings: Settings = get_settings()