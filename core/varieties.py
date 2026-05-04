# core/varieties.py

from typing import Dict, NamedTuple, Optional


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
    "local": VarietyInfo(
        display_name="Durian Lokal",
        local_name="Local / Durian Lokal",
        origin="Indonesia",
        description="Durian asli dari Indonesia dengan ciri khas dan rasa yang unik.",
    ),
}

_UNKNOWN_VARIETY = VarietyInfo(
    display_name="Varietas Tidak Dikenal",
    local_name="Unknown",
    origin="Tidak diketahui",
    description="Kode varietas tidak ditemukan dalam database.",
)


def get_variety_info(code: str) -> VarietyInfo:
    return VARIETY_MAP.get(code.strip().upper(), _UNKNOWN_VARIETY)


def get_display_name(code: str) -> str:
    info = VARIETY_MAP.get(code.strip().upper())
    return info.display_name if info else code