# core/file_validator.py
# Single source of truth untuk validasi file upload.
# routes.py WAJIB import dari sini — jangan mendefinisikan ulang MAGIC_BYTES di tempat lain.

from core.config import settings
from core.exceptions import (
    FileTooLargeException,
    InvalidImageException,
    UnsupportedFileTypeException,
)
from core.logger import get_logger
from core.middleware import AuditLogger

logger = get_logger(__name__)

# Header magic bytes per ekstensi.
# WebP butuh double-check: RIFF...WEBP di byte 0-4 dan 8-12.
MAGIC_BYTES: dict[str, list[bytes]] = {
    "jpg":  [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],
}


def check_magic_bytes(data: bytes, ext: str) -> bool:
    """
    Validasi magic bytes file. Kembalikan True jika cocok (atau ekstensi tidak dikenal).
    WebP divalidasi dua tahap: header RIFF + penanda WEBP di offset 8.
    """
    if ext not in MAGIC_BYTES:
        return True

    header = data[:12]
    for magic in MAGIC_BYTES[ext]:
        if not header.startswith(magic):
            continue
        if ext == "webp":
            return (
                len(data) >= 12
                and data[0:4] == b"RIFF"
                and data[8:12] == b"WEBP"
            )
        return True
    return False


def validate_upload(
    data:       bytes,
    filename:   str,
    request_id: str,
    client_ip:  str,
) -> str:
    """
    Validasi penuh file upload: ukuran, ekstensi, dan magic bytes.

    Returns:
        str: Ekstensi file yang valid (misal 'jpg', 'png').

    Raises:
        InvalidImageException: File kosong atau tidak bisa dibaca.
        FileTooLargeException: Ukuran melebihi MAX_FILE_SIZE_MB.
        UnsupportedFileTypeException: Ekstensi atau magic bytes tidak valid.
    """
    if len(data) == 0:
        raise InvalidImageException(detail="File kosong — tidak ada data gambar.")

    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeException(
            detail=(
                f"File terlalu besar ({len(data) / (1024 * 1024):.1f}MB). "
                f"Maksimum {settings.MAX_FILE_SIZE_MB}MB."
            )
        )

    # Sanitasi nama file — hanya izinkan karakter aman
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    ext = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""

    if not ext:
        raise UnsupportedFileTypeException(
            detail=(
                "File tidak memiliki ekstensi. "
                f"Format yang didukung: {settings.ALLOWED_EXTENSIONS}"
            )
        )

    if ext not in settings.allowed_extensions_set:
        raise UnsupportedFileTypeException(
            detail=(
                f"Format '.{ext}' tidak didukung. "
                f"Format yang diizinkan: {settings.ALLOWED_EXTENSIONS}"
            )
        )

    if not check_magic_bytes(data, ext):
        AuditLogger.suspicious_file(request_id, "Magic bytes mismatch", filename, client_ip)
        raise UnsupportedFileTypeException(
            detail=(
                f"Konten file tidak cocok dengan ekstensi '.{ext}'. "
                "Pastikan file tidak diubah atau dimanipulasi."
            )
        )

    return ext