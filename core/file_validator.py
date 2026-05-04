# services/file_validator.py

from core.config import settings
from core.exceptions import (
    FileTooLargeException,
    InvalidImageException,
    UnsupportedFileTypeException,
)
from core.logger import get_logger
from core.middleware import AuditLogger

logger = get_logger(__name__)

_MAGIC_BYTES: dict[str, list[bytes]] = {
    "jpg":  [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],
}


def _check_magic_bytes(data: bytes, ext: str) -> bool:
    if ext not in _MAGIC_BYTES:
        return True

    header = data[:12]
    for magic in _MAGIC_BYTES[ext]:
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

    if len(data) == 0:
        raise InvalidImageException(detail="File kosong — tidak ada data gambar.")

    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeException(
            detail=(
                f"File terlalu besar ({len(data) / (1024 * 1024):.1f}MB). "
                f"Maksimum {settings.MAX_FILE_SIZE_MB}MB."
            )
        )

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

    if not _check_magic_bytes(data, ext):
        AuditLogger.suspicious_file(request_id, "Magic bytes mismatch", filename, client_ip)
        raise UnsupportedFileTypeException(
            detail=(
                f"Konten file tidak cocok dengan ekstensi '.{ext}'. "
                "Pastikan file tidak diubah atau dimanipulasi."
            )
        )

    return ext