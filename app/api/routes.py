# app/api/routes.py

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.core_dependencies import AuthResult, require_scope, verify_api_key
from core.config import settings
from core.exceptions import (
    DurianServiceException,
    FileTooLargeException,
    InvalidImageException,
    UnsupportedFileTypeException,
)
from core.logger import get_logger
from core.middleware import AuditLogger
from core.security import KeyScope
from schemas.request import PredictionRequestBase64
from schemas.response import MarketContextResponse, PredictionResponse, VarietyScore
from services.image_processor import ImageProcessor
from services.inference_service import InferenceService
from services.clip_service import CLIPService

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])

# Magic bytes untuk validasi header file
MAGIC_BYTES: dict = {
    "jpg":  [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
    "png":  [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],
}


def _check_magic_bytes(data: bytes, ext: str) -> bool:
    """Validasi header file berdasarkan extension."""
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


async def _read_file_async(file: UploadFile) -> bytes:
    try:
        return await asyncio.to_thread(file.file.read)
    except Exception as e:
        raise InvalidImageException(detail="Tidak dapat membaca file yang diunggah.") from e


def _validate_file(data: bytes, filename: str, request_id: str, client_ip: str) -> str:
    """Validasi file: ukuran, tipe, dan magic bytes. Kembalikan extension."""
    if len(data) == 0:
        raise InvalidImageException(detail="File kosong — tidak ada data gambar.")

    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeException(
            detail=f"File terlalu besar ({len(data) / (1024*1024):.1f}MB). "
                   f"Maksimum {settings.MAX_FILE_SIZE_MB}MB."
        )

    # Sanitasi nama file
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    ext = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""

    if not ext:
        raise UnsupportedFileTypeException(
            detail="File tidak memiliki ekstensi. "
                   f"Format yang didukung: {settings.ALLOWED_EXTENSIONS}"
        )

    if ext not in settings.allowed_extensions_set:
        raise UnsupportedFileTypeException(
            detail=f"Format '.{ext}' tidak didukung. "
                   f"Format yang diizinkan: {settings.ALLOWED_EXTENSIONS}"
        )

    if not _check_magic_bytes(data, ext):
        AuditLogger.suspicious_file(request_id, "Magic bytes mismatch", filename, client_ip)
        raise UnsupportedFileTypeException(
            detail=f"Konten file tidak cocok dengan ekstensi '.{ext}'. "
                   "Pastikan file tidak diubah atau dimanipulasi."
        )

    return ext



@router.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Klasifikasi Varietas Durian",
    description    = (
        "Klasifikasi varietas durian dari gambar menggunakan EfficientNetB0.\n\n"
        "**Input:** File upload (`multipart/form-data`) atau Base64 JSON payload.\n\n"
        "**Output:** Varietas terdeteksi, confidence semua kelas, metadata enhancement, "
        "dan ringkasan harga pasar terkini (jika data tersedia).\n\n"
        "**Memerlukan API key scope `predict`.**"
    ),
)
async def predict_durian(
    request:  Request,
    response: Response,
    file:     Optional[UploadFile] = File(
        default     = None,
        description = "File gambar (JPG/PNG/WebP). Maks "
                      f"{settings.MAX_FILE_SIZE_MB}MB.",
    ),
    payload: Optional[PredictionRequestBase64] = Body(default=None),
    auth:    AuthResult = Depends(require_scope(KeyScope.PREDICT)),
) -> PredictionResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    client_ip  = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )

    # Set rate limit headers pada response
    rate_headers = getattr(request.state, "rate_headers", {})
    for k, v in rate_headers.items():
        response.headers[k] = v

    if auth.deprecated:
        response.headers["Warning"] = (
            '299 - "API key ini deprecated. Segera ganti dengan key baru."'
        )

    has_file    = file is not None and bool(file.filename)
    has_payload = payload is not None

    if has_file and has_payload:
        raise HTTPException(
            status_code=400,
            detail="Kirim HANYA file (multipart) ATAU payload JSON, bukan keduanya.",
        )
    if not has_file and not has_payload:
        raise HTTPException(
            status_code=400,
            detail="Data gambar tidak ada. Kirim file via multipart atau image_base64 via JSON.",
        )

    try:
        raw_input = None

        if has_file:
            data = await _read_file_async(file)
            _validate_file(data, file.filename or "", request_id, client_ip)
            raw_input = data

        elif has_payload:
            if payload.filename:
                ext = (
                    payload.filename.rsplit(".", 1)[-1].lower()
                    if "." in payload.filename else ""
                )
                if ext and ext not in settings.allowed_extensions_set:
                    raise UnsupportedFileTypeException(
                        detail=f"Format '.{ext}' tidak didukung. "
                               f"Format yang diizinkan: {settings.ALLOWED_EXTENSIONS}"
                    )
            raw_input = payload.image_base64

        if raw_input is None:
            raise InvalidImageException(detail="Gagal mengekstrak data gambar.")

        # Jalankan CLIP validation + image processing secara paralel
        clip_task    = asyncio.to_thread(CLIPService.is_durian, raw_input)
        process_task = asyncio.to_thread(ImageProcessor.process, raw_input)

        (is_valid_durian, (tensor, enhanced, preproc_ms)) = await asyncio.gather(
            clip_task,
            process_task,
        )

        if not is_valid_durian:
            raise InvalidImageException(
                detail=(
                    "Gambar ditolak. Sistem mendeteksi ini bukan gambar buah durian. "
                    "Pastikan gambar menampilkan buah durian utuh dengan jelas."
                )
            )

        pred_response = await asyncio.to_thread(
            InferenceService.predict, tensor, enhanced, preproc_ms
        )
        pred_response.request_id = request_id

        logger.info(
            f"[{request_id}] ✓ {pred_response.prediction.variety_name} "
            f"({pred_response.prediction.confidence_score:.4f}) "
            f"key={auth.key_prefix} "
            f"inf={pred_response.inference_time_ms:.1f}ms "
            f"preproc={pred_response.preprocessing_time_ms:.1f}ms "
        )
        return pred_response

    except DurianServiceException as e:
        raise e.to_http_exception()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error tidak terduga saat memproses prediksi. Silakan coba lagi.",
        )