# app/api/routes.py

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from fastapi import APIRouter, Body, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.core_dependencies import AuthResult, require_scope, verify_api_key
from core.config import settings
from core.exceptions import DurianServiceException, InvalidImageException
from core.file_validator import validate_upload  # ← single source, tidak ada duplikasi
from core.logger import get_logger
from core.security import KeyScope
from schemas.request import PredictionRequestBase64
from schemas.response import PredictionResponse
from services.clip_service import CLIPService
from services.image_processor import ImageProcessor
from services.inference_service import InferenceService

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])


async def _read_file_async(file: UploadFile) -> bytes:
    try:
        return await asyncio.to_thread(file.file.read)
    except Exception as e:
        raise InvalidImageException(detail="Tidak dapat membaca file yang diunggah.") from e


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
        description = f"File gambar (JPG/PNG/WebP). Maks {settings.MAX_FILE_SIZE_MB}MB.",
    ),
    payload: Optional[PredictionRequestBase64] = Body(default=None),
    auth:    AuthResult = Depends(require_scope(KeyScope.PREDICT)),
) -> PredictionResponse:

    request_id = getattr(request.state, "request_id", str(uuid.uuid4())[:8])
    client_ip  = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )

    # Propagasi rate-limit headers ke response
    for k, v in getattr(request.state, "rate_headers", {}).items():
        response.headers[k] = v

    if auth.deprecated:
        response.headers["Warning"] = (
            '299 - "API key ini deprecated. Segera ganti dengan key baru."'
        )

    has_file    = file is not None and bool(file.filename)
    has_payload = payload is not None

    if has_file and has_payload:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Kirim HANYA file (multipart) ATAU payload JSON, bukan keduanya.",
        )
    if not has_file and not has_payload:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "Data gambar tidak ada. Kirim file via multipart atau image_base64 via JSON.",
        )

    try:
        raw_input: bytes | str | None = None

        if has_file:
            data = await _read_file_async(file)
            validate_upload(data, file.filename or "", request_id, client_ip)
            raw_input = data

        else:  # has_payload
            if payload.filename:
                ext = (
                    payload.filename.rsplit(".", 1)[-1].lower()
                    if "." in payload.filename else ""
                )
                if ext and ext not in settings.allowed_extensions_set:
                    from core.exceptions import UnsupportedFileTypeException
                    raise UnsupportedFileTypeException(
                        detail=(
                            f"Format '.{ext}' tidak didukung. "
                            f"Format yang diizinkan: {settings.ALLOWED_EXTENSIONS}"
                        )
                    )
            raw_input = payload.image_base64

        if raw_input is None:
            raise InvalidImageException(detail="Gagal mengekstrak data gambar.")

        # Jalankan CLIP validation + image processing secara paralel
        clip_task    = asyncio.to_thread(CLIPService.is_durian, raw_input)
        process_task = asyncio.to_thread(ImageProcessor.process, raw_input)

        is_valid_durian, (tensor, enhanced, preproc_ms) = await asyncio.gather(
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
            f"preproc={pred_response.preprocessing_time_ms:.1f}ms"
        )
        return pred_response

    except DurianServiceException as e:
        raise e.to_http_exception()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unhandled exception: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Error tidak terduga saat memproses prediksi. Silakan coba lagi.",
        )