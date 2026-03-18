"""
Inference service for the Durian Classification API.

Facilitates running an image tensor through the ONNX model session,
mapping the output probabilities back to human-readable durian class names.

CATATAN ARSITEKTUR:
  Model ditraining dengan output layer: Dense(num_classes, activation="softmax")
  → output ONNX sudah berupa probability [0.0, 1.0] yang sum = 1.0
  → TIDAK perlu softmax manual di sini (double softmax = hasil salah/terlalu flat)

  Jika suatu saat model diganti dengan output logits (tanpa softmax),
  aktifkan kembali fungsi softmax() dan panggil di bawah.
"""

import time
from typing import Dict

import numpy as np

from core.config import settings
from core.exceptions import InferenceException, ModelNotLoadedException
from core.logger import get_logger
from models.model_loader import get_model_loader
from schemas.response import PredictionResponse, PredictionResult

logger = get_logger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax untuk digunakan HANYA jika model output adalah logits (bukan probability).
    Model saat ini sudah output softmax — fungsi ini TIDAK dipanggil.

    Args:
        x: Array logits shape (..., num_classes).

    Returns:
        Probability array dengan shape yang sama.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-8)


def _validate_probabilities(probs: np.ndarray) -> np.ndarray:
    """
    Validasi dan normalisasi output model.

    Jika output ONNX ternyata bukan probability (sum jauh dari 1.0),
    terapkan softmax otomatis sebagai safety net. Ini menangani edge case
    di mana model diekspor tanpa softmax layer.

    Args:
        probs: Array output model shape (num_classes,).

    Returns:
        Array probability yang valid, sum = 1.0.
    """
    prob_sum = float(probs.sum())

    if abs(prob_sum - 1.0) > 0.05:
        # Output bukan probability — terapkan softmax
        logger.warning(
            f"Output model sum={prob_sum:.4f} (bukan ~1.0). "
            f"Menerapkan softmax otomatis. "
            f"Periksa apakah model diekspor dengan benar."
        )
        return _softmax(probs)

    return probs


class InferenceService:
    """Service class untuk menjalankan prediksi ONNX model."""

    @staticmethod
    def predict(image_tensor: np.ndarray) -> PredictionResponse:
        """
        Jalankan inferensi pada preprocessed image tensor.

        Pipeline:
          1. Ambil ONNX session dari singleton ModelLoader
          2. Run inference → output probability (sudah softmax dari model)
          3. Validasi output (safety net jika ternyata logits)
          4. Map ke class names dari settings
          5. Return PredictionResponse

        Args:
            image_tensor: Numpy array shape (1, 224, 224, 3) dtype float32.
                          Nilai pixel 0–255 (EfficientNetB0 handle normalisasi internal).

        Returns:
            PredictionResponse dengan top prediction, semua confidence scores,
            dan inference timing.

        Raises:
            ModelNotLoadedException: Jika model belum di-load.
            InferenceException: Jika ONNX session gagal atau output tidak valid.
        """
        logger.debug(f"Memulai inferensi, input shape: {image_tensor.shape}")

        # ── Validasi input tensor ─────────────────────────────────────────────
        if image_tensor.ndim != 4:
            raise InferenceException(
                detail=f"Input tensor harus 4D (batch, H, W, C), dapat {image_tensor.ndim}D."
            )
        if image_tensor.shape[1:] != (224, 224, 3):
            raise InferenceException(
                detail=f"Input shape tidak valid: {image_tensor.shape}. "
                       f"Diharapkan (1, 224, 224, 3)."
            )
        if image_tensor.dtype != np.float32:
            logger.debug(f"Cast input dari {image_tensor.dtype} ke float32.")
            image_tensor = image_tensor.astype(np.float32)

        # ── Ambil ONNX session ────────────────────────────────────────────────
        model_loader = get_model_loader()

        try:
            session     = model_loader.session
            input_name  = model_loader.input_name
            output_name = model_loader.output_name
        except ModelNotLoadedException:
            raise  # Biarkan propagasi ke route handler sebagai 503

        # ── Run ONNX inference ────────────────────────────────────────────────
        try:
            start_time = time.perf_counter()

            raw_outputs = session.run(
                [output_name],
                {input_name: image_tensor},
            )

            inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            logger.debug(f"Inferensi selesai dalam {inference_time_ms:.2f} ms")

        except Exception as e:
            logger.error(f"ONNX session.run() gagal: {str(e)}")
            raise InferenceException(
                detail="Gagal menjalankan prediksi model."
            ) from e

        # ── Post-process output ───────────────────────────────────────────────
        try:
            # raw_outputs[0] shape: (1, num_classes)
            raw_probs = raw_outputs[0][0]   # ambil batch pertama → shape (num_classes,)

            # Model output sudah softmax → langsung pakai sebagai probability
            # Tapi tetap validasi untuk safety (handle edge case logits output)
            probabilities = _validate_probabilities(raw_probs)

            # ── Validasi jumlah kelas ─────────────────────────────────────────
            class_names = settings.class_names_list
            num_classes  = len(class_names)

            if len(probabilities) != num_classes:
                logger.error(
                    f"Mismatch jumlah kelas: model output {len(probabilities)} kelas, "
                    f"config mendefinisikan {num_classes} kelas. "
                    f"Periksa CLASS_NAMES di .env atau config.py."
                )
                raise InferenceException(
                    detail=(
                        f"Konfigurasi kelas tidak sinkron: model={len(probabilities)}, "
                        f"config={num_classes}. Hubungi administrator."
                    )
                )

            # ── Map ke class names ────────────────────────────────────────────
            confidence_map: Dict[str, float] = {
                name: round(float(prob), 6)
                for name, prob in zip(class_names, probabilities)
            }

            # ── Ambil top-1 prediction ────────────────────────────────────────
            top_idx        = int(np.argmax(probabilities))
            top_class_name = class_names[top_idx]
            top_confidence = float(probabilities[top_idx])

            logger.info(
                f"Prediksi: {top_class_name} "
                f"(confidence={top_confidence:.4f}, "
                f"waktu={inference_time_ms:.1f}ms)"
            )

            # ── Bangun response ───────────────────────────────────────────────
            return PredictionResponse(
                success=True,
                prediction=PredictionResult(
                    class_name=top_class_name,
                    confidence_score=round(top_confidence, 6),
                ),
                confidence_scores=confidence_map,
                inference_time_ms=round(inference_time_ms, 2),
                model_version=settings.APP_VERSION,
            )

        except InferenceException:
            raise  # Propagasi tanpa wrap
        except Exception as e:
            logger.error(f"Gagal memproses output inferensi: {str(e)}")
            raise InferenceException(
                detail="Gagal memproses hasil prediksi model."
            ) from e