# services/inference_service.py
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from core.config import get_display_name, get_variety_info, settings
from core.exceptions import InferenceException, ModelNotLoadedException
from core.logger import get_logger
from models.model_loader import get_model_loader
from schemas.response import PredictionResponse, PredictionResult, VarietyScore

logger = get_logger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-8)


def _ensure_probabilities(probs: np.ndarray) -> np.ndarray:
    prob_sum = float(probs.sum())
    if abs(prob_sum - 1.0) > 0.05:
        logger.warning(
            f"Output model sum={prob_sum:.4f} (bukan ≈1.0). "
            "Menerapkan softmax otomatis — periksa export ONNX."
        )
        return _softmax(probs)
    return probs


class InferenceService:

    @staticmethod
    def predict(
        image_tensor: np.ndarray,
        enhanced:     bool  = False,
        preproc_ms:   float = 0.0,
    ) -> PredictionResponse:
        logger.debug(f"Inferensi: shape={image_tensor.shape}, enhanced={enhanced}")

        if image_tensor.ndim != 4:
            raise InferenceException(
                detail=f"Input harus 4D, dapat {image_tensor.ndim}D."
            )
        expected_shape = (settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)
        if image_tensor.shape[1:] != expected_shape:
            raise InferenceException(
                detail=(
                    f"Input shape tidak valid: {image_tensor.shape}. "
                    f"Diharapkan (1, {settings.IMAGE_SIZE}, {settings.IMAGE_SIZE}, 3)."
                )
            )
        if image_tensor.dtype != np.float32:
            image_tensor = image_tensor.astype(np.float32)

        loader = get_model_loader()
        try:
            session     = loader.session
            input_name  = loader.input_name
            output_name = loader.output_name
        except ModelNotLoadedException:
            raise

        try:
            t_start = time.perf_counter()
            outputs = session.run([output_name], {input_name: image_tensor})
            inf_ms  = (time.perf_counter() - t_start) * 1000.0
            logger.debug(f"ONNX selesai dalam {inf_ms:.2f}ms")
        except Exception as e:
            logger.error(f"ONNX session.run() gagal: {str(e)}")
            raise InferenceException(detail="Gagal menjalankan prediksi model.") from e

        try:
            raw_probs     = outputs[0][0]
            probabilities = _ensure_probabilities(raw_probs)

            codes       = settings.class_names_list
            num_classes = len(codes)

            if len(probabilities) != num_classes:
                raise InferenceException(
                    detail=(
                        f"Konfigurasi tidak sinkron: model={len(probabilities)} kelas, "
                        f"config={num_classes} kelas. Periksa CLASS_NAMES di .env."
                    )
                )

            # Upgrade: key = variety_code (bukan nama) untuk determinisme lookup
            confidence_scores: Dict[str, float] = {}
            all_varieties:     List[VarietyScore] = []

            for code, prob in zip(codes, probabilities):
                score = round(float(prob), 6)
                confidence_scores[code] = score
                all_varieties.append(VarietyScore(
                    variety_code     = code,
                    confidence_score = score,
                ))

            all_varieties.sort(key=lambda v: v.confidence_score, reverse=True)

            top_idx  = int(np.argmax(probabilities))
            top_code = codes[top_idx]
            top_info = get_variety_info(top_code)
            top_conf = round(float(probabilities[top_idx]), 6)

            logger.info(
                f"Prediksi: [{top_code}] {top_info.display_name} "
                f"(conf={top_conf:.4f}, inf={inf_ms:.1f}ms, "
                f"preproc={preproc_ms:.1f}ms, enhanced={enhanced})"
            )

            return PredictionResponse(
                success    = True,
                prediction = PredictionResult(
                    variety_code     = top_code,
                    variety_name     = top_info.display_name,
                    local_name       = top_info.local_name,
                    origin           = top_info.origin,
                    description      = top_info.description,
                    confidence_score = top_conf,
                ),
                all_varieties         = all_varieties,
                confidence_scores     = confidence_scores,
                image_enhanced        = enhanced,
                inference_time_ms     = round(inf_ms, 2),
                preprocessing_time_ms = round(preproc_ms, 2),
                model_version         = settings.APP_VERSION,
                market_context        = None,  # Diisi oleh routes.py setelah predict
            )

        except InferenceException:
            raise
        except Exception as e:
            logger.error(f"Gagal memproses output inferensi: {str(e)}")
            raise InferenceException(detail="Gagal memproses hasil prediksi model.") from e
