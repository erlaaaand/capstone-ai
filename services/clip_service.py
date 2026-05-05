# services/clip_service.py

import base64
import io
import threading
from typing import Union

from PIL import Image

from core.clip_labels import DURIAN_LABEL_INDEX, LABEL_NAMES, LABEL_PROMPTS
from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class CLIPService:
    _model:          object         = None
    _processor:      object         = None
    _lock:           threading.Lock = threading.Lock()
    _load_attempted: bool           = False

    @classmethod
    def _ensure_loaded(cls) -> bool:
        # Fast path — sudah dicoba load sebelumnya.
        if cls._load_attempted:
            return cls._model is not None

        with cls._lock:
            if cls._load_attempted:
                return cls._model is not None

            cls._load_attempted = True
            model_id = settings.CLIP_MODEL_ID
            revision = settings.clip_revision

            logger.info(
                f"[CLIPService] Memuat model CLIP: '{model_id}' "
                f"(revision={revision or 'latest'})..."
            )

            try:
                from transformers import CLIPModel, CLIPProcessor

                cls._model = CLIPModel.from_pretrained(model_id, revision=revision)
                cls._processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
                cls._model.eval()

                logger.info(
                    f"[CLIPService] Model CLIP berhasil dimuat: '{model_id}' "
                    f"revision={revision or 'latest'}."
                )
                return True

            except Exception as exc:
                logger.error(
                    f"[CLIPService] Gagal memuat model CLIP '{model_id}': {exc}. "
                    "Validasi CLIP dinonaktifkan — semua gambar akan diizinkan.",
                    exc_info=True,
                )
                cls._model     = None
                cls._processor = None
                return False

    @classmethod
    def warmup(cls) -> bool:
        return cls._ensure_loaded()

    @staticmethod
    def is_durian(raw_input: Union[bytes, str]) -> bool:
        if not CLIPService._ensure_loaded():
            logger.warning(
                "[CLIPService] Model tidak tersedia — gambar diizinkan tanpa validasi CLIP."
            )
            return True

        non_durian_threshold = settings.CLIP_NON_DURIAN_THRESHOLD

        try:
            import torch

            image_bytes = base64.b64decode(raw_input) if isinstance(raw_input, str) else raw_input
            image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            inputs = CLIPService._processor(
                text           = LABEL_PROMPTS,
                images         = image,
                return_tensors = "pt",
                padding        = True,
            )

            with torch.no_grad():
                outputs = CLIPService._model(**inputs)

            probs      = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            best_idx   = int(probs.argmax())
            best_score = float(probs[best_idx])

            if best_idx != DURIAN_LABEL_INDEX and best_score > non_durian_threshold:
                logger.warning(
                    f"[CLIPService] Bukan durian — terdeteksi sebagai "
                    f"'{LABEL_NAMES[best_idx]}' (confidence={best_score:.2f})"
                )
                return False

            return True

        except Exception as exc:
            # Fail-open: jika error saat inferensi, izinkan gambar lanjut ke ONNX.
            logger.error(f"[CLIPService] Error saat inferensi: {exc}", exc_info=True)
            return True