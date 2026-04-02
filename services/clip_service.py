import io
import base64
import threading
from typing import Optional

from PIL import Image

from core.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID           = "openai/clip-vit-base-patch32"
_NON_DURIAN_THRESHOLD = 0.40

_LABELS = [
    "a photo of a durian fruit",
    "a photo of a person",
    "a photo of an animal",
    "a photo of a vehicle",
    "a photo of random objects or scenery",
]


class CLIPService:
    _model     = None
    _processor = None
    _lock      = threading.Lock()
    _load_attempted = False
    @classmethod
    def _ensure_loaded(cls) -> bool:
        if cls._model is not None or cls._load_attempted:
            return cls._model is not None

        with cls._lock:
            if cls._model is not None or cls._load_attempted:
                return cls._model is not None

            cls._load_attempted = True
            logger.info(f"[CLIPService] Memuat model CLIP: '{_MODEL_ID}'...")

            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                cls._model     = CLIPModel.from_pretrained(_MODEL_ID)
                cls._processor = CLIPProcessor.from_pretrained(_MODEL_ID)

                cls._model.eval()

                logger.info("[CLIPService] Model CLIP berhasil dimuat.")
                return True

            except Exception as exc:
                logger.error(
                    f"[CLIPService] Gagal memuat model CLIP: {exc}. "
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
    def is_durian(raw_input: bytes | str) -> bool:
        if not CLIPService._ensure_loaded():
            logger.warning(
                "[CLIPService] Model tidak tersedia — gambar diizinkan tanpa validasi CLIP."
            )
            return True

        try:
            import torch

            if isinstance(raw_input, str):
                image_bytes = base64.b64decode(raw_input)
            else:
                image_bytes = raw_input

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            inputs = CLIPService._processor(
                text=_LABELS,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = CLIPService._model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            best_idx   = int(probs.argmax())
            best_score = float(probs[best_idx])

            if best_idx != 0 and best_score > _NON_DURIAN_THRESHOLD:
                logger.warning(
                    f"[CLIPService] Bukan durian — terdeteksi sebagai: "
                    f"'{_LABELS[best_idx]}' (confidence={best_score:.2f})"
                )
                return False

            return True

        except Exception as exc:
            logger.error(
                f"[CLIPService] Error saat inferensi: {exc}",
                exc_info=True,
            )
            return True
