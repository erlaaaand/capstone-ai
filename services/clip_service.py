import base64
import io
import threading
from typing import Optional, Union

from PIL import Image

from core.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID             = "openai/clip-vit-base-patch32"
_NON_DURIAN_THRESHOLD = 0.40

# Kunci: prompt teks (Inggris) | Nilai: label tampil (Indonesia)
_LABELS: dict[str, str] = {
    "a raw, unedited, high-quality photograph of a real durian fruit, "
    "distinctly showing its sharp natural green-brown thorns or fresh yellow fleshy pods inside.":
        "Durian Asli",
    "a digital illustration, 3d render, vector graphic, cartoon, anime, "
    "painting, sketch, drawing, or ai-generated synthetic art of a fruit.":
        "Ilustrasi / Render 3D",
    "a digital screenshot, meme, promotional flyer, poster, graphic design, "
    "or an image containing visible text, words, icons, and UI elements.":
        "Tangkapan Layar / Teks",
    "a photograph prominently featuring a human face, a person, crowds, "
    "or visible human hands holding, opening, or interacting with objects.":
        "Manusia / Anggota Tubuh",
    "a photograph of cooked meals, yellow rice, curry, plated dishes on banana leaves, "
    "or durian-flavored desserts, ice cream, pastries, and cakes.":
        "Makanan Olahan / Hidangan",
    "a photograph of similar rough green fruits like jackfruit, breadfruit, or soursop, "
    "or just a random pile of green leaves, grass, tree branches, and plants.":
        "Buah Lain / Dedaunan",
    "a photograph of furry animals, pets, or spiky animals like hedgehogs or porcupines.":
        "Hewan",
    "a general photograph of everyday household items, electronics, indoor furniture, "
    "vehicles, buildings, or landscape scenery without a clear main subject.":
        "Objek Acak / Pemandangan",
}

_LABEL_PROMPTS: list[str] = list(_LABELS.keys())
_LABEL_NAMES:   list[str] = list(_LABELS.values())


class CLIPService:
    _model          = None
    _processor      = None
    _lock           = threading.Lock()
    _load_attempted = False

    @classmethod
    def _ensure_loaded(cls) -> bool:
        # Fast path — sudah dicoba load sebelumnya
        if cls._load_attempted:
            return cls._model is not None

        with cls._lock:
            if cls._load_attempted:
                return cls._model is not None

            cls._load_attempted = True
            logger.info(f"[CLIPService] Memuat model CLIP: '{_MODEL_ID}'...")

            try:
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
    def is_durian(raw_input: Union[bytes, str]) -> bool:
        """
        Kembalikan True jika gambar kemungkinan besar adalah durian.
        Jika model tidak tersedia, izinkan semua gambar (fail-open).
        """
        if not CLIPService._ensure_loaded():
            logger.warning(
                "[CLIPService] Model tidak tersedia — gambar diizinkan tanpa validasi CLIP."
            )
            return True

        try:
            import torch

            image_bytes = base64.b64decode(raw_input) if isinstance(raw_input, str) else raw_input
            image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            inputs = CLIPService._processor(
                text       = _LABEL_PROMPTS,
                images     = image,
                return_tensors = "pt",
                padding    = True,
            )

            with torch.no_grad():
                outputs = CLIPService._model(**inputs)

            probs      = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            best_idx   = int(probs.argmax())
            best_score = float(probs[best_idx])

            # Index 0 = "Durian Asli" — jika label lain menang dengan confidence tinggi → tolak
            if best_idx != 0 and best_score > _NON_DURIAN_THRESHOLD:
                logger.warning(
                    f"[CLIPService] Bukan durian — terdeteksi sebagai "
                    f"'{_LABEL_NAMES[best_idx]}' (confidence={best_score:.2f})"
                )
                return False

            return True

        except Exception as exc:
            # Fail-open: jika error saat inferensi, izinkan gambar lanjut ke ONNX
            logger.error(f"[CLIPService] Error saat inferensi: {exc}", exc_info=True)
            return True