import io
import base64
import threading
from typing import Optional

from PIL import Image

from core.logger import get_logger

logger = get_logger(__name__)

_MODEL_ID           = "openai/clip-vit-base-patch32"
_NON_DURIAN_THRESHOLD = 0.40

_LABELS = {
    "a raw, unedited, high-quality photograph of a real durian fruit, distinctly showing its sharp natural green-brown thorns or fresh yellow fleshy pods inside.": "Durian Asli",
    "a digital illustration, 3d render, vector graphic, cartoon, anime, painting, sketch, drawing, or ai-generated synthetic art of a fruit.": "Ilustrasi / Render 3D",
    "a digital screenshot, meme, promotional flyer, poster, graphic design, or an image containing visible text, words, icons, and UI elements.": "Tangkapan Layar / Teks",
    "a photograph prominently featuring a human face, a person, crowds, or visible human hands holding, opening, or interacting with objects.": "Manusia / Anggota Tubuh",
    "a photograph of cooked meals, yellow rice, curry, plated dishes on banana leaves, or durian-flavored desserts, ice cream, pastries, and cakes.": "Makanan Olahan / Hidangan",
    "a photograph of similar rough green fruits like jackfruit, breadfruit, or soursop, or just a random pile of green leaves, grass, tree branches, and plants.": "Buah Lain / Dedaunan",
    "a photograph of furry animals, pets, or spiky animals like hedgehogs or porcupines.": "Hewan",
    "a general photograph of everyday household items, electronics, indoor furniture, vehicles, buildings, or landscape scenery without a clear main subject.": "Objek Acak / Pemandangan"
}

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

            # 1. Ekstrak keys (prompt bahasa Inggris) dan values (nama label) menjadi list
            label_prompts = list(_LABELS.keys())
            label_names = list(_LABELS.values())

            # 2. Masukkan label_prompts (berupa List[str]) ke processor
            inputs = CLIPService._processor(
                text=label_prompts,
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
                # 3. Gunakan label_names untuk mengakses nama berdasarkan index angka
                logger.warning(
                    f"[CLIPService] Bukan durian — terdeteksi sebagai: "
                    f"'{label_names[best_idx]}' (confidence={best_score:.2f})"
                )
                return False

            return True

        except Exception as exc:
            logger.error(
                f"[CLIPService] Error saat inferensi: {exc}",
                exc_info=True,
            )
            return True

        except Exception as exc:
            logger.error(
                f"[CLIPService] Error saat inferensi: {exc}",
                exc_info=True,
            )
            return True
