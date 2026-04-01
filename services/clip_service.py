import io
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from core.logger import get_logger

logger = get_logger(__name__)

logger.info("Memuat model CLIP (Zero-Shot Classification)...")
try:
    model_id = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_id)
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    logger.info("Model CLIP berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat model CLIP: {e}")
    clip_model = None
    clip_processor = None


class CLIPService:
    @staticmethod
    def is_durian(raw_input: bytes | str) -> bool:
        if clip_model is None or clip_processor is None:
            return True

        try:
            if isinstance(raw_input, str):
                image_bytes = base64.b64decode(raw_input)
            else:
                image_bytes = raw_input

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            labels = [
                "a photo of a durian fruit",
                "a photo of a person",
                "a photo of an animal",
                "a photo of a vehicle",
                "a photo of random objects or scenery"
            ]

            inputs = clip_processor(
                text=labels, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )

            with torch.no_grad():
                outputs = clip_model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

            best_match_index = int(probs.argmax())
            best_score = float(probs[best_match_index])

            if best_match_index != 0 and best_score > 0.40:
                logger.warning(f"CLIP mendeteksi bukan durian. Terdeteksi sebagai: '{labels[best_match_index]}' ({best_score:.2f})")
                return False

            return True

        except Exception as e:
            logger.error(f"Error saat inferensi CLIP: {e}", exc_info=True)
            return True