import base64
import io
import time
from typing import Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

from core.config import settings
from core.exceptions import ImageProcessingException, InvalidImageException
from core.logger import get_logger

logger = get_logger(__name__)


def _auto_white_balance(arr: np.ndarray) -> np.ndarray:
    means = arr.mean(axis=(0, 1))
    if np.any(means < 1e-6):
        return arr
    gray  = means.mean()
    scale = gray / (means + 1e-8)
    result = np.clip(arr * scale[np.newaxis, np.newaxis, :], 0, 255)
    return result


def _apply_clahe(arr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    pil      = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    ycbcr    = pil.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y_arr    = np.array(y, dtype=np.float32)
    hist, _  = np.histogram(y_arr.flatten(), bins=256, range=(0, 256))
    cdf      = hist.cumsum().astype(np.float64)
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8) * 255.0
    y_eq     = cdf_norm[y_arr.astype(np.int32).clip(0, 255)]

    alpha    = min(clip_limit / 4.0, 1.0) * 0.40
    y_new    = np.clip((1 - alpha) * y_arr + alpha * y_eq, 0, 255).astype(np.uint8)

    merged   = Image.merge("YCbCr", (Image.fromarray(y_new, "L"), cb, cr)).convert("RGB")
    return np.array(merged, dtype=np.float32)


def _unsharp_mask(arr: np.ndarray, radius: int = 2, amount: float = 0.45) -> np.ndarray:
    pil      = Image.fromarray(arr.astype(np.uint8))
    blurred  = np.array(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)
    sharpened = arr + amount * (arr - blurred)
    return np.clip(sharpened, 0, 255)


def enhance_image(arr: np.ndarray) -> np.ndarray:
    if settings.ENABLE_WHITE_BALANCE:
        arr = _auto_white_balance(arr)
    if settings.ENABLE_CLAHE:
        arr = _apply_clahe(arr, clip_limit=settings.CLAHE_CLIP_LIMIT)
    if settings.ENABLE_SHARPENING:
        arr = _unsharp_mask(arr)
    return arr


def _letterbox_resize(
    image: Image.Image,
    target: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Image.Image:
    tw, th   = target
    ow, oh   = image.size
    scale    = min(tw / ow, th / oh)
    nw, nh   = int(round(ow * scale)), int(round(oh * scale))

    resized  = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas   = Image.new("RGB", target, color=pad_color)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


class ImageProcessor:

    @staticmethod
    def _decode_bytes(data: bytes) -> Image.Image:
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
            return Image.open(io.BytesIO(data))
        except UnidentifiedImageError as e:
            raise InvalidImageException(
                detail="File bukan gambar yang valid atau format tidak didukung."
            ) from e
        except Exception as e:
            raise InvalidImageException(
                detail=f"Tidak dapat membaca data gambar: {str(e)}"
            ) from e

    @staticmethod
    def _decode_base64(b64: str) -> Image.Image:
        try:
            pad = len(b64) % 4
            if pad:
                b64 += "=" * (4 - pad)
            raw = base64.b64decode(b64, validate=True)
            return ImageProcessor._decode_bytes(raw)
        except base64.binascii.Error as e:
            raise InvalidImageException(detail="String bukan Base64 yang valid.") from e
        except InvalidImageException:
            raise
        except Exception as e:
            raise InvalidImageException(detail=f"Gagal decode Base64: {str(e)}") from e

    @staticmethod
    def process(
        image_input: Union[bytes, str],
    ) -> Tuple[np.ndarray, bool, float]:
        t0 = time.perf_counter()

        if isinstance(image_input, bytes):
            img = ImageProcessor._decode_bytes(image_input)
        elif isinstance(image_input, str):
            img = ImageProcessor._decode_base64(image_input)
        else:
            raise ImageProcessingException(
                detail=f"Tipe input tidak didukung: {type(image_input)}."
            )

        try:
            orig_size, orig_mode = img.size, img.mode
            logger.debug(f"Gambar asli: ukuran={orig_size}, mode={orig_mode}")

            if img.mode != "RGB":
                img = img.convert("RGB")

            target = settings.image_size_tuple
            if img.size != target:
                scale_info = min(target[0] / orig_size[0], target[1] / orig_size[1])
                img = _letterbox_resize(img, target)
                logger.debug(f"Letterbox {orig_size}→{target} scale={scale_info:.3f}")

            arr = np.array(img, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ImageProcessingException(
                    detail=f"Shape tidak valid: {arr.shape}. Diharapkan (H,W,3)."
                )

            enhanced = False
            if settings.ENABLE_ENHANCEMENT:
                arr      = enhance_image(arr)
                enhanced = True

            tensor    = np.expand_dims(arr, axis=0)
            preproc_ms = (time.perf_counter() - t0) * 1000.0

            logger.debug(
                f"Processing selesai: shape={tensor.shape}, "
                f"range=[{tensor.min():.0f},{tensor.max():.0f}], "
                f"enhanced={enhanced}, t={preproc_ms:.1f}ms"
            )
            return tensor, enhanced, preproc_ms

        except (InvalidImageException, ImageProcessingException):
            raise
        except Exception as e:
            logger.error(f"Gagal preprocessing: {str(e)}")
            raise ImageProcessingException(
                detail="Gagal memproses gambar untuk inferensi."
            ) from e