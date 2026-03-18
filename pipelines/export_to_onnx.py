"""
Standalone model export script.

Converts a trained TensorFlow/Keras (.keras / .h5) model into ONNX format
for high-performance production inference via `onnxruntime`.

PENTING — Mengapa augmentation layer harus di-strip sebelum export:
  Model yang ditraining menyertakan layer augmentasi (RandomFlip, RandomRotation, dll)
  di dalam arsitektur. Layer ini AKTIF saat training, TIDAK AKTIF saat inference
  (karena dipanggil dengan training=False). Namun saat dikonversi ke ONNX via
  tf2onnx.convert.from_keras(), layer augmentasi ikut ter-include dan bisa
  menyebabkan output non-deterministik di production (setiap inferensi hasilnya
  berbeda karena augmentasi aktif secara random).

  Solusi: bangun inference-only model (tanpa augmentation layer) menggunakan
  bobot yang sama, lalu konversi model bersih ini ke ONNX.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("export_pipeline")


def build_inference_model(trained_model: tf.keras.Model) -> tf.keras.Model:
    """
    Strip augmentation layers dan buat inference-only model.

    Augmentation layer (Sequential bernama 'gpu_augment') TIDAK boleh
    ikut ke ONNX karena akan membuat prediksi non-deterministik.
    EfficientNetB0 sudah punya built-in preprocessing (rescaling ke [-1,1])
    di dalam backbone — ImageProcessor di services/ cukup kirim float32 [0,255].

    Args:
        trained_model: Model Keras hasil training lengkap dengan augmentasi.

    Returns:
        Model Keras baru tanpa augmentation layer, siap untuk ONNX export.
    """
    # Ambil backbone dari model yang sudah ditraining
    try:
        backbone = trained_model.get_layer("backbone")
    except ValueError:
        # Fallback: cari layer EfficientNetB0 dengan nama apapun
        backbone = None
        for layer in trained_model.layers:
            if "efficientnet" in layer.name.lower():
                backbone = layer
                break
        if backbone is None:
            raise ValueError(
                "Tidak dapat menemukan backbone EfficientNetB0 di dalam model. "
                "Pastikan layer backbone diberi nama 'backbone' saat build_model()."
            )

    logger.info(f"Backbone ditemukan: '{backbone.name}'")

    # Bangun inference model: input → backbone → head layers (tanpa augmentation)
    # Kita telusuri graph dari input baru melewati semua layer kecuali augmentation
    img_input = tf.keras.Input(shape=(224, 224, 3), name="image_input")

    # backbone dalam inference mode (BN pakai running statistics)
    x = backbone(img_input, training=False)

    # Ambil head layers dari model asli (setelah backbone, sebelum output)
    # Head layers: GAP → BN → Dropout → Dense512 → BN → Dropout → Dense256 → BN → Dropout → predictions
    head_layer_names = [
        "gap", "bn_0", "drop_0",
        "dense_512", "bn_1", "drop_1",
        "dense_256", "bn_2", "drop_2",
        "predictions",
    ]

    for name in head_layer_names:
        try:
            layer = trained_model.get_layer(name)
            # Dropout tidak aktif saat training=False (inference mode)
            x = layer(x, training=False)
            logger.info(f"  Layer '{name}' ditambahkan ke inference model.")
        except ValueError:
            logger.warning(f"  Layer '{name}' tidak ditemukan, dilewati.")

    inference_model = tf.keras.Model(
        inputs=img_input,
        outputs=x,
        name="DurianClassifier_InferenceOnly",
    )

    # Verifikasi output shape
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out   = inference_model(dummy, training=False)
    logger.info(f"Inference model output shape: {out.shape}  (harus: (1, num_classes))")

    return inference_model


def export_to_onnx(input_path: str, onnx_path: str) -> None:
    """
    Load model Keras (.keras atau .h5), strip augmentation, export ke ONNX.

    Args:
        input_path: Path ke model Keras (.keras atau .h5).
        onnx_path:  Path output file .onnx.
    """
    input_file  = Path(input_path)
    output_file = Path(onnx_path)

    if not input_file.exists():
        logger.error(f"Input model tidak ditemukan: {input_file}")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model Keras ──────────────────────────────────────────────────────
    logger.info(f"Loading Keras model dari: {input_file}")
    try:
        trained_model = tf.keras.models.load_model(str(input_file))
        logger.info("Model Keras berhasil di-load.")
        trained_model.summary(line_length=80)
    except Exception as e:
        logger.error(f"Gagal load Keras model: {str(e)}")
        sys.exit(1)

    # ── Strip augmentation, bangun inference-only model ───────────────────────
    logger.info("Membangun inference-only model (tanpa augmentation layer)...")
    try:
        inference_model = build_inference_model(trained_model)
        logger.info("Inference model berhasil dibangun.")
    except Exception as e:
        logger.error(f"Gagal membangun inference model: {str(e)}")
        logger.warning("Fallback: menggunakan model asli tanpa stripping augmentation.")
        inference_model = trained_model

    # ── Konversi ke ONNX ──────────────────────────────────────────────────────
    logger.info("Mengkonversi ke ONNX format...")
    try:
        input_signature = [
            tf.TensorSpec(
                shape=(None, 224, 224, 3),
                dtype=tf.float32,
                name="image_input",
            )
        ]

        _, _ = tf2onnx.convert.from_keras(
            inference_model,
            input_signature=input_signature,
            opset=13,
            output_path=str(output_file),
        )

        size_mb = output_file.stat().st_size / 1e6
        logger.info(f"ONNX export berhasil → {output_file}  ({size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"Gagal konversi ONNX: {str(e)}")
        sys.exit(1)

    # ── Verifikasi ONNX dengan onnxruntime ────────────────────────────────────
    logger.info("Verifikasi ONNX model dengan onnxruntime...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(output_file),
            providers=["CPUExecutionProvider"],
        )
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        outputs     = sess.run([output_name], {input_name: dummy_input})

        logger.info(
            f"Verifikasi OK | input='{input_name}' | output='{output_name}' "
            f"| output_shape={outputs[0].shape}"
        )

        # Pastikan output adalah probability (sum ≈ 1.0)
        prob_sum = float(outputs[0][0].sum())
        if abs(prob_sum - 1.0) < 0.01:
            logger.info(f"Output probability valid (sum={prob_sum:.6f} ≈ 1.0) ✓")
        else:
            logger.warning(
                f"Output sum={prob_sum:.6f} — mungkin bukan softmax probability. "
                f"Periksa inference_service.py apakah perlu softmax manual."
            )

    except Exception as e:
        logger.warning(f"Verifikasi ONNX gagal (model mungkin tetap valid): {str(e)}")


def main() -> None:
    """Execute the export pipeline."""
    parser = argparse.ArgumentParser(
        description="Konversi Keras model (.keras/.h5) ke ONNX untuk production inference."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="models/weights/best_model.keras",
        help="Path ke trained Keras model (.keras atau .h5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/weights/efficientnet_b0.onnx",
        help="Path output file .onnx",
    )
    args = parser.parse_args()

    input_path  = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    export_to_onnx(str(input_path), str(output_path))


if __name__ == "__main__":
    main()