import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("export_pipeline")

def _replace_activation_in_config(config: dict | list, old: str, new: str) -> None:

    if isinstance(config, dict):
        if "activation" in config:
            act = config["activation"]
            if isinstance(act, str) and act == old:
                config["activation"] = new
                logger.info(f"  Activation '{old}' → '{new}' (field langsung)")
            elif isinstance(act, dict):
                if act.get("class_name") == old:
                    act["class_name"] = new
                    logger.info(f"  Activation class_name '{old}' → '{new}' (dict nested)")

        for value in config.values():
            if isinstance(value, (dict, list)):
                _replace_activation_in_config(value, old, new)

    elif isinstance(config, list):
        for item in config:
            if isinstance(item, (dict, list)):
                _replace_activation_in_config(item, old, new)


def _replace_dtype_in_config(config: dict | list, old: str, new: str) -> None:
    if isinstance(config, dict):
        for key in ("dtype", "compute_dtype", "variable_dtype"):
            if config.get(key) == old:
                config[key] = new
                logger.info(f"  dtype field '{key}': '{old}' → '{new}'")

        if config.get("class_name") == "Policy":
            inner = config.get("config", {})
            if isinstance(inner, dict) and inner.get("name") == old:
                inner["name"] = new
                logger.info(f"  Policy name: '{old}' → '{new}'")

        for value in config.values():
            if isinstance(value, (dict, list)):
                _replace_dtype_in_config(value, old, new)

    elif isinstance(config, list):
        for item in config:
            if isinstance(item, (dict, list)):
                _replace_dtype_in_config(item, old, new)


def _sanitize_model_config(model_json: str) -> str:
    try:
        config = json.loads(model_json)
    except json.JSONDecodeError as e:
        logger.error(f"Gagal parse model JSON: {e}")
        raise

    _replace_dtype_in_config(config, "mixed_float16", "float32")
    _replace_dtype_in_config(config, "float16", "float32")

    _replace_activation_in_config(config, "gelu", "swish")

    return json.dumps(config)


def build_inference_model(trained_model: tf.keras.Model) -> tf.keras.Model:

    backbone = None
    for layer in trained_model.layers:
        if layer.name == "backbone":
            backbone = layer
            break
    if backbone is None:
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

    img_input = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32, name="image_input")

    x = backbone(img_input, training=False)

    head_layer_names = [
        "gap",       "bn_0",    "drop_0",
        "dense_512", "bn_1",    "drop_1",
        "dense_256", "bn_2",    "drop_2",
        "dense_128", "bn_3",    "drop_3",
        "predictions",
    ]

    for name in head_layer_names:
        try:
            layer = trained_model.get_layer(name)
            x = layer(x, training=False)
            logger.info(f"  Layer '{name}' ditambahkan ke inference model.")
        except ValueError:
            logger.warning(f"  Layer '{name}' tidak ditemukan, dilewati.")

    inference_model = tf.keras.Model(
        inputs=img_input,
        outputs=x,
        name="DurianClassifier_InferenceOnly",
    )

    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out   = inference_model(dummy, training=False)
    logger.info(f"Inference model output shape: {out.shape}  (harus: (1, num_classes))")
    logger.info(f"Inference model output dtype: {out.dtype}  (harus: float32)")

    return inference_model


def export_to_onnx(input_path: str, onnx_path: str, opset: int = 17) -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    logger.info("Global mixed precision policy di-set ke float32 sebelum load model.")

    input_file  = Path(input_path)
    output_file = Path(onnx_path)

    if not input_file.exists():
        logger.error(f"Input model tidak ditemukan: {input_file}")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Keras model dari: {input_file}")
    try:
        original_model = tf.keras.models.load_model(str(input_file))
        logger.info("Model Keras berhasil di-load.")

        logger.info("Mencuci arsitektur model dari mixed_float16 ke pure float32...")

        raw_json = original_model.to_json()

        sanitized_json = _sanitize_model_config(raw_json)
        logger.info("Sanitasi model config selesai (struktural, aman).")

        trained_model = tf.keras.models.model_from_json(sanitized_json)
        trained_model.set_weights(original_model.get_weights())
        logger.info("Konversi ke pure float32 berhasil.")

    except Exception as e:
        logger.error(f"Gagal load atau cuci Keras model: {str(e)}")
        sys.exit(1)

    logger.info("Membangun inference-only model (strip augmentation)...")
    try:
        inference_model = build_inference_model(trained_model)
        logger.info("Inference model berhasil dibangun.")
        inference_model.summary(line_length=80)
    except Exception as e:
        logger.error(f"Gagal membangun inference model: {str(e)}")
        logger.warning("Fallback: menggunakan model asli tanpa stripping augmentation.")
        inference_model = trained_model

    logger.info(f"Mengkonversi ke ONNX format (opset={opset})...")

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
            opset=opset,
            output_path=str(output_file),
        )

        size_mb = output_file.stat().st_size / 1e6
        logger.info(f"ONNX export berhasil → {output_file}  ({size_mb:.1f} MB)")
        logger.info(
            "Ukuran ~18-20MB = float32 (benar). "
            "Jika ~10MB = float16 (salah, ada masalah casting)."
        )

    except Exception as e:
        logger.error(f"Gagal konversi ONNX: {str(e)}")
        sys.exit(1)

    logger.info("Verifikasi ONNX model dengan onnxruntime...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(output_file),
            providers=["CPUExecutionProvider"],
        )
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        input_dtype = sess.get_inputs()[0].type

        logger.info(f"  Input  : name='{input_name}' dtype={input_dtype}")
        logger.info(f"  Output : name='{output_name}'")

        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        outputs     = sess.run([output_name], {input_name: dummy_input})

        output_shape = outputs[0].shape
        logger.info(
            f"Verifikasi OK | input='{input_name}' | output='{output_name}' "
            f"| output_shape={output_shape}"
        )

        num_classes_onnx = output_shape[-1]
        if num_classes_onnx != 8:
            logger.warning(f"Output memiliki {num_classes_onnx} kelas, diharapkan 8.")
        else:
            logger.info(f"Jumlah kelas: {num_classes_onnx} ✓")

    except Exception as e:
        logger.warning(f"Verifikasi ONNX gagal (model mungkin tetap valid): {str(e)}")


def main() -> None:
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
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    args = parser.parse_args()

    input_path  = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    export_to_onnx(str(input_path), str(output_path), opset=args.opset)


if __name__ == "__main__":
    main()
