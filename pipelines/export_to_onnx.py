"""
Standalone model export script.

Converts a trained TensorFlow/Keras (.h5) model into the ONNX (.onnx) format
for high-performance production inference via `onnxruntime`.
"""

import argparse
import sys
from pathlib import Path

import tensorflow as tf
import tf2onnx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("export_pipeline")


def export_h5_to_onnx(h5_path: str, onnx_path: str) -> None:
    """Load a Keras .h5 model and export it to .onnx.

    Args:
        h5_path: Path to the input Keras HDF5 model file.
        onnx_path: Path to the output ONNX model file.
    """
    input_file = Path(h5_path)
    output_file = Path(onnx_path)

    if not input_file.exists():
        logger.error(f"Input model file not found: {input_file}")
        sys.exit(1)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Keras model from: {input_file}")
    try:
        model = tf.keras.models.load_model(str(input_file))
        logger.info("Keras model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Keras model: {str(e)}")
        sys.exit(1)

    logger.info("Converting model to ONNX format...")
    try:
        # Get the input signature from the loaded model
        # EfficientNetB0 expects (None, 224, 224, 3) where None is the batch dimension
        input_signature = [
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input_1")
        ]

        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signature,
            opset=13,  # Opset 13 is highly compatible with onnxruntime
            output_path=str(output_file)
        )
        
        logger.info(f"ONNX conversion successful. Saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed during ONNX conversion: {str(e)}")
        sys.exit(1)


def main() -> None:
    """Execute the export pipeline."""
    parser = argparse.ArgumentParser(description="Convert Keras H5 model to ONNX for production.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="models/weights/best_model.keras", 
        help="Path to the trained Keras .h5 model"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="models/weights/efficientnet_b0.onnx", 
        help="Path to save the resulting .onnx model"
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    export_h5_to_onnx(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
