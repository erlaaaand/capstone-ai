"""
Standalone evaluation script for the Durian Classification Model.

Uses TensorFlow & Keras to load the trained .h5 model and evaluates it
against a test dataset (e.g., data/raw/test/). Calculates overall metrics
like Accuracy, Precision, Recall, F1-Score, and generates a visual Confusion
Matrix using Matplotlib and Seaborn.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("evaluate_pipeline")


def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: list[str], 
    output_path: Path
) -> None:
    """Plot and save a seaborn confusion matrix.

    Args:
        cm: Confusion matrix numpy array.
        class_names: List of class labels.
        output_path: Path to save the plot image.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Durian Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")


def evaluate_model(model_path: str, test_data_dir: str, batch_size: int = 32) -> None:
    """Load model and evaluate against test data.

    Args:
        model_path: Path to the trained Keras .h5 model.
        test_data_dir: Path to the test images directory.
        batch_size: Batch size for inference matching training.
    """
    model_file = PROJECT_ROOT / model_path
    test_dir = PROJECT_ROOT / test_data_dir

    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)
    if not test_dir.exists():
        logger.error(f"Test data directory not found: {test_dir}")
        sys.exit(1)

    logger.info(f"Loading Keras model from {model_file}")
    try:
        model = tf.keras.models.load_model(str(model_file))
    except Exception as e:
        logger.error(f"Failed to load Keras model: {str(e)}")
        sys.exit(1)

    logger.info(f"Loading test dataset from {test_dir}")
    
    # We set shuffle=False to ensure labels align with predictions perfectly
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=False,
    )
    
    class_names = test_ds.class_names
    logger.info(f"Classes found: {class_names}")

    logger.info("Running model predictions on test dataset...")
    # Get true labels and predicted labels
    true_labels = []
    for images, labels in test_ds:
        # labels are one-hot encoded, get the argmax
        true_labels.extend(np.argmax(labels.numpy(), axis=-1))
        
    true_labels = np.array(true_labels)
    
    # Predict
    predictions = model.predict(test_ds)
    pred_labels = np.argmax(predictions, axis=1)

    # 1. Classification Report (Accuracy, Precision, Recall, F1)
    logger.info("================ Classification Report ================")
    report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)
    for line in report.split('\n'):
        if line.strip():
            logger.info(line)
    logger.info("=======================================================")

    # 2. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Ensure output dir exists
    plot_dir = PROJECT_ROOT / "models" / "evaluation"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "confusion_matrix.png"
    
    plot_confusion_matrix(cm, class_names, plot_path)


def main() -> None:
    """Execute evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate Durian Classification Model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="models/weights/efficientnet_b0.h5", 
        help="Path to the trained Keras .h5 model"
    )
    parser.add_argument(
        "--test_dir", 
        type=str, 
        default="data/raw/test/", 
        help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for evaluation"
    )
    args = parser.parse_args()

    evaluate_model(args.model, args.test_dir, args.batch_size)


if __name__ == "__main__":
    main()
