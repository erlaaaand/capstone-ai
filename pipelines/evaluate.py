"""
Standalone evaluation script untuk Durian Classification Model.

Load model Keras (.keras atau .h5), evaluasi terhadap test/valid dataset,
hasilkan classification report dan confusion matrix.

Konsisten dengan pipeline training:
  - EfficientNetB0 dengan two-phase training
  - Input pixel [0, 255] float32 (normalisasi built-in di backbone)
  - 8 kelas: D101, D13, D197, D198, D2, D200, D24, D88
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
    class_names: list,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """
    Plot dan simpan confusion matrix.

    Args:
        cm: Confusion matrix array.
        class_names: Nama kelas.
        output_path: Path simpan gambar.
        normalize: Jika True, tampilkan sebagai proporsi (0.0–1.0).
    """
    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        cm_plot = cm
        fmt, title = "d", "Confusion Matrix (Raw Counts)"

    n = len(class_names)
    fig_size = max(10, n * 1.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        vmin=0.0 if normalize else None,
        vmax=1.0 if normalize else None,
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix disimpan: {output_path}")


def evaluate_model(
    model_path: str,
    test_data_dir: str,
    batch_size: int = 32,
    use_tta: bool = False,
    n_tta: int = 3,
) -> None:
    """
    Load model Keras dan evaluasi terhadap dataset.

    Args:
        model_path: Path ke model Keras (.keras atau .h5).
        test_data_dir: Path ke folder gambar test (subfolder = kelas).
        batch_size: Batch size inferensi.
        use_tta: Jika True, gunakan Test-Time Augmentation.
        n_tta: Jumlah augmentasi TTA (aktif jika use_tta=True).
    """
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = PROJECT_ROOT / model_path

    test_dir = Path(test_data_dir)
    if not test_dir.is_absolute():
        test_dir = PROJECT_ROOT / test_data_dir

    # ── Validasi path ─────────────────────────────────────────────────────────
    if not model_file.exists():
        logger.error(f"Model tidak ditemukan: {model_file}")
        sys.exit(1)
    if not test_dir.exists():
        logger.error(f"Folder test tidak ditemukan: {test_dir}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model dari: {model_file}")
    try:
        model = tf.keras.models.load_model(str(model_file))
        logger.info("Model berhasil di-load.")
    except Exception as e:
        logger.error(f"Gagal load model: {str(e)}")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset dari: {test_dir}")
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(test_dir),
            labels="inferred",
            label_mode="categorical",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=(224, 224),
            shuffle=False,   # WAJIB False agar label selaras dengan prediksi
        )
    except Exception as e:
        logger.error(f"Gagal load dataset: {str(e)}")
        sys.exit(1)

    class_names = test_ds.class_names
    logger.info(f"Kelas ditemukan ({len(class_names)}): {class_names}")

    # ── Optimasi pipeline ─────────────────────────────────────────────────────
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # ── Kumpulkan true labels ─────────────────────────────────────────────────
    logger.info("Mengumpulkan true labels...")
    true_labels = np.concatenate([
        np.argmax(labels.numpy(), axis=1) for _, labels in test_ds
    ])

    # ── Prediksi ──────────────────────────────────────────────────────────────
    if use_tta:
        logger.info(f"Menjalankan prediksi dengan TTA (n={n_tta})...")
        tta_aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
        ])

        all_probs = []
        for imgs, _ in test_ds:
            batch_probs = model(imgs, training=False).numpy()
            for _ in range(n_tta):
                aug_imgs    = tta_aug(imgs, training=True)
                batch_probs = batch_probs + model(aug_imgs, training=False).numpy()
            batch_probs /= (n_tta + 1)
            all_probs.append(batch_probs)

        predictions = np.concatenate(all_probs, axis=0)
    else:
        logger.info("Menjalankan prediksi standar...")
        predictions = model.predict(test_ds, verbose=1)

    pred_labels = np.argmax(predictions, axis=1)

    # ── Classification Report ─────────────────────────────────────────────────
    overall_acc = np.mean(true_labels == pred_labels)
    logger.info("=" * 65)
    logger.info("  CLASSIFICATION REPORT")
    logger.info("=" * 65)
    logger.info(f"  Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    logger.info("")

    report = classification_report(
        true_labels, pred_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    for line in report.split("\n"):
        if line.strip():
            logger.info(line)
    logger.info("=" * 65)

    # ── Simpan grafik ─────────────────────────────────────────────────────────
    plot_dir = PROJECT_ROOT / "models" / "evaluation"
    plot_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(true_labels, pred_labels)

    # Normalized
    plot_confusion_matrix(
        cm, class_names,
        output_path=plot_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    # Raw counts
    plot_confusion_matrix(
        cm, class_names,
        output_path=plot_dir / "confusion_matrix_counts.png",
        normalize=False,
    )

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    plt.figure(figsize=(max(10, len(class_names) * 1.5), 5))
    bars = plt.bar(class_names, per_class_acc * 100, color="steelblue", edgecolor="white")
    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc*100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    plt.title("Per-Class Accuracy", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(str(plot_dir / "per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class accuracy chart disimpan: {plot_dir / 'per_class_accuracy.png'}")

    logger.info(f"Semua hasil evaluasi disimpan di: {plot_dir}")


def main() -> None:
    """Execute evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Evaluasi Durian Classification Model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best_model.keras",    # .keras (Keras 3.x native)
        help="Path ke trained Keras model (.keras atau .h5)",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/raw/test",
        help="Path ke folder test dataset (subfolder = nama kelas)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size untuk evaluasi",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Gunakan Test-Time Augmentation untuk akurasi lebih tinggi",
    )
    parser.add_argument(
        "--n_tta",
        type=int,
        default=3,
        help="Jumlah augmentasi TTA (default: 3)",
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_data_dir=args.test_dir,
        batch_size=args.batch_size,
        use_tta=args.tta,
        n_tta=args.n_tta,
    )


if __name__ == "__main__":
    main()