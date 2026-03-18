"""
Standalone training script for the Durian Classification ML Model (8 Classes).
Optimized for EfficientNetB0 with Fine-Tuning and Data Augmentation.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("train_pipeline")

def get_data_generators(
    data_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """Load datasets from train and valid subdirectories for all 8 classes."""
    
    # Path spesifik ke sub-folder hasil export Roboflow
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "valid")
    
    logger.info(f"Loading training data from: {train_path}")
    logger.info(f"Loading validation data from: {val_path}")

    # Load Training Dataset (Otomatis membaca semua sub-folder sebagai kelas)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        seed=42,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Load Validation Dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        seed=42,
        image_size=target_size,
        batch_size=batch_size,
        label_mode="categorical"
    )
    
    class_names = train_ds.class_names
    logger.info(f"Successfully loaded {len(class_names)} classes: {class_names}")

    # Optimasi performa dataset
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3), 
    num_classes: int = 8
) -> tf.keras.Model:
    """Build EfficientNetB0 with Data Augmentation and Fine-Tuning enabled."""
    logger.info(f"Building model for {num_classes} classes with Fine-Tuning.")
    
    # 1. Data Augmentation Layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="augmentation_layer")

    # 2. Base Model (EfficientNetB0)
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    
    # Unfreeze base model for fine-tuning
    base_model.trainable = True

    # 3. Model Architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = base_model(x, training=False) # BN layers in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile with a slightly lower learning rate for Fine-Tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def plot_history(history, output_dir: Path):
    """Saves training_history.png for Bab 4 Report."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Accuracy History')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    
    plt.savefig(output_dir / "training_history.png")
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    data_dir_path = PROJECT_ROOT / args.data_dir
    output_filepath = PROJECT_ROOT / "models" / "weights" / "efficientnet_b0.h5"
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Execute Pipeline
    train_ds, val_ds, class_names = get_data_generators(str(data_dir_path), batch_size=args.batch_size)
    
    model = build_model(num_classes=len(class_names))
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(output_filepath), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
    ]

    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(str(output_filepath))
    plot_history(history, output_filepath.parent.parent.parent) # Simpan di root folder
    logger.info("Training finished. Graphs and Model saved.")

if __name__ == "__main__":
    main()