import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.logger import get_logger

logger = get_logger("train_pipeline")

IMG_SIZE     = (224, 224)
SEED         = 42
DROPOUT      = 0.45
L2_REG       = 1e-4
LABEL_SMOOTH = 0.10
MIXUP_ALPHA  = 0.30
FINE_TUNE_AT = 120

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_datasets(
    data_dir: str,
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], Dict[int, float]]:
    train_path = Path(data_dir) / "train"
    val_path   = Path(data_dir) / "valid"

    for p in [train_path, val_path]:
        if not p.is_dir():
            logger.error(f"Folder tidak ditemukan: {p}")
            sys.exit(1)

    kw = dict(seed=SEED, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical")

    train_raw = tf.keras.utils.image_dataset_from_directory(
        str(train_path), shuffle=True, **kw)
    val_raw   = tf.keras.utils.image_dataset_from_directory(
        str(val_path), shuffle=False, **kw)

    class_names = train_raw.class_names
    logger.info(f"Kelas ({len(class_names)}): {class_names}")

    all_labels = np.concatenate([
        np.argmax(y.numpy(), axis=1) for _, y in train_raw
    ])
    cw_vals           = compute_class_weight("balanced",
                                             classes=np.unique(all_labels),
                                             y=all_labels)
    class_weight_dict = dict(enumerate(cw_vals.tolist()))
    logger.info(f"Class Weights: { {class_names[k]: round(v,3) for k,v in class_weight_dict.items()} }")

    AUTOTUNE  = tf.data.AUTOTUNE
    train_ds  = (train_raw
                 .shuffle(min(2000, len(all_labels)), seed=SEED, reshuffle_each_iteration=True)
                 .prefetch(AUTOTUNE))
    val_ds    = val_raw.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, class_weight_dict


@tf.function(jit_compile=False)
def mixup_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    alpha     = tf.constant(MIXUP_ALPHA, dtype=tf.float32)
    rng_seed  = tf.random.uniform([2], minval=0, maxval=2**31 - 1, dtype=tf.int32)
    g1        = tf.random.stateless_gamma(shape=[1], seed=rng_seed,     alpha=alpha)
    g2        = tf.random.stateless_gamma(shape=[1], seed=rng_seed + 1, alpha=alpha)
    lam       = tf.squeeze(g1 / (g1 + g2 + 1e-8))
    lam       = tf.clip_by_value(lam, 0.3, 0.7)
    lam       = tf.cast(lam, tf.float32)

    batch_size = tf.shape(images)[0]
    perm       = tf.random.shuffle(tf.range(batch_size))

    img_f   = tf.cast(images, tf.float32)
    mixed_x = lam * img_f + (1.0 - lam) * tf.gather(img_f, perm)
    lbl_f   = tf.cast(labels, tf.float32)
    mixed_y = lam * lbl_f + (1.0 - lam) * tf.gather(lbl_f, perm)
    return mixed_x, mixed_y


def build_model(num_classes: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
    logger.info(f"Membangun model untuk {num_classes} kelas.")

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.20),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
    ], name="gpu_augment")

    base = EfficientNetB0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    base._name     = "backbone"

    inp = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x   = augment(inp)
    x   = base(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_0")(x)
    x = layers.Dropout(DROPOUT, name="drop_0")(x)

    x = layers.Dense(
            512, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="dense_512",
        )(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(DROPOUT * 0.80, name="drop_1")(x)

    x = layers.Dense(
            256, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="dense_256",
        )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(DROPOUT * 0.55, name="drop_2")(x)

    out = layers.Dense(
              num_classes, activation="softmax",
              dtype="float32", name="predictions",
          )(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="DurianClassifier")
    return model, base


def make_callbacks(phase: int, save_path: str) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10 if phase == 2 else 7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(PROJECT_ROOT / f"training_phase{phase}.csv"),
            append=False,
        ),
    ]


def plot_history(h1, h2, output_dir: Path) -> None:
    def merge(key):
        return h1.history[key] + h2.history[key]

    acc, val_acc = merge("accuracy"), merge("val_accuracy")
    loss, vloss  = merge("loss"),     merge("val_loss")
    ep    = range(1, len(acc) + 1)
    split = len(h1.history["accuracy"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Durian EfficientNetB0", fontsize=13, fontweight="bold")

    ax[0].plot(ep, acc,     "b-", lw=2, label="Train")
    ax[0].plot(ep, val_acc, "r-", lw=2, label="Val")
    ax[0].axvline(split, color="gray", ls="--", alpha=0.7, label=f"Phase1→2 (ep{split})")
    ax[0].fill_between(ep, acc, val_acc,
                       where=[a > v for a, v in zip(acc, val_acc)],
                       alpha=0.10, color="red")
    ax[0].set(title="Accuracy", xlabel="Epoch", ylim=(0, 1.05))
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, loss,  "b-", lw=2, label="Train")
    ax[1].plot(ep, vloss, "r-", lw=2, label="Val")
    ax[1].axvline(split, color="gray", ls="--", alpha=0.7)
    ax[1].set(title="Loss", xlabel="Epoch")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "training_history.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"History plot disimpan: {out}")

    gap      = acc[-1] - val_acc[-1]
    best_val = max(val_acc)
    logger.info(f"Train Acc={acc[-1]:.4f}  Val Acc={val_acc[-1]:.4f}  "
                f"Best Val={best_val:.4f}  Gap={gap:.4f}")
    if gap > 0.15:
        logger.warning("OVERFITTING terdeteksi — naikkan DROPOUT atau LABEL_SMOOTH.")
    elif best_val < 0.65:
        logger.warning("UNDERFITTING — kurangi FINE_TUNE_AT atau naikkan kapasitas head.")
    else:
        logger.info("Model SEHAT.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Durian Classification Model")
    parser.add_argument("--data_dir",    type=str, default="data/raw")
    parser.add_argument("--epochs_p1",  type=int, default=20,
                        help="Epochs Phase 1 (backbone frozen)")
    parser.add_argument("--epochs_p2",  type=int, default=50,
                        help="Epochs Phase 2 (fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mixed_prec", action="store_true", default=False,
                        help="Aktifkan mixed precision float16 (butuh GPU)")
    args = parser.parse_args()

    if args.mixed_prec and tf.config.list_physical_devices("GPU"):
        mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision float16 aktif.")
    else:
        mixed_precision.set_global_policy("float32")
        logger.info("Mixed precision TIDAK aktif (float32).")

    data_dir   = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / "models" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_path  = str(output_dir / "best_model.keras")
    final_path = str(output_dir / "efficientnet_b0_durian.keras")

    train_ds, val_ds, class_names, class_weights = load_datasets(
        str(data_dir), batch_size=args.batch_size)
    num_classes = len(class_names)

    AUTOTUNE       = tf.data.AUTOTUNE
    train_ds_mixup = train_ds.map(mixup_batch, num_parallel_calls=AUTOTUNE)

    model, backbone = build_model(num_classes)
    model.summary(line_length=90)

    logger.info("=" * 60)
    logger.info("  PHASE 1 — Feature Extraction (backbone FROZEN)")
    logger.info("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )

    history_p1 = model.fit(
        train_ds_mixup,
        validation_data=val_ds,
        epochs=args.epochs_p1,
        class_weight=class_weights,
        callbacks=make_callbacks(1, best_path),
        verbose=1,
    )
    logger.info(f"Phase 1 selesai. Best Val Acc: {max(history_p1.history['val_accuracy']):.4f}")

    logger.info("=" * 60)
    logger.info(f"  PHASE 2 — Fine-Tuning (unfreeze dari layer {FINE_TUNE_AT})")
    logger.info("=" * 60)

    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    n_trainable = sum(1 for l in backbone.layers if l.trainable)
    logger.info(f"Trainable backbone layers: {n_trainable} / {len(backbone.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy",
                 tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
    )

    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_p2,
        class_weight=class_weights,
        callbacks=make_callbacks(2, best_path),
        verbose=1,
    )
    logger.info(f"Phase 2 selesai. Best Val Acc: {max(history_p2.history['val_accuracy']):.4f}")

    model.save(final_path)
    logger.info(f"Model final disimpan: {final_path}")
    logger.info(f"Best checkpoint     : {best_path}")
    logger.info(f"Ukuran model        : {Path(final_path).stat().st_size / 1e6:.1f} MB")

    plot_history(history_p1, history_p2, PROJECT_ROOT)

    logger.info("Training selesai.")
    logger.info(f"Jalankan export: python pipelines/export_to_onnx.py --input {best_path}")


if __name__ == "__main__":
    main()