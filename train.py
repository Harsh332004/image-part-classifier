"""Train a machine-part image classifier (transfer learning + fine-tuning).

Pipeline
--------
1. Collect images, drop exact duplicate files (prevents train/test leakage).
2. Stratified train / validation / test split (70 / 15 / 15).
3. Phase 1 - "warm-up": freeze EfficientNetB0, train only the new head.
4. Phase 2 - "fine-tune": unfreeze the top of the backbone with a tiny
   learning rate (BatchNorm layers stay frozen).
5. Evaluate ONCE on the untouched test set, save model + metrics + plots.

Usage
-----
    python train.py --data_dir "C:/path/to/blnw-images-224"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from common import (IMG_SIZE, LABELS_PATH, METRICS_PATH, MODEL_DIR, MODEL_PATH,
                    dataset_roots, load_rgb, part_id, to_square)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_DATA_DIR = os.environ.get(
    "PART_DATA_DIR", r"C:/Users/hp340/OneDrive/Desktop/archive/dataset"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                   help="Folder with one sub-folder per class.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--head_epochs", type=int, default=8)
    p.add_argument("--finetune_epochs", type=int, default=25)
    p.add_argument("--unfreeze_layers", type=int, default=60,
                   help="How many top backbone layers to fine-tune.")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_finetune", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def collect_files(data_dir: str):
    """Gather images from every split folder, dropping exact duplicate files.

    The dataset ships with training/ and testing/ folders that contain the SAME
    parts (and some byte-identical files), so we merge them and make our own
    leak-free split instead of trusting the shipped one.
    """
    roots = dataset_roots(data_dir)
    print("Reading from:", [str(r) for r in roots])
    classes = sorted({p.name for r in roots for p in r.iterdir()
                      if p.is_dir() and not p.name.startswith(".")})
    if len(classes) < 2:
        raise SystemExit(f"Need >= 2 class sub-folders, found {classes}.")

    paths, labels, groups, seen, dupes = [], [], [], set(), 0
    for idx, cls in enumerate(classes):
        for root in roots:
            if not (root / cls).is_dir():
                continue
            for f in sorted((root / cls).rglob("*")):
                if f.suffix.lower() not in EXTS:
                    continue
                digest = hashlib.md5(f.read_bytes()).hexdigest()
                if digest in seen:      # byte-identical file -> would leak
                    dupes += 1
                    continue
                seen.add(digest)
                paths.append(str(f))
                labels.append(idx)
                groups.append(part_id(f, cls))

    print(f"Classes: {classes}")
    print(f"Images kept: {len(paths)}  (exact duplicates removed: {dupes})")
    print(f"Distinct parts: {len(set(groups))}")
    for i, cls in enumerate(classes):
        print(f"  {cls:<15} images={labels.count(i):<6} "
              f"parts={len({g for g, l in zip(groups, labels) if l == i})}")
    return np.array(paths), np.array(labels), np.array(groups), classes


def _load_py(path):
    """Identical preprocessing to the web app (common.load_rgb + to_square)."""
    img = to_square(load_rgb(path.decode("utf-8")))
    return np.asarray(img, dtype=np.uint8)


def make_dataset(paths, labels, num_classes, batch_size, training, seed):
    def load(path, label):
        img = tf.numpy_function(_load_py, [path], tf.uint8)
        img = tf.cast(img, tf.float32)            # stays 0-255, model rescales
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((list(paths), np.asarray(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int, dropout: float):
    L = tf.keras.layers
    augmentation = tf.keras.Sequential(
        [
            L.RandomFlip("horizontal_and_vertical"),
            L.RandomRotation(0.15, fill_mode="reflect"),
            L.RandomZoom((-0.2, 0.2), fill_mode="reflect"),
            L.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            L.RandomContrast(0.2),
            L.RandomBrightness(0.2, value_range=(0, 255)),
        ],
        name="augmentation",  # only active during training
    )
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    backbone.trainable = False

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = backbone(x, training=False)       # keep BatchNorm in inference mode
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(num_classes, activation="softmax", dtype="float32",
                      name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="part_classifier"), backbone


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=3, min_lr=1e-7, verbose=1),
    ]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def plot_history(histories, path):
    acc = sum((h.history["accuracy"] for h in histories), [])
    val_acc = sum((h.history["val_accuracy"] for h in histories), [])
    loss = sum((h.history["loss"] for h in histories), [])
    val_loss = sum((h.history["val_loss"] for h in histories), [])
    boundary = len(histories[0].history["loss"]) - 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, tr, va, name in ((ax[0], acc, val_acc, "Accuracy"),
                            (ax[1], loss, val_loss, "Loss")):
        a.plot(tr, label="train")
        a.plot(va, label="validation")
        a.axvline(boundary, ls="--", c="grey", label="fine-tune starts")
        a.set_title(name)
        a.set_xlabel("epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 1.4 * len(classes) + 2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    MODEL_DIR.mkdir(exist_ok=True)

    paths, labels, groups, classes = collect_files(args.data_dir)
    n_cls = len(classes)

    # Group-aware split: all views of one part land in exactly one subset.
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    tmp_idx, test_idx = next(outer.split(paths, labels, groups))
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed)
    tr_rel, va_rel = next(inner.split(paths[tmp_idx], labels[tmp_idx], groups[tmp_idx]))
    train_idx, val_idx = tmp_idx[tr_rel], tmp_idx[va_rel]

    assert not (set(groups[train_idx]) & set(groups[test_idx])), "part leakage train/test"
    assert not (set(groups[train_idx]) & set(groups[val_idx])), "part leakage train/val"
    assert not (set(groups[val_idx]) & set(groups[test_idx])), "part leakage val/test"

    p_train, y_train = paths[train_idx], labels[train_idx]
    p_val, y_val = paths[val_idx], labels[val_idx]
    p_test, y_test = paths[test_idx], labels[test_idx]
    print(f"\nSplit (by part, no leakage) -> train {len(p_train)} | "
          f"val {len(p_val)} | test {len(p_test)}")

    train_ds = make_dataset(p_train, y_train, n_cls, args.batch_size, True, args.seed)
    val_ds = make_dataset(p_val, y_val, n_cls, args.batch_size, False, args.seed)
    test_ds = make_dataset(p_test, y_test, n_cls, args.batch_size, False, args.seed)

    weights = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("Class weights:", {classes[i]: round(w, 2) for i, w in class_weight.items()})

    model, backbone = build_model(n_cls, args.dropout)

    print("\n=== Phase 1: training the new head (backbone frozen) ===")
    compile_model(model, args.lr_head)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                   class_weight=class_weight, callbacks=callbacks())

    print("\n=== Phase 2: fine-tuning top backbone layers ===")
    backbone.trainable = True
    for layer in backbone.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    for layer in backbone.layers:          # never update BatchNorm statistics
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, args.lr_finetune)  # must recompile after changing trainable
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                   class_weight=class_weight, callbacks=callbacks())

    print("\n=== Final evaluation on the held-out TEST set ===")
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(1)
    report = classification_report(y_test, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    wrong = np.where(y_pred != y_test)[0]
    if len(wrong):
        print("\nMost confidently WRONG test images (check for label noise):")
        for i in wrong[np.argsort(-probs[wrong].max(1))][:10]:
            print(f"  {p_test[i]}  true={classes[y_test[i]]}  "
                  f"pred={classes[y_pred[i]]}  conf={probs[i].max():.2f}")

    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps({"classes": classes}, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps({
        "test_accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": {c: report[c] for c in classes},
        "n_train": int(len(p_train)), "n_val": int(len(p_val)),
        "n_test": int(len(p_test)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
    }, indent=2), encoding="utf-8")
    plot_history([h1, h2], MODEL_DIR / "training_curves.png")
    plot_confusion(cm, classes, MODEL_DIR / "confusion_matrix.png")
    print(f"\nSaved model + reports to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
