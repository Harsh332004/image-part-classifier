"""Shared constants and preprocessing used by training, the CLI and the web app.

Keeping preprocessing in ONE place guarantees that what the model sees at
inference time is identical to what it saw during training (the original
project broke exactly because train.py used RGB and app.py used BGR).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 224
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "part_classifier.keras"
LABELS_PATH = MODEL_DIR / "labels.json"
METRICS_PATH = MODEL_DIR / "metrics.json"


def dataset_roots(path) -> list[Path]:
    """Resolve a dataset path into the folder(s) that directly hold class folders.

    Handles  <root>/<class>/*.png,  <root>/{training,testing}/<class>/*.png
    and one extra wrapper folder (e.g. archive/dataset/...).
    """
    root = Path(path)
    for _ in range(3):
        subs = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
        splits = [subs[n] for n in ("training", "train", "testing", "test", "val", "validation")
                  if n in subs]
        if splits:
            return splits
        if len(subs) == 1:          # wrapper folder like archive/dataset
            root = next(iter(subs.values()))
            continue
        return [root]
    return [root]


def part_id(file_path, class_name: str) -> str:
    """Identify the physical part a render belongs to.

    Files are named <part-name>_<view-number>.png, so removing the trailing
    view number groups all views of one part together.
    """
    base = re.sub(r"_\d+$", "", Path(file_path).stem)
    return f"{class_name}/{base}"


def load_rgb(source) -> Image.Image:
    """Open an image (path / file-like) as a clean RGB PIL image.

    * applies EXIF rotation (phone photos)
    * flattens transparency onto white instead of letting it become black
    """
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img.convert("RGB")


def _border_colour(img: Image.Image) -> tuple[int, int, int]:
    """Median colour of the image border, used to pad non-square images."""
    arr = np.asarray(img)
    border = np.concatenate(
        [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
    )
    return tuple(int(v) for v in np.median(border, axis=0))


def to_square(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Pad to a square (without distorting the part) and resize."""
    w, h = img.size
    side = max(w, h)
    if w != h:
        canvas = Image.new("RGB", (side, side), _border_colour(img))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    return img.resize((size, size), Image.Resampling.LANCZOS)


def load_artifacts():
    """Load the trained Keras model and the class-name list."""
    import tensorflow as tf  # imported lazily so common.py stays light

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found in '{MODEL_DIR}'. Run `python train.py` first."
        )
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    classes = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["classes"]
    return model, classes


def predict_proba(model, img: Image.Image, tta: bool = True) -> np.ndarray:
    """Return a probability vector for one PIL image.

    With ``tta`` (test-time augmentation) the prediction is averaged over the
    original and mirrored/flipped copies, which reduces variance for free.
    Pixel values stay in the 0-255 range: EfficientNet rescales internally.
    """
    arr = np.asarray(to_square(img), dtype=np.float32)
    views = [arr]
    if tta:
        views += [arr[:, ::-1, :], arr[::-1, :, :]]
    batch = np.stack(views, axis=0)
    probs = np.asarray(model(batch, training=False))
    return probs.mean(axis=0)
