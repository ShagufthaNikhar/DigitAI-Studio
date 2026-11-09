"""Model loading and image preprocessing utilities."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CNN_PATH = MODEL_DIR / "mnist_cnn.keras"
MLP_PATH = MODEL_DIR / "mnist_mlp.keras"


def load_cnn():
    if not CNN_PATH.exists():
        raise FileNotFoundError("Model missing. Run `python train.py` first.")
    return tf.keras.models.load_model(CNN_PATH)


def load_mlp():
    if not MLP_PATH.exists():
        raise FileNotFoundError("MLP model missing. Run `python train.py` first.")
    return tf.keras.models.load_model(MLP_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert a single dark handwritten digit on light paper into an MNIST-like
    28x28 tensor. This is a heuristic; clean, close-up images work best.
    """
    image = ImageOps.exif_transpose(image).convert("L")
    image = ImageOps.autocontrast(image)
    arr = np.array(image, dtype=np.uint8, copy=True)

    # Threshold dark ink. Ignore tiny specks, then keep the largest connected
    # component (often the digit). A thresholded crop reduces blank margins.
    mask = arr < 110
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for sy, sx in zip(*np.where(mask)):
        if visited[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        visited[sy, sx] = True
        points = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(points) >= 8:
            components.append(points)

    if not components:
        # Fallback: treat the whole image as an ink-on-paper image.
        cropped = image
        cropped = ImageOps.invert(cropped)
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(cropped, ((28-cropped.width)//2, (28-cropped.height)//2))
        return (np.asarray(canvas, dtype=np.float32) / 255.0)[None, ..., None]

    points = max(components, key=len)
    ys = [p[0] for p in points]
    xs = [p[1] for p in points]
    span = max(max(xs)-min(xs)+1, max(ys)-min(ys)+1)
    margin = max(3, int(span * 0.18))
    x1, x2 = max(0, min(xs)-margin), min(w, max(xs)+margin+1)
    y1, y2 = max(0, min(ys)-margin), min(h, max(ys)+margin+1)

    cropped = image.crop((x1, y1, x2, y2))
    cropped = ImageOps.invert(cropped)
    crop_arr = np.array(cropped, dtype=np.uint8, copy=True)
    crop_arr[crop_arr < 35] = 0
    cropped = Image.fromarray(crop_arr)
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, ((28-cropped.width)//2, (28-cropped.height)//2))
    normalized = np.asarray(canvas, dtype=np.float32) / 255.0
    return normalized[None, ..., None]


def preprocess_canvas(canvas_rgba: np.ndarray) -> np.ndarray:
    """Preprocess a drawable-canvas RGBA array with dark ink on light background."""
    rgb = np.asarray(canvas_rgba)[..., :3].astype(np.uint8)
    image = Image.fromarray(rgb).convert("L")
    return preprocess_image(image)
