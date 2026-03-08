"""
Person image preprocessing – Phase 2 (lightweight).
EXIF fix, center crop to portrait, resize to model input. All CPU, no GPU load.
"""
import logging

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Target size for CatVTON (width, height)
TARGET_SIZE = (768, 1024)


def preprocess_person(image: Image.Image) -> dict:
    """
    Standardize a person image for CatVTON inference.
    Lightweight — no GPU models loaded. All CPU operations.

    Pipeline:
      1. Fix EXIF rotation
      2. Center crop to 3:4 portrait ratio
      3. Resize + white-pad to 768×1024

    Returns:
        dict with:
          - "image": preprocessed PIL Image (768×1024)
          - "original_size": (w, h)
    """
    # ── 1. Fix EXIF rotation ───────────────────────────────────
    image = ImageOps.exif_transpose(image)
    original_size = image.size  # (w, h)

    # ── 2. Center crop to portrait ratio ───────────────────────
    image = _center_crop_portrait(image)

    # ── 3. Resize + pad to target ──────────────────────────────
    image = _resize_and_pad(image, TARGET_SIZE)

    logger.info("Person preprocessed: %s → 768×1024", original_size)

    return {
        "image": image,
        "original_size": original_size,
    }


def _center_crop_portrait(image: Image.Image) -> Image.Image:
    """Center-crop to ~3:4 portrait aspect ratio."""
    w, h = image.size
    target_ratio = 3 / 4  # width / height

    current_ratio = w / h
    if current_ratio > target_ratio:
        # Too wide — crop width
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        image = image.crop((offset, 0, offset + new_w, h))
    elif current_ratio < target_ratio * 0.8:
        # Way too tall — crop height a bit
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        image = image.crop((0, offset, w, offset + new_h))

    return image


def _resize_and_pad(image: Image.Image, target_size: tuple) -> Image.Image:
    """Resize maintaining aspect ratio, then white-pad to exact target size."""
    tw, th = target_size
    image.thumbnail((tw, th), Image.LANCZOS)

    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    x_off = (tw - image.width) // 2
    y_off = (th - image.height) // 2
    canvas.paste(image, (x_off, y_off))

    return canvas
