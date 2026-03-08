"""
Garment image preprocessing – Phase 2.
Removes background, crops to garment region, centers and resizes.
"""
import logging

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Target size for CatVTON (width, height)
TARGET_SIZE = (768, 1024)


def preprocess_garment(
    image: Image.Image,
    enable_bg_removal: bool = True,
) -> dict:
    """
    Standardize a garment image for CatVTON inference.

    Pipeline:
      1. Fix EXIF rotation
      2. Remove background (rembg / U2Net)
      3. Crop to garment bounding box
      4. Center on white canvas
      5. Resize + pad to 768×1024

    Args:
        image: Raw PIL image from upload.
        enable_bg_removal: Whether to run background removal.

    Returns:
        dict with:
          - "image": preprocessed PIL Image (768×1024)
          - "original_size": (w, h) of input
          - "bg_removed": bool
    """
    # ── 1. Fix EXIF rotation ───────────────────────────────────
    image = ImageOps.exif_transpose(image)
    original_size = image.size

    meta = {
        "original_size": original_size,
        "bg_removed": False,
    }

    # ── 2. Background removal ──────────────────────────────────
    if enable_bg_removal:
        try:
            result = _remove_background(image)
            if result is not None:
                image = result
                meta["bg_removed"] = True
                logger.info("Garment background removed")
        except Exception as exc:
            logger.warning("Background removal failed: %s. Using original.", exc)

    # ── 3. Crop to garment region ──────────────────────────────
    image = _crop_to_content(image)

    # ── 4. Center on white canvas + resize ─────────────────────
    image = _center_and_resize(image, TARGET_SIZE)
    meta["image"] = image

    return meta


def _remove_background(image: Image.Image) -> Image.Image | None:
    """Remove background using rembg (U2Net). Returns RGBA image or None."""
    try:
        from rembg import remove
    except ImportError:
        logger.warning(
            "rembg not installed. Run: pip install rembg[gpu]"
        )
        return None

    # rembg returns RGBA with transparent background
    result = remove(image)

    # Convert transparent → white background
    if result.mode == "RGBA":
        white_bg = Image.new("RGB", result.size, (255, 255, 255))
        white_bg.paste(result, mask=result.split()[3])
        return white_bg

    return result.convert("RGB")


def _crop_to_content(image: Image.Image) -> Image.Image:
    """Crop to the non-white bounding box of the garment."""
    img_arr = np.array(image)

    # Find non-white pixels (threshold: < 240 in any channel)
    if img_arr.ndim == 3:
        mask = np.any(img_arr < 240, axis=2)
    else:
        mask = img_arr < 240

    if not mask.any():
        # All white — return as-is
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add small padding (5% of dimensions)
    h, w = img_arr.shape[:2]
    pad_x = max(int(w * 0.05), 5)
    pad_y = max(int(h * 0.05), 5)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)

    return image.crop((x_min, y_min, x_max, y_max))


def _center_and_resize(
    image: Image.Image,
    target_size: tuple,
) -> Image.Image:
    """Resize garment maintaining aspect ratio, center on white canvas."""
    tw, th = target_size

    # Resize to fit within target (leave some margin)
    margin = 0.9  # use 90% of canvas
    max_w = int(tw * margin)
    max_h = int(th * margin)
    image.thumbnail((max_w, max_h), Image.LANCZOS)

    # Create white canvas and paste centered
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    x_off = (tw - image.width) // 2
    y_off = (th - image.height) // 2
    canvas.paste(image, (x_off, y_off))

    return canvas
