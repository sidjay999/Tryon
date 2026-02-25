"""
Image preprocessing utilities.
"""
import base64
import io
import logging
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_MB = 20


async def load_image_from_upload(upload: UploadFile) -> Image.Image:
    """Read an UploadFile and return a PIL Image (RGB)."""
    if upload.content_type not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported image type: {upload.content_type}")
    data = await upload.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise ValueError(f"Image exceeds {MAX_MB}MB limit")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def resize_to_square(image: Image.Image, size: int = 1024) -> Image.Image:
    """Pad and resize image to exactly (size x size) maintaining aspect ratio."""
    image = ImageOps.exif_transpose(image)  # fix EXIF rotation
    image.thumbnail((size, size), Image.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    x_off = (size - image.width) // 2
    y_off = (size - image.height) // 2
    square.paste(image, (x_off, y_off))
    return square


def encode_image_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 string (fallback when S3 unavailable)."""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()
