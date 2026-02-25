"""
Human parsing and segmentation pipeline.
Uses Segformer B2 Clothes to extract:
  - Full body mask
  - Upper-body clothing mask
  - Lower-body clothing mask
"""
import logging
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from app.models.loader import get_model

logger = logging.getLogger(__name__)

# Segformer B2 Clothes label map
LABEL_MAP = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

CLOTHING_LABELS = {4, 5, 6, 7, 8, 16, 17}   # everything clothing / accessories
BODY_LABELS = {11, 12, 13, 14, 15}            # face + limbs (preserve identity)


class SegmentationResult(NamedTuple):
    clothing_mask: Image.Image    # white = clothing region to replace
    body_mask: Image.Image        # white = full human silhouette
    face_mask: Image.Image        # white = face region (preserved)
    label_map: np.ndarray         # raw per-pixel label array


def segment_person(person_image: Image.Image) -> SegmentationResult:
    """
    Run Segformer on person_image.
    Returns binary masks (PIL, mode='L', 0/255) for clothing, body, and face.
    """
    processor = get_model("seg_processor")
    model = get_model("seg_model")
    device = get_model("device")

    original_size = person_image.size  # (W, H)

    inputs = processor(images=person_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits                       # (1, num_labels, H', W')
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(original_size[1], original_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    labels = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def _make_mask(label_set: set) -> Image.Image:
        binary = np.isin(labels, list(label_set)).astype(np.uint8) * 255
        return Image.fromarray(binary, mode="L")

    clothing_mask = _make_mask(CLOTHING_LABELS)
    body_mask = _make_mask(CLOTHING_LABELS | BODY_LABELS)
    face_mask = _make_mask({11})  # face only

    logger.debug("Segmentation complete — unique labels: %s", np.unique(labels).tolist())

    return SegmentationResult(
        clothing_mask=clothing_mask,
        body_mask=body_mask,
        face_mask=face_mask,
        label_map=labels,
    )


def dilate_mask(mask: Image.Image, kernel_size: int = 15) -> Image.Image:
    """Slightly expand the mask edges for smoother inpainting transitions."""
    import cv2

    arr = np.array(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated, mode="L")
