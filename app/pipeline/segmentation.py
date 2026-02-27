"""
Human parsing and segmentation pipeline – Phase 2.
Now also extracts face bounding box for hard mask exclusion.
"""
import logging
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from app.models.loader import get_model

logger = logging.getLogger(__name__)

CLOTHING_LABELS = {4, 5, 6, 7, 8, 16, 17}
BODY_LABELS = {11, 12, 13, 14, 15}
FACE_LABEL = 11


class SegmentationResult(NamedTuple):
    clothing_mask: Image.Image
    body_mask: Image.Image
    face_mask: Image.Image
    face_bbox: tuple[int, int, int, int] | None  # (x1, y1, x2, y2) or None
    label_map: np.ndarray


def segment_person(person_image: Image.Image) -> SegmentationResult:
    processor = get_model("seg_processor")
    model = get_model("seg_model")
    device = get_model("device")

    original_size = person_image.size  # (W, H)

    inputs = processor(images=person_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    labels = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def _make_mask(label_set: set) -> Image.Image:
        binary = np.isin(labels, list(label_set)).astype(np.uint8) * 255
        return Image.fromarray(binary, mode="L")

    clothing_mask = _make_mask(CLOTHING_LABELS)
    body_mask = _make_mask(CLOTHING_LABELS | BODY_LABELS)
    face_mask = _make_mask({FACE_LABEL})

    # Extract face bounding box for hard mask exclusion
    face_bbox = _extract_face_bbox(labels, FACE_LABEL)
    logger.debug("Segmentation complete — face_bbox: %s", face_bbox)

    return SegmentationResult(
        clothing_mask=clothing_mask,
        body_mask=body_mask,
        face_mask=face_mask,
        face_bbox=face_bbox,
        label_map=labels,
    )


def _extract_face_bbox(
    labels: np.ndarray,
    face_label: int,
) -> tuple[int, int, int, int] | None:
    """Return (x1, y1, x2, y2) of face region from segmentation labels."""
    ys, xs = np.where(labels == face_label)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def get_face_bbox_insightface(
    person_image: Image.Image,
) -> tuple[int, int, int, int] | None:
    """
    Use InsightFace detector for a more precise face bounding box.
    Falls back to segmentation bbox if InsightFace not loaded.
    """
    try:
        face_app = get_model("face_app")
        if face_app is None:
            return None
        import numpy as np, cv2
        arr = cv2.cvtColor(np.array(person_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = face_app.get(arr)
        if not faces:
            return None
        # Return the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        return x1, y1, x2, y2
    except Exception as exc:
        logger.warning("InsightFace bbox failed: %s", exc)
        return None


def dilate_mask(mask: Image.Image, kernel_size: int = 15) -> Image.Image:
    import cv2
    arr = np.array(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated, mode="L")


def exclude_face_from_mask(
    mask: Image.Image,
    face_bbox: tuple[int, int, int, int] | None,
    padding: int = 30,
) -> Image.Image:
    """
    Zero out the face region in the inpaint mask.
    This is the primary identity preservation mechanism — diffusion never modifies this area.
    """
    if face_bbox is None:
        return mask
    arr = np.array(mask).copy()
    x1, y1, x2, y2 = face_bbox
    H, W = arr.shape
    # Apply padding with bounds clipping
    y1p = max(0, y1 - padding)
    y2p = min(H, y2 + padding)
    x1p = max(0, x1 - padding)
    x2p = min(W, x2 + padding)
    arr[y1p:y2p, x1p:x2p] = 0
    return Image.fromarray(arr, mode="L")
