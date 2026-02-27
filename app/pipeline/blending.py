"""
Blending – Phase 2 (simplified).
The face is now hard-protected upstream (mask exclusion + IP-Adapter FaceID).
This module acts as a safety net:
  - Pastes original face back at full opacity in the face region
  - Applies Poisson cloning at garment mask boundaries
  - Matches lighting histogram
"""
import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.zeros_like(source)
    for c in range(3):
        src_cdf = np.histogram(source[:, :, c].ravel(), 256, [0, 256])[0].cumsum()
        ref_cdf = np.histogram(reference[:, :, c].ravel(), 256, [0, 256])[0].cumsum()
        src_cdf_n = src_cdf / src_cdf[-1]
        ref_cdf_n = ref_cdf / ref_cdf[-1]
        lut = np.interp(src_cdf_n, ref_cdf_n, np.arange(256))
        result[:, :, c] = lut[source[:, :, c]]
    return result.astype(np.uint8)


def blend_result(
    original_person: Image.Image,
    generated_image: Image.Image,
    clothing_mask: Image.Image,
    face_bbox: tuple[int, int, int, int] | None = None,
    face_padding: int = 30,
) -> Image.Image:
    """
    Phase 2 blend:
      1. Poisson clone at clothing mask boundaries for seamless edges
      2. Hard-paste original face back (safety net on top of hard mask + FaceID)
      3. Histogram-match lighting
    """
    target_size = generated_image.size
    orig_bgr = _pil_to_bgr(original_person.resize(target_size, Image.LANCZOS))
    gen_bgr = _pil_to_bgr(generated_image)
    mask_np = np.array(clothing_mask.resize(target_size, Image.LANCZOS).convert("L"))

    # 1. Poisson seamless cloning on clothing boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_eroded = cv2.erode(mask_np, kernel, iterations=2)
    if mask_eroded.sum() > 0:
        try:
            H, W = orig_bgr.shape[:2]
            center = (W // 2, H // 2)
            blended = cv2.seamlessClone(gen_bgr, orig_bgr, mask_eroded, center, cv2.NORMAL_CLONE)
        except cv2.error:
            alpha = mask_np[:, :, np.newaxis] / 255.0
            blended = (gen_bgr * alpha + orig_bgr * (1 - alpha)).astype(np.uint8)
    else:
        alpha = mask_np[:, :, np.newaxis] / 255.0
        blended = (gen_bgr * alpha + orig_bgr * (1 - alpha)).astype(np.uint8)

    # 2. Safety-net: hard-paste original face region back
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        H, W = blended.shape[:2]
        y1p = max(0, y1 - face_padding)
        y2p = min(H, y2 + face_padding)
        x1p = max(0, x1 - face_padding)
        x2p = min(W, x2 + face_padding)

        # Soft feathered paste using Gaussian-blended border
        face_region = orig_bgr[y1p:y2p, x1p:x2p]
        fh, fw = face_region.shape[:2]
        feather = cv2.GaussianBlur(
            np.ones((fh, fw), dtype=np.float32),
            (min(fw // 4 * 2 + 1, 51), min(fh // 4 * 2 + 1, 51)),
            0,
        )
        feather /= feather.max()
        f3 = feather[:, :, np.newaxis]
        blended[y1p:y2p, x1p:x2p] = (
            face_region * f3 + blended[y1p:y2p, x1p:x2p] * (1 - f3)
        ).astype(np.uint8)

    # 3. Histogram-match lighting to original
    blended = _histogram_match(blended, orig_bgr)

    return _bgr_to_pil(blended)
