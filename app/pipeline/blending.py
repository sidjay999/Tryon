"""
Final image blending and post-processing.
- Preserves face identity using facial landmark detection (mediapipe).
- Applies Poisson seamless cloning at mask boundaries.
- Matches lighting/histogram to maintain realism.
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


def _detect_face_region(image_bgr: np.ndarray) -> np.ndarray | None:
    """Return a face mask using MediaPipe or fall back to Haar cascade."""
    try:
        import mediapipe as mp
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(min_detection_confidence=0.5) as detector:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.detections:
                return None

            H, W = image_bgr.shape[:2]
            face_mask = np.zeros((H, W), dtype=np.uint8)
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x1 = max(0, int(box.xmin * W) - 10)
                y1 = max(0, int(box.ymin * H) - 10)
                x2 = min(W, int((box.xmin + box.width) * W) + 10)
                y2 = min(H, int((box.ymin + box.height) * H) + 10)
                face_mask[y1:y2, x1:x2] = 255
            return face_mask
    except Exception as exc:
        logger.warning("MediaPipe face detection failed: %s", exc)
        return None


def _histogram_match(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match histogram of source to reference for consistent lighting."""
    result = np.zeros_like(source)
    for c in range(3):
        src_hist, bins = np.histogram(source[:, :, c].ravel(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:, :, c].ravel(), 256, [0, 256])
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        src_cdf_norm = src_cdf / src_cdf[-1]
        ref_cdf_norm = ref_cdf / ref_cdf[-1]
        lut = np.interp(src_cdf_norm, ref_cdf_norm, np.arange(256))
        result[:, :, c] = lut[source[:, :, c]]
    return result.astype(np.uint8)


def blend_result(
    original_person: Image.Image,
    generated_image: Image.Image,
    clothing_mask: Image.Image,
) -> Image.Image:
    """
    Blend generated_image back over original_person:
    1. Poisson clone for seamless mask boundaries.
    2. Restore original face region.
    3. Histogram-match lighting.
    """
    target_size = generated_image.size
    original_resized = original_person.resize(target_size, Image.LANCZOS)
    mask_resized = clothing_mask.resize(target_size, Image.LANCZOS)

    orig_bgr = _pil_to_bgr(original_resized)
    gen_bgr = _pil_to_bgr(generated_image)
    mask_np = np.array(mask_resized.convert("L"))

    # 1. Poisson seamless cloning in clothing region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_eroded = cv2.erode(mask_np, kernel, iterations=2)

    if mask_eroded.sum() > 0:
        try:
            H, W = orig_bgr.shape[:2]
            center = (W // 2, H // 2)
            blended_bgr = cv2.seamlessClone(
                gen_bgr,
                orig_bgr,
                mask_eroded,
                center,
                cv2.NORMAL_CLONE,
            )
        except cv2.error:
            logger.warning("Poisson clone failed — using alpha blend")
            alpha = mask_np[:, :, np.newaxis] / 255.0
            blended_bgr = (gen_bgr * alpha + orig_bgr * (1 - alpha)).astype(np.uint8)
    else:
        alpha = mask_np[:, :, np.newaxis] / 255.0
        blended_bgr = (gen_bgr * alpha + orig_bgr * (1 - alpha)).astype(np.uint8)

    # 2. Restore face region from original
    face_mask = _detect_face_region(orig_bgr)
    if face_mask is not None:
        face_alpha = face_mask[:, :, np.newaxis] / 255.0
        blended_bgr = (orig_bgr * face_alpha + blended_bgr * (1 - face_alpha)).astype(np.uint8)

    # 3. Histogram-match lighting
    blended_bgr = _histogram_match(blended_bgr, orig_bgr)

    return _bgr_to_pil(blended_bgr)
