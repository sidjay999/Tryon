"""
Clothing warping module.
Aligns and warps the clothing image to match the person's body pose using:
  1. Bounding-box based affine pre-alignment
  2. Homography refinement for perspective correction
  3. Thin-Plate Spline (TPS) for non-rigid deformation
"""
import logging
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class WarpResult(NamedTuple):
    warped_clothing: Image.Image   # clothing warped to person geometry
    clothing_mask: Image.Image     # mask of warped clothing region (L mode)


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _mask_to_np(mask: Image.Image) -> np.ndarray:
    return np.array(mask.convert("L"))


def _bbox_from_mask(mask: np.ndarray):
    """Return (x1, y1, x2, y2) bounding box of non-zero region."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _affine_warp(
    clothing: np.ndarray,
    cloth_bbox,
    person_bbox,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale and translate clothing to roughly match person clothing region."""
    cx1, cy1, cx2, cy2 = cloth_bbox
    px1, py1, px2, py2 = person_bbox

    src = np.float32([
        [cx1, cy1], [cx2, cy1], [cx2, cy2], [cx1, cy2]
    ])
    dst = np.float32([
        [px1, py1], [px2, py1], [px2, py2], [px1, py2]
    ])

    M, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    warped = cv2.warpPerspective(clothing, M, (target_size[0], target_size[1]))
    return warped


class _TPS:
    """Minimal Thin-Plate Spline implementation using scipy."""

    def __init__(self, source_pts: np.ndarray, target_pts: np.ndarray):
        from scipy.interpolate import RBFInterpolator
        self._rbf_x = RBFInterpolator(source_pts, target_pts[:, 0], kernel="thin_plate_spline")
        self._rbf_y = RBFInterpolator(source_pts, target_pts[:, 1], kernel="thin_plate_spline")

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        H, W = image.shape[:2]
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        flat = np.column_stack([grid_y.ravel(), grid_x.ravel()])
        mapped_y = self._rbf_x(flat).reshape(H, W).astype(np.float32)
        mapped_x = self._rbf_y(flat).reshape(H, W).astype(np.float32)
        warped = cv2.remap(image, mapped_x, mapped_y, cv2.INTER_LINEAR)
        return warped


def warp_clothing(
    clothing_image: Image.Image,
    person_clothing_mask: Image.Image,
    target_size: tuple[int, int] = (1024, 1024),
) -> WarpResult:
    """
    Warp clothing_image to fit the clothing region of the person.
    person_clothing_mask indicates WHERE on the person clothing should appear.
    """
    W, H = target_size

    clothing_bgr = _pil_to_bgr(clothing_image.resize(target_size, Image.LANCZOS))
    person_mask_np = _mask_to_np(person_clothing_mask.resize(target_size, Image.LANCZOS))

    # Create a full-clothing mask (assume entire clothing image is the garment)
    cloth_mask_np = np.ones((H, W), dtype=np.uint8) * 255
    cloth_mask_np[:10, :] = 0
    cloth_mask_np[-10:, :] = 0
    cloth_mask_np[:, :10] = 0
    cloth_mask_np[:, -10:] = 0

    cloth_bbox = _bbox_from_mask(cloth_mask_np)
    person_bbox = _bbox_from_mask(person_mask_np)

    if cloth_bbox is None or person_bbox is None:
        logger.warning("No valid bounding box found — returning original clothing")
        return WarpResult(
            warped_clothing=clothing_image.resize(target_size, Image.LANCZOS),
            clothing_mask=Image.fromarray(person_mask_np, mode="L"),
        )

    # Step 1 – Affine / homography alignment
    warped_bgr = _affine_warp(clothing_bgr, cloth_bbox, person_bbox, (W, H))

    # Step 2 – TPS refinement using boundary keypoints
    try:
        cx1, cy1, cx2, cy2 = cloth_bbox
        px1, py1, px2, py2 = person_bbox

        # Sample points along bounding-box edges
        src_pts = np.float32([
            [cy1, cx1], [cy1, (cx1 + cx2) // 2], [cy1, cx2],
            [(cy1 + cy2) // 2, cx2], [cy2, cx2],
            [cy2, (cx1 + cx2) // 2], [cy2, cx1],
            [(cy1 + cy2) // 2, cx1],
        ])
        dst_pts = np.float32([
            [py1, px1], [py1, (px1 + px2) // 2], [py1, px2],
            [(py1 + py2) // 2, px2], [py2, px2],
            [py2, (px1 + px2) // 2], [py2, px1],
            [(py1 + py2) // 2, px1],
        ])

        tps = _TPS(src_pts, dst_pts)
        warped_bgr = tps.warp_image(warped_bgr)
    except Exception as exc:
        logger.warning("TPS warp failed (%s) — using affine only", exc)

    warped_pil = _bgr_to_pil(warped_bgr)
    mask_pil = Image.fromarray(person_mask_np, mode="L")

    return WarpResult(warped_clothing=warped_pil, clothing_mask=mask_pil)
