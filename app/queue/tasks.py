"""
Pipeline orchestration – CatVTON Production Pipeline.
3-stage pipeline: AutoMasker → CatVTON Diffusion → Output
"""
import io
import logging
import time

import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from app.models.loader import get_model

logger = logging.getLogger(__name__)


def run_tryon_pipeline_sync(
    person_image_bytes: bytes,
    clothing_image_bytes: bytes,
    job_id: str,
    garment_category: str = "upper",
) -> dict:
    """
    Run the full CatVTON virtual try-on pipeline synchronously.

    Args:
        person_image_bytes: Raw bytes of person image
        clothing_image_bytes: Raw bytes of garment image
        job_id: Unique job identifier
        garment_category: "upper" | "lower" | "overall"
    Returns:
        dict with result_image (base64) and metadata
    """
    t_start = time.time()
    pipeline = get_model("pipeline")
    automasker = get_model("automasker")
    settings = get_model("settings")

    # Map our garment categories to CatVTON's
    category_map = {
        "upper": "upper",
        "lower": "lower",
        "full": "overall",
        "overall": "overall",
    }
    mask_type = category_map.get(garment_category, "upper")

    # ── Stage 1: Load and preprocess images ────────────────────
    logger.info("[%s] Stage 1/3 – Preprocessing", job_id)
    person_img = Image.open(io.BytesIO(person_image_bytes)).convert("RGB")
    cloth_img = Image.open(io.BytesIO(clothing_image_bytes)).convert("RGB")

    # Import CatVTON utilities (already on sys.path from loader)
    from utils import resize_and_crop, resize_and_padding

    target_size = (settings.output_width, settings.output_height)
    person_resized = resize_and_crop(person_img, target_size)
    cloth_resized = resize_and_padding(cloth_img, target_size)

    # ── Stage 2: Generate agnostic mask via AutoMasker ─────────
    logger.info("[%s] Stage 2/3 – AutoMasker (DensePose + SCHP)", job_id)
    mask_result = automasker(person_resized, mask_type=mask_type)
    mask = mask_result["mask"]

    # Blur the mask edges for smoother results
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )
    mask = mask_processor.blur(mask, blur_factor=9)

    # ── Stage 3: CatVTON Diffusion ─────────────────────────────
    logger.info("[%s] Stage 3/3 – CatVTON diffusion (%d steps, CFG %.1f)",
                job_id, settings.num_inference_steps, settings.guidance_scale)

    generator = torch.Generator(device="cuda").manual_seed(settings.seed)

    result_images = pipeline(
        image=person_resized,
        condition_image=cloth_resized,
        mask=mask,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        height=settings.output_height,
        width=settings.output_width,
        generator=generator,
    )
    result_image = result_images[0]

    elapsed = time.time() - t_start
    logger.info("[%s] Pipeline complete in %.1fs", job_id, elapsed)

    # ── Return result ──────────────────────────────────────────
    from app.utils.image import encode_image_base64
    result_b64 = encode_image_base64(result_image, fmt="PNG")

    return {
        "job_id": job_id,
        "result_image_base64": result_b64,
        "garment_category": garment_category,
        "resolution": f"{settings.output_width}x{settings.output_height}",
        "inference_time_seconds": round(elapsed, 1),
    }
