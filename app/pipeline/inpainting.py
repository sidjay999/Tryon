"""
SDXL Inpainting pipeline – SD 1.5 Prototype Mode.
Uses StableDiffusionInpaintPipeline (SD 1.5) which fits natively in 6GB VRAM.
Inference: 15-20 seconds on RTX 4050 Laptop GPU.
"""
import logging

import numpy as np
import torch
from PIL import Image

from app.config import get_settings
from app.models.loader import free_gpu_memory, get_model
from app.pipeline.segmentation import exclude_face_from_mask, get_face_bbox_insightface

logger = logging.getLogger(__name__)
settings = get_settings()


def run_inpainting(
    person_image: Image.Image,
    warped_clothing: Image.Image,
    clothing_mask: Image.Image,
    pose_image: Image.Image,
    seg_face_bbox: tuple | None = None,
    prompt: str = (
        "a person wearing the clothing, photorealistic, fashion photography, "
        "high quality, detailed fabric texture, natural lighting"
    ),
    negative_prompt: str = (
        "deformed, blurry, bad anatomy, ugly, duplicate, low quality, "
        "watermark, text, changed face, different person, cartoon"
    ),
) -> Image.Image:
    """
    SD 1.5 Inpainting pipeline:
      1. Build composite (warped clothing pasted over person)
      2. Hard-exclude face region from mask (identity protection)
      3. Run SD 1.5 inpainting (direct GPU, no offloading)
      4. Return result (no refiner — SD 1.5 doesn't use one)
    """
    inpaint_pipe = get_model("inpaint_pipe")
    device = get_model("device")
    target_size = (settings.output_size, settings.output_size)  # 512x512

    # Resize all inputs to 512x512
    person_resized = person_image.resize(target_size, Image.LANCZOS).convert("RGB")
    mask_resized = clothing_mask.resize(target_size, Image.LANCZOS).convert("L")
    warped_resized = warped_clothing.resize(target_size, Image.LANCZOS).convert("RGB")

    # Composite: paste warped clothing over person
    composite = person_resized.copy()
    composite.paste(warped_resized, mask=mask_resized)

    # Hard face protection: exclude face region from inpaint mask
    face_bbox = get_face_bbox_insightface(person_resized) or seg_face_bbox
    inpaint_mask = exclude_face_from_mask(
        mask_resized,
        face_bbox,
        padding=settings.face_mask_padding,
    )
    if face_bbox:
        logger.info("Face region hard-excluded from mask — bbox: %s", face_bbox)
    else:
        logger.warning("No face bbox detected — using full clothing mask")

    # SD 1.5 inpainting — runs directly on GPU
    # Note: do NOT pass strength/height/width to SD 1.5 inpainting pipeline.
    # strength=<1.0 causes scheduler.get_timesteps() to return None in diffusers 0.27.x.
    # Image is already resized to 512px above, so height/width are redundant.
    generator = torch.Generator(device="cpu").manual_seed(42)
    logger.info("Running SD 1.5 inpainting (%d steps)...", settings.num_inference_steps)

    with torch.inference_mode():
        result = inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            mask_image=inpaint_mask,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            generator=generator,
        )

    generated = result.images[0]
    logger.info("Inpainting complete — output size: %s", generated.size)
    return generated
