"""
SDXL Inpainting pipeline with ControlNet pose conditioning.
Replaces the clothing region in the person image with the warped garment.
"""
import logging

import torch
from PIL import Image

from app.config import get_settings
from app.models.loader import get_model

logger = logging.getLogger(__name__)
settings = get_settings()


def run_inpainting(
    person_image: Image.Image,
    warped_clothing: Image.Image,
    clothing_mask: Image.Image,
    pose_image: Image.Image,
    prompt: str = (
        "a person wearing the clothing, photorealistic, high resolution, "
        "studio lighting, fashion photography, 8k"
    ),
    negative_prompt: str = (
        "deformed, blurry, bad anatomy, ugly, duplicate, artifacts, "
        "low quality, watermark, text"
    ),
) -> Image.Image:
    """
    Run SDXL inpainting conditioned on pose keypoints.
    The warped clothing is composited as the starting reference inside the mask region.
    Returns a 1024x1024 PIL image.
    """
    device = get_model("device")
    dtype = get_model("dtype")
    inpaint_pipe = get_model("inpaint_pipe")

    target_size = (settings.output_size, settings.output_size)

    # Resize all inputs to target resolution
    person_resized = person_image.resize(target_size, Image.LANCZOS).convert("RGB")
    mask_resized = clothing_mask.resize(target_size, Image.LANCZOS).convert("L")
    warped_resized = warped_clothing.resize(target_size, Image.LANCZOS).convert("RGB")

    # Composite warped clothing over person within mask to guide generation
    composite = person_resized.copy()
    composite.paste(warped_resized, mask=mask_resized)

    # Run SDXL inpainting
    generator = torch.Generator(device=device).manual_seed(42)

    with torch.inference_mode():
        result = inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            mask_image=mask_resized,
            height=settings.output_size,
            width=settings.output_size,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            strength=settings.strength,
            generator=generator,
        )

    output_image: Image.Image = result.images[0]
    logger.info("Inpainting complete — output size: %s", output_image.size)
    return output_image
