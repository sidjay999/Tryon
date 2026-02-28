"""
SDXL Inpainting pipeline – Phase 2 upgrade.

Key changes:
  1. Face region is EXCLUDED from the inpaint mask (hard protection)
  2. IP-Adapter FaceID conditions generation on face identity embeddings
  3. SDXL Refiner pass added for detail sharpening
  4. Synchronous — no Celery dependency
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


def _extract_face_embedding(person_image: Image.Image):
    """Extract ArcFace embedding vector using InsightFace."""
    try:
        import cv2
        face_app = get_model("face_app")
        if face_app is None:
            return None
        arr = cv2.cvtColor(np.array(person_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = face_app.get(arr)
        if not faces:
            logger.warning("No face detected by InsightFace — skipping FaceID conditioning")
            return None
        # Largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return torch.from_numpy(face.normed_embedding).unsqueeze(0)
    except Exception as exc:
        logger.warning("Face embedding extraction failed: %s", exc)
        return None


def _run_with_ip_adapter(
    ip_model,
    composite: Image.Image,
    inpaint_mask: Image.Image,
    face_embedding,
    prompt: str,
    negative_prompt: str,
    generator,
) -> Image.Image:
    """Run IP-Adapter FaceID conditioned inpainting."""
    result = ip_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=composite,
        mask_image=inpaint_mask,
        faceid_embeds=face_embedding,
        face_image=composite,           # reference image for IP-Adapter
        shortcut=True,                  # FaceID Plus shortcut for better identity
        scale=settings.ip_adapter_scale,
        height=settings.output_size,
        width=settings.output_size,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        strength=settings.strength,
        generator=generator,
    )
    return result.images[0]


def _run_standard_inpaint(
    inpaint_pipe,
    composite: Image.Image,
    inpaint_mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    generator,
) -> Image.Image:
    """Fallback: standard SDXL inpainting without IP-Adapter."""
    with torch.inference_mode():
        result = inpaint_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            mask_image=inpaint_mask,
            height=settings.output_size,
            width=settings.output_size,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            strength=settings.strength,
            generator=generator,
        )
    return result.images[0]


def _run_refiner(generated: Image.Image, prompt: str, generator) -> Image.Image:
    """SDXL Refiner — sharpens texture details without altering identity."""
    try:
        refiner_pipe = get_model("refiner_pipe")
        with torch.inference_mode():
            result = refiner_pipe(
                prompt=prompt,
                image=generated,
                strength=settings.refiner_strength,
                num_inference_steps=settings.refiner_steps,
                generator=generator,
            )
        return result.images[0]
    except Exception as exc:
        logger.warning("Refiner failed (%s) — returning unrefined result", exc)
        return generated


def run_inpainting(
    person_image: Image.Image,
    warped_clothing: Image.Image,
    clothing_mask: Image.Image,
    pose_image: Image.Image,
    seg_face_bbox: tuple | None = None,
    prompt: str = (
        "a person wearing the clothing, photorealistic, fashion photography, "
        "studio lighting, 8k, high detail, sharp focus"
    ),
    negative_prompt: str = (
        "deformed, blurry, bad anatomy, ugly, duplicate, artifacts, "
        "low quality, watermark, text, changed face, different person"
    ),
) -> Image.Image:
    """
    Phase 2 inpainting:
      1. Build composite (warped clothing pasted over person)
      2. Get face bbox via InsightFace (fallback: segmentation bbox)
      3. Exclude face region from inpaint mask (HARD PROTECTION)
      4. Run IP-Adapter FaceID conditioned gen (if available) else standard SDXL
      5. Optional SDXL Refiner pass
    """
    device = get_model("device")
    inpaint_pipe = get_model("inpaint_pipe")
    target_size = (settings.output_size, settings.output_size)

    # Resize all inputs
    person_resized = person_image.resize(target_size, Image.LANCZOS).convert("RGB")
    mask_resized = clothing_mask.resize(target_size, Image.LANCZOS).convert("L")
    warped_resized = warped_clothing.resize(target_size, Image.LANCZOS).convert("RGB")

    # Composite: paste warped clothing over person
    composite = person_resized.copy()
    composite.paste(warped_resized, mask=mask_resized)

    # ── STEP 1: Hard face protection ─────────────────────────
    # Try InsightFace first (more precise), fall back to Segformer bbox
    face_bbox = get_face_bbox_insightface(person_resized) or seg_face_bbox
    inpaint_mask = exclude_face_from_mask(
        mask_resized,
        face_bbox,
        padding=settings.face_mask_padding,
    )
    if face_bbox:
        logger.info("Face region excluded from inpaint mask — bbox: %s", face_bbox)
    else:
        logger.warning("No face bbox found — full mask used (identity may drift)")

    # ── STEP 2: Face embedding for IP-Adapter ─────────────────
    face_embedding = None
    if settings.face_identity_enabled:
        face_embedding = _extract_face_embedding(person_resized)

    # Generator on CPU is safe with both full-GPU and cpu-offload modes
    generator = torch.Generator(device="cpu").manual_seed(42)

    # ── STEP 3: Generate ──────────────────────────────────────
    ip_model = get_model("ip_adapter")
    if ip_model is not None and face_embedding is not None:
        logger.info("Running IP-Adapter FaceID conditioned inpainting …")
        generated = _run_with_ip_adapter(
            ip_model, composite, inpaint_mask, face_embedding,
            prompt, negative_prompt, generator,
        )
    else:
        logger.info("Running standard SDXL inpainting (no FaceID) …")
        generated = _run_standard_inpaint(
            inpaint_pipe, composite, inpaint_mask,
            prompt, negative_prompt, generator,
        )

    # ── STEP 4: Refiner pass (skipped on low VRAM — refiner_pipe will be None) ─
    refiner_pipe = get_model("refiner_pipe", optional=True)
    if settings.use_refiner and refiner_pipe is not None:
        logger.info("Running SDXL Refiner (strength=%.2f) ...", settings.refiner_strength)
        generated = _run_refiner(generated, prompt, generator)
    elif refiner_pipe is None:
        logger.info("Refiner skipped (not loaded — low VRAM mode)")

    logger.info("Inpainting complete — size: %s", generated.size)
    return generated
