"""
Pipeline orchestration – CatVTON Production Pipeline (Phase 2 slim).
Pipeline: Preprocess → AutoMasker → CatVTON Diffusion → Output
No extra GPU models loaded — preprocessing is CPU only.
"""
import io
import logging
import time

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from app.models.loader import get_model
from app.monitoring.monitor import PipelineMonitor
from app.preprocessing.person_preprocess import preprocess_person
from app.preprocessing.garment_preprocess import preprocess_garment

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Structured pipeline error with code."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def run_tryon_pipeline_sync(
    person_image_bytes: bytes,
    clothing_image_bytes: bytes,
    job_id: str,
    garment_category: str = "upper",
) -> dict:
    """
    Run the full CatVTON virtual try-on pipeline synchronously.

    Pipeline (all preprocessing is CPU, only AutoMasker + CatVTON use GPU):
      1. Person image: EXIF fix → center crop → resize to 768×1024  [CPU]
      2. Garment image: bg removal (rembg) → crop → center → resize  [CPU]
      3. AutoMasker: DensePose + SCHP → agnostic mask  [GPU]
      4. CatVTON diffusion → result image  [GPU]
    """
    pipeline = get_model("pipeline")
    automasker = get_model("automasker")
    settings = get_model("settings")

    category_map = {
        "upper": "upper",
        "lower": "lower",
        "full": "overall",
        "overall": "overall",
    }
    mask_type = category_map.get(garment_category, "upper")

    with PipelineMonitor(request_id=job_id) as monitor:

        # ── Stage 1: Person preprocessing (CPU) ────────────────
        logger.info("[%s] Stage 1/4 – Person preprocessing", job_id)
        try:
            person_img = Image.open(io.BytesIO(person_image_bytes)).convert("RGB")
        except Exception:
            raise PipelineError("INVALID_IMAGE", "Could not read person image.")

        person_result = preprocess_person(person_img)
        person_preprocessed = person_result["image"]

        # ── Stage 2: Garment preprocessing (CPU) ───────────────
        logger.info("[%s] Stage 2/4 – Garment preprocessing", job_id)
        try:
            cloth_img = Image.open(io.BytesIO(clothing_image_bytes)).convert("RGB")
        except Exception:
            raise PipelineError("INVALID_IMAGE", "Could not read garment image.")

        garment_result = preprocess_garment(
            cloth_img,
            enable_bg_removal=settings.enable_bg_removal,
        )
        cloth_preprocessed = garment_result["image"]

        logger.info(
            "[%s] Preprocessed: person %s→768×1024, garment bg_removed=%s",
            job_id, person_result["original_size"], garment_result["bg_removed"],
        )

        # ── Stage 3: AutoMasker (GPU – already loaded) ─────────
        logger.info("[%s] Stage 3/4 – AutoMasker (DensePose + SCHP)", job_id)
        mask_result = automasker(person_preprocessed, mask_type=mask_type)
        mask = mask_result["mask"]

        # Validate: if mask is essentially empty, person wasn't detected
        mask_arr = np.array(mask.convert("L"))
        mask_coverage = (mask_arr > 128).sum() / mask_arr.size
        if mask_coverage < 0.01:
            raise PipelineError(
                "PERSON_NOT_DETECTED",
                "AutoMasker could not detect a person in the image. "
                "Please upload a clear, full-body photo.",
            )

        logger.info("[%s] Mask coverage: %.1f%%", job_id, mask_coverage * 100)

        # Blur mask edges for smoother transitions
        mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        mask = mask_processor.blur(mask, blur_factor=9)

        # ── Stage 4: CatVTON Diffusion (GPU – already loaded) ─
        logger.info(
            "[%s] Stage 4/4 – CatVTON diffusion (%d steps, CFG %.1f)",
            job_id, settings.num_inference_steps, settings.guidance_scale,
        )

        generator = torch.Generator(device="cuda").manual_seed(settings.seed)

        try:
            result_images = pipeline(
                image=person_preprocessed,
                condition_image=cloth_preprocessed,
                mask=mask,
                num_inference_steps=settings.num_inference_steps,
                guidance_scale=settings.guidance_scale,
                height=settings.output_height,
                width=settings.output_width,
                generator=generator,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise PipelineError(
                "GPU_OOM",
                "GPU ran out of memory. Try closing other apps or restart WSL.",
            )

        result_image = result_images[0]

    # ── Build response ─────────────────────────────────────────
    from app.utils.image import encode_image_base64
    result_b64 = encode_image_base64(result_image, fmt="PNG")

    return {
        "job_id": job_id,
        "result_image_base64": result_b64,
        "garment_category": garment_category,
        "resolution": f"{settings.output_width}x{settings.output_height}",
        "monitoring": {
            "gpu": monitor.metrics.get("gpu", ""),
            "vram_peak_gb": monitor.metrics.get("vram_peak_gb", 0),
            "inference_time_s": monitor.metrics.get("inference_time_s", 0),
        },
        "preprocessing": {
            "bg_removed": garment_result["bg_removed"],
            "mask_coverage_pct": round(mask_coverage * 100, 1),
        },
    }
