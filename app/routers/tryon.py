"""
Try-On API router – CatVTON Production Pipeline (Phase 2).
POST /api/tryon → runs full preprocessing + CatVTON pipeline
"""
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.queue.tasks import PipelineError, run_tryon_pipeline_sync
from app.utils.image import image_to_bytes, load_image_from_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tryon", tags=["Try-On"])

# Structured error codes
ERROR_CODES = {
    "INVALID_IMAGE": 400,
    "PERSON_NOT_DETECTED": 400,
    "GPU_OOM": 503,
    "PIPELINE_ERROR": 500,
}


@router.post("", summary="Run virtual try-on (CatVTON)")
async def submit_tryon(
    person_image: UploadFile = File(..., description="Full-body photo of the person"),
    clothing_image: UploadFile = File(..., description="Photo of the clothing item"),
    garment_category: Literal["upper", "lower", "overall"] = Form(
        default="upper",
        description="upper = shirt/jacket | overall = dress/saree | lower = jeans/trousers",
    ),
):
    """
    Runs the CatVTON virtual try-on pipeline:
      1. Person preprocessing (detect, crop, resize)
      2. Garment preprocessing (bg removal, center, resize)
      3. AutoMasker – pixel-level human parsing (DensePose + SCHP)
      4. CatVTON – diffusion-based garment transfer (1024×768)
    Returns result image as base64 PNG with monitoring data.
    """
    job_id = str(uuid.uuid4())

    # ── Validate uploads ───────────────────────────────────────
    try:
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "INVALID_IMAGE", "message": str(exc), "job_id": job_id},
        )

    person_bytes = image_to_bytes(person_pil)
    clothing_bytes = image_to_bytes(clothing_pil)

    # ── Run pipeline ───────────────────────────────────────────
    try:
        result = run_tryon_pipeline_sync(
            person_image_bytes=person_bytes,
            clothing_image_bytes=clothing_bytes,
            job_id=job_id,
            garment_category=garment_category,
        )
        return JSONResponse({
            "job_id": job_id,
            "status": "completed",
            "garment_category": garment_category,
            **{k: v for k, v in result.items() if k != "job_id"},
        })

    except PipelineError as exc:
        status_code = ERROR_CODES.get(exc.code, 500)
        logger.error("[%s] PipelineError %s: %s", job_id, exc.code, exc.message)
        raise HTTPException(
            status_code=status_code,
            detail={"error": exc.code, "message": exc.message, "job_id": job_id},
        )

    except Exception as exc:
        import traceback
        logger.error("[%s] Unexpected error:\n%s", job_id, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PIPELINE_ERROR",
                "message": str(exc),
                "job_id": job_id,
            },
        )
