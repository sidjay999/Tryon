"""
Try-On API router – Phase 2 (synchronous).
POST /api/tryon       → runs pipeline in-process, returns result directly
GET  /api/tryon/health → alias for health check
"""
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.queue.tasks import run_tryon_pipeline_sync
from app.utils.image import image_to_bytes, load_image_from_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tryon", tags=["Try-On"])


@router.post("", summary="Run virtual try-on (synchronous)")
async def submit_tryon(
    person_image: UploadFile = File(..., description="Full-body photo of the person"),
    clothing_image: UploadFile = File(..., description="Photo of the clothing item"),
    garment_category: Literal["upper", "full", "lower"] = Form(
        default="upper",
        description="upper = shirt/jacket | full = dress/saree | lower = jeans/trousers",
    ),
):
    """
    Runs the full 5-stage try-on pipeline synchronously.
    Returns the result image directly (base64 or S3 URL).
    Expected response time: 10–20 seconds on GPU.
    """
    job_id = str(uuid.uuid4())
    try:
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    person_bytes = image_to_bytes(person_pil)
    clothing_bytes = image_to_bytes(clothing_pil)

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
    except Exception as exc:
        logger.error("Pipeline error for job %s: %s", job_id, exc)
        raise HTTPException(status_code=500, detail={"error": str(exc), "job_id": job_id})
