"""
Try-On API router.
POST /api/tryon       → enqueue job, return job_id
GET  /api/tryon/{id}  → poll job status / result
"""
import logging
import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.queue.tasks import run_tryon_pipeline
from app.queue.worker import celery_app
from app.utils.image import image_to_bytes, load_image_from_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tryon", tags=["Try-On"])


@router.post("", summary="Submit a virtual try-on job")
async def submit_tryon(
    person_image: UploadFile = File(..., description="Photo of the person"),
    clothing_image: UploadFile = File(..., description="Photo of the clothing item"),
):
    """
    Accept two images and enqueue an async try-on pipeline.
    Returns immediately with a `job_id` for polling.
    """
    try:
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    person_bytes = image_to_bytes(person_pil, fmt="PNG")
    clothing_bytes = image_to_bytes(clothing_pil, fmt="PNG")

    job_id = str(uuid.uuid4())
    task = run_tryon_pipeline.apply_async(
        kwargs={
            "person_image_bytes": person_bytes,
            "clothing_image_bytes": clothing_bytes,
            "job_id": job_id,
        },
        task_id=job_id,
    )
    logger.info("Queued try-on job: %s", job_id)
    return JSONResponse({"job_id": task.id, "status": "queued"}, status_code=202)


@router.get("/{job_id}", summary="Poll try-on job status")
async def get_tryon_status(job_id: str):
    """
    Returns job status and result URL (or base64) when complete.
    """
    result = AsyncResult(job_id, app=celery_app)
    state = result.state

    if state == "PENDING":
        return {"job_id": job_id, "status": "pending", "progress": 0}

    if state == "PROGRESS":
        meta = result.info or {}
        return {
            "job_id": job_id,
            "status": "processing",
            "step": meta.get("step", ""),
            "progress": meta.get("progress", 0),
        }

    if state == "SUCCESS":
        r = result.result or {}
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 100,
            "result_url": r.get("result_url"),
            "result_b64": r.get("result_b64"),
        }

    if state == "FAILURE":
        info = result.info or {}
        raise HTTPException(
            status_code=500,
            detail={"error": str(info.get("exc_message", "Unknown error")), "job_id": job_id},
        )

    return {"job_id": job_id, "status": state.lower()}
