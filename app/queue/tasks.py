"""
Celery tasks – orchestrates the full try-on pipeline end-to-end.
"""
import io
import logging
import traceback
import uuid
from pathlib import Path

from celery import states
from PIL import Image

from app.config import get_settings
from app.models.loader import free_gpu_memory
from app.pipeline.blending import blend_result
from app.pipeline.inpainting import run_inpainting
from app.pipeline.pose import extract_pose
from app.pipeline.segmentation import dilate_mask, segment_person
from app.pipeline.warping import warp_clothing
from app.queue.worker import celery_app
from app.storage import s3
from app.utils.image import encode_image_base64, image_to_bytes, resize_to_square

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(
    bind=True,
    name="tryon.run_pipeline",
    acks_late=True,
    max_retries=1,
)
def run_tryon_pipeline(
    self,
    person_image_bytes: bytes,
    clothing_image_bytes: bytes,
    job_id: str | None = None,
) -> dict:
    """
    Full try-on pipeline:
      1. Segmentation  → clothing + face masks
      2. Pose          → keypoint image for ControlNet
      3. Warp          → align clothing to body
      4. Inpaint       → SDXL generates try-on result
      5. Blend         → Poisson + face restore + histogram match
      6. Upload        → S3 (or base64 fallback)
    """
    job_id = job_id or str(uuid.uuid4())
    self.update_state(state="PROGRESS", meta={"step": "segmentation", "progress": 10})

    try:
        person_image = Image.open(io.BytesIO(person_image_bytes)).convert("RGB")
        clothing_image = Image.open(io.BytesIO(clothing_image_bytes)).convert("RGB")

        person_image = resize_to_square(person_image, settings.output_size)
        clothing_image = resize_to_square(clothing_image, settings.output_size)

        # 1. Segmentation
        logger.info("[%s] Step 1/5 – Segmentation", job_id)
        seg = segment_person(person_image)
        dilated_mask = dilate_mask(seg.clothing_mask, kernel_size=20)
        self.update_state(state="PROGRESS", meta={"step": "pose", "progress": 25})

        # 2. Pose
        logger.info("[%s] Step 2/5 – Pose extraction", job_id)
        pose = extract_pose(person_image)
        self.update_state(state="PROGRESS", meta={"step": "warp", "progress": 40})

        # 3. Warp
        logger.info("[%s] Step 3/5 – Clothing warp", job_id)
        warp = warp_clothing(
            clothing_image=clothing_image,
            person_clothing_mask=dilated_mask,
            target_size=(settings.output_size, settings.output_size),
        )
        self.update_state(state="PROGRESS", meta={"step": "inpainting", "progress": 55})

        # 4. Inpaint
        logger.info("[%s] Step 4/5 – Inpainting", job_id)
        generated = run_inpainting(
            person_image=person_image,
            warped_clothing=warp.warped_clothing,
            clothing_mask=dilated_mask,
            pose_image=pose.pose_image,
        )
        self.update_state(state="PROGRESS", meta={"step": "blend", "progress": 85})

        # 5. Blend
        logger.info("[%s] Step 5/5 – Blending", job_id)
        final_image = blend_result(
            original_person=person_image,
            generated_image=generated,
            clothing_mask=seg.clothing_mask,
        )
        free_gpu_memory()

        # 6. Upload result
        result_payload: dict = {"job_id": job_id, "step": "done", "progress": 100}
        if s3.is_configured():
            key = s3.upload_image(final_image, key=f"results/{job_id}.png")
            result_payload["result_url"] = s3.get_presigned_url(key)
            result_payload["s3_key"] = key
        else:
            result_payload["result_b64"] = encode_image_base64(final_image)

        logger.info("[%s] ✅ Pipeline complete", job_id)
        return result_payload

    except Exception as exc:
        logger.error("[%s] Pipeline failed: %s\n%s", job_id, exc, traceback.format_exc())
        free_gpu_memory()
        self.update_state(
            state=states.FAILURE,
            meta={"exc_type": type(exc).__name__, "exc_message": str(exc), "job_id": job_id},
        )
        raise
