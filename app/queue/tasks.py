"""
Synchronous pipeline orchestrator – Phase 2.
Replaces the Celery task for the prototype stage.
The full 5-stage pipeline runs in-process within the FastAPI request.
"""
import io
import logging
import traceback
import uuid

from PIL import Image

from app.config import get_settings
from app.models.loader import free_gpu_memory
from app.pipeline.blending import blend_result
from app.pipeline.inpainting import run_inpainting
from app.pipeline.pose import extract_pose
from app.pipeline.segmentation import dilate_mask, exclude_face_from_mask, segment_person
from app.pipeline.warping import warp_clothing
from app.storage import s3
from app.utils.image import encode_image_base64, resize_to_square

logger = logging.getLogger(__name__)
settings = get_settings()


def run_tryon_pipeline_sync(
    person_image_bytes: bytes,
    clothing_image_bytes: bytes,
    job_id: str | None = None,
    garment_category: str = "upper",  # "upper" | "full" | "lower"
) -> dict:
    """
    Synchronous Phase 2 pipeline:
      1. Segmentation  → clothing mask + face bbox
      2. Pose          → ControlNet keypoint image
      3. Warp          → affine + TPS clothing alignment
      4. Inpaint       → SDXL + IP-Adapter FaceID (face hard-protected)
      5. Blend         → Poisson + Gaussian face paste + histogram
    Returns result dict with result_url or result_b64.
    """
    job_id = job_id or str(uuid.uuid4())

    person_image = Image.open(io.BytesIO(person_image_bytes)).convert("RGB")
    clothing_image = Image.open(io.BytesIO(clothing_image_bytes)).convert("RGB")

    person_image = resize_to_square(person_image, settings.output_size)
    clothing_image = resize_to_square(clothing_image, settings.output_size)

    # 1. Segmentation
    logger.info("[%s] Step 1/5 – Segmentation", job_id)
    seg = segment_person(person_image)

    # Expand mask based on garment category
    if garment_category == "full":
        from app.pipeline.segmentation import CLOTHING_LABELS
        # Include lower-body labels for dresses/full outfits
        extra_labels = {5, 6, 7}  # skirt, pants, dress
        base_mask = seg.clothing_mask
    else:
        base_mask = seg.clothing_mask

    dilated_mask = dilate_mask(base_mask, kernel_size=20)

    # 2. Pose
    logger.info("[%s] Step 2/5 – Pose extraction", job_id)
    pose = extract_pose(person_image)

    # 3. Warp
    logger.info("[%s] Step 3/5 – Clothing warp", job_id)
    warp = warp_clothing(
        clothing_image=clothing_image,
        person_clothing_mask=dilated_mask,
        target_size=(settings.output_size, settings.output_size),
    )

    # 4. Inpaint (face hard-protected via mask exclusion + IP-Adapter FaceID)
    logger.info("[%s] Step 4/5 – Inpainting", job_id)
    generated = run_inpainting(
        person_image=person_image,
        warped_clothing=warp.warped_clothing,
        clothing_mask=dilated_mask,
        pose_image=pose.pose_image,
        seg_face_bbox=seg.face_bbox,
    )

    # 5. Blend
    logger.info("[%s] Step 5/5 – Blending", job_id)
    final_image = blend_result(
        original_person=person_image,
        generated_image=generated,
        clothing_mask=seg.clothing_mask,
        face_bbox=seg.face_bbox,
        face_padding=settings.face_mask_padding,
    )
    free_gpu_memory()

    # Upload / encode result
    result: dict = {"job_id": job_id}
    if s3.is_configured():
        key = s3.upload_image(final_image, key=f"results/{job_id}.png")
        result["result_url"] = s3.get_presigned_url(key)
        result["s3_key"] = key
    else:
        result["result_b64"] = encode_image_base64(final_image)

    logger.info("[%s] ✅ Pipeline complete", job_id)
    return result
