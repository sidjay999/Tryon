"""
Pose keypoint extraction using OpenPose via controlnet_aux.
Returns a pose-visualization image suitable for ControlNet conditioning.
"""
import logging
from typing import NamedTuple

from PIL import Image

from app.models.loader import get_model

logger = logging.getLogger(__name__)


class PoseResult(NamedTuple):
    pose_image: Image.Image    # RGB image with skeleton overlay (for ControlNet)


def extract_pose(person_image: Image.Image) -> PoseResult:
    """
    Run OpenPose detection on the person image.
    Returns a pose keypoint visualization image at the same resolution.
    """
    detector = get_model("pose_detector")

    pose_image = detector(
        person_image,
        hand_and_face=True,
        output_type="pil",
    )

    # Ensure output matches input dimensions
    if pose_image.size != person_image.size:
        pose_image = pose_image.resize(person_image.size, Image.LANCZOS)

    logger.debug("Pose extracted — image size: %s", pose_image.size)
    return PoseResult(pose_image=pose_image)
