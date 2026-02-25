"""
S3-compatible object storage adapter.
Works with AWS S3, MinIO, Cloudflare R2, DigitalOcean Spaces.
"""
import io
import logging
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_client():
    kwargs: dict = {
        "region_name": settings.aws_region,
        "aws_access_key_id": settings.aws_access_key_id or None,
        "aws_secret_access_key": settings.aws_secret_access_key or None,
        "config": Config(signature_version="s3v4"),
    }
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    return boto3.client("s3", **kwargs)


def upload_image(image: Image.Image, key: str | None = None, fmt: str = "PNG") -> str:
    """Upload PIL image to S3 and return the object key."""
    if key is None:
        key = f"results/{uuid.uuid4()}.png"
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    client = _get_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=buf,
        ContentType=f"image/{fmt.lower()}",
    )
    logger.info("Uploaded result to s3://%s/%s", settings.s3_bucket, key)
    return key


def get_presigned_url(key: str) -> str:
    """Generate a pre-signed URL valid for result_url_expiry_seconds."""
    client = _get_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=settings.result_url_expiry_seconds,
    )
    return url


def delete_object(key: str) -> None:
    """Delete an object from S3."""
    try:
        client = _get_client()
        client.delete_object(Bucket=settings.s3_bucket, Key=key)
    except ClientError as exc:
        logger.error("Failed to delete s3://%s/%s: %s", settings.s3_bucket, key, exc)


def is_configured() -> bool:
    """Return True if S3 credentials are available."""
    return bool(settings.aws_access_key_id and settings.aws_secret_access_key)
