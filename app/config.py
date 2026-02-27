"""
Configuration – Phase 2 additions:
  - use_refiner: toggle SDXL refiner pass
  - refiner_strength: how strongly the refiner changes the image
  - face_mask_padding: extra pixels around detected face to exclude from inpaint mask
  - ip_adapter_scale: how strongly IP-Adapter FaceID conditions the generation
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Model ──────────────────────────────────────────────────
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model_id: str = "diffusers/controlnet-openpose-sdxl-1.0"
    inpainting_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    # ── Runtime ────────────────────────────────────────────────
    device: str = "cuda"
    use_fp16: bool = True
    use_xformers: bool = True
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    output_size: int = 1024
    strength: float = 0.85
    controlnet_conditioning_scale: float = 0.8

    # ── Phase 2: Identity Preservation ────────────────────────
    use_refiner: bool = True              # SDXL refiner pass for detail sharpening
    refiner_strength: float = 0.2        # keep low (0.15-0.3) to avoid identity shift
    refiner_steps: int = 20
    face_mask_padding: int = 30          # px padding around face bbox excluded from mask
    ip_adapter_scale: float = 0.7        # strength of FaceID identity conditioning
    face_identity_enabled: bool = True   # auto-disables if InsightFace not installed

    # ── Storage ────────────────────────────────────────────────
    s3_bucket: str = "tryon-results"
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    result_url_expiry_seconds: int = 3600

    # ── API ────────────────────────────────────────────────────
    max_upload_size_mb: int = 20
    allowed_origins: list[str] = ["*"]
    api_title: str = "AI Virtual Try-On API"
    api_version: str = "2.0.0"

    # ── Paths ──────────────────────────────────────────────────
    tmp_dir: str = "/tmp/tryon"
    models_cache_dir: str = "/app/model_cache"

    # ── Redis (disabled by default in Phase 2) ─────────────────
    redis_url: str = "redis://redis:6379/0"
    celery_max_concurrency: int = 1


@lru_cache()
def get_settings() -> Settings:
    return Settings()
