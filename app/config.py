"""
Configuration management for the AI Virtual Try-On system.
All settings are loaded from environment variables with sensible defaults.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Model ──────────────────────────────────────────────────
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model_id: str = "diffusers/controlnet-openpose-sdxl-1.0"
    inpainting_model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    # ── Runtime ────────────────────────────────────────────────
    device: str = "cuda"                 # "cuda" | "cpu"
    use_fp16: bool = True
    use_xformers: bool = True
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    output_size: int = 1024              # px (square)
    strength: float = 0.85              # inpainting denoising strength
    controlnet_conditioning_scale: float = 0.8

    # ── Redis / Celery ─────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    celery_max_concurrency: int = 1      # 1 worker per GPU instance

    # ── S3-compatible Storage ──────────────────────────────────
    s3_bucket: str = "tryon-results"
    s3_endpoint_url: str = ""            # empty → AWS, set for MinIO/Cloudflare
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    result_url_expiry_seconds: int = 3600

    # ── API ────────────────────────────────────────────────────
    max_upload_size_mb: int = 20
    allowed_origins: list[str] = ["*"]
    api_title: str = "AI Virtual Try-On API"
    api_version: str = "1.0.0"

    # ── Paths ──────────────────────────────────────────────────
    tmp_dir: str = "/tmp/tryon"
    models_cache_dir: str = "/app/model_cache"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
