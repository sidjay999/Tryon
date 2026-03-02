"""
Configuration – CatVTON Production Pipeline.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── CatVTON Model ─────────────────────────────────────────
    base_model_id: str = "booksforcharlie/stable-diffusion-inpainting"
    catvton_repo_id: str = "zhengchong/CatVTON"
    attn_ckpt_version: str = "mix"  # "mix" | "vitonhd" | "dresscode"

    # ── Runtime ───────────────────────────────────────────────
    device: str = "cuda"
    mixed_precision: str = "fp16"  # "no" | "fp16" | "bf16"
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    seed: int = 42
    output_height: int = 1024
    output_width: int = 768

    # ── Paths ─────────────────────────────────────────────────
    catvton_dir: str = ""  # Auto-detected
    models_cache_dir: str = os.path.expanduser("~/tryon_models")
    tmp_dir: str = "/tmp/tryon"
    output_dir: str = "/tmp/tryon/output"

    # ── S3-Compatible Storage ─────────────────────────────────
    s3_bucket: str = "tryon-results"
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    result_url_expiry_seconds: int = 3600

    # ── API ───────────────────────────────────────────────────
    max_upload_size_mb: int = 20
    allowed_origins: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
