"""
Model loader – stable version for 6GB GPU (RTX 4050 Laptop).

Memory strategy:
  - FP16 precision (halves VRAM requirement)
  - enable_model_cpu_offload() for SDXL on < 10GB VRAM
    (moves full model submodules to GPU one at a time — stable in diffusers 0.27.x)
  - Refiner: always skipped on < 10GB (would OOM)
  - Refiner_pipe key always set to None so pipeline code never crashes
"""
import gc
import logging
import os
from typing import Any

import torch
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionXLInpaintPipeline
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_models: dict[str, Any] = {}

VRAM_THRESHOLD_GB = 10.0  # GPUs below this use CPU offload


def _dtype():
    return torch.float16 if settings.use_fp16 else torch.float32


def _get_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _load_insightface(device):
    try:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0 if device.type == "cuda" else -1, det_size=(640, 640))
        logger.info("InsightFace loaded")
        return face_app
    except Exception as exc:
        logger.warning("InsightFace unavailable (%s) — Segformer-based face bbox fallback active", exc)
        return None


def _from_pretrained_with_retry(cls, model_id, **kwargs):
    """Retry with force_download=True if a file integrity check fails."""
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        if "Consistency check failed" in str(exc) or "size" in str(exc):
            logger.warning("Corrupted cached file — retrying with force_download=True")
            kwargs.pop("resume_download", None)
            return cls.from_pretrained(model_id, force_download=True, resume_download=False, **kwargs)
        raise


def load_all_models() -> None:
    """Load all models. Auto-switches to CPU offload on GPUs with < 10GB VRAM."""

    # Enable faster HF downloads
    try:
        import hf_transfer  # noqa
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        logger.info("hf_transfer enabled")
    except ImportError:
        pass

    os.makedirs(settings.models_cache_dir, exist_ok=True)
    os.makedirs(settings.tmp_dir, exist_ok=True)

    vram_gb = _get_vram_gb()
    low_vram = vram_gb > 0 and vram_gb < VRAM_THRESHOLD_GB
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype() if device.type == "cuda" else torch.float32

    logger.info("GPU VRAM: %.1fGB | Mode: %s | Device: %s",
                vram_gb, "CPU-OFFLOAD" if low_vram else "FULL-GPU", device)

    _models["device"] = device
    _models["dtype"] = dtype
    _models["refiner_pipe"] = None   # always set — prevents KeyError on low VRAM
    _models["ip_adapter"] = None     # set now; overwritten if successfully loaded below
    _models["face_app"] = None

    # ── 1. SDXL Inpainting pipeline ───────────────────────────
    logger.info("Loading SDXL Inpainting pipeline...")
    inpaint_pipe = _from_pretrained_with_retry(
        StableDiffusionXLInpaintPipeline,
        settings.inpainting_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=settings.models_cache_dir,
        low_cpu_mem_usage=True,
    )

    if low_vram:
        logger.info("Enabling model_cpu_offload (6GB GPU mode)...")
        inpaint_pipe.enable_model_cpu_offload()   # stable in diffusers 0.27.x
        inpaint_pipe.enable_vae_slicing()
        inpaint_pipe.enable_vae_tiling()
    else:
        inpaint_pipe.to(device)
        if settings.use_xformers:
            try:
                inpaint_pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                logger.warning("xFormers not available")
        inpaint_pipe.enable_attention_slicing()
        inpaint_pipe.enable_vae_tiling()

    _models["inpaint_pipe"] = inpaint_pipe
    logger.info("SDXL Inpainting ready")

    # ── 2. SDXL Refiner ───────────────────────────────────────
    if settings.use_refiner and not low_vram:
        logger.info("Loading SDXL Refiner...")
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            refiner = _from_pretrained_with_retry(
                StableDiffusionXLImg2ImgPipeline,
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                cache_dir=settings.models_cache_dir,
                text_encoder_2=inpaint_pipe.text_encoder_2,
                vae=inpaint_pipe.vae,
                low_cpu_mem_usage=True,
            )
            refiner.to(device)
            refiner.enable_attention_slicing()
            _models["refiner_pipe"] = refiner
            logger.info("SDXL Refiner ready")
        except Exception as exc:
            logger.warning("Refiner failed to load (%s) — skipped", exc)
    else:
        logger.info("Refiner skipped (%.1fGB VRAM < %.0fGB required)", vram_gb, VRAM_THRESHOLD_GB)

    # ── 3. Segformer (human parsing) ─────────────────────────
    logger.info("Loading Segformer...")
    seg_processor = AutoImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=settings.models_cache_dir,
    )
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=settings.models_cache_dir,
    ).to(device)
    seg_model.eval()
    _models["seg_processor"] = seg_processor
    _models["seg_model"] = seg_model
    logger.info("Segformer ready")

    # ── 4. OpenPose ───────────────────────────────────────────
    logger.info("Loading OpenPose...")
    pose_detector = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=settings.models_cache_dir,
    )
    _models["pose_detector"] = pose_detector
    logger.info("OpenPose ready")

    # ── 5. InsightFace (optional) ─────────────────────────────
    _models["face_app"] = _load_insightface(device)

    logger.info(
        "=== All models loaded | VRAM: %.1fGB | Mode: %s | Refiner: %s | FaceID: %s ===",
        vram_gb,
        "CPU-OFFLOAD" if low_vram else "FULL-GPU",
        "YES" if _models.get("refiner_pipe") else "NO (skipped)",
        "YES" if _models.get("face_app") else "NO (install C++ Build Tools)",
    )


def get_model(name: str, optional: bool = False) -> Any:
    if name not in _models:
        if optional:
            return None
        raise RuntimeError(f"Model '{name}' not loaded. Call load_all_models() first.")
    return _models[name]


def models_ready() -> bool:
    return "inpaint_pipe" in _models


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
