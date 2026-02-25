"""
Model loader – preloads SDXL + ControlNet + Segmentation models at startup.
Uses FP16 precision and xFormers memory-efficient attention.
"""
import gc
import logging
import os
from typing import Any

import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline,
)
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Singleton model registry ──────────────────────────────────────────────────
_models: dict[str, Any] = {}


def _dtype():
    return torch.float16 if settings.use_fp16 else torch.float32


def load_all_models() -> None:
    """Preload every model into GPU memory. Called once at startup."""
    logger.info("🚀 Loading AI models — this may take several minutes on first run …")

    os.makedirs(settings.models_cache_dir, exist_ok=True)
    os.makedirs(settings.tmp_dir, exist_ok=True)

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype() if device.type == "cuda" else torch.float32

    # 1. ControlNet (OpenPose SDXL)
    logger.info("Loading ControlNet …")
    controlnet = ControlNetModel.from_pretrained(
        settings.controlnet_model_id,
        torch_dtype=dtype,
        cache_dir=settings.models_cache_dir,
    )
    _models["controlnet"] = controlnet

    # 2. SDXL ControlNet pipeline (for pose-conditioned generation)
    logger.info("Loading SDXL + ControlNet pipeline …")
    sdxl_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        settings.sdxl_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=settings.models_cache_dir,
    )
    sdxl_pipe = sdxl_pipe.to(device)
    if settings.use_xformers and device.type == "cuda":
        try:
            sdxl_pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ xFormers enabled")
        except Exception:
            logger.warning("xFormers not available — falling back to standard attention")
    sdxl_pipe.enable_attention_slicing()
    _models["sdxl_pipe"] = sdxl_pipe

    # 3. SDXL Inpainting pipeline
    logger.info("Loading SDXL Inpainting pipeline …")
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        settings.inpainting_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=settings.models_cache_dir,
    )
    inpaint_pipe = inpaint_pipe.to(device)
    if settings.use_xformers and device.type == "cuda":
        try:
            inpaint_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    inpaint_pipe.enable_attention_slicing()
    inpaint_pipe.enable_vae_tiling()
    _models["inpaint_pipe"] = inpaint_pipe

    # 4. Segmentation model (clothing / human parsing)
    logger.info("Loading Segformer clothing parser …")
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

    # 5. OpenPose detector
    logger.info("Loading OpenPose detector …")
    pose_detector = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=settings.models_cache_dir,
    )
    _models["pose_detector"] = pose_detector

    _models["device"] = device
    _models["dtype"] = dtype
    logger.info("✅ All models loaded successfully")


def get_model(name: str) -> Any:
    """Retrieve a preloaded model by name."""
    if name not in _models:
        raise RuntimeError(f"Model '{name}' not loaded. Was load_all_models() called?")
    return _models[name]


def free_gpu_memory() -> None:
    """Force-free unused GPU tensors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def models_ready() -> bool:
    return bool(_models)
