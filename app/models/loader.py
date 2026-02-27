"""
Model loader – Phase 2 upgrade.
Loads SDXL + ControlNet + Segformer + OpenPose + InsightFace + IP-Adapter FaceID + SDXL Refiner.
"""
import gc
import logging
import os
from typing import Any

import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_models: dict[str, Any] = {}


def _dtype():
    return torch.float16 if settings.use_fp16 else torch.float32


def _load_insightface(device):
    """Load InsightFace ArcFace model for face embedding extraction."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=0 if device.type == "cuda" else -1, det_size=(640, 640))
        logger.info("✅ InsightFace loaded")
        return face_app
    except Exception as exc:
        logger.warning("InsightFace not available (%s) — face identity conditioning disabled", exc)
        return None


def _load_ip_adapter(inpaint_pipe, device):
    """Load IP-Adapter FaceID Plus weights into the inpaint pipeline."""
    try:
        from ip_adapter import IPAdapterFaceIDPlus
        # IP-Adapter FaceID Plus for SDXL
        image_encoder_path = os.path.join(settings.models_cache_dir, "ip_adapter", "image_encoder")
        ip_ckpt = os.path.join(settings.models_cache_dir, "ip_adapter", "ip-adapter-faceid-plusv2_sdxl.bin")

        if not os.path.isfile(ip_ckpt):
            logger.info("Downloading IP-Adapter FaceID weights …")
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(ip_ckpt), exist_ok=True)
            hf_hub_download(
                repo_id="h94/IP-Adapter-FaceID",
                filename="ip-adapter-faceid-plusv2_sdxl.bin",
                local_dir=os.path.dirname(ip_ckpt),
            )
            # Download image encoder if needed
            from huggingface_hub import snapshot_download
            if not os.path.isdir(image_encoder_path):
                snapshot_download(
                    repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                    local_dir=image_encoder_path,
                )

        ip_model = IPAdapterFaceIDPlus(
            inpaint_pipe,
            image_encoder_path,
            ip_ckpt,
            device,
        )
        logger.info("✅ IP-Adapter FaceID loaded")
        return ip_model
    except Exception as exc:
        logger.warning("IP-Adapter not available (%s) — falling back to standard inpainting", exc)
        return None


def load_all_models() -> None:
    """Preload all models into GPU memory at startup."""
    logger.info("🚀 Loading AI models (Phase 2: with FaceID + Refiner) …")

    os.makedirs(settings.models_cache_dir, exist_ok=True)
    os.makedirs(settings.tmp_dir, exist_ok=True)

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    dtype = _dtype() if device.type == "cuda" else torch.float32

    # 1. SDXL Inpainting pipeline (primary generation pipeline)
    logger.info("Loading SDXL Inpainting pipeline …")
    inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        settings.inpainting_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=settings.models_cache_dir,
    ).to(device)

    if settings.use_xformers and device.type == "cuda":
        try:
            inpaint_pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ xFormers enabled on inpaint pipeline")
        except Exception:
            logger.warning("xFormers not available")
    inpaint_pipe.enable_attention_slicing()
    inpaint_pipe.enable_vae_tiling()
    _models["inpaint_pipe"] = inpaint_pipe

    # 2. SDXL Refiner (lightweight quality boost pass)
    if settings.use_refiner:
        logger.info("Loading SDXL Refiner …")
        try:
            refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
                cache_dir=settings.models_cache_dir,
                text_encoder_2=inpaint_pipe.text_encoder_2,
                vae=inpaint_pipe.vae,  # share VAE to save VRAM
            ).to(device)
            if settings.use_xformers and device.type == "cuda":
                try:
                    refiner_pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            refiner_pipe.enable_attention_slicing()
            _models["refiner_pipe"] = refiner_pipe
            logger.info("✅ SDXL Refiner loaded (shared VAE)")
        except Exception as exc:
            logger.warning("Refiner failed to load (%s) — skipping", exc)

    # 3. Segmentation model (human parsing)
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

    # 4. OpenPose detector
    logger.info("Loading OpenPose detector …")
    pose_detector = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=settings.models_cache_dir,
    )
    _models["pose_detector"] = pose_detector

    # 5. InsightFace (ArcFace face embeddings)
    _models["face_app"] = _load_insightface(device)

    # 6. IP-Adapter FaceID (identity-conditioned generation)
    _models["ip_adapter"] = _load_ip_adapter(inpaint_pipe, device)

    _models["device"] = device
    _models["dtype"] = dtype
    logger.info("✅ All models loaded — face identity pipeline: %s",
                "ENABLED" if _models.get("ip_adapter") else "DISABLED (fallback)")


def get_model(name: str) -> Any:
    if name not in _models:
        raise RuntimeError(f"Model '{name}' not loaded.")
    return _models[name]


def models_ready() -> bool:
    return bool(_models)


def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
