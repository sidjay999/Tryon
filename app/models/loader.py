"""
Model loader – CatVTON Production Pipeline.
Loads: CatVTON diffusion pipeline + AutoMasker (DensePose + SCHP).
"""
import logging
import os
import sys
import torch
from huggingface_hub import snapshot_download

from app.config import get_settings

logger = logging.getLogger(__name__)

# Global model registry
_models: dict = {}


def _init_weight_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    if precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_all_models():
    """Load CatVTON pipeline + AutoMasker. Called once at startup."""
    settings = get_settings()
    device = settings.device
    weight_dtype = _init_weight_dtype(settings.mixed_precision)

    logger.info("GPU: %s | Device: %s | Precision: %s",
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                device, settings.mixed_precision)

    # ── 1. Download CatVTON checkpoint from HuggingFace ────────
    logger.info("Downloading CatVTON checkpoint (first run only)...")
    repo_path = snapshot_download(
        repo_id=settings.catvton_repo_id,
        cache_dir=settings.models_cache_dir,
    )
    logger.info("CatVTON checkpoint at: %s", repo_path)
    _models["repo_path"] = repo_path

    # ── 2. Find CatVTON source directory ───────────────────────
    # CatVTON source is cloned alongside our project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catvton_dir = settings.catvton_dir or os.path.join(project_root, "CatVTON")
    if not os.path.isdir(catvton_dir):
        # Try alternative location
        catvton_dir = os.path.join(project_root, "catvton_lib")
    if not os.path.isdir(catvton_dir):
        raise FileNotFoundError(
            f"CatVTON source not found. Expected at {catvton_dir}. "
            "Clone it: git clone https://github.com/Zheng-Chong/CatVTON.git CatVTON"
        )
    logger.info("CatVTON source at: %s", catvton_dir)

    # Add CatVTON to Python path so its imports work
    if catvton_dir not in sys.path:
        sys.path.insert(0, catvton_dir)

    # ── 3. Load CatVTON Pipeline ───────────────────────────────
    from model.pipeline import CatVTONPipeline  # noqa: E402

    logger.info("Loading CatVTON pipeline (base: %s)...", settings.base_model_id)
    pipeline = CatVTONPipeline(
        base_ckpt=settings.base_model_id,
        attn_ckpt=repo_path,
        attn_ckpt_version=settings.attn_ckpt_version,
        weight_dtype=weight_dtype,
        device=device,
        skip_safety_check=True,  # We handle safety ourselves
        use_tf32=True,
    )
    _models["pipeline"] = pipeline
    logger.info("CatVTON pipeline loaded (%s)", settings.mixed_precision)

    # ── 4. Load AutoMasker (DensePose + SCHP) ──────────────────
    from model.cloth_masker import AutoMasker  # noqa: E402

    logger.info("Loading AutoMasker (DensePose + SCHP)...")
    automasker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=device,
    )
    _models["automasker"] = automasker
    logger.info("AutoMasker loaded")

    # ── 5. Store metadata ──────────────────────────────────────
    _models["device"] = device
    _models["dtype"] = weight_dtype
    _models["settings"] = settings

    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3 if torch.cuda.is_available() else 0
    logger.info("=== All models loaded | VRAM: %.1fGB | CatVTON | DensePose+SCHP ===", vram)


def get_model(key: str, optional: bool = False):
    """Get a loaded model by key."""
    if key not in _models:
        if optional:
            return None
        raise RuntimeError(
            f"Model '{key}' not loaded. Available: {list(_models.keys())}"
        )
    return _models[key]
