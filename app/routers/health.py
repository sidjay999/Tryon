"""
Health check router – CatVTON Pipeline.
"""
import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.loader import get_model

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
        }

    models_loaded = get_model("pipeline", optional=True) is not None
    automasker_loaded = get_model("automasker", optional=True) is not None

    return JSONResponse({
        "status": "ok",
        "pipeline": "CatVTON (ICLR 2025)",
        "models_loaded": models_loaded,
        "automasker_loaded": automasker_loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
    })
