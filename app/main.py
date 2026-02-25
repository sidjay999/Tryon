"""
FastAPI application entry point.
Models are preloaded at startup via lifespan context manager.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.loader import load_all_models
from app.routers import health, tryon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    logger.info("=== AI Virtual Try-On API starting up ===")
    load_all_models()
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Production-grade AI Virtual Try-On API powered by "
        "Stable Diffusion XL + ControlNet."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(health.router)
app.include_router(tryon.router)

# Serve frontend static files (when running without Nginx)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
