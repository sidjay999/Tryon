# TryOnAI — Virtual Try-On (CatVTON)

Production-grade AI virtual try-on powered by **CatVTON** (ICLR 2025).  
Upload a person photo + a garment → get a photorealistic 1024×768 result.

## Features

- **CatVTON diffusion** — learned garment transfer via latent concatenation
- **DensePose + SCHP** — pixel-level human parsing (no bounding box hacks)
- **1024×768 HD output** — single-pass generation, no blending artifacts
- **Garment types** — upper / lower / full-body
- **FastAPI backend** — REST API + web UI with before/after slider
- **6GB VRAM** — runs on RTX 4050 Laptop GPU via FP16

## Requirements

- **WSL2 (Ubuntu)** with NVIDIA GPU driver
- Python 3.12 + PyTorch + CUDA
- ~5GB disk for model weights (downloaded automatically on first run)

## Quick Start

```bash
# 1. Open WSL
wsl -d Ubuntu
cd /mnt/c/Users/JAY/OneDrive/Desktop/tryon

# 2. First-time setup (installs everything)
bash setup_wsl.sh

# 3. Start the server
bash start_wsl.sh
```

Open `http://localhost:8000` in your Windows browser.

## Project Structure

```
app/
├── config.py           # CatVTON settings (resolution, steps, precision)
├── main.py             # FastAPI app with model preloading
├── models/
│   └── loader.py       # Loads CatVTON pipeline + AutoMasker
├── queue/
│   └── tasks.py        # 3-stage pipeline orchestration
├── routers/
│   ├── health.py       # GET /health
│   └── tryon.py        # POST /api/tryon
├── storage/
│   └── s3.py           # Optional S3 upload
└── utils/
    └── image.py        # Image preprocessing
frontend/
├── index.html          # Web UI
├── css/style.css       # Dark glassmorphism theme
└── js/                 # Upload, progress, slider modules
setup_wsl.sh            # WSL environment setup
start_wsl.sh            # Server launcher
```

## Pipeline

```
Person Image + Garment Image
        ↓
  AutoMasker (DensePose + SCHP)
        ↓ agnostic mask
  CatVTON Diffusion (50 steps, CFG 2.5)
        ↓
  1024×768 Result
```

## API

```bash
# Try-on
curl -X POST http://localhost:8000/api/tryon \
  -F person_image=@person.jpg \
  -F clothing_image=@garment.jpg \
  -F garment_category=upper

# Health check
curl http://localhost:8000/health
```

Interactive docs at `http://localhost:8000/docs`.

## License

CatVTON model: see [Zheng-Chong/CatVTON](https://github.com/Zheng-Chong/CatVTON).
