# 🚀 Setup Guide — AI Virtual Try-On (No Docker)

> **Simple path:** Python + pip directly on Windows. No Docker needed.

---

## What You Need

- Windows 10/11
- NVIDIA GPU (≥16GB VRAM recommended)
- [Python 3.11](https://python.org/downloads) — check "Add to PATH" during install
- [Git](https://git-scm.com) (already have it)

---

## Step 1 — Verify NVIDIA Driver

```powershell
nvidia-smi
```
→ Should show your GPU name and CUDA version. If not, [download drivers](https://nvidia.com/drivers).

---

## Step 2 — Clone the Project

```powershell
git clone https://github.com/sidjay999/Tryon.git tryon
cd tryon
```

---

## Step 3 — Install Everything (One Command)

```powershell
.\install.ps1
```

This script automatically:
- Creates a Python virtual environment (`venv/`)
- Installs PyTorch 2.2 with CUDA 12.1 (~2.5GB)
- Installs all project dependencies (~3GB)
- Installs IP-Adapter FaceID from GitHub
- Creates required temp/cache folders
- Sets up `.env` with Windows-friendly paths

> ⏳ Takes 10–15 minutes depending on internet speed.

**If PowerShell blocks scripts, run this first:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Step 4 — Start the App

```powershell
.\start.ps1
```

Open your browser:
```
http://localhost:8000         ← Main UI
http://localhost:8000/docs    ← API docs
http://localhost:8000/health  ← GPU status
```

> ⏳ **First start downloads ~15GB of AI models** (SDXL, Refiner, Segformer, IP-Adapter).
> Takes 20–40 minutes once. Cached at `C:\tryon_models` forever after.
>
> **Ready when you see:**
> ```
> ✅ All models loaded — face identity pipeline: ENABLED
> ```

---

## Verify GPU Is Working

```powershell
# With venv active:
.\venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
→ Prints your GPU name. Or check `http://localhost:8000/health` for live GPU memory info.

---

## How Libraries Are Used

Everything installs into `venv/` — you never touch system Python.

| Library | What it does | Where used |
|---|---|---|
| `torch` | GPU tensor computation | All pipeline stages |
| `diffusers` | SDXL inpainting + refiner | `pipeline/inpainting.py` |
| `transformers` | Segformer human parsing | `pipeline/segmentation.py` |
| `insightface` | ArcFace face detection | `pipeline/segmentation.py` |
| `ip-adapter` | FaceID identity conditioning | `pipeline/inpainting.py` |
| `controlnet_aux` | OpenPose keypoints | `pipeline/pose.py` |
| `opencv-python-headless` | Poisson blending | `pipeline/blending.py` |
| `fastapi` | Web API server | `app/main.py` |

---

## Updating Configuration

Edit `.env` in the project root, then restart `.\start.ps1`:

| Variable | What it does |
|---|---|
| `NUM_INFERENCE_STEPS=20` | Faster (lower quality) |
| `NUM_INFERENCE_STEPS=50` | Slower (best quality) |
| `USE_REFINER=false` | Disable refiner (save ~6GB VRAM) |
| `IP_ADAPTER_SCALE=0.8` | Stronger face identity lock |
| `FACE_MASK_PADDING=50` | Protect larger area around face |
| `OUTPUT_SIZE=768` | Smaller output for low VRAM |

---

## Daily Workflow

```powershell
cd C:\Users\JAY\OneDrive\Desktop\tryon

# Start app
.\start.ps1

# Stop: Ctrl+C in the terminal

# After code changes, just restart:
.\start.ps1
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `install.ps1` blocked | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `torch.cuda.is_available()` = False | Reinstall PyTorch: `pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121` |
| `CUDA out of memory` | Set `USE_REFINER=false` and `OUTPUT_SIZE=768` in `.env` |
| `insightface` install fails | Try: `pip install insightface --no-build-isolation` |
| Port 8000 already in use | Change: `uvicorn app.main:app --port 8001` in `start.ps1` |
| Models keep redownloading | Check `MODELS_CACHE_DIR=C:/tryon_models` is set in `.env` |
