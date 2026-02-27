# 🚀 Complete Setup Guide — AI Virtual Try-On (Phase 2)

> Updated for Phase 2: simplified infrastructure (2 Docker services), IP-Adapter FaceID, SDXL Refiner.
> You already have Docker installed and a GPU available.

---

## 🗺️ Overview

```
Your PC
 ├── NVIDIA GPU (≥18GB VRAM recommended)   ← You have this
 ├── NVIDIA Drivers                         ← Step 1
 ├── Docker Desktop                         ← You have this
 └── NVIDIA Container Toolkit              ← Step 2 (critical)

Inside Docker (automatic):
 ├── Python 3.11 + PyTorch 2.2 (CUDA 12.1)
 ├── SDXL Inpainting + SDXL Refiner
 ├── InsightFace (ArcFace face embeddings)
 ├── IP-Adapter FaceID Plus
 ├── Segformer B2 Clothes (human parsing)
 └── OpenPose (pose keypoints)
```

You do **not** install Python, transformers, or PyTorch manually. Everything runs inside the Docker container.

---

## ✅ Step 1 — Verify NVIDIA Driver

```powershell
nvidia-smi
```

Expected output:
```
NVIDIA-SMI 535.x   Driver Version: 535.x   CUDA Version: 12.x
GPU Name: GeForce RTX ...
```

→ **If not found:** Download from [nvidia.com/drivers](https://www.nvidia.com/drivers), install, reboot.

**Minimum driver version:** 525 (for CUDA 12.1 support)

---

## ✅ Step 2 — Install NVIDIA Container Toolkit (Critical)

This allows Docker containers to access your GPU.

Open **WSL2** (search "Ubuntu" in Start Menu):

```bash
# Add NVIDIA container toolkit repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use GPU
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Verify it works:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
→ Should print your GPU table. ✅

---

## ✅ Step 3 — Enable GPU in Docker Desktop

1. Open **Docker Desktop** → ⚙️ Settings
2. **General** → ✅ "Use the WSL 2 based engine"
3. **Resources → WSL Integration** → Enable your Ubuntu distro
4. Click **Apply & Restart**

---

## ✅ Step 4 — Clone the Project

```powershell
git clone https://github.com/sidjay999/Tryon.git tryon
cd tryon
```

> If you already have the folder from previous steps, just `cd` into it.

---

## ✅ Step 5 — Configure Environment

```powershell
Copy-Item .env.example .env
```

**Minimum settings to get running locally** (no S3 needed):

```env
DEVICE=cuda
USE_FP16=true
USE_XFORMERS=true
USE_REFINER=true
REFINER_STRENGTH=0.2
IP_ADAPTER_SCALE=0.7
FACE_IDENTITY_ENABLED=true

# Leave S3 fields blank — results returned as base64:
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

> **Low VRAM (16GB)?** Set `USE_REFINER=false` and `NUM_INFERENCE_STEPS=20` to reduce memory usage.

---

## ✅ Step 6 — Launch the Stack

```powershell
docker compose up --build
```

### What starts:
| Service | Port | Purpose |
|---|---|---|
| `api` | 8000 | FastAPI + GPU inference (all models) |
| `nginx` | 80 | Serves frontend UI + proxies /api/* to FastAPI |

### What downloads (first run — one time only):
| Model | Size | Purpose |
|---|---|---|
| SDXL Inpainting | ~7GB | Core clothing generation |
| SDXL Refiner | ~6GB | Texture sharpening pass |
| Segformer B2 Clothes | ~400MB | Human parsing |
| OpenPose | ~300MB | Body keypoints |
| IP-Adapter FaceID | ~600MB | Face identity conditioning |
| InsightFace buffalo_l | ~500MB | ArcFace face detection |

> ⏳ **First run: 30–50 minutes** (downloading ~15GB). Saved to Docker volume.
> ⚡ **After first run: ~90 seconds** to start.

### Ready signal (watch logs for this):
```
api_1  | ✅ All models loaded — face identity pipeline: ENABLED
```

---

## ✅ Step 7 — Open the App

```
http://localhost          ← Main UI
http://localhost/docs     ← Swagger API docs
http://localhost/health   ← GPU + model status
```

---

## ✅ Step 8 — Verify GPU Is Being Used

### Method 1 — GPU usage during inference
```bash
# In WSL2, while making a request:
watch -n 1 nvidia-smi
# GPU-Util should spike to 80-100% during generation
```

### Method 2 — Health endpoint
```powershell
curl http://localhost/health
```
Check that `cuda_available: true` and your GPU name appears.

### Method 3 — Inside the container
```powershell
docker exec -it tryon-api-1 python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## ✅ Step 9 — Run a Try-On

1. Go to `http://localhost`
2. Drop a **full-body person photo** in the left zone
3. Drop a **clothing item photo** in the right zone
4. Click **✦ Generate Try-On**
5. Wait 15–20 seconds — watch the pipeline steps
6. Drag the **Before / After slider** to compare
7. Click **⬇ Download HD** to save

---

## 🔧 Updating Configuration

### Tune identity preservation strength
```env
IP_ADAPTER_SCALE=0.5   # less locked to face (more creative)
IP_ADAPTER_SCALE=0.9   # stronger face identity lock
FACE_MASK_PADDING=50   # protect a larger area around face
```
Restart: `docker compose restart api`

### Tune quality vs speed
```env
NUM_INFERENCE_STEPS=20   # faster (~10s)
NUM_INFERENCE_STEPS=50   # best quality (~30s)
REFINER_STRENGTH=0.3     # more texture detail (slight identity risk above 0.4)
```

### Disable refiner (low VRAM)
```env
USE_REFINER=false
```

### Switch to CPU (no GPU)
```env
DEVICE=cpu
USE_FP16=false
USE_XFORMERS=false
FACE_IDENTITY_ENABLED=false   # InsightFace GPU backend won't work
```
> ⚠️ CPU inference takes ~5–10 minutes per image.

### View live logs
```powershell
docker compose logs -f api      # FastAPI inference logs
docker compose logs -f nginx    # Request logs
```

### Stop the stack
```powershell
docker compose down
```

### Full model cache reset (re-downloads everything)
```powershell
docker compose down -v   # ⚠️ deletes model_cache volume
```

---

## 🧯 Troubleshooting

| Problem | Solution |
|---|---|
| `cuda_available: false` in /health | Redo Steps 2 & 3 — Container Toolkit not configured |
| `CUDA out of memory` | Set `USE_REFINER=false` and/or lower `OUTPUT_SIZE=768` in `.env` |
| `ip_adapter` shows DISABLED in logs | IP-Adapter weights didn't download — check internet, retry `docker compose up` |
| Port 80 already in use | Change `"80:80"` → `"8080:80"` in `docker-compose.yml` → visit `http://localhost:8080` |
| Models never finish loading | First-run download. Wait 30–50 min. Watch `docker compose logs -f api` |
| Face still changes slightly | Increase `FACE_MASK_PADDING=50` and lower `IP_ADAPTER_SCALE=0.8` |

---

## 📋 Quick Reference Commands

```powershell
# Start
docker compose up --build

# Start (background)
docker compose up -d --build

# Check GPU
docker exec -it tryon-api-1 python -c "import torch; print(torch.cuda.is_available())"

# Health check
curl http://localhost/health

# Restart after .env change
docker compose restart api

# View logs
docker compose logs -f api

# Stop
docker compose down
```
