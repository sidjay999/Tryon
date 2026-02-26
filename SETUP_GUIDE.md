# 🚀 Complete Setup Guide — AI Virtual Try-On

> **Audience:** Beginner-friendly. You already have Docker installed and a GPU available.
> **Goal:** Run the full working prototype locally and verify GPU is working.

---

## 🗺️ Overview: What You Need

```
Your PC
 ├── NVIDIA GPU                  ← You already have this
 ├── NVIDIA Drivers              ← Step 1 (verify/install)
 ├── Docker Desktop              ← You already have this
 ├── NVIDIA Container Toolkit    ← Step 2 (lets Docker use GPU)
 └── Project Code (tryon/)      ← Already built + pushed to GitHub
```

The project itself has CUDA/PyTorch/transformers *baked into the Docker image* — you don't write CUDA code manually. You just need to make sure Docker can **see** your GPU.

---

## ✅ Step 1 — Verify Your NVIDIA Driver

Your GPU needs an NVIDIA driver ≥ 525.

Open PowerShell and run:
```powershell
nvidia-smi
```

You should see a table like this:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.x   Driver Version: 535.x   CUDA Version: 12.2             |
+-------------------------------+----------------------+----------------------+
| GeForce RTX ...               | ...                  |                      |
```

> **If you get "command not found"** → Download and install drivers from:
> [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers)
> Then reboot and try again.

---

## ✅ Step 2 — Install NVIDIA Container Toolkit (Critical!)

This is what allows Docker containers to access your GPU. Without it, the AI models won't use GPU acceleration.

### Windows (WSL2 path — required for Docker Desktop on Windows)

Docker Desktop on Windows runs containers through WSL2. Do this **inside WSL2**:

**Open WSL2** (from Start Menu → search "Ubuntu" or "WSL"):
```bash
# Add NVIDIA package repo
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Verify it worked:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
You should see your GPU listed inside that Docker output. ✅

---

## ✅ Step 3 — Enable GPU in Docker Desktop (Windows)

1. Open **Docker Desktop**
2. Go to ⚙️ Settings → **Resources** → **WSL Integration**
3. Enable integration for your Ubuntu/WSL2 distro
4. Click **Apply & Restart**

Then in Docker Desktop Settings → **General**:
- Make sure **"Use the WSL 2 based engine"** is checked ✅

---

## ✅ Step 4 — Clone the Project

```powershell
# In PowerShell or terminal:
cd C:\Users\JAY\OneDrive\Desktop
git clone https://github.com/sidjay999/Tryon.git tryon
cd tryon
```

> If you already have the folder, just `cd` into it — no need to clone again.

---

## ✅ Step 5 — Configure Environment Variables

```powershell
Copy-Item .env.example .env
```

Open `.env` in any text editor. The **minimum required settings** to run locally:

```env
# These defaults work out of the box (no GPU needed to edit):
DEVICE=cuda
USE_FP16=true
USE_XFORMERS=true
NUM_INFERENCE_STEPS=30
OUTPUT_SIZE=1024
REDIS_URL=redis://redis:6379/0

# S3 Storage — LEAVE BLANK for local testing (uses base64 fallback):
S3_BUCKET=tryon-results
S3_ENDPOINT_URL=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

> **Note:** Leaving S3 credentials empty is fine. Results will be returned as base64 images instead of URLs — the frontend handles this automatically.

---

## ✅ Step 6 — Understanding How CUDA Is Used in This Project

You don't write CUDA code directly. Here's exactly where GPU acceleration happens:

### Where it's configured — `app/config.py`
```python
device: str = "cuda"       # tells PyTorch to use GPU
use_fp16: bool = True      # FP16 halves VRAM usage
use_xformers: bool = True  # faster attention kernels
```

### Where models load onto GPU — `app/models/loader.py`
```python
# Models are loaded in FP16 directly onto CUDA:
controlnet = ControlNetModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16   # ← FP16 on GPU
)
sdxl_pipe = sdxl_pipe.to(device)  # ← moves model to CUDA GPU
sdxl_pipe.enable_xformers_memory_efficient_attention()  # ← GPU optimization
```

### Where transformers is used — `app/models/loader.py`
```python
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Segformer (the human parsing model) comes from HuggingFace transformers:
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
).to(device)
```

> **`transformers`** is already in `requirements.txt` and is installed automatically when the Docker image is built. You never need to install it manually.

### How each stage uses GPU:
| Stage | File | GPU Operation |
|---|---|---|
| Segmentation | `pipeline/segmentation.py` | Segformer inference on CUDA |
| Pose | `pipeline/pose.py` | OpenPose on CUDA |
| Inpainting | `pipeline/inpainting.py` | SDXL diffusion steps on CUDA |
| Blending | `pipeline/blending.py` | CPU (OpenCV) — no GPU needed |

---

## ✅ Step 7 — Launch the Full Stack

```powershell
cd C:\Users\JAY\OneDrive\Desktop\tryon
docker compose up --build
```

### What happens:
| Service | What it does |
|---|---|
| `redis` | Starts in seconds — job queue broker |
| `api` | Builds image, **downloads ~15GB of models** (first run only), starts FastAPI on port 8000 |
| `worker` | Celery GPU worker — processes inference jobs |
| `nginx` | Starts on port 80 — serves the UI and proxies API |

> ⏳ **First run takes 20–40 minutes** because it downloads:
> - Stable Diffusion XL base model (~7GB)
> - ControlNet OpenPose SDXL (~2.5GB)
> - SDXL Inpainting model (~7GB)
> - Segformer B2 Clothes (~400MB)
> - OpenPose weights (~300MB)
>
> **Subsequent starts: ~90 seconds** (models cached in Docker volume)

Watch for this log line to know it's ready:
```
api_1  | ✅ All models loaded successfully
```

---

## ✅ Step 8 — Open the App

Once you see "All models loaded", open your browser:
```
http://localhost
```

- **UI** → `http://localhost`
- **API Docs (Swagger)** → `http://localhost/docs`
- **Health Check** → `http://localhost/health`

---

## ✅ Step 9 — Verify GPU Is Actually Being Used

Open a second terminal while the app is running:

### Method 1 — nvidia-smi watch (live GPU usage)
```powershell
# In WSL2 or on host:
watch -n 1 nvidia-smi
```
When you submit a try-on job, GPU utilization should spike to 80-100%.

### Method 2 — Health API endpoint
```powershell
curl http://localhost/health
```
Response will show:
```json
{
  "status": "ok",
  "models_loaded": true,
  "cuda_available": true,
  "gpu": {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "memory_allocated_gb": 8.24,
    "memory_reserved_gb": 10.1,
    "total_memory_gb": 23.69
  }
}
```
If `cuda_available` is `true` and `gpu` shows your card → GPU is working ✅

### Method 3 — Inside the container
```powershell
docker exec -it tryon-api-1 python -c "import torch; print(torch.cuda.get_device_name(0))"
```
Should print your GPU name.

---

## ✅ Step 10 — Test a Try-On

1. Go to `http://localhost`
2. Drag a **full-body photo** into the left drop zone
3. Drag a **clothing image** (shirt, dress, jacket) into the right drop zone
4. Click **✦ Generate Try-On**
5. Watch the step-by-step progress bar
6. Compare before/after with the slider
7. Click **⬇ Download HD** to save the result

**Expected time:** 10–15 seconds on an RTX 3090/4090, A10G.

---

## 🔧 How to Update Configuration Later

### Change inference quality/speed
Edit `.env`:
```env
NUM_INFERENCE_STEPS=20   # faster (lower quality)
NUM_INFERENCE_STEPS=50   # slower (best quality)
```
Then restart: `docker compose restart api worker`

### Switch to CPU (no GPU)
Edit `.env`:
```env
DEVICE=cpu
USE_FP16=false
USE_XFORMERS=false
```
> ⚠️ CPU inference takes ~5 minutes per image.

### Add S3 storage (for production)
Edit `.env`:
```env
S3_BUCKET=your-bucket-name
S3_ENDPOINT_URL=             # blank for AWS, or https://your-minio-url
AWS_ACCESS_KEY_ID=AKIAXXXXXX
AWS_SECRET_ACCESS_KEY=xxxxxxxxxx
AWS_REGION=us-east-1
```
Restart: `docker compose restart api worker`

### Change output resolution
```env
OUTPUT_SIZE=768    # faster, lower res
OUTPUT_SIZE=1024   # default
```

### Rebuild after code changes
```powershell
docker compose up --build
```

### View live logs
```powershell
docker compose logs -f api      # FastAPI logs
docker compose logs -f worker   # Celery/inference logs
docker compose logs -f nginx    # Nginx logs
```

### Stop everything
```powershell
docker compose down
```

### Full reset (including downloaded models)
```powershell
docker compose down -v    # ⚠️ deletes model cache, next start re-downloads
```

---

## 🧯 Troubleshooting

| Problem | Fix |
|---|---|
| `nvidia-smi` not found | Install NVIDIA drivers, reboot |
| `docker: Error: unknown flag --gpus` | Install NVIDIA Container Toolkit (Step 2) |
| `CUDA out of memory` | Lower `NUM_INFERENCE_STEPS` or `OUTPUT_SIZE` in `.env` |
| Models fail to download | Check internet connection; retry `docker compose up` |
| Port 80 already in use | Change `"80:80"` to `"8080:80"` in `docker-compose.yml` and visit `http://localhost:8080` |
| `cuda_available: false` in /health | Container can't see GPU — redo Steps 2 & 3 |
| Celery worker crashes | Check `docker compose logs worker` — usually a VRAM issue |

---

## 📋 Quick Reference Commands

```powershell
# Start everything
docker compose up --build

# Start in background
docker compose up -d --build

# Check GPU inside container
docker exec -it tryon-api-1 python -c "import torch; print(torch.cuda.is_available())"

# Health check
curl http://localhost/health

# Watch GPU usage live
# (in WSL2): watch -n 1 nvidia-smi

# View inference logs
docker compose logs -f worker

# Stop
docker compose down
```
