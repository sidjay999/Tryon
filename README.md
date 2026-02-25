# 🧥 AI Virtual Try-On

> Production-grade SaaS virtual try-on platform powered by **Stable Diffusion XL + ControlNet**.
> Upload a person photo and a clothing image — get a photorealistic 1024px try-on result in under 15 seconds.

![Architecture](docs/architecture.md)

---

## ✨ Features

- **SDXL + ControlNet** – pose-conditioned inpainting for photorealistic results
- **Segformer B2 Clothes** – accurate human parsing and clothing mask extraction
- **TPS Clothing Warp** – affine + thin-plate-spline geometric fitting
- **Poisson Blending** – seamless boundary compositing + face identity preservation
- **FP16 + xFormers** – memory-efficient inference on 24GB+ GPUs
- **Async Queue** – Celery + Redis for non-blocking, scalable inference
- **S3-Compatible Storage** – AWS S3, MinIO, Cloudflare R2
- **Modern UI** – glassmorphism design, drag-and-drop, before/after slider

---

## 📁 Project Structure

```
tryon/
├── app/                   # FastAPI backend
│   ├── main.py            # App entry point + lifespan model loading
│   ├── config.py          # Pydantic Settings (env-driven)
│   ├── models/loader.py   # Model preloader (SDXL, ControlNet, Segformer, OpenPose)
│   ├── pipeline/          # 5-stage ML pipeline
│   │   ├── segmentation.py
│   │   ├── pose.py
│   │   ├── warping.py
│   │   ├── inpainting.py
│   │   └── blending.py
│   ├── routers/           # API endpoints
│   ├── queue/             # Celery worker + tasks
│   ├── storage/           # S3 adapter
│   └── utils/             # Image utilities
├── frontend/              # Vanilla HTML/CSS/JS SPA
├── nginx/                 # Reverse proxy config
├── scripts/               # Startup scripts
├── docs/                  # Architecture, API, optimization notes
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose v2
- NVIDIA GPU with ≥24GB VRAM + NVIDIA Container Toolkit
- ~20GB disk space (model weights)

### 1. Clone and configure

```bash
git clone <your-repo> tryon
cd tryon
cp .env.example .env
# Edit .env — add S3 credentials if desired
```

### 2. Launch

```bash
docker compose up --build
```

> ⚠️ **First run:** models are downloaded from Hugging Face (~15GB). This takes 20–40 minutes. Subsequent starts load from the `model_cache` volume in ~90 seconds.

### 3. Open the UI

```
http://localhost
```

API Docs: `http://localhost/docs`

---

## ☁️ Deployment

### AWS EC2 (g5.xlarge – A10G 24GB)

```bash
# 1. Launch g5.xlarge with Deep Learning AMI (Ubuntu 22.04)
# 2. Install Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ct.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-ct.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(. /etc/os-release; echo $ID$VERSION_ID) /" | sudo tee /etc/apt/sources.list.d/nvidia-ct.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Deploy
git clone <your-repo> tryon && cd tryon
cp .env.example .env   # fill in S3 credentials
docker compose up -d --build
```

### RunPod

1. Create a pod with **NVIDIA A4000/A6000**, runtime image: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
2. Open HTTP port 80
3. Clone repo, configure `.env`, run `docker compose up --build`

---

## ⚙️ Configuration

See `.env.example` for all options. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `USE_FP16` | `true` | Enable FP16 precision |
| `USE_XFORMERS` | `true` | Enable xFormers attention |
| `NUM_INFERENCE_STEPS` | `30` | Diffusion steps (20=fast, 50=best) |
| `OUTPUT_SIZE` | `1024` | Output image size in pixels |
| `S3_BUCKET` | `tryon-results` | S3 bucket for results |
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker URL |

---

## 📖 Documentation

| Doc | Description |
|---|---|
| [Architecture](docs/architecture.md) | System overview + Mermaid diagrams |
| [API Reference](docs/api.md) | Endpoint docs + curl / Python examples |
| [Model Optimization](docs/model_optimization.md) | FP16, xFormers, VAE tiling, batching |

---

## 📊 Performance Targets

| GPU | Inference Time | Quality |
|---|---|---|
| A10G (24GB) | ~10-13s | ✅ Production |
| A100 (40GB) | ~6-9s | ✅ Best |
| RTX 4090 (24GB) | ~8-11s | ✅ Production |
| RTX 3090 (24GB) | ~12-15s | ✅ Acceptable |

---

## 📜 License

MIT — feel free to use for commercial and non-commercial projects.
