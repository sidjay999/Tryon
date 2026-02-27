# 🧥 AI Virtual Try-On (Phase 2)

> Production-grade SaaS virtual try-on platform powered by **SDXL + IP-Adapter FaceID + SDXL Refiner**.
> Preserves user identity and facial realism. Generates 1024px photorealistic results in under 20 seconds.

---

## ✨ What's New in Phase 2

| Upgrade | Effect |
|---|---|
| **Hard face mask exclusion** | Diffusion never touches the face region — zero identity drift |
| **InsightFace ArcFace** | Precise face bounding box + embedding extraction |
| **IP-Adapter FaceID Plus** | Conditions entire generation on face identity embeddings |
| **SDXL Refiner pass** | Sharpens fabric/texture detail at 0.2 strength without altering identity |
| **Synchronous API** | No Redis/Celery needed — single POST returns result directly |
| **Simplified Docker** | 2 services (api + nginx) instead of 4 |
| **Multi-garment support** | `garment_category` param: `upper` / `full` / `lower` |

---

## 🏗 Architecture

```
Person Photo + Clothing Image
       │
       ├─→ Segformer B2 Clothes: clothing mask + face bbox
       ├─→ OpenPose: body keypoints (ControlNet conditioning)
       └─→ InsightFace: ArcFace face embedding
              │
              ↓
         Mask = clothing_mask MINUS face_region (hard exclusion)
              │
              ↓
         IP-Adapter FaceID Plus
         (conditions SDXL on face identity embedding)
              │
              ↓
         SDXL Inpainting (only touches clothing region)
              │
              ↓
         SDXL Refiner (strength=0.2 — texture detail only)
              │
              ↓
         Blend: Poisson clone + Gaussian face paste + histogram match
              │
              ↓
         1024px output — face identical to input photo
```

---

## 📁 Project Structure

```
tryon/
├── app/
│   ├── main.py             # FastAPI entry point + model preload
│   ├── config.py           # All settings (env-driven, Pydantic)
│   ├── models/loader.py    # SDXL + InsightFace + IP-Adapter + Refiner
│   ├── pipeline/
│   │   ├── segmentation.py # Segformer + face bbox + mask exclusion
│   │   ├── pose.py         # OpenPose keypoints
│   │   ├── warping.py      # Affine + TPS clothing warp
│   │   ├── inpainting.py   # SDXL + IP-Adapter FaceID + Refiner
│   │   └── blending.py     # Poisson + Gaussian face paste + histogram
│   ├── queue/tasks.py      # Synchronous pipeline orchestrator
│   ├── routers/tryon.py    # POST /api/tryon (synchronous)
│   └── storage/s3.py       # S3 adapter (optional)
├── frontend/               # Vanilla HTML/CSS/JS SPA
├── nginx/nginx.conf        # Reverse proxy
├── Dockerfile
├── docker-compose.yml      # 2 services: api + nginx
├── .env.example
├── requirements.txt
├── SETUP_GUIDE.md          # Full beginner setup guide
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose v2
- NVIDIA GPU ≥ 18GB VRAM (24GB recommended)
- NVIDIA Container Toolkit installed (see `SETUP_GUIDE.md`)

### 1. Clone and configure
```bash
git clone https://github.com/sidjay999/Tryon.git tryon
cd tryon
cp .env.example .env
```

### 2. Launch
```bash
docker compose up --build
```

> ⏳ **First run:** Downloads ~17GB of model weights (SDXL + Refiner + Segformer + IP-Adapter).
> Takes 30–50 minutes. Cached in Docker volume for all subsequent starts (~90s).

### 3. Open the app
```
http://localhost           # UI
http://localhost/docs      # API Swagger docs
http://localhost/health    # GPU + model status
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `USE_FP16` | `true` | FP16 to halve VRAM |
| `USE_XFORMERS` | `true` | xFormers attention |
| `NUM_INFERENCE_STEPS` | `30` | Quality/speed tradeoff |
| `OUTPUT_SIZE` | `1024` | Output resolution |
| `USE_REFINER` | `true` | SDXL Refiner pass |
| `REFINER_STRENGTH` | `0.2` | 0.15–0.3 recommended |
| `FACE_MASK_PADDING` | `30` | Face exclusion padding in px |
| `IP_ADAPTER_SCALE` | `0.7` | FaceID identity lock strength |
| `FACE_IDENTITY_ENABLED` | `true` | Disable if InsightFace unavailable |

---

## 🧪 API

```bash
# Upload and get result in one call
curl -X POST http://localhost/api/tryon \
  -F person_image=@person.jpg \
  -F clothing_image=@shirt.jpg \
  -F garment_category=upper

# Garment categories:
# upper  → t-shirts, shirts, jackets (recommended for best quality)
# full   → dresses, sarees, full outfits
# lower  → jeans, trousers (experimental)
```

---

## 🔧 Production Scaling (When Ready)

To re-enable Celery + Redis for concurrent inference:
1. Uncomment the `redis` and `worker` sections in `docker-compose.yml`
2. Set `REDIS_URL=redis://redis:6379/0` in `.env`
3. Switch `routers/tryon.py` back to async enqueue pattern

---

## 📊 Performance

| GPU | Inference Time | VRAM |
|---|---|---|
| A10G (24GB) | ~13–18s | ~17GB |
| A100 (40GB) | ~8–12s | ~17GB |
| RTX 4090 (24GB) | ~10–15s | ~17GB |
| RTX 3090 (24GB) | ~15–20s | ~17GB (tight — disable refiner if OOM) |

---

## 📖 Docs

| Doc | Description |
|---|---|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Full beginner GPU/Docker setup guide |
| [docs/architecture.md](docs/architecture.md) | System diagrams |
| [docs/api.md](docs/api.md) | API reference |
| [docs/model_optimization.md](docs/model_optimization.md) | FP16, xFormers, batching notes |
