# AI Virtual Try-On – Architecture

## System Overview

```mermaid
flowchart TD
    USER["🧑 User Browser"] -->|"POST /api/tryon\n(person + clothing images)"| NGINX["Nginx\nReverse Proxy :80"]
    NGINX -->|"/api/*"| API["FastAPI Service :8000\n(preloaded models)"]
    NGINX -->|"/"| FE["Frontend Static\n(Nginx :80)"]
    API -->|"enqueue task"| REDIS["Redis :6379\n(broker + result backend)"]
    REDIS -->|"consume"| WORKER["Celery GPU Worker"]
    WORKER -->|"upload result"| S3["S3-Compatible\nObject Storage"]
    WORKER -->|"store job state"| REDIS
    USER -->|"GET /api/tryon/{id}\n(polling)"| API
    API -->|"read job state"| REDIS
    API -->|"presigned URL"| USER
    USER -->|"download result"| S3
```

## Pipeline Detail

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        P["Person Image"]
        C["Clothing Image"]
    end

    subgraph PIPE["Celery Pipeline"]
        S1["1. Segmentation\n(Segformer B2 Clothes)\n→ clothing mask\n→ face mask"]
        S2["2. Pose Detection\n(OpenPose / controlnet_aux)\n→ keypoint image"]
        S3["3. Clothing Warp\n(Affine + TPS)\n→ aligned garment"]
        S4["4. SDXL Inpainting\n(+ ControlNet Pose)\n→ raw result 1024px"]
        S5["5. Blending\n(Poisson clone +\nface restore +\nhistogram match)\n→ final image"]
    end

    subgraph OUTPUT["Output"]
        R["Result PNG\n1024 × 1024"]
    end

    P --> S1
    P --> S2
    C --> S3
    S1 -->|mask| S3
    S1 -->|mask| S4
    S2 -->|pose image| S4
    S3 -->|warped clothing| S4
    S4 -->|generated| S5
    P -->|original| S5
    S1 -->|face mask| S5
    S5 --> R
```

## Technology Stack

| Layer | Technology |
|---|---|
| Base Model | Stable Diffusion XL (stabilityai/sdxl-base-1.0) |
| Pose Conditioning | ControlNet OpenPose SDXL |
| Human Parsing | Segformer B2 Clothes |
| Pose Detector | controlnet_aux OpenposeDetector |
| Blending | OpenCV (Poisson clone), MediaPipe (face) |
| API Framework | FastAPI + Uvicorn (uvloop) |
| Async Queue | Celery 5 + Redis 7 |
| Object Storage | S3-compatible (AWS / MinIO / Cloudflare R2) |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| Reverse Proxy | Nginx Alpine |
| Containerization | Docker + Docker Compose |
| GPU Target | NVIDIA A10G / A100 / RTX 3090+ |

## Deployment Targets

| Platform | Instance Type | Notes |
|---|---|---|
| AWS EC2 | g5.xlarge (A10G, 24GB) | Recommended for production |
| RunPod | RTX A4000 / A6000 | Cost-effective scaling |
| Lambda Labs | A100 80GB | High-throughput batch |
| Local Dev | RTX 3090 24GB | FP16 required |
