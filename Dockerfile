# ══════════════════════════════════════════════════════════════
# AI Virtual Try-On – Dockerfile
# Base: NVIDIA CUDA 12.1 + Python 3.11
# ══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-distutils \
    git curl wget libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Alias python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Install Python deps first (leverage Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/

# Create required directories
RUN mkdir -p /tmp/tryon /app/model_cache

# Expose FastAPI port
EXPOSE 8000

# Environment defaults (override via docker-compose or .env)
ENV DEVICE=cuda \
    USE_FP16=true \
    USE_XFORMERS=true \
    NUM_INFERENCE_STEPS=30 \
    OUTPUT_SIZE=1024

# Startup
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
