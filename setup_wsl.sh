#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CatVTON – WSL Setup Script
# Sets up the full production-grade virtual try-on pipeline
# Run this inside WSL Ubuntu: bash setup_wsl.sh
# ═══════════════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════════════"
echo "  CatVTON Virtual Try-On – WSL Setup"
echo "═══════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12-venv python3-pip git wget curl build-essential libgl1 libglib2.0-0

# ── 2. Project directory ──────────────────────────────────────
PROJECT_DIR="$HOME/tryon"
CATVTON_DIR="$PROJECT_DIR/CatVTON"

if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p "$PROJECT_DIR"
fi

# Clone CatVTON if not already present
if [ ! -d "$CATVTON_DIR" ]; then
    echo "[2/7] Cloning CatVTON..."
    git clone https://github.com/Zheng-Chong/CatVTON.git "$CATVTON_DIR"
else
    echo "[2/7] CatVTON already cloned, pulling latest..."
    cd "$CATVTON_DIR" && git pull && cd "$PROJECT_DIR"
fi

# ── 3. Python venv ────────────────────────────────────────────
echo "[3/7] Setting up Python virtual environment..."
cd "$PROJECT_DIR"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# ── 4. PyTorch with CUDA ──────────────────────────────────────
echo "[4/7] Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ── 5. CatVTON dependencies ──────────────────────────────────
echo "[5/7] Installing CatVTON dependencies..."
pip install accelerate==0.31.0 \
    diffusers \
    transformers \
    matplotlib \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    pillow \
    PyYAML \
    scipy \
    scikit-image \
    tqdm \
    fvcore \
    cloudpickle \
    omegaconf \
    pycocotools \
    av \
    peft \
    huggingface_hub \
    hf_transfer

# ── 6. Server dependencies ───────────────────────────────────
echo "[6/7] Installing server dependencies..."
pip install fastapi uvicorn python-multipart python-dotenv boto3 aiofiles

# ── 7. Verify GPU ────────────────────────────────────────────
echo "[7/7] Verifying GPU access..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('WARNING: CUDA not available! GPU inference will not work.')
"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete!"
echo "  To start the server:  bash start_wsl.sh"
echo "═══════════════════════════════════════════════════"
