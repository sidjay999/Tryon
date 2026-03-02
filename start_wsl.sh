#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CatVTON – WSL Start Script
# Run inside WSL: bash start_wsl.sh
# ═══════════════════════════════════════════════════════════════
set -e

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════"
echo "  CatVTON Virtual Try-On Server"
echo "═══════════════════════════════════════════════════"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "$HOME/tryon/venv" ]; then
    source "$HOME/tryon/venv/bin/activate"
else
    echo "ERROR: No venv found. Run setup_wsl.sh first."
    exit 1
fi

# Check GPU
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'GPU: {gpu} ({vram:.1f}GB VRAM)')
else:
    print('WARNING: No GPU detected')
"

# Ensure CatVTON is on Python path
export PYTHONPATH="$SCRIPT_DIR/CatVTON:$SCRIPT_DIR/catvton_lib:$PYTHONPATH"

echo ""
echo "Starting server..."
echo "  UI     → http://localhost:8000"
echo "  API    → http://localhost:8000/docs"
echo "  Health → http://localhost:8000/health"
echo ""
echo "First run: downloads AI models (~5GB). Wait 10-20 min."
echo "After that: starts in ~60 seconds each time."
echo ""
echo "Press Ctrl+C to stop."
echo ""

# Start uvicorn
python3 -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
