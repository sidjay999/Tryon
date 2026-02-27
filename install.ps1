# AI Virtual Try-On - One-Click Installer (No Docker)
# Run once to set up the virtual environment and install all dependencies

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== AI Virtual Try-On - Setup ===" -ForegroundColor Cyan

# 1. Check Python
try {
    $pyVersion = python --version 2>&1
    Write-Host "OK: $pyVersion found" -ForegroundColor Green
}
catch {
    Write-Host "ERROR: Python not found. Install from https://python.org/downloads" -ForegroundColor Red
    exit 1
}

# 2. Create virtual environment
if (-Not (Test-Path ".\venv")) {
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}
else {
    Write-Host "OK: Virtual environment already exists" -ForegroundColor Green
}

# 3. Activate
Write-Host "Activating venv..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip --quiet

# 5. Install PyTorch with CUDA 12.1 first (downloads ~2.5GB)
Write-Host ""
Write-Host "Installing PyTorch with CUDA 12.1 (downloads ~2.5GB)..." -ForegroundColor Yellow
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 6. Verify CUDA
Write-Host ""
Write-Host "Verifying CUDA..." -ForegroundColor Yellow
$cudaAvail = python -c "import torch; print(int(torch.cuda.is_available()))"
if ($cudaAvail -eq "1") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "OK: CUDA available - GPU: $gpuName" -ForegroundColor Green
}
else {
    Write-Host "WARNING: CUDA not detected. Check NVIDIA drivers. Will run on CPU (slow)." -ForegroundColor Yellow
}

# 7. Install all project dependencies (downloads ~3-4GB)
Write-Host ""
Write-Host "Installing project dependencies (downloads ~3-4GB)..." -ForegroundColor Yellow
pip install -r requirements.txt

# 8. Install IP-Adapter from GitHub
Write-Host ""
Write-Host "Installing IP-Adapter FaceID..." -ForegroundColor Yellow
pip install "git+https://github.com/tencent-ailab/IP-Adapter.git"

# 9. Create required directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "C:\tmp\tryon" | Out-Null
New-Item -ItemType Directory -Force -Path "C:\tryon_models" | Out-Null
Write-Host "OK: C:\tmp\tryon and C:\tryon_models created" -ForegroundColor Green

# 10. Set up .env
if (-Not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    $envContent = Get-Content ".\.env"
    $envContent = $envContent -replace "TMP_DIR=.*", "TMP_DIR=C:/tmp/tryon"
    $envContent = $envContent -replace "MODELS_CACHE_DIR=.*", "MODELS_CACHE_DIR=C:/tryon_models"
    $envContent | Set-Content ".\.env"
    Write-Host "OK: .env created from template" -ForegroundColor Green
}
else {
    Write-Host "OK: .env already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next - start the app:" -ForegroundColor White
Write-Host "  .\start.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: First start downloads AI models (~15GB)." -ForegroundColor Yellow
Write-Host "      Takes 20-40 minutes once, then cached forever." -ForegroundColor Yellow
Write-Host ""
