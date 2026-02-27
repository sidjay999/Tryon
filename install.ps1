# ============================================================
# AI Virtual Try-On — One-Click Windows Installer (No Docker)
# Run this ONCE to set up the virtual environment and install deps
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "`n=== AI Virtual Try-On — Setup ===" -ForegroundColor Cyan

# 1. Check Python
try { $pyVersion = python --version 2>&1; Write-Host "✅ $pyVersion found" -ForegroundColor Green }
catch { Write-Host "❌ Python not found. Install from https://python.org/downloads" -ForegroundColor Red; exit 1 }

# 2. Create virtual environment
if (-Not (Test-Path ".\venv")) {
    Write-Host "`n📦 Creating virtual environment…" -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# 3. Activate
Write-Host "`n⚡ Activating venv…" -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip --quiet

# 5. Install PyTorch with CUDA 12.1 first
Write-Host "`n🔥 Installing PyTorch with CUDA 12.1…" -ForegroundColor Yellow
Write-Host "   (This downloads ~2.5GB — takes a few minutes)" -ForegroundColor Gray
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 --quiet

# 6. Verify CUDA
Write-Host "`n🔍 Verifying CUDA availability…" -ForegroundColor Yellow
$cudaAvail = python -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')"
if ($cudaAvail -eq "YES") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "✅ CUDA available — GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Host "⚠️  CUDA not detected — will run on CPU (slow). Check your NVIDIA drivers." -ForegroundColor Yellow
}

# 7. Install project dependencies
Write-Host "`n📚 Installing project dependencies…" -ForegroundColor Yellow
Write-Host "   (This downloads ~3-4GB of packages)" -ForegroundColor Gray
pip install -r requirements.txt --quiet

# 8. Install IP-Adapter from GitHub
Write-Host "`n🎭 Installing IP-Adapter (FaceID)…" -ForegroundColor Yellow
pip install git+https://github.com/tencent-ailab/IP-Adapter.git --quiet

# 9. Create required directories
Write-Host "`n📁 Creating directories…" -ForegroundColor Yellow
$tmpDir = "C:\tmp\tryon"
$cacheDir = "C:\tryon_models"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
Write-Host "✅ Directories created: $tmpDir, $cacheDir" -ForegroundColor Green

# 10. Set up .env if not already there
if (-Not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    # Set Windows-friendly paths
    (Get-Content ".\.env") `
        -replace 'TMP_DIR=.*', 'TMP_DIR=C:/tmp/tryon' `
        -replace 'MODELS_CACHE_DIR=.*', 'MODELS_CACHE_DIR=C:/tryon_models' |
        Set-Content ".\.env"
    Write-Host "✅ .env created from template" -ForegroundColor Green
} else {
    Write-Host "✅ .env already exists" -ForegroundColor Green
}

Write-Host "`n" + ("=" * 50) -ForegroundColor Cyan
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next step — run the app:" -ForegroundColor White
Write-Host "   .\start.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏳ First start downloads AI models (~15GB). " -ForegroundColor Yellow
Write-Host "   This takes 20-40 minutes once, then cached forever." -ForegroundColor Yellow
Write-Host ""
