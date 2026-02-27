# ============================================================
# AI Virtual Try-On — Start Server (No Docker)
# Run this every time you want to start the app
# ============================================================

$ErrorActionPreference = "Stop"

# Check venv exists
if (-Not (Test-Path ".\venv")) {
    Write-Host "❌ Virtual environment not found. Run install.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate venv
.\venv\Scripts\Activate.ps1

Write-Host "`n=== AI Virtual Try-On ===" -ForegroundColor Cyan

# Quick CUDA check
$cudaAvail = python -c "import torch; print('YES' if torch.cuda.is_available() else 'NO')" 2>$null
if ($cudaAvail -eq "YES") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "✅ GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Host "⚠️  GPU not detected — running on CPU (very slow)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Starting server…" -ForegroundColor Cyan
Write-Host "   UI  → http://localhost:8000" -ForegroundColor White
Write-Host "   API → http://localhost:8000/docs" -ForegroundColor White
Write-Host "   Health → http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "⏳ First run: downloading AI models (~15GB). Please wait 20-40 min." -ForegroundColor Yellow
Write-Host "   Subsequent starts: ~90 seconds" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

# Start FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000
