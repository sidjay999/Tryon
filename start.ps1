# AI Virtual Try-On - Start Server (No Docker)
# Run this every time you want to start the app

# Check venv exists
if (-Not (Test-Path ".\venv")) {
    Write-Host "ERROR: Run install.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate venv
.\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "=== AI Virtual Try-On ===" -ForegroundColor Cyan

# CUDA check
$cudaAvail = python -c "import torch; print(int(torch.cuda.is_available()))"
if ($cudaAvail -eq "1") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "GPU: $gpuName" -ForegroundColor Green
}
else {
    Write-Host "WARNING: GPU not detected - running on CPU (very slow)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Cyan
Write-Host "  UI     -> http://localhost:8000" -ForegroundColor White
Write-Host "  API    -> http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Health -> http://localhost:8000/health" -ForegroundColor White
Write-Host ""
Write-Host "First run: downloads AI models (~15GB). Wait 20-40 min." -ForegroundColor Yellow
Write-Host "After that: starts in ~90 seconds each time." -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

uvicorn app.main:app --host 0.0.0.0 --port 8000
