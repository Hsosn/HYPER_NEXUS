Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "   NEXUS AGENT - STARTING" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Force UTF-8 for all Python subprocess I/O
$env:PYTHONIOENCODING = "utf-8"

Set-Location $PSScriptRoot

# ── 1. Ensure Redis container is running (for Celery task queue) ──
Write-Host "[1/4] Checking Redis container..." -ForegroundColor Gray
$redisRunning = docker ps --filter "name=nexus-redis" --format "{{.Names}}" 2>$null
if (-not $redisRunning) {
    Write-Host "      Redis container not running. Starting..." -ForegroundColor Yellow
    docker run -d --name nexus-redis -p 6379:6379 redis:7-alpine 2>$null
}
Write-Host "      Redis container ready." -ForegroundColor Green

# ── 2. Install Python dependencies ───────────────────────────
Write-Host "[2/4] Installing Python dependencies..." -ForegroundColor Gray
python -m pip install espeakng-loader --quiet 2>$null

python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip install failed. Check the error above from requirements.txt." -ForegroundColor Red
    Write-Host "        You can retry manually: python -m pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "      Python dependencies installed." -ForegroundColor Green

# ── 3. Start Celery worker in background ─────────────────────
Write-Host "[3/4] Starting Celery worker..." -ForegroundColor Gray
$worker = Start-Job -ScriptBlock {
    param($dir)
    Set-Location $dir
    python -m celery -A nexus.celery_app worker -l info --pool=solo --concurrency=1
} -ArgumentList $PSScriptRoot
Write-Host "      Celery worker started (PID: $($worker.Id))" -ForegroundColor Green

# ── 4. Start the Nexus server ────────────────────────────────
Write-Host "[4/4] Starting Nexus server..." -ForegroundColor Gray
Write-Host ""
Write-Host " WebUI:  http://127.0.0.1:8765" -ForegroundColor Green
Write-Host " Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python run.py
