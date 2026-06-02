@echo off
setlocal enabledelayedexpansion

REM Force UTF-8 for all Python subprocess I/O (prevents cp1252 decode errors on Windows)
set PYTHONIOENCODING=utf-8
echo.
echo ================================
echo   NEXUS AGENT - STARTING
echo ================================
echo.
cd /d "%~dp0"

REM ── 1. Ensure Redis container is running (for Celery task queue) ──
echo [1/4] Checking Redis container...
docker ps --filter "name=nexus-redis" --format "{{.Names}}" | findstr /r "." >nul
set redis_running=%errorlevel%

if !redis_running! neq 0 (
    echo       Redis container not running. Starting...
    docker run -d --name nexus-redis -p 6379:6379 redis:7-alpine 2>nul
)
echo       Redis container ready.

REM ── 2. Install Python dependencies ──────────────────────────
echo [2/4] Installing Python dependencies...
python -m pip install espeakng-loader --quiet 2>nul

python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] pip install failed. Check the error above from requirements.txt.
    pause
    exit /b 1
)
echo       Python dependencies installed.

REM ── 3. Start Celery worker in background ────────────────────
echo [3/4] Starting Celery worker...
start /B python -m celery -A nexus.celery_app worker -l info --pool=solo --concurrency=1 > celery_worker.log 2>&1
timeout /t 3 /nobreak >nul
echo       Celery worker started (check celery_worker.log if issues)

REM ── 4. Start the Nexus server ───────────────────────────────
echo [4/4] Starting Nexus server...
echo.
echo  WebUI: http://127.0.0.1:8765
echo  Press Ctrl+C to stop
echo.
python run.py
