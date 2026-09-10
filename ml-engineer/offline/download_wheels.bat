@echo off
REM Batch script to pre-fetch wheels for offline environment
echo =======================================================
echo Downloading Python wheels for offline air-gapped demo
echo =======================================================

python offline\download_wheels.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to download offline wheels.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] Offline package bundle ready.
pause
