@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title XAI Dashboard
color 0A

echo ========================================
echo  XAI - API + React Dashboard
echo ========================================
echo.
echo Working folder:
echo   %CD%
echo.

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "API_URL=http://127.0.0.1:8000"
set "UI_URL=http://localhost:3000"

REM --- Find a system Python if venv is missing ---
set "SYS_PY="
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PY=%%I"
)
if not defined SYS_PY (
    where python >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "SYS_PY=%%I"
    )
)

REM --- Create venv if missing ---
if not exist "%VENV_PY%" (
    echo [1/5] Creating virtual environment...
    if not defined SYS_PY (
        echo [ERROR] Python was not found.
        echo Install Python 3.10+ from python.org and tick "Add python.exe to PATH".
        goto :fail
    )
    "%SYS_PY%" -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create .venv
        goto :fail
    )
) else (
    echo [1/5] Virtual environment found.
)

if not exist "%VENV_PY%" (
    echo [ERROR] .venv\Scripts\python.exe is missing.
    goto :fail
)

REM --- Install Python dependencies ---
echo [2/5] Checking Python dependencies...
"%VENV_PY%" -m pip install --upgrade pip >nul
"%VENV_PY%" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed.
    goto :fail
)

REM --- Frontend dependencies ---
echo [3/5] Checking frontend dependencies...
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm was not found. Install Node.js LTS from https://nodejs.org
    goto :fail
)
if not exist "%CD%\frontend\node_modules" (
    pushd "%CD%\frontend"
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] npm install failed.
        goto :fail
    )
    popd
) else (
    echo Frontend node_modules found.
)

REM --- Pipeline: only if not already trained ---
echo [4/5] Checking training status...
if exist "%CD%\checkpoints\train.done" (
    echo Training already completed - skipping pipeline.
) else (
    echo Training not found - running full pipeline...
    echo This can take a long time on first run.
    echo.
    "%VENV_PY%" -m src.pipeline all
    if errorlevel 1 (
        echo.
        echo [ERROR] Pipeline failed.
        echo Put CICIDS-2017 CSV files in:
        echo   data\raw\CICIDS2017\
        echo Then run this file again.
        goto :fail
    )
)

REM --- Launch API + React ---
echo [5/5] Starting FastAPI (%API_URL%) and React (%UI_URL%)
echo.
echo Keep this window open while using the app.
echo Close this window to stop both servers.
echo.

start "xai-api" cmd /c ""%VENV_PY%" -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"
start "open-browser" cmd /c "timeout /t 4 /nobreak >nul & start %UI_URL%"

pushd "%CD%\frontend"
call npm run dev
popd

echo.
echo Frontend stopped. Stopping API window if still open...
taskkill /FI "WINDOWTITLE eq xai-api*" /T /F >nul 2>&1
pause
exit /b 0

:fail
echo.
echo Launcher stopped with an error.
pause
exit /b 1
