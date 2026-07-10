@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title XAI Cyber Threat Classification
color 0A

echo ========================================
echo  Explainable AI - Local Launcher
echo ========================================
echo.
echo Working folder:
echo   %CD%
echo.

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "PORT=8501"
set "URL=http://localhost:%PORT%"

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
    echo [1/4] Creating virtual environment...
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
    echo [1/4] Virtual environment found.
)

if not exist "%VENV_PY%" (
    echo [ERROR] .venv\Scripts\python.exe is missing.
    goto :fail
)

REM --- Install dependencies (uses venv python directly; no activate needed) ---
echo [2/4] Checking dependencies...
"%VENV_PY%" -m pip install --upgrade pip >nul
"%VENV_PY%" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed.
    goto :fail
)

REM --- Pipeline: only if not already trained ---
echo [3/4] Checking training status...
if exist "%CD%\checkpoints\train.done" (
    echo Training already completed - skipping pipeline.
    echo Existing reports/models will be shown in the UI.
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

REM --- Launch Streamlit on localhost ---
echo [4/4] Starting Streamlit on %URL%
echo.
echo Keep this window open while using the app.
echo Close this window to stop the server.
echo.

if not exist "%CD%\ui\app.py" (
    echo [ERROR] ui\app.py not found.
    goto :fail
)

REM Open browser after a short delay so the server can start
start "open-browser" cmd /c "timeout /t 3 /nobreak >nul & start %URL%"

"%VENV_PY%" -m streamlit run "%CD%\ui\app.py" --server.address localhost --server.port %PORT% --browser.gatherUsageStats false --server.headless true
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit failed to start.
    goto :fail
)

echo.
echo App stopped.
pause
exit /b 0

:fail
echo.
echo Launcher stopped with an error.
pause
exit /b 1
