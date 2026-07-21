@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
python -c "import fastapi" 2>nul || (
  echo Installing API dependencies...
  pip install fastapi uvicorn python-multipart
)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
