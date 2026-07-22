"""Streamlit UI removed.

Use the React frontend with the FastAPI backend:

    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
    cd frontend && npm install && npm run dev

Open http://localhost:3000
"""

raise SystemExit(
    "Streamlit UI has been replaced by the React frontend.\n"
    "1) uvicorn api.main:app --reload --host 127.0.0.1 --port 8000\n"
    "2) cd frontend && npm install && npm run dev\n"
    "3) Open http://localhost:3000\n"
    "Or run: run_app.bat"
)
