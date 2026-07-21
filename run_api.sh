#!/usr/bin/env sh
cd "$(dirname "$0")"
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
