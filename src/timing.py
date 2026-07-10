from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import REPORTS_DIR, write_json

TIMING_JSON = REPORTS_DIR / "timing.json"


def _empty_timing() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "total_wall_seconds": 0.0,
        "explainability": {},
    }


def load_timing() -> dict[str, Any]:
    if not TIMING_JSON.exists():
        return _empty_timing()
    try:
        payload = json.loads(TIMING_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_timing()
    payload.setdefault("steps", {})
    payload.setdefault("explainability", {})
    payload.setdefault("total_wall_seconds", 0.0)
    return payload


def save_timing(payload: dict[str, Any]) -> Path:
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    steps = payload.get("steps", {})
    payload["total_wall_seconds"] = round(
        sum(float(entry.get("wall_seconds", 0.0)) for entry in steps.values()),
        4,
    )
    return write_json(payload, TIMING_JSON)


def record_step(
    step_name: str,
    wall_seconds: float,
    *,
    skipped: bool = False,
) -> None:
    payload = load_timing()
    payload["steps"][step_name] = {
        "wall_seconds": round(wall_seconds, 4),
        "skipped": skipped,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    save_timing(payload)


def record_explainability(
    step_name: str,
    model_name: str,
    *,
    shap_seconds: float | None = None,
    lime_seconds: float | None = None,
) -> None:
    payload = load_timing()
    explainability: dict[str, Any] = payload.setdefault("explainability", {})
    step_entry: dict[str, Any] = explainability.setdefault(step_name, {})
    model_entry: dict[str, Any] = step_entry.setdefault(model_name, {})
    if shap_seconds is not None:
        model_entry["shap_seconds"] = round(
            model_entry.get("shap_seconds", 0.0) + shap_seconds, 4
        )
    if lime_seconds is not None:
        model_entry["lime_seconds"] = round(
            model_entry.get("lime_seconds", 0.0) + lime_seconds, 4
        )
    save_timing(payload)
