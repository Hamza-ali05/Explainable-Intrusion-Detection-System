from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import REPORTS_DIR, ROOT_DIR, write_json

ENV_JSON = REPORTS_DIR / "env.json"
LOCK_FILE = ROOT_DIR / "requirements.lock.txt"


def _package_version(name: str) -> str:
    try:
        return str(__import__(name).__version__)
    except (ImportError, AttributeError):
        pass
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"


def capture_env_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "packages": {
            "sklearn": _package_version("sklearn"),
            "xgboost": _package_version("xgboost"),
            "shap": _package_version("shap"),
            "lime": _package_version("lime"),
        },
    }
    write_json(payload, ENV_JSON)
    return payload


def write_requirements_lock() -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    LOCK_FILE.write_text(result.stdout, encoding="utf-8")
    return LOCK_FILE


def finalize_reproducibility() -> dict[str, Any]:
    env = capture_env_snapshot()
    lock_path = write_requirements_lock()
    return {"env_json": str(ENV_JSON), "requirements_lock": str(lock_path), "env": env}
