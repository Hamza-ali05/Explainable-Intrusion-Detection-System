from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

@dataclass
class Config:
    # Paths
    root_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    raw_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    manual_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    # Project constants
    RANDOM_SEED: int = 42
    THREAT_CLASSES: list[str] = field(
        default_factory=lambda: [
            "BENIGN",
            "DDoS",
            "DoS",
            "PortScan",
            "BruteForce",
            "WebAttack",
            "Infiltration",
            "Botnet",
        ]
    )
    MODEL_NAMES: list[str] = field(default_factory=lambda: ["decision_tree", "random_forest", "xgboost"])

    def __post_init__(self) -> None:
        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.manual_dir = self.data_dir / "manual"
        self.artifacts_dir = self.root_dir / "artifacts"
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.reports_dir = self.root_dir / "reports"
        self.predictions_dir = self.reports_dir / "predictions"
        self.logs_dir = self.root_dir / "logs"

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        cfg = cls()
        path = config_path or (cfg.root_dir / "config.yaml")
        if not path.exists():
            return cfg
        if yaml is None:
            # Allow project import/test execution even when PyYAML isn't installed yet.
            return cfg

        overrides = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(overrides, dict):
            return cfg

        # Support either flat keys or a nested `paths:` block.
        path_overrides = overrides.pop("paths", None)
        if isinstance(path_overrides, dict):
            overrides = {**overrides, **path_overrides}

        for key, value in overrides.items():
            if not hasattr(cfg, key):
                continue
            current = getattr(cfg, key)
            if isinstance(current, Path):
                setattr(cfg, key, Path(value))
            else:
                setattr(cfg, key, value)

        return cfg


CONFIG = Config.load()

RANDOM_SEED: int = CONFIG.RANDOM_SEED
THREAT_CLASSES: list[str] = CONFIG.THREAT_CLASSES
MODEL_NAMES: list[str] = CONFIG.MODEL_NAMES

ROOT_DIR = CONFIG.root_dir
DATA_DIR = CONFIG.data_dir
RAW_DIR = CONFIG.raw_dir
INTERIM_DIR = CONFIG.interim_dir
PROCESSED_DIR = CONFIG.processed_dir
MANUAL_DIR = CONFIG.manual_dir
ARTIFACTS_DIR = CONFIG.artifacts_dir
CHECKPOINTS_DIR = CONFIG.checkpoints_dir
REPORTS_DIR = CONFIG.reports_dir
PREDICTIONS_DIR = CONFIG.predictions_dir
LOGS_DIR = CONFIG.logs_dir


def get_logger(name: str) -> logging.Logger:
    """Create a stdout + file logger for the project."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / ("pipeline.log" if name == "ids_pipeline" else f"{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


LOGGER = get_logger("ids_pipeline")

def setup_logging() -> logging.Logger:
    """Backwards-compatible alias for older code."""
    return get_logger("ids_pipeline")


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(params, sort_keys=True, default=str))


def _collect_paths(value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, Path):
        paths.append(value)
    elif isinstance(value, str):
        candidate = Path(value)
        if candidate.exists():
            paths.append(candidate)
    elif isinstance(value, dict):
        for nested in value.values():
            paths.extend(_collect_paths(nested))
    elif isinstance(value, (list, tuple, set)):
        for nested in value:
            paths.extend(_collect_paths(nested))
    return paths


def compute_input_hash(input_paths: list[Path], params: dict[str, Any]) -> str:
    """Hash sorted input metadata and params."""
    hasher = hashlib.sha256()
    for path in sorted({p.resolve() for p in input_paths if p.exists()}):
        stat = path.stat()
        hasher.update(str(path).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime).encode("utf-8"))
    hasher.update(json.dumps(_normalize_params(params), sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def _atomic_write(target_path: Path, write_cb: Callable[[Path], None]) -> Path:
    """Write to a sibling `.tmp` file then atomically rename into place."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        write_cb(tmp_path)
        if not tmp_path.exists():
            raise FileNotFoundError(f"Temporary write did not produce: {tmp_path}")
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target_path


def atomic_copy(src_path: Path, target_path: Path) -> Path:
    """Copy a file via a sibling `.tmp` file then atomically rename into place."""
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        shutil.copy2(src_path, tmp_path)
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return target_path


# --- IO helpers (parquet / joblib / json / yaml) ---
def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    return _atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


def read_joblib(path: Path) -> Any:
    return joblib.load(path)


def write_joblib(obj: Any, path: Path) -> Path:
    return _atomic_write(path, lambda tmp: joblib.dump(obj, tmp))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> Path:
    def _write(tmp: Path) -> None:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    return _atomic_write(path, _write)


def read_yaml(path: Path) -> Any:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for YAML reads. Install `pyyaml`.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(obj: Any, path: Path) -> Path:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required for YAML writes. Install `pyyaml`.")
    def _write(tmp: Path) -> None:
        tmp.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")

    return _atomic_write(path, _write)


def checkpoint_path(step_name: str) -> Path:
    return CHECKPOINTS_DIR / f"{step_name}.done"


def is_done(step_name: str, input_hash: str) -> tuple[bool, dict[str, Any] | None]:
    path = checkpoint_path(step_name)
    if not path.exists():
        return False, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, None
    return payload.get("input_hash") == input_hash, payload


def mark_done(
    step_name: str,
    input_hash: str,
    output_paths: list[Path],
    params: dict[str, Any],
) -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "step_name": step_name,
        "input_hash": input_hash,
        "output_paths": [str(path) for path in output_paths],
        "params": _normalize_params(params),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    done_path = checkpoint_path(step_name)
    done_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return done_path


def _load_one_output(path: Path) -> Any:
    """Reload a checkpointed artifact. Binary files return their path string."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return read_json(path)
    if suffix in {".yml", ".yaml"}:
        return read_yaml(path)
    if suffix in {".pkl", ".joblib"}:
        return read_joblib(path)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".pdf", ".npy", ".npz"}:
        return str(path)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return str(path)


def load_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for raw_path in payload.get("output_paths", []):
        path = Path(raw_path)
        if path.exists():
            outputs[str(path)] = _load_one_output(path)
    return {"loaded_from_checkpoint": True, "outputs": outputs, "metadata": payload}


def checkpointed(step_name: str) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Decorator for hash-based checkpoint skip/load behavior."""

    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            force: bool = bool(kwargs.pop("force", False))

            # Preferred call pattern: `func(params_dict)` where `params_dict` drives the step.
            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                params = args[0]
                input_paths = _collect_paths(params)
            else:
                # Backwards-compatible pattern: `func(**kwargs)` / `func(*args, **kwargs)`.
                params = {k: v for k, v in kwargs.items() if k != "force"}
                input_paths: list[Path] = []
                for value in [*args, kwargs]:
                    input_paths.extend(_collect_paths(value))

            current_hash = compute_input_hash(input_paths, params)
            done, payload = is_done(step_name, current_hash)
            if done and payload and not force:
                LOGGER.info("Step '%s' skipped via checkpoint.", step_name)
                return load_outputs(payload)

            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                result = func(params)
            else:
                result = func(*args, **kwargs)

            raw_outputs = result.get("output_paths", [])
            output_paths = [Path(path) for path in raw_outputs]
            mark_done(step_name, current_hash, output_paths, params)
            LOGGER.info("Step '%s' completed and checkpointed.", step_name)
            return result

        return wrapper

    return decorator


if __name__ == "__main__":
    LOGGER.info("Core utilities ready.")
    print("src.core: initialized.")

