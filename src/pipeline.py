from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PREDICTIONS_DIR,
    RAW_DIR,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
    REPORTS_DIR,
    MODEL_NAMES,
    LOGGER,
    checkpoint_path,
)

import src.data as data
import src.model as model
import src.prep as prep
import src.reproducibility as reproducibility
import src.xai as xai
from src.timing import record_step


STEP_ORDER: list[str] = [
    "discover",
    "load",
    "integrate",
    "eda",
    "clean",
    "split",
    "preprocess",
    "balance",
    "train",
    "evaluate",
    "explain-global",
    "explain-local",
    "explanation-quality",
    "predict",
]


def _downstream_steps(step_name: str) -> list[str]:
    idx = STEP_ORDER.index(step_name)
    return STEP_ORDER[idx:]


def _step_params(step_name: str) -> dict[str, Any]:
    # These params are included in checkpoint hashing; keep them deterministic.
    if step_name == "discover":
        return {"raw_dir": RAW_DIR}
    if step_name == "load":
        return {"raw_dir": RAW_DIR, "interim_dir": INTERIM_DIR}
    if step_name == "integrate":
        return {"loaded_path": INTERIM_DIR / "loaded.parquet", "interim_dir": INTERIM_DIR}
    if step_name == "eda":
        return {"integrated_path": INTERIM_DIR / "integrated.parquet"}
    if step_name == "clean":
        return {"integrated_path": INTERIM_DIR / "integrated.parquet"}
    if step_name == "split":
        return {
            "cleaned_path": INTERIM_DIR / "cleaned.parquet",
            "random_seed": RANDOM_SEED,
        }
    if step_name == "preprocess":
        return {
            "cleaned_path": INTERIM_DIR / "cleaned.parquet",
            "splits_path": PROCESSED_DIR / "splits.json",
        }
    if step_name == "balance":
        return {
            "unscaled_path": PROCESSED_DIR / "tabular_unscaled.parquet",
            "splits_path": PROCESSED_DIR / "splits.json",
            "random_seed": RANDOM_SEED,
        }
    if step_name == "train":
        return {
            "train_path": PROCESSED_DIR / "train_balanced.parquet",
            "unscaled_path": PROCESSED_DIR / "tabular_unscaled.parquet",
            "random_seed": RANDOM_SEED,
        }
    if step_name == "evaluate":
        return {"unscaled_path": PROCESSED_DIR / "tabular_unscaled.parquet"}
    if step_name in {"explain-global", "explain-local", "explanation-quality", "explain"}:
        return {"unscaled_path": PROCESSED_DIR / "tabular_unscaled.parquet"}
    if step_name == "predict":
        return {"unscaled_path": PROCESSED_DIR / "tabular_unscaled.parquet"}
    return {}


def _load_checkpoint_payload(step: str) -> dict[str, Any] | None:
    marker = checkpoint_path(step)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def checkpoint_status_rows() -> list[dict[str, str]]:
    """Build status rows matching the Streamlit Status tab."""
    rows: list[dict[str, str]] = []
    for step in STEP_ORDER:
        payload = _load_checkpoint_payload(step)
        if payload:
            ts = payload.get("timestamp", "")
            try:
                last_run = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except ValueError:
                last_run = ts or "—"
            outputs = payload.get("output_paths", [])
            output_text = ", ".join(Path(p).name for p in outputs[:5])
            if len(outputs) > 5:
                output_text += f" (+{len(outputs) - 5} more)"
            status = "Done"
            input_hash = payload.get("input_hash", "—")
        else:
            last_run = "—"
            output_text = "—"
            status = "Pending"
            input_hash = "—"
        rows.append(
            {
                "Step": step,
                "Status": status,
                "Last run": last_run,
                "Input hash": input_hash[:12] + "…" if len(input_hash) > 12 else input_hash,
                "Outputs": output_text,
            }
        )
    return rows


def _print_status_table(rows: list[dict[str, str]]) -> None:
    columns = ["Step", "Status", "Last run", "Input hash", "Outputs"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    def fmt_row(values: dict[str, str]) -> str:
        return " | ".join(str(values[col]).ljust(widths[col]) for col in columns)

    separator = "-+-".join("-" * widths[col] for col in columns)
    print(fmt_row({col: col for col in columns}))
    print(separator)
    for row in rows:
        print(fmt_row(row))


def run_step(step_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    params = _step_params(step_name)
    if step_name == "discover":
        result = data.discover(params)
    elif step_name == "load":
        result = data.load(params)
    elif step_name == "integrate":
        result = data.integrate(params)
    elif step_name == "clean":
        result = prep.clean(params)
    elif step_name == "eda":
        result = prep.eda(params)
    elif step_name == "split":
        result = prep.split(params)
    elif step_name == "preprocess":
        result = prep.preprocess(params)
    elif step_name == "balance":
        result = prep.balance(params)
    elif step_name == "train":
        result = model.train(params)
    elif step_name == "evaluate":
        result = model.evaluate(params)
    elif step_name == "explain-global":
        result = xai.explain_global(params)
    elif step_name == "explain-local":
        result = xai.explain_local(params)
    elif step_name == "explanation-quality":
        result = xai.explanation_quality(params)
    elif step_name == "explain":
        result = xai.explain(params)
    elif step_name == "predict":
        result = model.predict_step(params)
    else:
        raise ValueError(f"Unknown step: {step_name}")

    elapsed = time.perf_counter() - started
    record_step(step_name, elapsed, skipped=bool(result.get("loaded_from_checkpoint")))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cybersecurity XAI pipeline runner (idempotent via checkpoints).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for step in STEP_ORDER:
        subparsers.add_parser(step, help=f"Run step: {step}")

    subparsers.add_parser("all", help="Run all steps in order.")
    subparsers.add_parser("status", help="Show checkpoint marker status.")

    reset_parser = subparsers.add_parser(
        "reset", help="Delete a step marker and every downstream marker."
    )
    reset_parser.add_argument("--step", required=True, choices=STEP_ORDER, help="Step name to reset from.")

    return parser.parse_args()


def cmd_all() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for step_name in STEP_ORDER:
        LOGGER.info("Running step: %s", step_name)
        results[step_name] = run_step(step_name)
    repro = reproducibility.finalize_reproducibility()
    LOGGER.info("Reproducibility snapshot written: %s", repro["env_json"])
    return {"steps": results, "reproducibility": repro}


def cmd_status() -> list[dict[str, str]]:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return checkpoint_status_rows()


def _extra_train_markers() -> list[Path]:
    return [CHECKPOINTS_DIR / f"train_{name}.done" for name in MODEL_NAMES]


def cmd_reset(step_name: str, *, confirm: bool = True) -> dict[str, Any]:
    downstream = _downstream_steps(step_name)
    if confirm and sys.stdin.isatty():
        print(
            f"This will delete checkpoint markers for: {', '.join(downstream)} "
            f"(and per-model train_* markers if resetting from train or later)."
        )
        answer = input("Continue? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Reset cancelled.")
            return {"reset_from": step_name, "deleted_markers": [], "cancelled": True}

    deleted: list[str] = []
    for downstream_step in downstream:
        marker = checkpoint_path(downstream_step)
        if marker.exists():
            marker.unlink()
            deleted.append(downstream_step)

    train_idx = STEP_ORDER.index("train")
    reset_idx = STEP_ORDER.index(step_name)
    if reset_idx <= train_idx:
        for marker in _extra_train_markers():
            if marker.exists():
                marker.unlink()
                deleted.append(marker.stem)

    return {"reset_from": step_name, "deleted_markers": deleted}


if __name__ == "__main__":
    args = parse_args()
    if args.command == "all":
        cmd_all()
        LOGGER.info("Command 'all' completed.")
        print("src.pipeline: command=all")
    elif args.command == "status":
        rows = cmd_status()
        _print_status_table(rows)
    elif args.command == "reset":
        result = cmd_reset(args.step)
        if result.get("cancelled"):
            sys.exit(1)
        LOGGER.info("Reset deleted markers: %s", result.get("deleted_markers"))
        print(f"src.pipeline: reset --step {args.step} ({len(result['deleted_markers'])} markers deleted)")
    else:
        run_step(args.command)
        LOGGER.info("Command '%s' completed.", args.command)
        print(f"src.pipeline: command={args.command}")

