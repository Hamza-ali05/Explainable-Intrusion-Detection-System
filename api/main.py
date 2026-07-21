from __future__ import annotations

import base64
import csv
import io
import json
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import (  # noqa: E402
    LOGGER,
    LOGS_DIR,
    MODEL_NAMES,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpoint_path,
    read_json,
)
from src.model import MODELS_DIR, predict_from_dataframe  # noqa: E402
from src.pipeline import STEP_ORDER, cmd_all, run_step  # noqa: E402

ModelName = Literal["decision_tree", "random_forest", "xgboost"]
PlotType = Literal["beeswarm", "bar", "dependence"]
ExplainerType = Literal["xgb_shap", "xgb_lime", "dt_shap"]
OutcomeType = Literal["correct", "wrong"]
QualityMetric = Literal["fidelity", "stability", "comprehensibility"]

EXPLAINER_PREFIX = {
    "xgb_shap": "xgb_shap_waterfall",
    "xgb_lime": "xgb_lime_bar",
    "dt_shap": "dt_shap_waterfall",
}

QUALITY_TABLES = {
    "fidelity": ("fidelity_table.csv", "fidelity_plot.png"),
    "stability": ("stability_table.csv", "stability_plot.png"),
    "comprehensibility": ("comprehensibility_table.csv", "comprehensibility_heatmap.png"),
}

_pipeline_lock = threading.Lock()
_pipeline_running = False
_pipeline_current_step: str | None = None


class PipelineRunRequest(BaseModel):
    step: str = Field(default="all", description='Pipeline step name or "all"')


app = FastAPI(title="Cybersecurity XAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_checkpoint_payload(step: str) -> dict[str, Any] | None:
    marker = checkpoint_path(step)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _encode_png(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_reports_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    reports_root = REPORTS_DIR.resolve()
    if reports_root not in candidate.parents and candidate != reports_root:
        raise HTTPException(status_code=400, detail="Path must be under reports/")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")
    return candidate


def _validate_model(model: str) -> str:
    if model not in MODEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Expected one of: {', '.join(MODEL_NAMES)}",
        )
    return model


def _run_pipeline_background(step: str) -> None:
    global _pipeline_running, _pipeline_current_step
    try:
        if step == "all":
            LOGGER.info("API pipeline run started: all steps")
            cmd_all()
            LOGGER.info("API pipeline run completed: all steps")
        else:
            LOGGER.info("API pipeline run started: %s", step)
            run_step(step)
            LOGGER.info("API pipeline run completed: %s", step)
    except Exception:
        LOGGER.exception("API pipeline run failed for step=%s", step)
        traceback.print_exc()
    finally:
        with _pipeline_lock:
            _pipeline_running = False
            _pipeline_current_step = None


@app.get("/api/status")
def api_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in STEP_ORDER:
        payload = _load_checkpoint_payload(step)
        rows.append(
            {
                "step": step,
                "done": payload is not None,
                "timestamp": payload.get("timestamp") if payload else None,
            }
        )
    return rows


@app.post("/api/pipeline/run")
def api_pipeline_run(body: PipelineRunRequest) -> dict[str, str]:
    global _pipeline_running, _pipeline_current_step

    step = body.step.strip()
    if step != "all" and step not in STEP_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step '{step}'. Expected 'all' or one of: {', '.join(STEP_ORDER)}",
        )

    with _pipeline_lock:
        if _pipeline_running:
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline already running (step={_pipeline_current_step})",
            )
        _pipeline_running = True
        _pipeline_current_step = step

    thread = threading.Thread(target=_run_pipeline_background, args=(step,), daemon=True)
    thread.start()
    return {"status": "started", "step": step}


@app.get("/api/pipeline/log")
def api_pipeline_log() -> dict[str, list[str]]:
    log_path = LOGS_DIR / "pipeline.log"
    if not log_path.exists():
        return {"lines": []}
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-200:]}


@app.get("/api/evaluate/leaderboard")
def api_evaluate_leaderboard() -> list[dict[str, Any]]:
    leaderboard: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        metrics_path = REPORTS_DIR / "evaluate" / model / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        leaderboard.append(
            {
                "model": model,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("f1", {}).get("macro"),
                "weighted_f1": metrics.get("f1", {}).get("weighted"),
                "mcc": metrics.get("mcc"),
                "pr_auc": metrics.get("pr_auc"),
            }
        )
    return leaderboard


@app.get("/api/evaluate/detail")
def api_evaluate_detail(model: ModelName) -> dict[str, Any]:
    model = _validate_model(model)
    eval_dir = REPORTS_DIR / "evaluate" / model
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for model '{model}'")

    payload: dict[str, Any] = {"model": model, "metrics": read_json(metrics_path), "images": {}}
    for png_path in sorted(eval_dir.glob("*.png")):
        payload["images"][png_path.stem] = _encode_png(png_path)
    return payload


@app.get("/api/evaluate/training-curve")
def api_evaluate_training_curve(model: ModelName) -> dict[str, str | None]:
    model = _validate_model(model)
    curve_path = REPORTS_DIR / "train" / f"{model}_train_curve.png"
    image = _encode_png(curve_path)
    return {"image": image}


@app.get("/api/xai/global")
def api_xai_global(model: ModelName, plot_type: PlotType) -> dict[str, str | None]:
    model = _validate_model(model)
    image_path = REPORTS_DIR / "xai" / "global" / f"{model}_shap_{plot_type}.png"
    return {"model": model, "plot_type": plot_type, "image": _encode_png(image_path)}


@app.get("/api/xai/local/instances")
def api_xai_local_instances() -> dict[str, Any]:
    instances_path = REPORTS_DIR / "xai" / "local" / "selected_instances.json"
    if not instances_path.exists():
        raise HTTPException(status_code=404, detail="selected_instances.json not found")
    return read_json(instances_path)


@app.get("/api/xai/local/image")
def api_xai_local_image(
    explainer: ExplainerType,
    true_class: str,
    outcome: OutcomeType,
    instance_idx: int,
) -> dict[str, Any]:
    prefix = EXPLAINER_PREFIX[explainer]
    filename = f"{prefix}_{true_class}_{outcome}_{instance_idx}.png"
    image_path = REPORTS_DIR / "xai" / "local" / filename
    return {
        "explainer": explainer,
        "true_class": true_class,
        "outcome": outcome,
        "instance_idx": instance_idx,
        "image": _encode_png(image_path),
    }


@app.get("/api/xai/quality")
def api_xai_quality(metric: QualityMetric) -> dict[str, Any]:
    table_name, plot_name = QUALITY_TABLES[metric]
    table_path = REPORTS_DIR / "xai" / "quality" / table_name
    plot_path = REPORTS_DIR / "xai" / "quality" / plot_name
    if not table_path.exists():
        raise HTTPException(status_code=404, detail=f"Quality table not found: {table_name}")

    return {
        "metric": metric,
        "rows": _read_csv_records(table_path),
        "image": _encode_png(plot_path),
    }


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    model: ModelName = Form(...),
) -> dict[str, Any]:
    model = _validate_model(model)
    model_path = MODELS_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {model_path.name}")

    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    try:
        result = predict_from_dataframe(df, model_path=model_path, audit=True)
    except Exception as exc:
        LOGGER.exception("Prediction failed for model=%s", model)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    predictions = result["predicted_label"].astype(str).tolist()
    counts = pd.Series(predictions).value_counts()
    label_counts = {str(cls): int(counts.get(cls, 0)) for cls in THREAT_CLASSES}
    for cls, count in counts.items():
        label_counts[str(cls)] = int(count)

    return {
        "model": model,
        "predictions": predictions,
        "label_counts": label_counts,
        "total_rows": int(len(result)),
    }


@app.get("/api/reports")
def api_reports() -> list[dict[str, str]]:
    if not REPORTS_DIR.exists():
        return []

    entries: list[dict[str, str]] = []
    for path in sorted(REPORTS_DIR.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"md", "csv", "json", "png"}:
            entries.append({"path": rel, "type": suffix})
    return entries


@app.get("/api/reports/file")
def api_reports_file(path: str = Query(..., description="Relative path under reports/")) -> dict[str, str]:
    normalized = path.replace("\\", "/")
    if not normalized.startswith("reports/"):
        normalized = f"reports/{normalized.lstrip('/')}"
    file_path = _resolve_reports_path(normalized)
    suffix = file_path.suffix.lower()

    if suffix == ".png":
        encoded = _encode_png(file_path)
        if encoded is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"content": encoded, "type": "image"}

    if suffix in {".md", ".csv", ".json", ".txt"}:
        return {"content": file_path.read_text(encoding="utf-8", errors="replace"), "type": "text"}

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


@app.get("/")
def api_root() -> dict[str, str]:
    return {"message": "Cybersecurity XAI API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
