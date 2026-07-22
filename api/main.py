from __future__ import annotations

import base64
import csv
import io
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import (  # noqa: E402
    LOGGER,
    LOGS_DIR,
    MANUAL_DIR,
    MODEL_NAMES,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpoint_path,
    read_json,
)
from src.model import MODELS_DIR, predict_from_dataframe  # noqa: E402
from src.pipeline import STEP_ORDER, cmd_all, run_step  # noqa: E402
from src.xai import explain_one  # noqa: E402

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

PREDICT_ROW_LIMIT = 500

_pipeline_lock = threading.Lock()
_pipeline_running = False
_pipeline_current_step: str | None = None


class PipelineRunRequest(BaseModel):
    step: str = Field(default="all", description='Pipeline step name or "all"')
    force: bool = Field(default=False, description="Delete checkpoint before running (single steps only)")


class ExplainRequest(BaseModel):
    model: ModelName
    row: dict[str, Any] = Field(..., description="Single flow feature map")


class SinglePredictRequest(BaseModel):
    model: ModelName
    features: dict[str, Any] = Field(..., description="Single flow feature map")


app = FastAPI(title="Cybersecurity XAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FIREBASE_VERIFY = False
_sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
if _sa_path and Path(_sa_path).is_file():
    import firebase_admin
    from firebase_admin import credentials

    firebase_admin.initialize_app(credentials.Certificate(_sa_path))
    _FIREBASE_VERIFY = True

MOCK_GOOGLE_ALERTS = {
    "alerts": [
        {
            "severity": "Low",
            "type": "Suspicious login",
            "timestamp": "2026-07-22T08:14:00Z",
            "detail": "Sign-in from new location detected",
        },
        {
            "severity": "Medium",
            "type": "Password reuse",
            "timestamp": "2026-07-21T16:45:00Z",
            "detail": "Reused password detected across services",
        },
        {
            "severity": "High",
            "type": "Phishing email",
            "timestamp": "2026-07-20T11:30:00Z",
            "detail": "Phishing attempt blocked by Gmail filter",
        },
    ]
}


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


def _run_pipeline_background(step: str, *, force: bool = False) -> None:
    global _pipeline_running, _pipeline_current_step
    try:
        if force and step != "all":
            marker = checkpoint_path(step)
            if marker.exists():
                marker.unlink()
                LOGGER.info("Deleted checkpoint for force rerun: %s", step)
        if step == "all":
            LOGGER.info("API pipeline run started: all steps")
            cmd_all()
            LOGGER.info("API pipeline run completed: all steps")
        else:
            LOGGER.info("API pipeline run started: %s (force=%s)", step, force)
            run_step(step)
            LOGGER.info("API pipeline run completed: %s", step)
    except Exception:
        LOGGER.exception("API pipeline run failed for step=%s", step)
        traceback.print_exc()
    finally:
        with _pipeline_lock:
            _pipeline_running = False
            _pipeline_current_step = None


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _series_to_feature_dict(row: pd.Series) -> dict[str, Any]:
    skip = {"threat_class", "threat_label", "source_file", "dataset_source"}
    features: dict[str, Any] = {}
    for col, value in row.items():
        if str(col).strip().lower() == "label" or col in skip:
            continue
        if pd.isna(value):
            continue
        features[str(col)] = _json_safe_value(value)
    return features


def _hand_coded_predict_samples() -> dict[str, dict[str, Any]]:
    base: dict[str, Any] = {
        "Destination Port": 443,
        "Flow Duration": 120000.0,
        "Total Fwd Packets": 12,
        "Total Backward Packets": 10,
        "Total Length of Fwd Packets": 620,
        "Total Length of Bwd Packets": 980,
        "Fwd Packet Length Max": 1460,
        "Fwd Packet Length Min": 0,
        "Fwd Packet Length Mean": 51.6,
        "Fwd Packet Length Std": 98.2,
        "Bwd Packet Length Max": 1460,
        "Bwd Packet Length Min": 0,
        "Bwd Packet Length Mean": 98.0,
        "Bwd Packet Length Std": 210.4,
        "Flow Bytes/s": 13.3,
        "Flow Packets/s": 0.18,
        "Flow IAT Mean": 545000.0,
        "Flow IAT Std": 620000.0,
        "Flow IAT Max": 1200000.0,
        "Flow IAT Min": 1.0,
        "Fwd IAT Total": 118000.0,
        "Fwd IAT Mean": 9800.0,
        "Fwd IAT Std": 22000.0,
        "Fwd IAT Max": 120000.0,
        "Fwd IAT Min": 0.0,
        "Bwd IAT Total": 119000.0,
        "Bwd IAT Mean": 11800.0,
        "Bwd IAT Std": 24000.0,
        "Bwd IAT Max": 130000.0,
        "Bwd IAT Min": 0.0,
        "Fwd PSH Flags": 0,
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Fwd Header Length": 240,
        "Bwd Header Length": 232,
        "Fwd Packets/s": 0.1,
        "Bwd Packets/s": 0.08,
        "Min Packet Length": 0,
        "Max Packet Length": 1460,
        "Packet Length Mean": 72.5,
        "Packet Length Std": 180.0,
        "Packet Length Variance": 32400.0,
        "FIN Flag Count": 0,
        "SYN Flag Count": 0,
        "RST Flag Count": 0,
        "PSH Flag Count": 2,
        "ACK Flag Count": 18,
        "URG Flag Count": 0,
        "CWE Flag Count": 0,
        "ECE Flag Count": 0,
        "Down/Up Ratio": 0.83,
        "Average Packet Size": 75.0,
        "Avg Fwd Segment Size": 51.0,
        "Avg Bwd Segment Size": 98.0,
        "Subflow Fwd Packets": 12,
        "Subflow Fwd Bytes": 620,
        "Subflow Bwd Packets": 10,
        "Subflow Bwd Bytes": 980,
        "Init_Win_bytes_forward": 8192,
        "Init_Win_bytes_backward": 65535,
        "act_data_pkt_fwd": 8,
        "min_seg_size_forward": 20,
        "Active Mean": 0.0,
        "Active Std": 0.0,
        "Active Max": 0.0,
        "Active Min": 0.0,
        "Idle Mean": 120000.0,
        "Idle Std": 0.0,
        "Idle Max": 120000.0,
        "Idle Min": 120000.0,
    }
    overrides: dict[str, dict[str, Any]] = {
        "BENIGN": {"Flow Duration": 85000.0, "Total Fwd Packets": 6, "SYN Flag Count": 0},
        "DDoS": {
            "Destination Port": 80,
            "Flow Duration": 2000.0,
            "Total Fwd Packets": 12000,
            "Total Backward Packets": 0,
            "Flow Packets/s": 6000.0,
            "SYN Flag Count": 12000,
            "ACK Flag Count": 0,
        },
        "DoS": {
            "Destination Port": 53,
            "Flow Duration": 5000.0,
            "Total Fwd Packets": 4000,
            "Total Backward Packets": 0,
            "Flow Packets/s": 800.0,
            "SYN Flag Count": 4000,
        },
        "PortScan": {
            "Destination Port": 22,
            "Flow Duration": 120.0,
            "Total Fwd Packets": 1,
            "Total Backward Packets": 0,
            "SYN Flag Count": 1,
            "ACK Flag Count": 0,
        },
        "BruteForce": {
            "Destination Port": 22,
            "Flow Duration": 45000.0,
            "Total Fwd Packets": 120,
            "Total Backward Packets": 95,
            "PSH Flag Count": 40,
            "ACK Flag Count": 180,
        },
        "WebAttack": {
            "Destination Port": 80,
            "Flow Duration": 15000.0,
            "Total Fwd Packets": 40,
            "Total Backward Packets": 35,
            "Total Length of Fwd Packets": 18000,
            "PSH Flag Count": 12,
        },
        "Infiltration": {
            "Destination Port": 445,
            "Flow Duration": 300000.0,
            "Total Fwd Packets": 80,
            "Total Backward Packets": 120,
            "Idle Mean": 240000.0,
        },
        "Botnet": {
            "Destination Port": 8080,
            "Flow Duration": 60000.0,
            "Total Fwd Packets": 25,
            "Total Backward Packets": 25,
            "Flow IAT Mean": 2400.0,
            "ACK Flag Count": 40,
        },
    }
    return {
        threat_class: {**base, **overrides.get(threat_class, {})}
        for threat_class in THREAT_CLASSES
    }


def _load_predict_samples() -> dict[str, dict[str, Any]]:
    from src.data import _standardize_label_text
    from src.prep import map_threat_class

    samples: dict[str, dict[str, Any]] = {}
    example_path = MANUAL_DIR / "example_flows.csv"
    if example_path.exists():
        df = pd.read_csv(example_path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        label_col = next(
            (c for c in df.columns if str(c).strip().lower() == "label"),
            None,
        )
        if label_col is not None:
            for _, row in df.iterrows():
                threat_class = map_threat_class(_standardize_label_text(row[label_col]))
                if threat_class in THREAT_CLASSES and threat_class not in samples:
                    samples[threat_class] = _series_to_feature_dict(row)

    fallback = _hand_coded_predict_samples()
    for threat_class in THREAT_CLASSES:
        if threat_class not in samples:
            samples[threat_class] = fallback[threat_class]
    return samples


def _serialize_prediction_frame(result: pd.DataFrame, model: str) -> dict[str, Any]:
    predictions = result["predicted_label"].astype(str).tolist()
    counts = pd.Series(predictions).value_counts()
    label_counts = {str(cls): int(counts.get(cls, 0)) for cls in THREAT_CLASSES}
    for cls, count in counts.items():
        label_counts[str(cls)] = int(count)

    preview = result.head(PREDICT_ROW_LIMIT).copy()
    if "confidence" in preview.columns:
        preview["confidence"] = preview["confidence"].astype(float)
    rows = json.loads(preview.to_json(orient="records", date_format="iso"))

    return {
        "model": model,
        "predictions": predictions,
        "label_counts": label_counts,
        "total_rows": int(len(result)),
        "columns": list(result.columns),
        "rows": rows,
    }


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


@app.get("/api/pipeline/state")
def api_pipeline_state() -> dict[str, Any]:
    with _pipeline_lock:
        return {"running": _pipeline_running, "step": _pipeline_current_step}


@app.post("/api/pipeline/run")
def api_pipeline_run(body: PipelineRunRequest) -> dict[str, Any]:
    global _pipeline_running, _pipeline_current_step

    step = body.step.strip()
    if step != "all" and step not in STEP_ORDER:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid step '{step}'. Expected 'all' or one of: {', '.join(STEP_ORDER)}",
        )
    if body.force and step == "all":
        raise HTTPException(
            status_code=400,
            detail="force=true is only supported for individual steps, not 'all'",
        )

    with _pipeline_lock:
        if _pipeline_running:
            raise HTTPException(
                status_code=409,
                detail=f"Pipeline already running (step={_pipeline_current_step})",
            )
        _pipeline_running = True
        _pipeline_current_step = step

    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(step,),
        kwargs={"force": body.force},
        daemon=True,
    )
    thread.start()
    return {"status": "started", "step": step, "force": body.force}


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


@app.get("/api/predict/samples")
def api_predict_samples() -> dict[str, dict[str, Any]]:
    """One feature dict per threat class from data/manual/example_flows.csv."""
    return _load_predict_samples()


@app.post("/api/predict/single")
def api_predict_single(body: SinglePredictRequest) -> dict[str, Any]:
    """Predict one flow from a feature dict; returns class label and per-class probabilities."""
    model = _validate_model(body.model)
    model_path = MODELS_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {model_path.name}")

    try:
        result = predict_from_dataframe(
            pd.DataFrame([body.features]),
            model_path=model_path,
            audit=False,
        )
    except Exception as exc:
        LOGGER.exception("Single prediction failed for model=%s", model)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    predicted_class = str(result["predicted_label"].iloc[0])
    probabilities: dict[str, float] = {}
    for col in result.columns:
        if str(col).startswith("predicted_proba_"):
            cls = str(col).removeprefix("predicted_proba_")
            probabilities[cls] = float(result[col].iloc[0])

    return {"predicted_class": predicted_class, "probabilities": probabilities}


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

    return _serialize_prediction_frame(result, model)


@app.post("/api/predict/demo")
async def api_predict_demo(model: ModelName = Form(...)) -> dict[str, Any]:
    model = _validate_model(model)
    model_path = MODELS_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {model_path.name}")

    example_path = MANUAL_DIR / "example_flows.csv"
    if not example_path.exists():
        raise HTTPException(
            status_code=404,
            detail="example_flows.csv not found. Run: python scripts/seed_example_flows.py",
        )

    try:
        df = pd.read_csv(example_path)
        result = predict_from_dataframe(df, model_path=model_path, audit=True)
    except Exception as exc:
        LOGGER.exception("Demo prediction failed for model=%s", model)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    payload = _serialize_prediction_frame(result, model)
    payload["source"] = "example_flows.csv"
    return payload


@app.post("/api/predict/explain")
def api_predict_explain(body: ExplainRequest) -> dict[str, Any]:
    model = _validate_model(body.model)
    model_path = MODELS_DIR / f"{model}.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model artifact not found: {model_path.name}")

    try:
        explanation = explain_one(body.row, model_name=model)
    except Exception as exc:
        LOGGER.exception("Live explanation failed for model=%s", model)
        raise HTTPException(status_code=400, detail=f"Explanation failed: {exc}") from exc

    # Data only — no narrative text for the UI.
    return {
        "model": explanation.get("model", model),
        "predicted_label": explanation.get("predicted_label"),
        "probabilities": explanation.get("probabilities", {}),
        "confidence": explanation.get("confidence"),
        "shap_top15": explanation.get("shap_top15", []),
        "lime_top10": explanation.get("lime_top10", []),
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


@app.get("/api/google/alerts")
def google_alerts(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    """Verify Firebase ID token (Bearer) and return mock account alerts for demo."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    if _FIREBASE_VERIFY:
        try:
            from firebase_admin import auth as firebase_auth

            firebase_auth.verify_id_token(token)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid Firebase ID token") from exc

    return MOCK_GOOGLE_ALERTS


@app.get("/")
def api_root() -> dict[str, str]:
    return {"message": "Cybersecurity XAI API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
