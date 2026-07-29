from __future__ import annotations

import base64
import csv
import ipaddress
import io
import json
import os
import random
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
from api.db import delete_session, get_sessions, init_db, save_session  # noqa: E402

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

# ---------------------------------------------------------------------------
# IDS module-level artefact loading
# ---------------------------------------------------------------------------
_xgb_model: Any = None
_preprocess_bundle: dict[str, Any] | None = None
_feature_columns: list[str] | None = None
_shap_explainer: Any = None
_model_feature_columns: list[str] | None = None
_preprocess_feature_columns: list[str] | None = None

_ARTIFACTS_DIR = ROOT / "artifacts"
_PROCESSED_DIR = ROOT / "data" / "processed"

try:
    _xgb_model = joblib.load(_ARTIFACTS_DIR / "models" / "xgboost.joblib")
    LOGGER.info("IDS: loaded xgboost.joblib")
    _model_n_features = getattr(_xgb_model, "n_features_in_", None)
    _inner = getattr(_xgb_model, "estimator", _xgb_model)
    _fni = getattr(_inner, "feature_names_in_", None)
    if _fni is not None:
        _model_feature_columns = [str(f) for f in _fni]
except Exception:
    LOGGER.warning("IDS: xgboost.joblib not found — IDS endpoints unavailable")

try:
    _preprocess_bundle = joblib.load(_ARTIFACTS_DIR / "preprocess" / "preprocess_bundle.joblib")
    LOGGER.info("IDS: loaded preprocess_bundle.joblib")
    if isinstance(_preprocess_bundle, dict):
        scaler = _preprocess_bundle.get("scaler")
        if scaler is not None:
            scaler_feature_names = getattr(scaler, "feature_names_in_", None)
            if scaler_feature_names is not None:
                _preprocess_feature_columns = [str(f) for f in scaler_feature_names]
except Exception:
    LOGGER.warning("IDS: preprocess_bundle.joblib not found")

try:
    _unscaled_path = _PROCESSED_DIR / "tabular_unscaled.parquet"
    _raw_cols = pd.read_parquet(_unscaled_path).columns.tolist()
    _feature_columns = [c for c in _raw_cols if c not in ("label", "threat_class", "threat_label")]
    LOGGER.info("IDS: %d feature columns loaded from tabular_unscaled.parquet", len(_feature_columns or []))
except Exception:
    LOGGER.warning("IDS: tabular_unscaled.parquet not found — falling back to model features")
    if _model_feature_columns is not None:
        _feature_columns = list(_model_feature_columns)

if _feature_columns and _model_feature_columns and len(_feature_columns) != len(_model_feature_columns):
    # Current pipeline artefacts expose a 3-feature shared schema, while the
    # persisted XGBoost model was trained on a 61-feature CICIDS-style schema.
    # We keep the parquet-derived list for audit visibility, but IDS inference
    # must use the model's own feature names to pad/truncate safely.
    LOGGER.warning(
        "IDS: feature mismatch detected (parquet=%d, model=%d). Using model feature names for inference.",
        len(_feature_columns),
        len(_model_feature_columns),
    )
    _feature_columns = list(_model_feature_columns)
elif not _feature_columns and _model_feature_columns:
    _feature_columns = list(_model_feature_columns)

try:
    import shap

    _tree_model = getattr(_xgb_model, "estimator", _xgb_model) if _xgb_model else None
    if _tree_model is not None:
        _shap_explainer = shap.TreeExplainer(_tree_model)
        LOGGER.info("IDS: SHAP TreeExplainer initialised")
except Exception:
    LOGGER.warning("IDS: SHAP explainer could not be initialised")

IDS_MAX_ROWS = 500

IDS_THREAT_CLASSES = [
    "BENIGN", "DDoS", "DoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet",
]

CLASS_NOTES: dict[str, str] = {
    "DDoS": "High packet counts and short flow durations are consistent with volumetric DDoS patterns.",
    "DoS": "Elevated byte rates and abnormal flow lengths suggest a denial-of-service attempt.",
    "PortScan": "Short-lived flows with low byte counts are characteristic of port scanning activity.",
    "BruteForce": "Repeated short connections with consistent packet sizes suggest credential brute-forcing.",
    "WebAttack": "Destination port and payload size patterns match known web attack signatures.",
    "Infiltration": "Unusual outbound byte volume and timing anomalies indicate potential data exfiltration.",
    "Botnet": "Regular inter-arrival times and idle periods are consistent with botnet command-and-control traffic.",
    "BENIGN": "No anomalous patterns were detected. Traffic matches normal network behaviour profiles.",
}

SEVERITY_MAP: dict[str, str] = {
    "BENIGN": "None",
    "DDoS": "Critical",
    "DoS": "Critical",
    "PortScan": "High",
    "BruteForce": "High",
    "WebAttack": "High",
    "Botnet": "High",
    "Infiltration": "Critical",
}

MITIGATIONS: dict[str, list[str]] = {
    "BENIGN": ["No action required.", "Continue monitoring normally."],
    "DDoS": [
        "Block source IP at the perimeter firewall immediately.",
        "Enable rate-limiting and traffic scrubbing on the affected interface.",
        "Contact upstream ISP to apply traffic filtering at network edge.",
        "Activate DDoS mitigation service (e.g., Cloudflare, Akamai) if available.",
    ],
    "DoS": [
        "Block the source IP address.",
        "Apply rate-limiting rules on the affected service port.",
        "Check server resource utilisation and restart services if necessary.",
    ],
    "PortScan": [
        "Block the scanning IP address.",
        "Review open ports and close any that are unnecessary.",
        "Enable stealth mode on the firewall to drop unsolicited packets.",
        "Check IDS/IPS logs for follow-up exploitation attempts from the same IP.",
    ],
    "BruteForce": [
        "Temporarily block the source IP.",
        "Enable account lockout policy after 5 failed login attempts.",
        "Enforce multi-factor authentication on all externally accessible services.",
        "Audit affected accounts for unauthorised changes.",
    ],
    "WebAttack": [
        "Block the source IP.",
        "Enable Web Application Firewall (WAF) rules for the targeted endpoint.",
        "Patch the affected web application to the latest version.",
        "Review access logs for any successful exploitation.",
    ],
    "Infiltration": [
        "Isolate the affected host from the network immediately.",
        "Perform memory and disk forensics on the host.",
        "Review outbound traffic logs for data exfiltration.",
        "Reset all credentials associated with the affected host.",
        "Notify the incident response team.",
    ],
    "Botnet": [
        "Isolate the affected host.",
        "Run a full malware scan using an updated endpoint protection tool.",
        "Block known C2 IP addresses and domains at the firewall.",
        "Rebuild the host from a clean image after forensics are complete.",
    ],
}

MALICIOUS_ASNS = {"AS16276", "AS14061", "AS63949", "AS396356", "AS211298"}


def _ids_preprocess_input(df: pd.DataFrame) -> np.ndarray:
    """Align raw DataFrame to model feature columns and return numpy array."""
    if _feature_columns is None:
        raise RuntimeError("Feature columns not loaded")

    incoming = df.copy()
    incoming.columns = [str(c).strip() for c in incoming.columns]

    unknown_columns = [c for c in incoming.columns if c not in _feature_columns]
    if unknown_columns:
        LOGGER.warning(
            "IDS: dropping %d unrecognised columns: %s",
            len(unknown_columns),
            ", ".join(unknown_columns[:12]),
        )
        incoming = incoming.drop(columns=unknown_columns, errors="ignore")

    aligned = incoming.reindex(columns=_feature_columns, fill_value=0.0)
    for col in aligned.columns:
        aligned[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0)

    scaler = None
    if isinstance(_preprocess_bundle, dict):
        scaler = _preprocess_bundle.get("scaler")

    if scaler is not None:
        scaler_feature_names = getattr(scaler, "feature_names_in_", None)
        if scaler_feature_names is not None:
            expected = [str(f) for f in scaler_feature_names]
            aligned_for_scaler = aligned.reindex(columns=expected, fill_value=0.0)
        elif _preprocess_feature_columns:
            aligned_for_scaler = aligned.reindex(columns=_preprocess_feature_columns, fill_value=0.0)
        else:
            aligned_for_scaler = aligned

        try:
            transformed = scaler.transform(aligned_for_scaler)
            transformed = np.asarray(transformed, dtype=np.float64)
            if transformed.shape[1] == len(_feature_columns):
                return transformed
            LOGGER.warning(
                "IDS: scaler output width %d does not match model feature width %d; using aligned raw values.",
                transformed.shape[1],
                len(_feature_columns),
            )
        except Exception:
            LOGGER.exception("IDS: scaler transform failed; using aligned raw values instead")

    expected_n = getattr(_xgb_model, "n_features_in_", None)
    array = aligned.to_numpy(dtype=np.float64)
    if expected_n is not None and array.shape[1] != int(expected_n):
        LOGGER.warning(
            "IDS: input width %d does not match model.n_features_in_=%d. Adjusting with zero padding/truncation.",
            array.shape[1],
            int(expected_n),
        )
        if array.shape[1] > int(expected_n):
            array = array[:, : int(expected_n)]
        else:
            pad = np.zeros((array.shape[0], int(expected_n) - array.shape[1]), dtype=np.float64)
            array = np.hstack([array, pad])
    return array


def _ids_generate_shap(X_processed: np.ndarray, row_idx: int, predicted_class_idx: int) -> list[dict[str, Any]]:
    """Return top-15 SHAP feature contributions for the predicted class."""
    if _shap_explainer is None or _feature_columns is None:
        return []
    try:
        shap_values = _shap_explainer.shap_values(X_processed[row_idx : row_idx + 1])
        row_shap = shap_values[0, :, predicted_class_idx]
        pairs = list(zip(_feature_columns, row_shap))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return [
            {
                "feature": feat,
                "shap_value": round(float(val), 6),
                "direction": "positive" if val > 0 else "negative",
            }
            for feat, val in pairs[:15]
        ]
    except Exception:
        LOGGER.exception("SHAP explanation failed for row %d", row_idx)
        return []


def _ids_generate_narrative(top_shap: list[dict[str, Any]], predicted_class: str, confidence: float) -> str:
    """Build a deterministic plain-language explanation."""
    f1 = top_shap[0] if len(top_shap) > 0 else {"feature": "unknown", "direction": "unknown"}
    f2 = top_shap[1] if len(top_shap) > 1 else {"feature": "unknown", "direction": "unknown"}
    note = CLASS_NOTES.get(predicted_class, "")
    return (
        f"This flow was classified as {predicted_class} with {confidence:.0%} confidence. "
        f"The primary contributing features were {f1['feature']} ({f1['direction']} contribution) "
        f"and {f2['feature']} ({f2['direction']} contribution). {note}"
    )


def _ids_analyze_rows(df: pd.DataFrame) -> dict[str, Any]:
    """Run IDS analysis on a DataFrame of raw network flows."""
    if _xgb_model is None or _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")

    if len(df) > IDS_MAX_ROWS:
        LOGGER.warning("IDS: input has %d rows, truncating to %d", len(df), IDS_MAX_ROWS)
        df = df.head(IDS_MAX_ROWS).copy()

    X = _ids_preprocess_input(df)
    probas = _xgb_model.predict_proba(X)
    classes = [str(c) for c in _xgb_model.classes_]

    results: list[dict[str, Any]] = []
    summary: dict[str, int] = {}

    for i in range(len(X)):
        proba_row = probas[i]
        pred_idx = int(np.argmax(proba_row))
        predicted_class = classes[pred_idx]
        confidence = float(proba_row[pred_idx])
        severity = SEVERITY_MAP.get(predicted_class, "Medium")
        mitigation = MITIGATIONS.get(predicted_class, [])

        shap_top15 = _ids_generate_shap(X, i, pred_idx)
        narrative = _ids_generate_narrative(shap_top15, predicted_class, confidence)

        prob_dict = {cls: round(float(proba_row[j]), 6) for j, cls in enumerate(classes)}
        feature_payload = {
            col: float(pd.to_numeric(df.iloc[i].get(col, 0.0), errors="coerce") if col in df.columns else 0.0)
            for col in _feature_columns
        }

        summary[predicted_class] = summary.get(predicted_class, 0) + 1

        results.append({
            "row_index": i,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "severity": severity,
            "probabilities": prob_dict,
            "shap_top15": shap_top15,
            "narrative": narrative,
            "mitigation": mitigation,
            "features": feature_payload,
            "model": "xgboost",
        })

    return {"total_rows": len(results), "summary": summary, "results": results}


class IDSSingleRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="Feature name to numeric value map")
    model: str = Field(default="xgboost")


class IDSExportRequest(BaseModel):
    row_index: int
    predicted_class: str
    confidence: float
    severity: str
    probabilities: dict[str, Any]
    shap_top15: list[dict[str, Any]]
    narrative: str
    mitigation: list[str]
    features: dict[str, Any]
    model: str = "xgboost"


def _ids_build_session_record(
    *,
    result: dict[str, Any],
    user_email: str | None,
    model_used: str,
    input_method: str,
) -> dict[str, Any]:
    summary = result.get("summary", {})
    benign_count = int(summary.get("BENIGN", 0))
    total_rows = int(result.get("total_rows", len(result.get("results", []))))
    threat_count = max(0, total_rows - benign_count)
    threat_only = {k: int(v) for k, v in summary.items() if k != "BENIGN" and int(v) > 0}
    if threat_only:
        top_threat, top_threat_count = max(threat_only.items(), key=lambda kv: kv[1])
    else:
        top_threat, top_threat_count = "None", 0
    return {
        "session_timestamp": datetime.now(timezone.utc).isoformat(),
        "user_email": user_email or "anonymous",
        "total_rows": total_rows,
        "benign_count": benign_count,
        "threat_count": threat_count,
        "top_threat": top_threat,
        "top_threat_count": int(top_threat_count),
        "model_used": model_used,
        "input_method": input_method,
    }


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

init_db()

_FIREBASE_VERIFY = False
_sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
if _sa_path and Path(_sa_path).is_file():
    import firebase_admin
    from firebase_admin import credentials

    firebase_admin.initialize_app(credentials.Certificate(_sa_path))
    _FIREBASE_VERIFY = True

MOCK_GOOGLE_ALERTS = [
    {
        "severity": "Critical",
        "type": "Suspicious login",
        "timestamp": "2026-07-28T03:14:22Z",
        "detail": "New sign-in from unrecognised device (Pakistan, Rawalpindi)",
    },
    {
        "severity": "High",
        "type": "Multiple failed logins",
        "timestamp": "2026-07-27T22:51:09Z",
        "detail": "5 failed login attempts to Gmail in 3 minutes",
    },
    {
        "severity": "Medium",
        "type": "New app authorised",
        "timestamp": "2026-07-26T11:30:00Z",
        "detail": "New OAuth application granted access to Google account",
    },
    {
        "severity": "Low",
        "type": "Password change",
        "timestamp": "2026-07-25T09:00:00Z",
        "detail": "Account password was changed from a new device",
    },
]


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


def _predict_sample_pool() -> dict[str, list[dict[str, Any]]]:
    """All feature dicts per threat class from data/manual/example_flows.csv."""
    from src.data import _standardize_label_text
    from src.prep import map_threat_class

    pool: dict[str, list[dict[str, Any]]] = {threat_class: [] for threat_class in THREAT_CLASSES}
    example_path = MANUAL_DIR / "example_flows.csv"
    if not example_path.exists():
        return pool

    df = pd.read_csv(example_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    label_col = next(
        (c for c in df.columns if str(c).strip().lower() == "label"),
        None,
    )
    if label_col is None:
        return pool

    for _, row in df.iterrows():
        threat_class = map_threat_class(_standardize_label_text(row[label_col]))
        if threat_class in THREAT_CLASSES:
            pool[threat_class].append(_series_to_feature_dict(row))
    return pool


def _jitter_predict_sample(
    base: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Apply light random variation to numeric hand-coded sample values."""
    out: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            scale = rng.uniform(0.75, 1.25)
            next_value = int(round(value * scale))
            out[key] = max(0, next_value)
        elif isinstance(value, float):
            scale = rng.uniform(0.75, 1.25)
            out[key] = float(value * scale)
        else:
            out[key] = value
    return out


def _refresh_predict_sample(threat_class: str) -> dict[str, Any]:
    """Return a random feature dict for the given threat class."""
    pool = _predict_sample_pool().get(threat_class, [])
    if pool:
        return dict(random.choice(pool))

    fallback = _hand_coded_predict_samples().get(threat_class)
    if fallback is None:
        raise ValueError(f"Unknown threat class: {threat_class}")
    return _jitter_predict_sample(fallback, random.Random())


def _load_predict_samples() -> dict[str, dict[str, Any]]:
    pool = _predict_sample_pool()
    fallback = _hand_coded_predict_samples()
    samples: dict[str, dict[str, Any]] = {}
    for threat_class in THREAT_CLASSES:
        rows = pool.get(threat_class, [])
        samples[threat_class] = rows[0] if rows else fallback[threat_class]
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


@app.get("/api/predict/samples/refresh")
def api_refresh_predict_sample(
    threat_class: str = Query(..., description="Threat class to sample"),
) -> dict[str, Any]:
    """Return a random feature dict for one threat class."""
    if threat_class not in THREAT_CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown threat class: {threat_class}")
    try:
        features = _refresh_predict_sample(threat_class)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"threat_class": threat_class, "features": features}


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


@app.get("/api/google/alerts")
def google_alerts(authorization: str | None = Header(default=None)) -> list[dict[str, Any]]:
    """Verify Firebase ID token (Bearer) and return mock account alerts for demo."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if _FIREBASE_VERIFY:
        try:
            from firebase_admin import auth as firebase_auth

            firebase_auth.verify_id_token(token)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid Firebase ID token") from exc

    return MOCK_GOOGLE_ALERTS


# ---------------------------------------------------------------------------
# IDS endpoints
# ---------------------------------------------------------------------------

@app.get("/api/ids/features")
def api_ids_features() -> dict[str, Any]:
    """Return the list of feature names the model expects."""
    if _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")
    return {"features": _feature_columns}


@app.get("/api/ids/ip-lookup")
async def api_ids_ip_lookup(ip: str = Query(..., description="Public IPv4 address")) -> dict[str, Any]:
    try:
        addr = ipaddress.IPv4Address(ip)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid public IPv4 address.") from exc

    if addr.is_private or addr.is_loopback or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
        raise HTTPException(status_code=422, detail="Invalid public IPv4 address.")

    country = None
    org = None
    asn = None
    is_vpn = None
    is_datacenter = None

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"https://ipapi.co/{ip}/json/", timeout=10)
            if resp.is_success:
                payload = resp.json()
                country = payload.get("country_name")
                org = payload.get("org")
                asn = payload.get("asn")
        except Exception:
            LOGGER.warning("IDS IP lookup: ipapi lookup failed for %s", ip)

        try:
            resp = await client.get(
                f"https://ip-api.com/json/{ip}?fields=proxy,hosting",
                timeout=10,
            )
            if resp.is_success:
                payload = resp.json()
                is_vpn = payload.get("proxy")
                is_datacenter = payload.get("hosting")
        except Exception:
            LOGGER.warning("IDS IP lookup: ip-api lookup failed for %s", ip)

    org_text = str(org or "").lower()
    is_tor = bool(is_vpn and "tor" in org_text)

    threat_score = 0
    if is_vpn:
        threat_score += 40
    if asn in MALICIOUS_ASNS:
        threat_score += 50
    if is_vpn and is_datacenter:
        threat_score += 30
    threat_score = min(threat_score, 100)

    if threat_score >= 70:
        suggested_class = "Infiltration"
    elif threat_score >= 50:
        suggested_class = "PortScan"
    elif threat_score >= 30:
        suggested_class = "DDoS"
    elif is_vpn:
        suggested_class = "BruteForce"
    elif is_datacenter:
        suggested_class = "BENIGN"
    else:
        suggested_class = "BENIGN"

    return {
        "ip": ip,
        "country": country,
        "org": org,
        "asn": asn,
        "is_vpn": is_vpn,
        "is_tor": is_tor,
        "is_datacenter": is_datacenter,
        "threat_score": threat_score,
        "suggested_class": suggested_class,
    }


@app.post("/api/ids/analyze")
async def api_ids_analyze(
    file: UploadFile = File(...),
    model: str = Form(default="xgboost"),
    x_user_email: str | None = Header(default=None),
) -> dict[str, Any]:
    """Analyse a CSV of network flows through the IDS pipeline."""
    if _xgb_model is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    result = _ids_analyze_rows(df)
    save_session(
        _ids_build_session_record(
            result=result,
            user_email=x_user_email,
            model_used=model,
            input_method="csv",
        )
    )
    return result


@app.post("/api/ids/analyze-single")
def api_ids_analyze_single(
    body: IDSSingleRequest,
    x_user_email: str | None = Header(default=None),
) -> dict[str, Any]:
    """Analyse a single network flow from a feature dict."""
    if _xgb_model is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")

    df = pd.DataFrame([body.features])
    result = _ids_analyze_rows(df)
    save_session(
        _ids_build_session_record(
            result=result,
            user_email=x_user_email,
            model_used=body.model,
            input_method="manual",
        )
    )
    if result["results"]:
        return result["results"][0]
    raise HTTPException(status_code=500, detail="Analysis produced no results.")


@app.post("/api/ids/shap-plot")
def api_ids_shap_plot(body: IDSSingleRequest) -> dict[str, Any]:
    if _xgb_model is None or _shap_explainer is None or _feature_columns is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")

    raw_df = pd.DataFrame([body.features])
    X_processed = _ids_preprocess_input(raw_df)
    raw_aligned = raw_df.reindex(columns=_feature_columns, fill_value=0.0).apply(
        lambda col: pd.to_numeric(col, errors="coerce").fillna(0.0)
    )
    proba = _xgb_model.predict_proba(X_processed)[0]
    predicted_idx = int(np.argmax(proba))

    try:
        shap_exp = _shap_explainer(X_processed)
        exp_single = shap.Explanation(
            values=shap_exp.values[0, :, predicted_idx],
            base_values=shap_exp.base_values[0, predicted_idx],
            data=raw_aligned.iloc[0].to_numpy(),
            feature_names=_feature_columns,
        )
        shap.plots.waterfall(exp_single, max_display=15, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close()
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return {"image": encoded}
    except Exception as exc:
        plt.close()
        raise HTTPException(status_code=400, detail=f"SHAP plot generation failed: {exc}") from exc


@app.post("/api/ids/export-pdf")
def api_ids_export_pdf(body: IDSExportRequest) -> StreamingResponse:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="PDF export requires reportlab. Run pip install reportlab.",
        ) from exc

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Explainable AI Intrusion Detection System — Threat Report", styles["Title"]))
    story.append(
        Paragraph(
            "Canterbury Christ Church University — MSc Cybersecurity | Kamran (100159510)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Detection timestamp: {datetime.now(timezone.utc).isoformat()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_table = Table(
        [
            ["Predicted Class", "Confidence", "Severity", "Model Used"],
            [body.predicted_class, f"{body.confidence:.1%}", body.severity, body.model],
        ]
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph(body.narrative, styles["BodyText"]))
    story.append(Spacer(1, 12))

    shap_rows = [["Feature", "SHAP Value", "Direction"]]
    for row in body.shap_top15:
        shap_rows.append(
            [
                str(row.get("feature", "")),
                str(row.get("shap_value", "")),
                str(row.get("direction", "")),
            ]
        )
    shap_table = Table(shap_rows)
    shap_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    story.append(shap_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Mitigation steps", styles["Heading3"]))
    for idx, step in enumerate(body.mitigation, start=1):
        story.append(Paragraph(f"{idx}. {step}", styles["BodyText"]))
    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "Generated by XAI-IDS Research System. For defensive research purposes only.",
            styles["Italic"],
        )
    )

    doc.build(story)
    buf.seek(0)
    filename = f"ids_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buf, media_type="application/pdf", headers=headers)


@app.get("/api/ids/history")
def api_ids_history(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    return {"sessions": get_sessions(limit)}


@app.delete("/api/ids/history/{session_id}")
def api_delete_ids_history(session_id: int) -> dict[str, bool]:
    return {"deleted": delete_session(session_id)}


# ---------------------------------------------------------------------------

@app.get("/")
def api_root() -> dict[str, str]:
    return {"message": "Cybersecurity XAI API", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok"}
