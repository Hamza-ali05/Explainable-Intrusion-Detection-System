from __future__ import annotations

import io
import json
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.core import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    MANUAL_DIR,
    MODEL_NAMES,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpoint_path,
    read_json,
)
from src.model import MODELS_DIR, predict_from_file
from src.pipeline import STEP_ORDER, cmd_all, checkpoint_status_rows, run_step
from src.xai import explain_one

XAI_DIR = REPORTS_DIR / "xai"
EVALUATE_DIR = REPORTS_DIR / "evaluate"
QUALITY_DIR = XAI_DIR / "quality"
LOG_PATH = LOGS_DIR / "pipeline.log"

st.set_page_config(page_title="Cyber threat XAI", layout="wide")


def _init_session_state() -> None:
    defaults = {
        "pipeline_running": False,
        "pipeline_error": None,
        "predictions_df": None,
        "predictions_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name).strip())


def _parse_instance_dir(dirname: str) -> tuple[int, str, str] | None:
    for true_label in THREAT_CLASSES:
        for pred_label in THREAT_CLASSES:
            suffix = f"_{_safe_filename(true_label)}_{_safe_filename(pred_label)}"
            if dirname.endswith(suffix):
                id_part = dirname[: -len(suffix)]
                if id_part.isdigit():
                    return int(id_part), true_label, pred_label
    return None


def _load_checkpoint(step: str) -> dict[str, Any] | None:
    marker = checkpoint_path(step)
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _checkpoint_status_rows() -> list[dict[str, Any]]:
    return checkpoint_status_rows()


def _tail_log(n_lines: int = 200) -> str:
    if not LOG_PATH.exists():
        return "(no log file yet)"
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read log: {exc})"
    return "\n".join(lines[-n_lines:]) if lines else "(empty log)"


def _run_pipeline_background() -> None:
    try:
        cmd_all()
        st.session_state["pipeline_error"] = None
    except Exception as exc:
        st.session_state["pipeline_error"] = str(exc)
    finally:
        st.session_state["pipeline_running"] = False


def _available_models() -> list[str]:
    return [name for name in MODEL_NAMES if (MODELS_DIR / f"{name}.joblib").exists()]


def _global_plot_paths(model: str) -> dict[str, list[Path]]:
    global_dir = XAI_DIR / model / "global"
    if not global_dir.exists():
        return {"beeswarm": [], "bar": [], "dependence": []}
    beeswarm = sorted(global_dir.glob("shap_beeswarm*.png"))
    bar = [global_dir / "shap_bar_mean_abs.png"]
    bar = [p for p in bar if p.exists()]
    dependence = sorted(global_dir.glob("shap_dependence_*.png"))
    return {"beeswarm": beeswarm, "bar": bar, "dependence": dependence}


def _list_local_instances(model: str) -> list[dict[str, Any]]:
    local_root = XAI_DIR / model / "local"
    if not local_root.exists():
        return []
    instances: list[dict[str, Any]] = []
    for inst_dir in sorted(local_root.iterdir()):
        if not inst_dir.is_dir():
            continue
        parsed = _parse_instance_dir(inst_dir.name)
        if parsed is None:
            continue
        instance_id, true_label, pred_label = parsed
        kind = "correct" if true_label == pred_label else "misclassified"
        label = f"{kind}: {instance_id} ({true_label} → {pred_label})"
        instances.append(
            {
                "label": label,
                "dir": inst_dir,
                "instance_id": instance_id,
                "true_label": true_label,
                "pred_label": pred_label,
                "kind": kind,
            }
        )
    instances.sort(key=lambda r: (r["kind"], r["instance_id"]))
    return instances


def _narrative_from_shap_json(shap_json_path: Path, predicted_label: str) -> str:
    if not shap_json_path.exists():
        return f"No SHAP data for {predicted_label}."
    data = read_json(shap_json_path)
    top_features = [
        {
            "feature": row["feature"],
            "contribution": row["shap_value"],
        }
        for row in data.get("features", [])[:15]
    ]
    if not top_features:
        return f"This flow was classified as {predicted_label} with no dominant feature attributions."

    attack_hints = {
        "BENIGN": "typical benign traffic characteristics",
        "DDoS": "volumetric attack patterns",
        "DoS": "denial-of-service behaviour",
        "PortScan": "reconnaissance or scanning activity",
        "BruteForce": "credential-guessing behaviour",
        "WebAttack": "web-layer exploitation attempts",
        "Infiltration": "lateral movement or infiltration",
        "Botnet": "botnet command-and-control traffic",
    }
    parts: list[str] = []
    for item in top_features[:3]:
        feat = str(item["feature"])
        direction = "elevated" if float(item["contribution"]) > 0 else "unusually low"
        parts.append(f"{feat} was {direction}")
    hint = attack_hints.get(predicted_label, "the observed threat class")
    joined = ", ".join(parts)
    return (
        f"This flow was classified as {predicted_label} primarily because {joined} "
        f"— consistent with {hint}."
    )


def _shap_waterfall_figure(shap_items: list[dict[str, Any]]) -> io.BytesIO:
    items = list(reversed(shap_items[:15]))
    features = [str(i["feature"]) for i in items]
    values = [float(i.get("contribution", i.get("shap_value", 0))) for i in items]
    colors = ["#ff0051" if v > 0 else "#008bfb" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP waterfall")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _lime_bar_figure(lime_items: list[dict[str, Any]]) -> io.BytesIO:
    items = lime_items[:10]
    features = [str(i["feature"]) for i in items]
    weights = [float(i["weight"]) for i in items]
    colors = ["#ff0051" if w > 0 else "#008bfb" for w in weights]
    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.35)))
    ax.barh(features, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight")
    ax.set_title("LIME explanation")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_three_panel_explanation(
    waterfall_source: Path | io.BytesIO | None,
    lime_source: Path | io.BytesIO | None,
    narrative: str,
) -> None:
    col_shap, col_lime, col_text = st.columns(3)
    with col_shap:
        st.caption("SHAP waterfall")
        if waterfall_source is not None:
            st.image(waterfall_source, use_container_width=True)
        else:
            st.write("—")
    with col_lime:
        st.caption("LIME bar")
        if lime_source is not None:
            st.image(lime_source, use_container_width=True)
        else:
            st.write("—")
    with col_text:
        st.caption("Narrative")
        st.markdown(narrative)


def _metrics_table_for_model(model: str) -> pd.DataFrame | None:
    metrics_path = EVALUATE_DIR / model / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    rows: list[dict[str, Any]] = []
    for cls in THREAT_CLASSES:
        rows.append(
            {
                "Class": cls,
                "Precision": metrics.get("precision", {}).get("per_class", {}).get(cls),
                "Recall": metrics.get("recall", {}).get("per_class", {}).get(cls),
                "F1": metrics.get("f1", {}).get("per_class", {}).get(cls),
            }
        )
    summary = pd.DataFrame(rows)
    macro = pd.DataFrame(
        [
            {
                "Class": "macro",
                "Precision": metrics.get("precision", {}).get("macro"),
                "Recall": metrics.get("recall", {}).get("macro"),
                "F1": metrics.get("f1", {}).get("macro"),
            },
            {
                "Class": "weighted",
                "Precision": metrics.get("precision", {}).get("weighted"),
                "Recall": metrics.get("recall", {}).get("weighted"),
                "F1": metrics.get("f1", {}).get("weighted"),
            },
        ]
    )
    return pd.concat([summary, macro], ignore_index=True)


def _collect_report_files() -> list[Path]:
    if not REPORTS_DIR.exists():
        return []
    files: list[Path] = []
    for path in sorted(REPORTS_DIR.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            files.append(path)
    return files


def _render_report_file(path: Path) -> None:
    suffix = path.suffix.lower()
    rel = path.relative_to(REPORTS_DIR)
    if suffix in {".md", ".txt"}:
        st.markdown(path.read_text(encoding="utf-8", errors="replace"))
    elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        st.image(str(path), use_container_width=True)
    elif suffix == ".csv":
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/csv",
            key=f"dl_csv_{rel.as_posix()}",
        )
    elif suffix == ".json":
        try:
            st.json(read_json(path))
        except (json.JSONDecodeError, OSError):
            st.code(path.read_text(encoding="utf-8", errors="replace"))
        st.download_button(
            "Download JSON",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/json",
            key=f"dl_json_{rel.as_posix()}",
        )
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        st.dataframe(df.head(200), use_container_width=True)
        st.caption(f"Showing up to 200 rows of {len(df)} total.")
    else:
        st.code(path.read_text(encoding="utf-8", errors="replace")[:8000])


_init_session_state()

tab_status, tab_run, tab_evaluate, tab_explain, tab_predict, tab_reports = st.tabs(
    ["Status", "Run", "Evaluate", "Explain", "Predict", "Reports"]
)

# --- Status ---
with tab_status:
    st.subheader("Pipeline status")
    rows = _checkpoint_status_rows()
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Force rerun")
    for step in STEP_ORDER:
        cols = st.columns([3, 1])
        with cols[0]:
            st.text(step)
        with cols[1]:
            if st.button("Force rerun", key=f"force_rerun_{step}"):
                marker = checkpoint_path(step)
                if marker.exists():
                    marker.unlink()
                try:
                    run_step(step)
                except Exception as exc:
                    st.error(f"{step}: {exc}")
                else:
                    st.rerun()

# --- Run ---
with tab_run:
    st.subheader("Run pipeline")
    if st.button("Run", disabled=st.session_state["pipeline_running"]):
        st.session_state["pipeline_running"] = True
        st.session_state["pipeline_error"] = None
        threading.Thread(target=_run_pipeline_background, daemon=True).start()

    if st.session_state["pipeline_error"]:
        st.error(st.session_state["pipeline_error"])

    if st.session_state["pipeline_running"]:
        st.caption("Pipeline running in background.")
        if st.button("Refresh log"):
            st.rerun()

    st.subheader("Log tail")
    st.code(_tail_log(), language=None)

# --- Evaluate ---
with tab_evaluate:
    st.subheader("Evaluation")
    leaderboard_path = EVALUATE_DIR / "final_leaderboard.md"
    if leaderboard_path.exists():
        st.markdown(leaderboard_path.read_text(encoding="utf-8"))
    else:
        st.caption("No final leaderboard yet. Run the evaluate step.")

    models = _available_models() or list(MODEL_NAMES)
    selected_model = st.selectbox("Model", models, key="eval_model")

    model_dir = EVALUATE_DIR / selected_model
    cm_path = model_dir / "confusion_matrix.png"
    roc_path = model_dir / "roc_curves.png"
    pr_path = model_dir / "pr_curves.png"

    img_cols = st.columns(3)
    with img_cols[0]:
        st.caption("Confusion matrix")
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.write("—")
    with img_cols[1]:
        st.caption("ROC curves")
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.write("—")
    with img_cols[2]:
        st.caption("PR curves")
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)
        else:
            st.write("—")

    st.subheader("Metrics")
    metrics_df = _metrics_table_for_model(selected_model)
    if metrics_df is not None:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No metrics for this model.")

# --- Explain ---
with tab_explain:
    st.subheader("Global explanations")
    global_models = _available_models() or list(MODEL_NAMES)
    global_model = st.selectbox("Model", global_models, key="global_model")
    plots = _global_plot_paths(global_model)

    if plots["beeswarm"]:
        st.caption("SHAP beeswarm")
        for img in plots["beeswarm"]:
            st.image(str(img), use_container_width=True)
    else:
        st.caption("No beeswarm plots. Run explain-global.")

    if plots["bar"]:
        st.caption("Mean |SHAP| bar chart")
        st.image(str(plots["bar"][0]), use_container_width=True)

    if plots["dependence"]:
        st.caption("Top dependence plots")
        dep_cols = st.columns(min(3, len(plots["dependence"])))
        for i, dep in enumerate(plots["dependence"]):
            with dep_cols[i % len(dep_cols)]:
                st.image(str(dep), use_container_width=True)

    st.divider()
    st.subheader("Local explanations")
    local_models = _available_models() or list(MODEL_NAMES)
    local_model = st.selectbox("Model", local_models, key="local_model")
    instances = _list_local_instances(local_model)
    if instances:
        instance_labels = [inst["label"] for inst in instances]
        chosen_label = st.selectbox("Instance", instance_labels, key="local_instance")
        chosen = next(inst for inst in instances if inst["label"] == chosen_label)
        inst_dir = chosen["dir"]
        waterfall = inst_dir / "shap_waterfall.png"
        lime_bar = inst_dir / "lime_bar.png"
        narrative = _narrative_from_shap_json(
            inst_dir / "shap_values.json",
            chosen["pred_label"],
        )
        _render_three_panel_explanation(
            waterfall if waterfall.exists() else None,
            lime_bar if lime_bar.exists() else None,
            narrative,
        )
    else:
        st.caption("No local explanations. Run explain-local.")

    st.divider()
    st.subheader("Explanation quality")
    comparison_path = QUALITY_DIR / "comparison.md"
    if comparison_path.exists():
        st.markdown(comparison_path.read_text(encoding="utf-8"))
    else:
        st.caption("No quality comparison. Run explanation-quality.")

    quality_models = _available_models() or list(MODEL_NAMES)
    for q_model in quality_models:
        st.caption(q_model)
        q_cols = st.columns(3)
        fidelity_path = QUALITY_DIR / f"fidelity_{q_model}.csv"
        stability_path = QUALITY_DIR / f"stability_{q_model}.csv"
        comp_path = QUALITY_DIR / f"comprehensibility_{q_model}.csv"
        with q_cols[0]:
            st.markdown("**Fidelity**")
            if fidelity_path.exists():
                st.dataframe(pd.read_csv(fidelity_path), use_container_width=True, hide_index=True)
            else:
                st.write("—")
        with q_cols[1]:
            st.markdown("**Stability**")
            if stability_path.exists():
                st.dataframe(pd.read_csv(stability_path), use_container_width=True, hide_index=True)
            else:
                st.write("—")
        with q_cols[2]:
            st.markdown("**Comprehensibility**")
            if comp_path.exists():
                st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
            else:
                st.write("—")

    summary_plot = QUALITY_DIR / "fidelity_summary.png"
    if summary_plot.exists():
        st.caption("Fidelity summary")
        st.image(str(summary_plot), use_container_width=True)

# --- Predict ---
with tab_predict:
    st.subheader("Predict from CSV")
    st.caption("Upload a CSV of flow records matching the CICIDS-2017 schema, or run the built-in demo set.")

    example_path = MANUAL_DIR / "example_flows.csv"
    demo_cols = st.columns(2)
    with demo_cols[0]:
        if example_path.exists() and st.button("Run demo (example_flows.csv)"):
            try:
                result_df = predict_from_file(example_path)
                st.session_state["predictions_df"] = result_df
                st.session_state["predictions_source"] = "example_flows.csv"
            except Exception as exc:
                st.error(str(exc))
    with demo_cols[1]:
        if example_path.exists():
            st.caption(f"{len(pd.read_csv(example_path))} hand-selected flows covering all threat classes.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="predict_upload")
    if uploaded is not None and st.button("Predict"):
        try:
            tmp_path = CHECKPOINTS_DIR.parent / "data" / "manual" / "_ui_upload.csv"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.getvalue())
            result_df = predict_from_file(tmp_path)
            st.session_state["predictions_df"] = result_df
            st.session_state["predictions_source"] = uploaded.name
        except Exception as exc:
            st.error(str(exc))

    pred_df: pd.DataFrame | None = st.session_state.get("predictions_df")
    if pred_df is not None:
        st.dataframe(pred_df, use_container_width=True)
        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

        st.subheader("Explain selected row")
        row_indices = list(range(len(pred_df)))
        row_idx = st.selectbox("Row", row_indices, format_func=lambda i: f"Row {i}", key="explain_row")
        explain_model = st.selectbox(
            "Model",
            _available_models() or list(MODEL_NAMES),
            key="explain_row_model",
        )
        if st.button("Explain"):
            try:
                row = pred_df.iloc[row_idx]
                feature_cols = [
                    c
                    for c in row.index
                    if not c.startswith("predicted_") and c != "confidence"
                    and not str(c).startswith("predicted_proba_")
                ]
                row_input = row[feature_cols].to_dict()
                explanation = explain_one(row_input, model_name=explain_model)
                shap_items = [
                    {
                        "feature": item["feature"],
                        "contribution": item["contribution"],
                    }
                    for item in explanation.get("shap_top15", [])
                ]
                lime_items = explanation.get("lime_top10", [])
                _render_three_panel_explanation(
                    _shap_waterfall_figure(shap_items) if shap_items else None,
                    _lime_bar_figure(lime_items) if lime_items else None,
                    explanation.get("narrative", ""),
                )
            except Exception as exc:
                st.error(str(exc))

# --- Reports ---
with tab_reports:
    st.subheader("Reports")
    report_files = _collect_report_files()
    if not report_files:
        st.caption("No reports found.")
    else:
        by_dir: dict[str, list[Path]] = {}
        for path in report_files:
            rel = path.relative_to(REPORTS_DIR)
            section = "root" if len(rel.parts) == 1 else rel.parts[0]
            by_dir.setdefault(section, []).append(path)

        for section in sorted(by_dir.keys()):
            with st.expander(section, expanded=False):
                for path in sorted(by_dir[section]):
                    rel = path.relative_to(REPORTS_DIR)
                    st.markdown(f"**{rel.as_posix()}**")
                    _render_report_file(path)
                    st.divider()
