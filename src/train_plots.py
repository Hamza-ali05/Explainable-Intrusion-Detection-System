"""Generate training artefact plots for the three tree models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree

from src.core import MODEL_NAMES, REPORTS_DIR, THREAT_CLASSES, read_joblib, read_parquet
from src.model import (
    MODELS_DIR,
    TRAIN_REPORTS_DIR,
    _feature_importance_frame,
    _feature_matrix,
    _load_splits,
    _target_column,
    tree_model_for_shap,
)
from src.prep import TABULAR_UNSCALED_PARQUET

PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_DPI = 150
TREE_DPI = 120
TOP_N = 20


def _load_feature_names() -> list[str]:
    splits = _load_splits()
    target_col = _target_column(splits)
    df = read_parquet(TABULAR_UNSCALED_PARQUET)
    X, _ = _feature_matrix(df.iloc[:1], target_col)
    return list(X.columns)


def _unwrap_model(model: Any) -> Any:
    return tree_model_for_shap(model)


def _cv_param_label(row: pd.Series) -> str:
    parts: list[str] = []
    for col in sorted(row.index):
        if not col.startswith("param_"):
            continue
        value = row[col]
        if pd.isna(value):
            continue
        parts.append(f"{col.replace('param_', '')}={value}")
    return ", ".join(parts) if parts else "default"


def _save_feature_importance_plot(
    model: Any,
    feature_names: list[str],
    title: str,
    out_path: Path,
) -> Path:
    fi = _feature_importance_frame(model, feature_names)
    if fi.empty:
        raise ValueError(f"No feature importances available for {out_path.name}")

    top = fi.head(TOP_N).sort_values("importance_gain", ascending=True)
    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["feature"], top["importance_gain"], color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out_path


def _save_cv_bar_plot(
    cv_df: pd.DataFrame,
    title: str,
    out_path: Path,
) -> Path:
    labels = [_cv_param_label(row) for _, row in cv_df.iterrows()]
    scores = cv_df["mean_test_score"].astype(float)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(scores))
        ax.bar(x, scores, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Hyperparameter set")
        ax.set_ylabel("Mean CV F1-macro")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out_path


def _save_cv_results_plot(cv_df: pd.DataFrame, model_name: str, out_path: Path) -> Path:
    param_cols = [c for c in cv_df.columns if c.startswith("param_")]
    scores = cv_df["mean_test_score"].astype(float)

    with plt.style.context(PLOT_STYLE):
        if len(param_cols) == 1:
            param = param_cols[0]
            grouped = cv_df.groupby(param)["mean_test_score"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                grouped[param].astype(str),
                grouped["mean_test_score"],
                marker="o",
                color="steelblue",
            )
            ax.set_xlabel(param.replace("param_", ""))
            ax.set_ylabel("Mean CV F1-macro")
        else:
            labels = [_cv_param_label(row) for _, row in cv_df.iterrows()]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(np.arange(len(scores)), scores, marker="o", color="steelblue")
            ax.set_xlabel("Hyperparameter combination index")
            ax.set_ylabel("Mean CV F1-macro")
            ax.set_xticks(np.arange(len(scores)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)

        ax.set_title(f"{model_name.replace('_', ' ').title()} — CV F1-Macro vs Hyperparameters")
        plt.tight_layout()
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out_path


def _save_decision_tree_structure(
    model: Any,
    feature_names: list[str],
    out_path: Path,
) -> Path:
    estimator = _unwrap_model(model)
    fig, ax = plt.subplots(figsize=(20, 10))
    class_names = [str(c) for c in getattr(model, "classes_", THREAT_CLASSES)]
    plot_tree(
        estimator,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree — Structure (Top 4 Levels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=TREE_DPI, bbox_inches="tight")
    plt.close()
    return out_path


def generate_train_plots() -> list[str]:
    TRAIN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = _load_feature_names()
    output_paths: list[str] = []

    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            continue

        model = read_joblib(model_path)
        train_curve_path = TRAIN_REPORTS_DIR / f"{model_name}_train_curve.png"
        cv_path = TRAIN_REPORTS_DIR / f"{model_name}_cv_results.csv"

        if model_name == "decision_tree":
            _save_decision_tree_structure(model, feature_names, train_curve_path)
        elif model_name == "random_forest":
            display = "Random Forest"
            _save_feature_importance_plot(
                model,
                feature_names,
                f"{display} — Feature Importances (Top {TOP_N})",
                train_curve_path,
            )
        elif model_name == "xgboost":
            if cv_path.exists():
                cv_df = pd.read_csv(cv_path)
                _save_cv_bar_plot(
                    cv_df,
                    "XGBoost — Cross-Validation F1-Macro by Hyperparameter Set",
                    train_curve_path,
                )
            else:
                _save_feature_importance_plot(
                    model,
                    feature_names,
                    f"XGBoost — Feature Importances (Top {TOP_N})",
                    train_curve_path,
                )

        output_paths.append(str(train_curve_path))

        if cv_path.exists():
            cv_plot_path = TRAIN_REPORTS_DIR / f"{model_name}_cv_results_plot.png"
            cv_df = pd.read_csv(cv_path)
            _save_cv_results_plot(cv_df, model_name, cv_plot_path)
            output_paths.append(str(cv_plot_path))

    print("\nTraining artefact plots saved:")
    for path in sorted(TRAIN_REPORTS_DIR.iterdir()):
        if path.is_file():
            size_kb = path.stat().st_size / 1024
            print(f"  {path}  ({size_kb:.1f} KB)")

    return output_paths


def run_default() -> dict[str, list[str]]:
    paths = generate_train_plots()
    return {"output_paths": paths}


if __name__ == "__main__":
    run_default()
