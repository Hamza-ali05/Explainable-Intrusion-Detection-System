from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score
from src.core import (
    ARTIFACTS_DIR,
    LOGGER,
    MODEL_NAMES,
    RANDOM_SEED,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpointed,
    read_joblib,
    read_json,
    read_parquet,
    write_json,
)
from src.model import (
    BEST_MODEL_META_PATH,
    BEST_MODEL_PATH,
    MODELS_DIR,
    PREPROCESS_BUNDLE_PATH,
    _encode_features,
    _load_inference_context,
    _load_splits,
    _prepare_raw_dataframe,
    _target_column,
    tree_model_for_shap,
)
from src.prep import TABULAR_UNSCALED_PARQUET
from src.timing import record_explainability

try:
    import shap
except Exception:  # pragma: no cover
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover
    LimeTabularExplainer = None


# --- paths ---
XAI_REPORTS_DIR = REPORTS_DIR / "xai"
GLOBAL_XAI_DIR = XAI_REPORTS_DIR / "global"
LOCAL_XAI_DIR = XAI_REPORTS_DIR / "local"
QUALITY_DIR = XAI_REPORTS_DIR / "quality"
IOC_FEATURE_MAP_PATH = ARTIFACTS_DIR / "ioc_feature_map.json"

GLOBAL_SAMPLE_SIZE = 5000
GLOBAL_PER_CLASS_TARGET = 500
TOP_GLOBAL_FEATURES = 20
PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_FIGSIZE = (12, 8)
PLOT_DPI = 150
LOCAL_WATERFALL_FIGSIZE = (10, 6)
LOCAL_LIME_FIGSIZE = (10, 5)

MODEL_DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
LOCAL_REFERENCE_MODEL = "xgboost"
LOCAL_CORRECT_PER_CLASS = 1
LOCAL_WRONG_PER_CLASS = 1
LOCAL_LIME_STABILITY_RUNS = 5
FIDELITY_SAMPLE_SIZE = 2000
FIDELITY_K_VALUES = [1, 3, 5, 10]
COMPREHENSIBILITY_SAMPLE_SIZE = 2000

IOC_REFERENCE: dict[str, list[str]] = {
    "DDoS": [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
    ],
    "DoS": [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s",
    ],
    "PortScan": [
        "Destination Port",
        "Flow Duration",
        "Total Fwd Packets",
        "Fwd Packet Length Min",
    ],
    "BruteForce": [
        "Flow Duration",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Total Fwd Packets",
    ],
    "WebAttack": [
        "Destination Port",
        "Flow Duration",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
    ],
    "Botnet": [
        "Flow Duration",
        "Flow IAT Mean",
        "Fwd IAT Mean",
        "Active Mean",
        "Idle Mean",
    ],
    "Infiltration": [
        "Flow Duration",
        "Flow Bytes/s",
        "Total Fwd Packets",
        "Init_Win_bytes_forward",
    ],
}
IOC_THREAT_CLASSES = list(IOC_REFERENCE.keys())
PERTURBATION_SIGMA = 0.01

# Module-level explainer cache (Streamlit / explain_one)
_EXPLAINER_CACHE: dict[str, Any] = {
    "shap": {},
    "lime": None,
    "lime_meta": None,
}


# --- helpers ---
def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(name).strip())


def _write_markdown(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_data_context(
    unscaled_path: Path | None = None,
) -> dict[str, Any]:
    unscaled_path = Path(unscaled_path or TABULAR_UNSCALED_PARQUET)
    if not unscaled_path.exists():
        raise FileNotFoundError(f"Unscaled tabular data not found: {unscaled_path}")

    splits = _load_splits()
    target_col = _target_column(splits)
    df = read_parquet(unscaled_path)

    train_df = df.iloc[splits["train"]].copy()
    test_df = df.iloc[splits["test"]].copy()
    drop_cols = {target_col, "threat_label"}
    feature_names = [c for c in test_df.columns if c not in drop_cols]

    X_train = train_df[feature_names]
    X_test = test_df[feature_names]
    y_test = test_df[target_col].astype(str).to_numpy()
    test_index = test_df.index.to_numpy()

    bundle = read_joblib(PREPROCESS_BUNDLE_PATH) if PREPROCESS_BUNDLE_PATH.exists() else {}
    scaler = bundle.get("scaler")

    return {
        "splits": splits,
        "target_col": target_col,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_index": test_index,
        "test_df": test_df,
        "scaler": scaler,
        "unscaled_path": unscaled_path,
    }


def _load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run train() first.")
    return read_joblib(path)


def _best_model_name() -> str:
    if BEST_MODEL_META_PATH.exists():
        meta = read_json(BEST_MODEL_META_PATH)
        if meta.get("best_model"):
            return str(meta["best_model"])
    for name in MODEL_NAMES:
        if (MODELS_DIR / f"{name}.joblib").exists():
            return name
    return MODEL_NAMES[0]


def _shap_explainer_for_model(model: Any, model_name: str) -> Any:
    if shap is None:
        raise RuntimeError("SHAP is not installed.")
    cached = _EXPLAINER_CACHE["shap"].get(model_name)
    if cached is not None:
        return cached
    kwargs: dict[str, Any] = {}
    if model_name == "xgboost":
        kwargs["model_output"] = "raw"
    explainer = shap.TreeExplainer(tree_model_for_shap(model), **kwargs)
    _EXPLAINER_CACHE["shap"][model_name] = explainer
    return explainer


def _lime_explainer_for_data(
    X_train: pd.DataFrame,
    class_names: list[str],
) -> LimeTabularExplainer:
    if LimeTabularExplainer is None:
        raise RuntimeError("LIME is not installed.")
    meta = (tuple(class_names), X_train.shape[1], len(X_train))
    if _EXPLAINER_CACHE["lime"] is not None and _EXPLAINER_CACHE["lime_meta"] == meta:
        return _EXPLAINER_CACHE["lime"]
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_train, dtype=np.float64),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        random_state=RANDOM_SEED,
    )
    _EXPLAINER_CACHE["lime"] = explainer
    _EXPLAINER_CACHE["lime_meta"] = meta
    return explainer


def _expected_value_json(explainer: Any, n_classes: int) -> Any:
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return [float(v) for v in np.asarray(ev).ravel()[:n_classes]]
    return float(ev)


def _local_shap_row(
    explainer: Any,
    shap_values: Any,
    class_index: int,
) -> tuple[np.ndarray, float]:
    """Extract one instance's SHAP vector and base value for the predicted class."""
    ev = np.asarray(explainer.expected_value).ravel()
    if isinstance(shap_values, list):
        values_row = np.asarray(shap_values[class_index][0], dtype=np.float64)
        return values_row, float(ev[class_index])
    matrix = np.asarray(shap_values, dtype=np.float64)
    if matrix.ndim == 3:
        return matrix[0, :, class_index].astype(np.float64), float(ev[class_index])
    values_row = np.asarray(matrix[0], dtype=np.float64)
    base_idx = min(class_index, len(ev) - 1)
    return values_row, float(ev[base_idx])


def _stack_shap_for_predictions(
    shap_values: Any,
    model: Any,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    X_arr = np.asarray(X)
    if isinstance(shap_values, list):
        preds = model.predict(X_arr)
        classes = list(model.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        stacked = np.zeros((len(X_arr), shap_values[0].shape[1]), dtype=np.float64)
        for i, pred in enumerate(preds):
            idx = class_to_idx.get(pred, 0)
            stacked[i] = shap_values[idx][i]
        return stacked
    matrix = np.asarray(shap_values, dtype=np.float64)
    if matrix.ndim == 3:
        preds = model.predict(X_arr)
        classes = list(model.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        stacked = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
        for i, pred in enumerate(preds):
            idx = class_to_idx.get(pred, 0)
            stacked[i] = matrix[i, :, idx]
        return stacked
    return matrix


def _global_mean_abs_shap(shap_values: Any) -> np.ndarray:
    if isinstance(shap_values, list):
        per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
        return np.mean(per_class, axis=0)
    matrix = np.asarray(shap_values, dtype=np.float64)
    if matrix.ndim == 3:
        return np.abs(matrix).mean(axis=(0, 2))
    return np.abs(matrix).mean(axis=0)


def _shap_values_as_class_list(shap_values: Any) -> list[np.ndarray]:
    """Normalize list or (n_samples, n_features, n_classes) SHAP output to per-class matrices."""
    if isinstance(shap_values, list):
        return [np.asarray(sv, dtype=np.float64) for sv in shap_values]
    matrix = np.asarray(shap_values, dtype=np.float64)
    if matrix.ndim == 3:
        return [matrix[:, :, i] for i in range(matrix.shape[2])]
    return [matrix]


def _dependence_shap_matrix(shap_values: Any) -> np.ndarray:
    """2D SHAP matrix for dependence plots (class-averaged when multi-output)."""
    per_class = _shap_values_as_class_list(shap_values)
    if len(per_class) == 1:
        return per_class[0]
    return np.mean(np.stack(per_class, axis=0), axis=0)


def _stratified_sample_indices(
    y: np.ndarray,
    n: int,
    random_state: int = RANDOM_SEED,
    per_class_target: int | None = None,
) -> np.ndarray:
    labels = np.asarray(y)
    rng = np.random.default_rng(random_state)
    unique = sorted(np.unique(labels))
    chosen: list[int] = []

    if per_class_target is not None:
        remaining_budget = n
        for cls in unique:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0 or remaining_budget <= 0:
                continue
            take = min(per_class_target, len(cls_idx), remaining_budget)
            picked = rng.choice(cls_idx, size=take, replace=False)
            chosen.extend(picked.tolist())
            remaining_budget -= take

        if len(chosen) < n:
            pool = sorted(set(range(len(labels))) - set(chosen))
            extra = min(n - len(chosen), len(pool))
            if extra > 0:
                chosen.extend(rng.choice(pool, size=extra, replace=False).tolist())
    else:
        label_series = pd.Series(labels)
        groups = label_series.groupby(label_series)
        weights = groups.transform("count") / len(label_series)
        target_counts = (weights * n).round().astype(int).clip(lower=1)
        target_counts = target_counts.groupby(label_series).transform("max")
        for cls in unique:
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) == 0:
                continue
            n_cls = min(int(target_counts[label_series == cls].iloc[0]), len(cls_idx))
            n_cls = min(n_cls, max(1, int(round(n * len(cls_idx) / len(labels)))))
            picked = rng.choice(cls_idx, size=n_cls, replace=False)
            chosen.extend(picked.tolist())

        if len(chosen) < n:
            pool = sorted(set(range(len(labels))) - set(chosen))
            extra = min(n - len(chosen), len(pool))
            if extra > 0:
                chosen.extend(rng.choice(pool, size=extra, replace=False).tolist())

    if len(chosen) > n:
        chosen = rng.choice(np.array(chosen), size=n, replace=False).tolist()

    return np.array(sorted(chosen), dtype=int)


def _top_feature_indices(values: np.ndarray, k: int = 10) -> set[int]:
    abs_vals = np.abs(values)
    top = np.argsort(abs_vals)[-k:]
    return set(int(i) for i in top.tolist())


def _jaccard_sets(a: set[Any], b: set[Any]) -> float:
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def _spearman_rank_corr(a: dict[str, float], b: dict[str, float]) -> float:
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return float("nan")
    va = [a[k] for k in shared]
    vb = [b[k] for k in shared]
    corr, _ = spearmanr(va, vb)
    return float(corr) if corr is not None else float("nan")


def _as_feature_frame(values: Any, feature_names: list[str]) -> pd.DataFrame:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return pd.DataFrame(array, columns=feature_names)


def _predict_proba_for_lime(model: Any, feature_names: list[str]):
    def predict_fn(values: Any) -> np.ndarray:
        return model.predict_proba(_as_feature_frame(values, feature_names))

    return predict_fn


def _perturb_row_scaled(
    row: np.ndarray,
    scaler: Any,
    sigma: float = PERTURBATION_SIGMA,
    rng: np.random.Generator | None = None,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng(RANDOM_SEED)
    if scaler is None:
        return row + rng.normal(0.0, sigma, size=row.shape)
    columns = feature_names or [f"f{i}" for i in range(len(row))]
    frame = _as_feature_frame(row, columns)
    scaled = scaler.transform(frame)
    noisy = scaled + rng.normal(0.0, sigma, size=scaled.shape)
    return scaler.inverse_transform(pd.DataFrame(noisy, columns=columns))[0]


def _load_ioc_map() -> dict[str, dict[str, str]]:
    if not IOC_FEATURE_MAP_PATH.exists():
        return {}
    return read_json(IOC_FEATURE_MAP_PATH)


def _ioc_alignment_fraction(top_features: list[str], ioc_map: dict[str, dict[str, str]]) -> float:
    if not top_features:
        return 0.0
    hits = sum(1 for f in top_features if f in ioc_map)
    return hits / len(top_features)


def _sparsity_80_mass(shap_row: np.ndarray) -> int:
    abs_vals = np.abs(shap_row)
    total = abs_vals.sum()
    if total <= 0:
        return 0
    order = np.argsort(-abs_vals)
    cumulative = 0.0
    for count, idx in enumerate(order, start=1):
        cumulative += abs_vals[idx]
        if cumulative / total >= 0.8:
            return count
    return len(order)


def _narrative_from_shap(
    predicted_label: str,
    top_features: list[dict[str, Any]],
    ioc_map: dict[str, dict[str, str]],
) -> str:
    if not top_features:
        return f"This flow was classified as {predicted_label} with no dominant feature attributions."

    parts: list[str] = []
    for item in top_features[:3]:
        feat = str(item["feature"])
        direction = "elevated" if float(item["contribution"]) > 0 else "unusually low"
        ioc = ioc_map.get(feat, {})
        hint = ioc.get("description", "")
        if hint:
            parts.append(f"{feat} was {direction} ({ioc.get('category', 'signal')})")
        else:
            parts.append(f"{feat} was {direction}")

    attack_hints = {
        "DDoS": "volumetric attack patterns",
        "DoS": "denial-of-service behaviour",
        "PortScan": "reconnaissance or scanning activity",
        "BruteForce": "credential-guessing behaviour",
        "WebAttack": "web-layer exploitation attempts",
        "Infiltration": "lateral movement or infiltration",
        "Botnet": "botnet command-and-control traffic",
        "BENIGN": "typical benign traffic characteristics",
    }
    hint = attack_hints.get(predicted_label, "the observed threat class")
    joined = ", ".join(parts)
    return (
        f"This flow was classified as {predicted_label} primarily because {joined} "
        f"— consistent with {hint}."
    )


def _save_shap_global_plots(
    model_name: str,
    shap_values: Any,
    X_sample: pd.DataFrame,
    feature_names: list[str],
    mean_abs: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if shap is None:
        return paths

    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    X_np = np.asarray(X_sample, dtype=np.float64)

    per_class_shap = _shap_values_as_class_list(shap_values)
    summary_shap = (
        per_class_shap[0]
        if len(per_class_shap) == 1
        else np.mean(np.stack(per_class_shap, axis=0), axis=0)
    )

    top_idx = np.argsort(-mean_abs)[: min(TOP_GLOBAL_FEATURES, len(mean_abs))]
    top_feature_names = [feature_names[i] for i in top_idx]
    summary_top = summary_shap[:, top_idx]
    X_top = X_np[:, top_idx]

    with plt.style.context(PLOT_STYLE):
        plt.figure(figsize=PLOT_FIGSIZE)
        shap.summary_plot(
            summary_top,
            X_top,
            feature_names=top_feature_names,
            max_display=TOP_GLOBAL_FEATURES,
            show=False,
        )
        plt.title(f"{display_name} — SHAP Beeswarm (Top {TOP_GLOBAL_FEATURES} Features)")
        beeswarm = out_dir / f"{model_name}_shap_beeswarm.png"
        plt.tight_layout()
        plt.savefig(beeswarm, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        paths.append(beeswarm)

        bar_df = pd.DataFrame(
            {"feature": top_feature_names, "mean_abs_shap": mean_abs[top_idx]}
        ).sort_values("mean_abs_shap", ascending=True)
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        ax.barh(bar_df["feature"], bar_df["mean_abs_shap"], color="steelblue")
        ax.set_title(f"{display_name} — Global SHAP Feature Importance")
        ax.set_xlabel("Mean |SHAP|")
        bar_path = out_dir / f"{model_name}_shap_bar.png"
        plt.tight_layout()
        plt.savefig(bar_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        paths.append(bar_path)

        dep_shap = _dependence_shap_matrix(shap_values)
        top_feat_idx = int(top_idx[0])
        second_feat_idx = int(top_idx[1]) if len(top_idx) > 1 else top_feat_idx
        top_feat_name = feature_names[top_feat_idx]
        dep_path = out_dir / f"{model_name}_shap_dependence.png"
        try:
            plt.figure(figsize=PLOT_FIGSIZE)
            shap.dependence_plot(
                top_feat_idx,
                dep_shap,
                X_np,
                feature_names=feature_names,
                interaction_index=second_feat_idx,
                show=False,
            )
            plt.title(f"{display_name} — SHAP Dependence: {top_feat_name}")
            plt.tight_layout()
            plt.savefig(dep_path, dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()
            paths.append(dep_path)
        except Exception as exc:
            plt.close("all")
            LOGGER.warning("Dependence plot failed for %s feature %s: %s", model_name, top_feat_name, exc)

    return paths


def _save_local_shap_waterfall(
    explainer: Any,
    shap_values: Any,
    row: np.ndarray,
    feature_names: list[str],
    class_index: int,
    out_path: Path,
    title: str,
) -> Path | None:
    if shap is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    values_row, base = _local_shap_row(explainer, shap_values, class_index)
    try:
        explanation = shap.Explanation(
            values=values_row,
            base_values=base,
            data=row,
            feature_names=feature_names,
        )
        plt.figure(figsize=LOCAL_WATERFALL_FIGSIZE)
        shap.plots.waterfall(explanation, show=False, max_display=15)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception as exc:
        plt.close("all")
        LOGGER.warning("Waterfall plot failed for %s: %s", out_path.name, exc)
        return None


def _save_lime_bar_chart(
    lime_explainer: LimeTabularExplainer,
    model: Any,
    row: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    title: str,
) -> tuple[Path | None, list[tuple[str, float]]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exp = lime_explainer.explain_instance(
        data_row=row,
        predict_fn=_predict_proba_for_lime(model, feature_names),
        num_features=10,
        top_labels=1,
    )
    label = exp.top_labels[0]
    pairs = exp.as_list(label=label)[:10]

    features = [feature for feature, _ in pairs]
    weights = [float(weight) for _, weight in pairs]
    colors = ["green" if weight > 0 else "red" for weight in weights]

    fig, ax = plt.subplots(figsize=LOCAL_LIME_FIGSIZE)
    ax.barh(features, weights, color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    return out_path, pairs


def _save_lime_stability_runs(
    lime_explainer: LimeTabularExplainer,
    model: Any,
    row: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    n_runs: int = LOCAL_LIME_STABILITY_RUNS,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predict_fn = _predict_proba_for_lime(model, feature_names)
    runs: list[dict[str, Any]] = []
    for run_idx in range(n_runs):
        exp = lime_explainer.explain_instance(
            data_row=row,
            predict_fn=predict_fn,
            num_features=10,
            top_labels=1,
        )
        label = exp.top_labels[0]
        pairs = exp.as_list(label=label)[:10]
        runs.append(
            {
                "run": run_idx,
                "label": int(label),
                "top10": [
                    {"rank": rank, "feature": feature, "weight": float(weight)}
                    for rank, (feature, weight) in enumerate(pairs, start=1)
                ],
            }
        )

    write_json({"n_runs": n_runs, "runs": runs}, out_path)
    return out_path


def _local_instance_stem(instance: dict[str, Any]) -> str:
    true_label = _safe_filename(instance["true_label"])
    kind = instance["kind"]
    idx = int(instance["instance_id"])
    return f"{true_label}_{kind}_{idx}"


def _select_local_instances_xgb(ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = _load_model(LOCAL_REFERENCE_MODEL)
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]
    test_index = ctx["test_index"]
    preds = model.predict(X_test)
    rng = np.random.default_rng(RANDOM_SEED)

    instances: list[dict[str, Any]] = []
    per_class: dict[str, Any] = {}

    for cls in THREAT_CLASSES:
        cls_positions = np.where(y_test == cls)[0]
        correct = cls_positions[preds[cls_positions] == cls]
        wrong = cls_positions[preds[cls_positions] != cls]

        picked_correct = int(rng.choice(correct)) if len(correct) else None
        picked_wrong = int(rng.choice(wrong)) if len(wrong) else None

        per_class[cls] = {
            "available_in_test": int(len(cls_positions)),
            "correct_available": int(len(correct)),
            "wrong_available": int(len(wrong)),
            "correct_position": picked_correct,
            "wrong_position": picked_wrong,
            "correct_instance_id": int(test_index[picked_correct]) if picked_correct is not None else None,
            "wrong_instance_id": int(test_index[picked_wrong]) if picked_wrong is not None else None,
        }

        if picked_correct is not None:
            instances.append(
                {
                    "position": picked_correct,
                    "instance_id": int(test_index[picked_correct]),
                    "true_label": cls,
                    "predicted_label": str(preds[picked_correct]),
                    "kind": "correct",
                }
            )
        if picked_wrong is not None:
            instances.append(
                {
                    "position": picked_wrong,
                    "instance_id": int(test_index[picked_wrong]),
                    "true_label": cls,
                    "predicted_label": str(preds[picked_wrong]),
                    "kind": "wrong",
                }
            )

    selection_log = {
        "reference_model": LOCAL_REFERENCE_MODEL,
        "per_class": per_class,
        "instances": instances,
    }
    return instances, selection_log


def _save_local_shap_plots(
    explainer: Any,
    shap_values: Any,
    row: np.ndarray,
    feature_names: list[str],
    class_index: int,
    out_dir: Path,
    instance_id: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    values_row, base = _local_shap_row(explainer, shap_values, class_index)

    json_rows = [
        {"feature": feature_names[i], "shap_value": float(values_row[i])}
        for i in range(len(feature_names))
    ]
    json_rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    shap_json_path = out_dir / "shap_values.json"
    write_json({"base_value": base, "features": json_rows[:50]}, shap_json_path)
    paths.append(shap_json_path)

    if shap is not None:
        try:
            explanation = shap.Explanation(
                values=values_row,
                base_values=base,
                data=row,
                feature_names=feature_names,
            )
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False, max_display=15)
            wf_path = out_dir / "shap_waterfall.png"
            plt.tight_layout()
            plt.savefig(wf_path, dpi=120, bbox_inches="tight")
            plt.close()
            paths.append(wf_path)
        except Exception as exc:
            LOGGER.warning("Waterfall plot failed for instance %s: %s", instance_id, exc)

        try:
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                base,
                values_row,
                row,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
            )
            fp_path = out_dir / "shap_force.png"
            plt.tight_layout()
            plt.savefig(fp_path, dpi=120, bbox_inches="tight")
            plt.close()
            paths.append(fp_path)
        except Exception as exc:
            LOGGER.warning("Force plot failed for instance %s: %s", instance_id, exc)

    return paths, json_rows


def _save_lime_local(
    lime_explainer: LimeTabularExplainer,
    model: Any,
    row: np.ndarray,
    out_dir: Path,
    feature_names: list[str],
) -> tuple[list[Path], list[tuple[str, float]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    exp = lime_explainer.explain_instance(
        data_row=row,
        predict_fn=_predict_proba_for_lime(model, feature_names),
        num_features=10,
        top_labels=1,
    )
    label = exp.top_labels[0]
    pairs = exp.as_list(label=label)[:10]
    lime_json_path = out_dir / "lime_top10.json"
    write_json(
        {
            "label": int(label),
            "contributions": [{"feature": f, "weight": float(w)} for f, w in pairs],
        },
        lime_json_path,
    )
    paths.append(lime_json_path)

    try:
        fig = exp.as_pyplot_figure(label=label)
        lime_png = out_dir / "lime_bar.png"
        fig.savefig(lime_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(lime_png)
    except Exception as exc:
        LOGGER.warning("LIME bar plot failed: %s", exc)

    return paths, pairs


def _select_local_instance_ids(
    ctx: dict[str, Any],
    reference_model_name: str | None = None,
) -> tuple[list[int], dict[str, Any]]:
    model_name = reference_model_name or _best_model_name()
    model = _load_model(model_name)
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]
    test_index = ctx["test_index"]
    preds = model.predict(X_test)

    log: dict[str, Any] = {"reference_model": model_name, "per_class": {}}
    selected: list[int] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for cls in sorted(set(y_test)):
        cls_mask = y_test == cls
        cls_positions = np.where(cls_mask)[0]
        correct = cls_positions[preds[cls_positions] == cls]
        wrong = cls_positions[preds[cls_positions] != cls]

        picked_correct = (
            rng.choice(correct, size=min(LOCAL_CORRECT_PER_CLASS, len(correct)), replace=False).tolist()
            if len(correct)
            else []
        )
        picked_wrong = (
            rng.choice(wrong, size=min(LOCAL_WRONG_PER_CLASS, len(wrong)), replace=False).tolist()
            if len(wrong)
            else []
        )
        log["per_class"][cls] = {
            "correct_positions": [int(p) for p in picked_correct],
            "misclassified_positions": [int(p) for p in picked_wrong],
            "correct_instance_ids": [int(test_index[p]) for p in picked_correct],
            "misclassified_instance_ids": [int(test_index[p]) for p in picked_wrong],
        }
        selected.extend(picked_correct)
        selected.extend(picked_wrong)

    return sorted(set(selected)), log


# --- EXPLAIN GLOBAL ---
@checkpointed("explain-global")
def explain_global(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ctx = _load_data_context(params.get("unscaled_path"))
    sample_size = int(params.get("sample_size", GLOBAL_SAMPLE_SIZE))

    if shap is None:
        raise RuntimeError("SHAP is required for explain_global.")

    sample_idx = _stratified_sample_indices(
        ctx["y_test"],
        n=sample_size,
        per_class_target=GLOBAL_PER_CLASS_TARGET,
    )
    X_sample = ctx["X_test"].iloc[sample_idx]
    feature_names = ctx["feature_names"]
    test_index = ctx["test_index"]

    GLOBAL_XAI_DIR.mkdir(parents=True, exist_ok=True)
    subsample_path = GLOBAL_XAI_DIR / "shap_subsample_indices.json"
    y_sample = ctx["y_test"][sample_idx]
    class_counts = {str(cls): int((y_sample == cls).sum()) for cls in sorted(np.unique(y_sample))}
    write_json(
        {
            "sample_size": int(len(sample_idx)),
            "positions": [int(i) for i in sample_idx.tolist()],
            "instance_ids": [int(test_index[i]) for i in sample_idx],
            "class_counts": class_counts,
            "per_class_target": GLOBAL_PER_CLASS_TARGET,
            "random_seed": RANDOM_SEED,
        },
        subsample_path,
    )

    output_paths: list[str] = [str(subsample_path)]
    per_model: dict[str, Any] = {}
    global_plot_paths: list[Path] = []

    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            LOGGER.warning("Skipping explain_global for missing model: %s", model_name)
            continue

        model = read_joblib(model_path)
        out_dir = XAI_REPORTS_DIR / model_name / "global"
        out_dir.mkdir(parents=True, exist_ok=True)

        explainer = _shap_explainer_for_model(model, model_name)
        shap_started = time.perf_counter()
        shap_values = explainer.shap_values(np.asarray(X_sample, dtype=np.float64))
        record_explainability(
            "explain-global",
            model_name,
            shap_seconds=time.perf_counter() - shap_started,
        )
        mean_abs = _global_mean_abs_shap(shap_values)

        np.save(out_dir / "shap_values.npy", shap_values, allow_pickle=True)
        ev_path = out_dir / "shap_expected_value.json"
        write_json(
            {
                "expected_value": _expected_value_json(explainer, len(model.classes_)),
                "feature_names": feature_names,
                "sample_size": int(len(X_sample)),
                "sample_positions": [int(i) for i in sample_idx.tolist()],
            },
            ev_path,
        )

        plot_paths = _save_shap_global_plots(
            model_name, shap_values, X_sample, feature_names, mean_abs, GLOBAL_XAI_DIR
        )
        global_plot_paths.extend(plot_paths)
        output_paths.extend(
            [
                str(out_dir / "shap_values.npy"),
                str(ev_path),
                *[str(p) for p in plot_paths],
            ]
        )
        per_model[model_name] = {"sample_size": len(X_sample), "n_plots": len(plot_paths)}
        LOGGER.info("Global SHAP complete for %s (%d samples).", model_name, len(X_sample))

    if not per_model:
        raise RuntimeError("No trained models available for explain_global.")

    print("\nGlobal SHAP plot outputs:")
    for plot_path in sorted(global_plot_paths):
        size_kb = plot_path.stat().st_size / 1024
        print(f"  {plot_path}  ({size_kb:.1f} KB)")

    return {"output_paths": output_paths, "models": per_model, "sample_size": len(X_sample)}


# --- EXPLAIN LOCAL ---
@checkpointed("explain-local")
def explain_local(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ctx = _load_data_context(params.get("unscaled_path"))
    LOCAL_XAI_DIR.mkdir(parents=True, exist_ok=True)

    if shap is None:
        raise RuntimeError("SHAP is required for explain_local.")
    if LimeTabularExplainer is None:
        raise RuntimeError("LIME is required for explain_local.")

    override_instances: list[dict[str, Any]] | None = params.get("instances")
    if override_instances is not None:
        instances = override_instances
        selection_log = {"instances": instances, "reference_model": LOCAL_REFERENCE_MODEL}
    elif params.get("instance_ids"):
        id_to_pos = {int(idx): pos for pos, idx in enumerate(ctx["test_index"])}
        xgb_model = _load_model(LOCAL_REFERENCE_MODEL)
        instances = []
        for instance_id in params["instance_ids"]:
            pos = id_to_pos.get(int(instance_id))
            if pos is None:
                continue
            true_label = str(ctx["y_test"][pos])
            row_df = ctx["X_test"].iloc[[pos]]
            pred_label = str(xgb_model.predict(row_df)[0])
            instances.append(
                {
                    "position": int(pos),
                    "instance_id": int(instance_id),
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "kind": "correct" if true_label == pred_label else "wrong",
                }
            )
        selection_log = {"instances": instances, "reference_model": LOCAL_REFERENCE_MODEL}
    else:
        instances, selection_log = _select_local_instances_xgb(ctx)

    if not instances:
        raise RuntimeError("No local instances selected — train/evaluate XGBoost first.")

    selected_path = LOCAL_XAI_DIR / "selected_instances.json"
    write_json(
        {
            "generated": datetime.now(timezone.utc).isoformat(),
            "reference_model": LOCAL_REFERENCE_MODEL,
            "selection": selection_log,
            "instances": instances,
        },
        selected_path,
    )

    xgb_model = _load_model(LOCAL_REFERENCE_MODEL)
    dt_model = _load_model("decision_tree")
    xgb_explainer = _shap_explainer_for_model(xgb_model, LOCAL_REFERENCE_MODEL)
    dt_explainer = _shap_explainer_for_model(dt_model, "decision_tree")
    xgb_class_to_idx = {cls: idx for idx, cls in enumerate(xgb_model.classes_)}
    dt_class_to_idx = {cls: idx for idx, cls in enumerate(dt_model.classes_)}

    class_names = [str(c) for c in THREAT_CLASSES]
    lime_explainer = _lime_explainer_for_data(ctx["X_train"], class_names)
    feature_names = ctx["feature_names"]

    output_paths: list[str] = [str(selected_path)]
    shap_png_count = 0
    lime_png_count = 0
    lime_stability_count = 0
    shap_seconds = 0.0
    lime_seconds = 0.0

    for instance in instances:
        pos = int(instance["position"])
        instance_id = int(instance["instance_id"])
        true_label = str(instance["true_label"])
        pred_label = str(instance["predicted_label"])
        kind = str(instance["kind"])
        stem = _local_instance_stem(instance)
        safe_true = _safe_filename(true_label)

        row_df = ctx["X_test"].iloc[[pos]]
        row = np.asarray(row_df, dtype=np.float64).ravel()
        xgb_pred_idx = xgb_class_to_idx.get(pred_label, 0)
        dt_pred_label = str(dt_model.predict(row_df)[0])
        dt_pred_idx = dt_class_to_idx.get(dt_pred_label, 0)

        xgb_title = (
            f"XGBoost SHAP Waterfall — True: {true_label}, "
            f"Predicted: {pred_label}, Instance #{instance_id}"
        )
        dt_title = (
            f"Decision Tree SHAP Waterfall — True: {true_label}, "
            f"Predicted: {dt_pred_label}, Instance #{instance_id}"
        )
        lime_title = (
            f"XGBoost LIME Explanation — True: {true_label}, "
            f"Predicted: {pred_label}, Instance #{instance_id}"
        )

        shap_started = time.perf_counter()
        xgb_shap_values = xgb_explainer.shap_values(row_df)
        dt_shap_values = dt_explainer.shap_values(row_df)
        shap_seconds += time.perf_counter() - shap_started

        xgb_wf_path = LOCAL_XAI_DIR / f"xgb_shap_waterfall_{stem}.png"
        dt_wf_path = LOCAL_XAI_DIR / f"dt_shap_waterfall_{stem}.png"
        lime_bar_path = LOCAL_XAI_DIR / f"xgb_lime_bar_{stem}.png"
        lime_stability_path = LOCAL_XAI_DIR / f"lime_stability_{safe_true}_{instance_id}_runs.json"

        xgb_saved = _save_local_shap_waterfall(
            xgb_explainer,
            xgb_shap_values,
            row,
            feature_names,
            xgb_pred_idx,
            xgb_wf_path,
            xgb_title,
        )
        if xgb_saved is not None:
            shap_png_count += 1
            output_paths.append(str(xgb_saved))

        dt_saved = _save_local_shap_waterfall(
            dt_explainer,
            dt_shap_values,
            row,
            feature_names,
            dt_pred_idx,
            dt_wf_path,
            dt_title,
        )
        if dt_saved is not None:
            shap_png_count += 1
            output_paths.append(str(dt_saved))

        lime_started = time.perf_counter()
        lime_saved, _ = _save_lime_bar_chart(
            lime_explainer,
            xgb_model,
            row,
            feature_names,
            lime_bar_path,
            lime_title,
        )
        stability_saved = _save_lime_stability_runs(
            lime_explainer,
            xgb_model,
            row,
            feature_names,
            lime_stability_path,
            n_runs=LOCAL_LIME_STABILITY_RUNS,
        )
        lime_seconds += time.perf_counter() - lime_started

        if lime_saved is not None:
            lime_png_count += 1
            output_paths.append(str(lime_saved))
        lime_stability_count += 1
        output_paths.append(str(stability_saved))

    record_explainability(
        "explain-local",
        LOCAL_REFERENCE_MODEL,
        shap_seconds=shap_seconds,
        lime_seconds=lime_seconds,
    )

    print("\nLocal XAI summary:")
    print(f"  Instances explained: {len(instances)}")
    print(f"  SHAP waterfall PNGs saved: {shap_png_count}")
    print(f"  LIME bar PNGs saved: {lime_png_count}")
    print(f"  LIME stability JSON files saved: {lime_stability_count}")

    return {
        "output_paths": output_paths,
        "instances": instances,
        "n_instances": len(instances),
        "n_shap_pngs": shap_png_count,
        "n_lime_pngs": lime_png_count,
        "n_lime_stability_json": lime_stability_count,
    }


def _parse_lime_stability_path(path: Path) -> tuple[str, int] | None:
    name = path.stem
    if not name.startswith("lime_stability_") or not name.endswith("_runs"):
        return None
    core = name[len("lime_stability_") : -len("_runs")]
    for cls in sorted(THREAT_CLASSES, key=len, reverse=True):
        prefix = f"{cls}_"
        if core.startswith(prefix):
            idx_str = core[len(prefix) :]
            if idx_str.isdigit():
                return cls, int(idx_str)
    return None


def _pairwise_stability_scores(
    runs: list[list[dict[str, Any]]],
) -> tuple[float, float]:
    jaccards: list[float] = []
    spearmans: list[float] = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            feats_i = {item["feature"] for item in runs[i]}
            feats_j = {item["feature"] for item in runs[j]}
            jaccards.append(_jaccard_sets(feats_i, feats_j))
            dict_i = {item["feature"]: float(item["weight"]) for item in runs[i]}
            dict_j = {item["feature"]: float(item["weight"]) for item in runs[j]}
            spearmans.append(_spearman_rank_corr(dict_i, dict_j))
    jaccard_mean = float(np.mean(jaccards)) if jaccards else float("nan")
    spearman_mean = float(np.nanmean(spearmans)) if spearmans else float("nan")
    return jaccard_mean, spearman_mean


def _shap_top10_for_instance(
    explainer: Any,
    model: Any,
    row_df: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    pred_label = str(model.predict(row_df)[0])
    class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
    class_index = class_to_idx.get(pred_label, 0)
    shap_values = explainer.shap_values(np.asarray(row_df, dtype=np.float64))
    values_row, _ = _local_shap_row(explainer, shap_values, class_index)
    order = np.argsort(-np.abs(values_row))[:10]
    return [
        {"feature": feature_names[i], "weight": float(values_row[i])}
        for i in order
    ]


def _compute_fidelity_table(ctx: dict[str, Any]) -> pd.DataFrame:
    if shap is None:
        raise RuntimeError("SHAP is required for fidelity measurement.")

    feature_names = ctx["feature_names"]
    X_train = ctx["X_train"]
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]
    train_means = X_train.mean(numeric_only=True)

    sample_idx = _stratified_sample_indices(ctx["y_test"], n=FIDELITY_SAMPLE_SIZE)
    X_sample = X_test.iloc[sample_idx]

    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        if not (MODELS_DIR / f"{model_name}.joblib").exists():
            continue

        model = _load_model(model_name)
        explainer = _shap_explainer_for_model(model, model_name)
        shap_values = explainer.shap_values(np.asarray(X_sample, dtype=np.float64))
        mean_abs = _global_mean_abs_shap(shap_values)
        ranked_idx = np.argsort(-mean_abs)

        baseline_preds = model.predict(X_test)
        baseline_accuracy = float(accuracy_score(y_test, baseline_preds))
        baseline_f1 = float(f1_score(y_test, baseline_preds, average="macro", zero_division=0))

        for k in FIDELITY_K_VALUES:
            X_ablated = X_test.copy()
            for feat_idx in ranked_idx[:k]:
                col = feature_names[int(feat_idx)]
                X_ablated[col] = train_means[col]
            ablated_preds = model.predict(X_ablated)
            ablated_accuracy = float(accuracy_score(y_test, ablated_preds))
            ablated_f1 = float(f1_score(y_test, ablated_preds, average="macro", zero_division=0))
            rows.append(
                {
                    "model": model_name,
                    "k_ablated": k,
                    "baseline_accuracy": baseline_accuracy,
                    "ablated_accuracy": ablated_accuracy,
                    "accuracy_drop": baseline_accuracy - ablated_accuracy,
                    "baseline_f1": baseline_f1,
                    "ablated_f1": ablated_f1,
                    "f1_drop": baseline_f1 - ablated_f1,
                }
            )

    if not rows:
        raise RuntimeError("No models available for fidelity measurement.")
    return pd.DataFrame(rows)


def _save_fidelity_plot(fidelity_df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in MODEL_NAMES:
        model_df = fidelity_df[fidelity_df["model"] == model_name]
        if model_df.empty:
            continue
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(
            model_df["k_ablated"],
            model_df["f1_drop"],
            marker="o",
            label=label,
        )
    ax.set_xlabel("k (top features ablated)")
    ax.set_ylabel("Macro F1 drop")
    ax.set_title("Fidelity: F1 Drop After Ablating Top-k SHAP Features")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    return out_path


def _compute_stability_table(ctx: dict[str, Any]) -> pd.DataFrame:
    if shap is None:
        raise RuntimeError("SHAP is required for stability measurement.")

    xgb_model = _load_model(LOCAL_REFERENCE_MODEL)
    xgb_explainer = _shap_explainer_for_model(xgb_model, LOCAL_REFERENCE_MODEL)
    feature_names = ctx["feature_names"]
    id_to_pos = {int(idx): pos for pos, idx in enumerate(ctx["test_index"])}

    stability_files = sorted(LOCAL_XAI_DIR.glob("lime_stability_*_runs.json"))
    if not stability_files:
        raise FileNotFoundError(
            f"No LIME stability JSON files found in {LOCAL_XAI_DIR}. Run explain-local first."
        )

    rows: list[dict[str, Any]] = []
    for path in stability_files:
        parsed = _parse_lime_stability_path(path)
        if parsed is None:
            continue
        true_class, instance_idx = parsed
        payload = read_json(path)
        lime_runs = [run["top10"] for run in payload.get("runs", [])]
        lime_jaccard, lime_spearman = _pairwise_stability_scores(lime_runs)

        pos = id_to_pos.get(instance_idx)
        if pos is None:
            shap_jaccard = float("nan")
            shap_spearman = float("nan")
        else:
            row_df = ctx["X_test"].iloc[[pos]]
            shap_runs = [
                _shap_top10_for_instance(xgb_explainer, xgb_model, row_df, feature_names)
                for _ in range(LOCAL_LIME_STABILITY_RUNS)
            ]
            shap_jaccard, shap_spearman = _pairwise_stability_scores(shap_runs)

        rows.append(
            {
                "instance_idx": instance_idx,
                "true_class": true_class,
                "lime_jaccard_mean": lime_jaccard,
                "lime_spearman_mean": lime_spearman,
                "shap_jaccard_mean": shap_jaccard,
                "shap_spearman_mean": shap_spearman,
            }
        )

    if not rows:
        raise RuntimeError("No stability rows computed from LIME stability JSON files.")
    return pd.DataFrame(rows)


def _save_stability_plot(stability_df: pd.DataFrame, out_path: Path) -> Path:
    lime_jaccard = float(stability_df["lime_jaccard_mean"].mean())
    lime_spearman = float(stability_df["lime_spearman_mean"].mean())
    shap_jaccard = float(stability_df["shap_jaccard_mean"].mean())
    shap_spearman = float(stability_df["shap_spearman_mean"].mean())

    metrics = ["Jaccard", "Spearman"]
    lime_means = [lime_jaccard, lime_spearman]
    shap_means = [shap_jaccard, shap_spearman]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, lime_means, width, label="LIME", color="steelblue")
    ax.bar(x + width / 2, shap_means, width, label="SHAP", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Mean stability score")
    ax.set_title("Stability: LIME vs SHAP Explanation Consistency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    return out_path


def _compute_comprehensibility_table(ctx: dict[str, Any]) -> pd.DataFrame:
    if shap is None:
        raise RuntimeError("SHAP is required for comprehensibility measurement.")

    feature_names = ctx["feature_names"]
    y_test = ctx["y_test"]
    sample_idx = _stratified_sample_indices(ctx["y_test"], n=COMPREHENSIBILITY_SAMPLE_SIZE)
    X_sample = ctx["X_test"].iloc[sample_idx]
    y_sample = y_test[sample_idx]

    rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        if not (MODELS_DIR / f"{model_name}.joblib").exists():
            continue

        model = _load_model(model_name)
        explainer = _shap_explainer_for_model(model, model_name)
        shap_values = explainer.shap_values(np.asarray(X_sample, dtype=np.float64))
        per_class_shap = _shap_values_as_class_list(shap_values)
        class_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}

        for threat_class in IOC_THREAT_CLASSES:
            if threat_class not in class_to_idx:
                continue
            class_idx = class_to_idx[threat_class]
            class_mask = y_sample == threat_class
            if not np.any(class_mask):
                rows.append(
                    {
                        "model": model_name,
                        "threat_class": threat_class,
                        "ioc_alignment_score": float("nan"),
                        "top10_features": "",
                    }
                )
                continue

            class_shap = per_class_shap[class_idx][class_mask]
            mean_abs = np.abs(class_shap).mean(axis=0)
            top10_idx = np.argsort(-mean_abs)[:10]
            top10_features = [feature_names[i] for i in top10_idx]
            ioc_set = set(IOC_REFERENCE[threat_class])
            hits = sum(1 for feat in top10_features if feat in ioc_set)
            rows.append(
                {
                    "model": model_name,
                    "threat_class": threat_class,
                    "ioc_alignment_score": hits / 10.0,
                    "top10_features": ", ".join(top10_features),
                }
            )

    if not rows:
        raise RuntimeError("No comprehensibility rows computed.")
    return pd.DataFrame(rows)


def _save_comprehensibility_heatmap(comp_df: pd.DataFrame, out_path: Path) -> Path:
    pivot = comp_df.pivot(index="threat_class", columns="model", values="ioc_alignment_score")
    pivot = pivot.reindex(index=IOC_THREAT_CLASSES, columns=MODEL_NAMES)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "IoC alignment score"},
        ax=ax,
    )
    ax.set_title("Comprehensibility: IoC Alignment of Top SHAP Features")
    ax.set_xlabel("Model")
    ax.set_ylabel("Threat class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()
    return out_path


def _dataframe_preview_markdown(df: pd.DataFrame, n: int = 5) -> list[str]:
    preview = df.head(n)
    headers = list(preview.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in preview.iterrows():
        cells = [str(row[col]).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _write_quality_summary(
    fidelity_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    comprehensibility_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> Path:
    best_fidelity = (
        fidelity_df.groupby("model")["f1_drop"].max().sort_values(ascending=False).index[0]
    )
    lime_jaccard_mean = float(stability_df["lime_jaccard_mean"].mean())
    shap_jaccard_mean = float(stability_df["shap_jaccard_mean"].mean())
    best_comp = (
        comprehensibility_df.groupby("model")["ioc_alignment_score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    lines = [
        "# XAI Quality Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Key findings",
        "",
        f"- **Fidelity:** `{best_fidelity}` shows the largest macro-F1 drop when top SHAP features are ablated, indicating its explanations track the most decision-critical features.",
        f"- **Stability:** Mean LIME Jaccard = {lime_jaccard_mean:.3f} vs SHAP Jaccard = {shap_jaccard_mean:.3f}; tree SHAP explanations are deterministic and serve as the stability reference.",
        f"- **Comprehensibility:** `{best_comp}` achieves the highest average IoC alignment between class-specific top-10 SHAP features and cybersecurity indicator references.",
        "",
        "## Fidelity table preview",
        "",
        *_dataframe_preview_markdown(fidelity_df),
        "",
        "## Stability table preview",
        "",
        *_dataframe_preview_markdown(stability_df),
        "",
        "## Comprehensibility table preview",
        "",
        *_dataframe_preview_markdown(comprehensibility_df),
        "",
        "## Generated plots",
        "",
        f"- Fidelity plot: `{plot_paths['fidelity']}`",
        f"- Stability plot: `{plot_paths['stability']}`",
        f"- Comprehensibility heatmap: `{plot_paths['comprehensibility']}`",
        "",
    ]
    out_path = QUALITY_DIR / "xai_quality_summary.md"
    return _write_markdown(out_path, lines)


# --- EXPLANATION QUALITY ---
@checkpointed("explanation-quality")
def explanation_quality(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ctx = _load_data_context(params.get("unscaled_path"))
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    fidelity_df = _compute_fidelity_table(ctx)
    fidelity_path = QUALITY_DIR / "fidelity_table.csv"
    fidelity_df.to_csv(fidelity_path, index=False)
    fidelity_plot_path = _save_fidelity_plot(fidelity_df, QUALITY_DIR / "fidelity_plot.png")

    stability_df = _compute_stability_table(ctx)
    stability_path = QUALITY_DIR / "stability_table.csv"
    stability_df.to_csv(stability_path, index=False)
    stability_plot_path = _save_stability_plot(stability_df, QUALITY_DIR / "stability_plot.png")

    comprehensibility_df = _compute_comprehensibility_table(ctx)
    comprehensibility_path = QUALITY_DIR / "comprehensibility_table.csv"
    comprehensibility_df.to_csv(comprehensibility_path, index=False)
    comprehensibility_plot_path = _save_comprehensibility_heatmap(
        comprehensibility_df,
        QUALITY_DIR / "comprehensibility_heatmap.png",
    )

    plot_paths = {
        "fidelity": fidelity_plot_path,
        "stability": stability_plot_path,
        "comprehensibility": comprehensibility_plot_path,
    }
    summary_path = _write_quality_summary(
        fidelity_df,
        stability_df,
        comprehensibility_df,
        plot_paths,
    )

    print("\nExplanation quality outputs:")
    for path in [
        fidelity_path,
        fidelity_plot_path,
        stability_path,
        stability_plot_path,
        comprehensibility_path,
        comprehensibility_plot_path,
        summary_path,
    ]:
        size_kb = path.stat().st_size / 1024
        print(f"  {path}  ({size_kb:.1f} KB)")

    output_paths = [str(p) for p in [
        fidelity_path,
        fidelity_plot_path,
        stability_path,
        stability_plot_path,
        comprehensibility_path,
        comprehensibility_plot_path,
        summary_path,
    ]]

    return {
        "output_paths": output_paths,
        "fidelity_rows": len(fidelity_df),
        "stability_rows": len(stability_df),
        "comprehensibility_rows": len(comprehensibility_df),
    }


# --- EXPLAIN ONE (Streamlit / interactive) ---
def explain_one(
    row_or_flow_dict: pd.DataFrame | pd.Series | dict[str, Any],
    model_name: str | None = None,
) -> dict[str, Any]:
    """Explain a single flow for the UI: prediction, SHAP, LIME, and a plain-language narrative."""
    resolved_name = model_name or _best_model_name()
    model_path = MODELS_DIR / f"{resolved_name}.joblib"
    if not model_path.exists():
        model_path = BEST_MODEL_PATH
    _load_inference_context(model_path)

    if isinstance(row_or_flow_dict, dict):
        row_df = pd.DataFrame([row_or_flow_dict])
    elif isinstance(row_or_flow_dict, pd.Series):
        row_df = row_or_flow_dict.to_frame().T
    else:
        row_df = row_or_flow_dict.copy()

    raw = _prepare_raw_dataframe(row_df)
    X = _encode_features(raw)
    model = read_joblib(model_path)
    feature_names = list(X.columns)

    pred = str(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    classes = [str(c) for c in model.classes_]
    probabilities = {cls: float(proba[i]) for i, cls in enumerate(classes)}

    explainer = _shap_explainer_for_model(model, resolved_name)
    shap_values = explainer.shap_values(np.asarray(X, dtype=np.float64))
    if isinstance(shap_values, list):
        class_idx = classes.index(pred) if pred in classes else 0
        shap_row = np.asarray(shap_values[class_idx][0], dtype=np.float64)
    else:
        shap_row = np.asarray(shap_values[0], dtype=np.float64)

    shap_items = [
        {
            "feature": feature_names[i],
            "contribution": float(shap_row[i]),
            "direction": "positive" if shap_row[i] > 0 else "negative",
            "magnitude": float(abs(shap_row[i])),
        }
        for i in range(len(feature_names))
    ]
    shap_items.sort(key=lambda r: r["magnitude"], reverse=True)
    shap_top15 = shap_items[:15]

    lime_top10: list[dict[str, Any]] = []
    if LimeTabularExplainer is not None:
        splits = _load_splits()
        ctx = _load_data_context()
        lime_explainer = _lime_explainer_for_data(ctx["X_train"], classes)
        row = np.asarray(X, dtype=np.float64).ravel()
        exp = lime_explainer.explain_instance(
            data_row=row,
            predict_fn=_predict_proba_for_lime(model, feature_names),
            num_features=10,
            top_labels=1,
        )
        label = exp.top_labels[0]
        lime_top10 = [
            {"feature": f, "weight": float(w)} for f, w in exp.as_list(label=label)[:10]
        ]

    ioc_map = _load_ioc_map()
    narrative = _narrative_from_shap(pred, shap_top15, ioc_map)

    return {
        "model": resolved_name,
        "predicted_label": pred,
        "probabilities": probabilities,
        "confidence": float(max(probabilities.values())),
        "shap_top15": shap_top15,
        "lime_top10": lime_top10,
        "narrative": narrative,
    }


# --- pipeline orchestration ---
def explain(params: dict[str, Any] | None = None, *, force: bool = False) -> dict[str, Any]:
    """Run all XAI checkpoint steps in sequence (global → local → quality)."""
    params = dict(params or {})
    params.pop("force", None)
    results = {
        "explain_global": explain_global(params, force=force),
        "explain_local": explain_local(params, force=force),
        "explanation_quality": explanation_quality(params, force=force),
    }
    output_paths: list[str] = []
    for step_result in results.values():
        output_paths.extend(step_result.get("output_paths", []))
    return {"output_paths": output_paths, "steps": results}


def run_default() -> dict[str, Any]:
    return explain({})


if __name__ == "__main__":
    result = run_default()
    loaded = result.get("loaded_from_checkpoint", False)
    state = "checkpoint" if loaded else "executed"
    print(f"src.xai: {state}.")
