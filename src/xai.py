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
from sklearn.metrics import f1_score
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
from src.prep import TABULAR_UNSCALED_PARQUET, TRAIN_BALANCED_PARQUET
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
QUALITY_DIR = XAI_REPORTS_DIR / "quality"
IOC_FEATURE_MAP_PATH = ARTIFACTS_DIR / "ioc_feature_map.json"

GLOBAL_SAMPLE_SIZE = 5000
LOCAL_CORRECT_PER_CLASS = 3
LOCAL_WRONG_PER_CLASS = 3
STABILITY_INSTANCES = 50
STABILITY_LIME_RUNS = 5
ABLATION_K_VALUES = [1, 3, 5, 10]
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


def _stratified_sample_indices(
    y: np.ndarray,
    n: int,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    labels = pd.Series(y)
    groups = labels.groupby(labels)
    weights = groups.transform("count") / len(labels)
    target_counts = (weights * n).round().astype(int).clip(lower=1)
    target_counts = target_counts.groupby(labels).transform("max")
    chosen: list[int] = []
    rng = np.random.default_rng(random_state)
    for cls in sorted(labels.unique()):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        n_cls = min(int(target_counts[labels == cls].iloc[0]), len(cls_idx))
        n_cls = min(n_cls, max(1, int(round(n * len(cls_idx) / len(y)))))
        picked = rng.choice(cls_idx, size=n_cls, replace=False)
        chosen.extend(picked.tolist())
    if len(chosen) > n:
        chosen = rng.choice(np.array(chosen), size=n, replace=False).tolist()
    elif len(chosen) < n:
        remaining = list(set(range(len(y))) - set(chosen))
        extra = min(n - len(chosen), len(remaining))
        if extra > 0:
            chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
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

    X_np = np.asarray(X_sample, dtype=np.float64)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_np, feature_names=feature_names, show=False)
    beeswarm = out_dir / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(beeswarm)

    if isinstance(shap_values, list):
        for i, sv in enumerate(shap_values):
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, X_np, feature_names=feature_names, show=False)
            class_path = out_dir / f"shap_beeswarm_class_{i}.png"
            plt.tight_layout()
            plt.savefig(class_path, dpi=120, bbox_inches="tight")
            plt.close()
            paths.append(class_path)

    top_idx = np.argsort(-mean_abs)[: min(20, len(mean_abs))]
    bar_df = pd.DataFrame(
        {"feature": [feature_names[i] for i in top_idx], "mean_abs_shap": mean_abs[top_idx]}
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=bar_df, y="feature", x="mean_abs_shap", ax=ax, color="steelblue")
    ax.set_title(f"Global mean |SHAP| — {model_name}")
    ax.set_xlabel("Mean |SHAP|")
    bar_path = out_dir / "shap_bar_mean_abs.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=120)
    plt.close()
    paths.append(bar_path)

    top5_idx = top_idx[:5]
    for idx in top5_idx:
        feat = feature_names[idx]
        plt.figure(figsize=(7, 5))
        if isinstance(shap_values, list):
            shap.dependence_plot(
                idx,
                np.asarray(shap_values[0]),
                X_np,
                feature_names=feature_names,
                show=False,
            )
        else:
            shap.dependence_plot(idx, shap_values, X_np, feature_names=feature_names, show=False)
        dep_path = out_dir / f"shap_dependence_{_safe_filename(feat)}.png"
        plt.tight_layout()
        plt.savefig(dep_path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(dep_path)

    return paths


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
    values_row: np.ndarray

    if isinstance(shap_values, list):
        values_row = np.asarray(shap_values[class_index][0], dtype=np.float64)
        base = float(np.asarray(explainer.expected_value).ravel()[class_index])
    else:
        values_row = np.asarray(shap_values[0], dtype=np.float64)
        base = float(explainer.expected_value)

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


def _retrain_lightweight(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    random_seed: int = RANDOM_SEED,
) -> Any:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X = X_train[feature_cols]
    if model_name == "decision_tree":
        est = DecisionTreeClassifier(max_depth=8, random_state=random_seed)
    elif model_name == "random_forest":
        est = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=random_seed, n_jobs=-1)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        est = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_seed,
            eval_metric="mlogloss",
            tree_method="hist",
            objective="multi:softprob",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    est.fit(X, y_train)
    return est


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

    sample_idx = _stratified_sample_indices(ctx["y_test"], n=sample_size)
    X_sample = ctx["X_test"].iloc[sample_idx]
    feature_names = ctx["feature_names"]

    output_paths: list[str] = []
    per_model: dict[str, Any] = {}

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
            model_name, shap_values, X_sample, feature_names, mean_abs, out_dir
        )
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

    return {"output_paths": output_paths, "models": per_model, "sample_size": len(X_sample)}


# --- EXPLAIN LOCAL ---
@checkpointed("explain-local")
def explain_local(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ctx = _load_data_context(params.get("unscaled_path"))
    instance_ids: list[int] | None = params.get("instance_ids")

    if instance_ids is None:
        instance_ids, selection_log = _select_local_instance_ids(ctx)
    else:
        selection_log = {"instance_ids": instance_ids}

    log_path = XAI_REPORTS_DIR / "local_instance_selection.json"
    write_json(
        {
            "generated": datetime.now(timezone.utc).isoformat(),
            "selection": selection_log,
            "instance_ids": instance_ids,
        },
        log_path,
    )

    id_to_pos = {int(idx): pos for pos, idx in enumerate(ctx["test_index"])}
    positions = [id_to_pos[i] for i in instance_ids if i in id_to_pos]

    class_names = [str(c) for c in THREAT_CLASSES]
    lime_explainer = _lime_explainer_for_data(ctx["X_train"], class_names)

    output_paths: list[str] = [str(log_path)]
    explained = 0

    for model_name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            continue
        model = read_joblib(model_path)
        explainer = _shap_explainer_for_model(model, model_name)
        classes = list(model.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        model_shap_seconds = 0.0
        model_lime_seconds = 0.0

        for pos in positions:
            instance_id = int(ctx["test_index"][pos])
            true_label = str(ctx["y_test"][pos])
            row_df = ctx["X_test"].iloc[[pos]]
            row = np.asarray(row_df, dtype=np.float64).ravel()
            pred_label = str(model.predict(row_df)[0])
            pred_idx = class_to_idx.get(pred_label, 0)

            inst_dir = (
                XAI_REPORTS_DIR
                / model_name
                / "local"
                / f"{instance_id}_{_safe_filename(true_label)}_{_safe_filename(pred_label)}"
            )

            shap_started = time.perf_counter()
            shap_values = explainer.shap_values(row_df)
            model_shap_seconds += time.perf_counter() - shap_started
            if isinstance(shap_values, list):
                sv_local = [np.asarray(sv) for sv in shap_values]
            else:
                sv_local = np.asarray(shap_values)

            shap_paths, _ = _save_local_shap_plots(
                explainer,
                sv_local if isinstance(sv_local, list) else sv_local.reshape(1, -1),
                row,
                ctx["feature_names"],
                pred_idx,
                inst_dir,
                instance_id,
            )
            lime_started = time.perf_counter()
            lime_paths, _ = _save_lime_local(
                lime_explainer, model, row, inst_dir, ctx["feature_names"]
            )
            model_lime_seconds += time.perf_counter() - lime_started
            output_paths.extend(str(p) for p in shap_paths + lime_paths)
            explained += 1

        record_explainability(
            "explain-local",
            model_name,
            shap_seconds=model_shap_seconds,
            lime_seconds=model_lime_seconds,
        )

    if explained == 0:
        raise RuntimeError("No local explanations generated — train models first.")

    return {
        "output_paths": output_paths,
        "instance_ids": instance_ids,
        "n_explanations": explained,
    }


# --- EXPLANATION QUALITY ---
def _fidelity_for_model(
    model_name: str,
    ctx: dict[str, Any],
    global_dir: Path,
) -> pd.DataFrame:
    shap_path = global_dir / "shap_values.npy"
    if not shap_path.exists():
        raise FileNotFoundError(f"Run explain_global first: missing {shap_path}")

    shap_values = np.load(shap_path, allow_pickle=True)
    mean_abs = _global_mean_abs_shap(shap_values)
    ranked_idx = np.argsort(-mean_abs)
    feature_names = ctx["feature_names"]

    train_df = read_parquet(TRAIN_BALANCED_PARQUET)
    target_col = ctx["target_col"]
    y_train = train_df[target_col].astype(str).to_numpy()
    X_train_full = train_df[ctx["feature_names"]]
    X_test = ctx["X_test"]
    y_test = ctx["y_test"]

    baseline_model = _retrain_lightweight(model_name, X_train_full, y_train, ctx["feature_names"])
    baseline_f1 = float(f1_score(y_test, baseline_model.predict(X_test), average="macro", zero_division=0))

    rows: list[dict[str, Any]] = []
    for k in ABLATION_K_VALUES:
        drop_idx = set(ranked_idx[:k].tolist())
        keep_cols = [feature_names[i] for i in range(len(feature_names)) if i not in drop_idx]
        ablated = _retrain_lightweight(model_name, X_train_full, y_train, keep_cols)
        ablated_f1 = float(f1_score(y_test, ablated.predict(X_test[keep_cols]), average="macro", zero_division=0))
        rows.append(
            {
                "model": model_name,
                "k_removed": k,
                "baseline_f1_macro": baseline_f1,
                "ablated_f1_macro": ablated_f1,
                "f1_drop": baseline_f1 - ablated_f1,
                "removed_features": [feature_names[i] for i in sorted(drop_idx)],
            }
        )
    return pd.DataFrame(rows)


def _stability_for_model(model_name: str, ctx: dict[str, Any]) -> pd.DataFrame:
    model = _load_model(model_name)
    class_names = [str(c) for c in model.classes_]
    lime_explainer = _lime_explainer_for_data(ctx["X_train"], class_names)

    feature_names = ctx["feature_names"]
    predict_proba = _predict_proba_for_lime(model, feature_names)

    sample_idx = _stratified_sample_indices(ctx["y_test"], n=STABILITY_INSTANCES)
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, Any]] = []

    for pos in sample_idx:
        row = np.asarray(ctx["X_test"].iloc[[pos]], dtype=np.float64).ravel()
        instance_id = int(ctx["test_index"][pos])

        base_runs: list[list[tuple[str, float]]] = []
        for run in range(STABILITY_LIME_RUNS):
            exp = lime_explainer.explain_instance(
                data_row=row,
                predict_fn=predict_proba,
                num_features=10,
                top_labels=1,
            )
            label = exp.top_labels[0]
            base_runs.append(exp.as_list(label=label)[:10])

        jaccards: list[float] = []
        spearmans: list[float] = []
        for i in range(STABILITY_LIME_RUNS):
            for j in range(i + 1, STABILITY_LIME_RUNS):
                feats_i = {f for f, _ in base_runs[i]}
                feats_j = {f for f, _ in base_runs[j]}
                jaccards.append(_jaccard_sets(feats_i, feats_j))
                dict_i = {f: w for f, w in base_runs[i]}
                dict_j = {f: w for f, w in base_runs[j]}
                spearmans.append(_spearman_rank_corr(dict_i, dict_j))

        perturbed = _perturb_row_scaled(
            row, ctx["scaler"], PERTURBATION_SIGMA, rng, feature_names=feature_names
        )
        pert_runs: list[list[tuple[str, float]]] = []
        for run in range(STABILITY_LIME_RUNS):
            exp = lime_explainer.explain_instance(
                data_row=perturbed,
                predict_fn=predict_proba,
                num_features=10,
                top_labels=1,
            )
            label = exp.top_labels[0]
            pert_runs.append(exp.as_list(label=label)[:10])

        pert_jaccards: list[float] = []
        pert_spearmans: list[float] = []
        for i in range(STABILITY_LIME_RUNS):
            for j in range(i + 1, STABILITY_LIME_RUNS):
                feats_i = {f for f, _ in pert_runs[i]}
                feats_j = {f for f, _ in pert_runs[j]}
                pert_jaccards.append(_jaccard_sets(feats_i, feats_j))
                dict_i = {f: w for f, w in pert_runs[i]}
                dict_j = {f: w for f, w in pert_runs[j]}
                pert_spearmans.append(_spearman_rank_corr(dict_i, dict_j))

        rows.append(
            {
                "model": model_name,
                "instance_id": instance_id,
                "jaccard_mean": float(np.mean(jaccards)) if jaccards else float("nan"),
                "spearman_mean": float(np.nanmean(spearmans)) if spearmans else float("nan"),
                "perturbed_jaccard_mean": float(np.mean(pert_jaccards)) if pert_jaccards else float("nan"),
                "perturbed_spearman_mean": float(np.nanmean(pert_spearmans)) if pert_spearmans else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _comprehensibility_for_model(
    model_name: str,
    ctx: dict[str, Any],
) -> pd.DataFrame:
    model = _load_model(model_name)
    sample_idx = _stratified_sample_indices(ctx["y_test"], n=min(500, len(ctx["y_test"])))
    X_sample = ctx["X_test"].iloc[sample_idx]
    explainer = _shap_explainer_for_model(model, model_name)
    shap_values = explainer.shap_values(np.asarray(X_sample, dtype=np.float64))
    stacked = _stack_shap_for_predictions(shap_values, model, X_sample)
    ioc_map = _load_ioc_map()
    feature_names = ctx["feature_names"]

    rows: list[dict[str, Any]] = []
    sparsities: list[int] = []
    ioc_fracs: list[float] = []

    for i in range(len(X_sample)):
        row = stacked[i]
        sp = _sparsity_80_mass(row)
        sparsities.append(sp)
        order = np.argsort(-np.abs(row))[:5]
        top_feats = [feature_names[j] for j in order]
        ioc_frac = _ioc_alignment_fraction(top_feats, ioc_map)
        ioc_fracs.append(ioc_frac)
        rows.append(
            {
                "model": model_name,
                "instance_position": int(sample_idx[i]),
                "sparsity_80_mass": sp,
                "ioc_top5_alignment": ioc_frac,
                "top5_features": top_feats,
            }
        )

    summary = pd.DataFrame(rows)
    summary.attrs["mean_sparsity"] = float(np.mean(sparsities))
    summary.attrs["mean_ioc_alignment"] = float(np.mean(ioc_fracs))
    return summary


def _write_comparison_md(summaries: dict[str, dict[str, float]]) -> Path:
    lines = [
        "# XAI quality comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Side-by-side comparison of explanation quality across the three tree models.",
        "",
        "| Model | Test F1 (macro) | Mean fidelity drop (K≤5) | LIME stability (Jaccard) | Perturbed stability | Sparsity (80% mass) | IoC top-5 alignment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in MODEL_NAMES:
        if name not in summaries:
            continue
        s = summaries[name]
        lines.append(
            f"| {name} | {s.get('test_f1_macro', float('nan')):.4f} | "
            f"{s.get('fidelity_drop_k5', float('nan')):.4f} | "
            f"{s.get('stability_jaccard', float('nan')):.4f} | "
            f"{s.get('stability_perturbed_jaccard', float('nan')):.4f} | "
            f"{s.get('sparsity', float('nan')):.2f} | "
            f"{s.get('ioc_alignment', float('nan')):.4f} |"
        )

    best_acc = max(summaries.items(), key=lambda kv: kv[1].get("test_f1_macro", 0.0))[0]
    best_fid = max(summaries.items(), key=lambda kv: kv[1].get("fidelity_drop_k5", 0.0))[0]
    best_stab = max(summaries.items(), key=lambda kv: kv[1].get("stability_jaccard", 0.0))[0]
    best_comp = max(
        summaries.items(),
        key=lambda kv: (kv[1].get("ioc_alignment", 0.0), -kv[1].get("sparsity", 999.0)),
    )[0]

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- **Highest test accuracy:** `{best_acc}`",
            f"- **Highest fidelity (feature-ablation drop):** `{best_fid}`",
            f"- **Most stable LIME explanations:** `{best_stab}`",
            f"- **Most analyst-recognisable (IoC alignment + sparsity):** `{best_comp}`",
            "",
            "### Dissertation conclusion",
            "",
            "For operational SOC use, prefer the model that **retains strong macro-F1 on the held-out "
            "test split** while scoring well on fidelity and stability — post-hoc explanations are only "
            "actionable when they track model behaviour and do not change arbitrarily under small input "
            "perturbations. When accuracy differences are small (<1–2 pp macro-F1), favour the model with "
            "higher IoC-aligned SHAP features and lower sparsity (fewer features explaining 80% of "
            "attribution mass), as these explanations map more directly to analyst playbooks.",
            "",
        ]
    )

    overall_scores: dict[str, float] = {}
    for name, s in summaries.items():
        f1 = s.get("test_f1_macro", 0.0)
        fid = s.get("fidelity_drop_k5", 0.0)
        stab = s.get("stability_jaccard", 0.0)
        ioc = s.get("ioc_alignment", 0.0)
        sparsity_penalty = 1.0 / max(s.get("sparsity", 1.0), 1.0)
        overall_scores[name] = 0.4 * f1 + 0.25 * fid + 0.15 * stab + 0.1 * ioc + 0.1 * sparsity_penalty

    if overall_scores:
        winner = max(overall_scores.items(), key=lambda kv: kv[1])[0]
        lines.append(
            f"**Balanced recommendation (accuracy + explainability):** `{winner}` "
            f"(composite score {overall_scores[winner]:.4f})."
        )

    out = QUALITY_DIR / "comparison.md"
    return _write_markdown(out, lines)


@checkpointed("explanation-quality")
def explanation_quality(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    ctx = _load_data_context(params.get("unscaled_path"))
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    output_paths: list[str] = []
    summaries: dict[str, dict[str, float]] = {}

    for model_name in MODEL_NAMES:
        global_dir = XAI_REPORTS_DIR / model_name / "global"
        if not (MODELS_DIR / f"{model_name}.joblib").exists():
            LOGGER.warning("Skipping explanation_quality for missing model: %s", model_name)
            continue

        fid_df = _fidelity_for_model(model_name, ctx, global_dir)
        fid_path = QUALITY_DIR / f"fidelity_{model_name}.csv"
        fid_df.to_csv(fid_path, index=False)
        output_paths.append(str(fid_path))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fid_df["k_removed"], fid_df["f1_drop"], marker="o", label=model_name)
        ax.set_xlabel("Top-K features removed")
        ax.set_ylabel("Macro-F1 drop")
        ax.set_title(f"Fidelity (feature ablation) — {model_name}")
        ax.grid(alpha=0.3)
        fid_plot = QUALITY_DIR / f"fidelity_{model_name}.png"
        plt.tight_layout()
        plt.savefig(fid_plot, dpi=120)
        plt.close()
        output_paths.append(str(fid_plot))

        stab_df = _stability_for_model(model_name, ctx)
        stab_path = QUALITY_DIR / f"stability_{model_name}.csv"
        stab_df.to_csv(stab_path, index=False)
        output_paths.append(str(stab_path))

        comp_df = _comprehensibility_for_model(model_name, ctx)
        comp_path = QUALITY_DIR / f"comprehensibility_{model_name}.csv"
        comp_df.to_csv(comp_path, index=False)
        output_paths.append(str(comp_path))

        metrics_path = REPORTS_DIR / "evaluate" / model_name / "metrics.json"
        test_f1 = float("nan")
        if metrics_path.exists():
            test_f1 = float(read_json(metrics_path)["f1"]["macro"])

        summaries[model_name] = {
            "test_f1_macro": test_f1,
            "fidelity_drop_k5": float(fid_df.loc[fid_df["k_removed"] == 5, "f1_drop"].mean()),
            "stability_jaccard": float(stab_df["jaccard_mean"].mean()),
            "stability_perturbed_jaccard": float(stab_df["perturbed_jaccard_mean"].mean()),
            "sparsity": float(comp_df["sparsity_80_mass"].mean()),
            "ioc_alignment": float(comp_df["ioc_top5_alignment"].mean()),
        }

    if not summaries:
        raise RuntimeError("No models available for explanation_quality.")

    comparison_path = _write_comparison_md(summaries)
    output_paths.append(str(comparison_path))

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in summaries:
        fid_path = QUALITY_DIR / f"fidelity_{model_name}.csv"
        if fid_path.exists():
            df = pd.read_csv(fid_path)
            ax.plot(df["k_removed"], df["f1_drop"], marker="o", label=model_name)
    ax.set_xlabel("Top-K features removed")
    ax.set_ylabel("Macro-F1 drop")
    ax.set_title("Fidelity comparison across models")
    ax.legend()
    ax.grid(alpha=0.3)
    summary_plot = QUALITY_DIR / "fidelity_summary.png"
    plt.tight_layout()
    plt.savefig(summary_plot, dpi=120)
    plt.close()
    output_paths.append(str(summary_plot))

    return {"output_paths": output_paths, "summaries": summaries}


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
