from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from src.core import (
    ARTIFACTS_DIR,
    CHECKPOINTS_DIR,
    INTERIM_DIR,
    LOGGER,
    MODEL_NAMES,
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpoint_path,
    checkpointed,
    compute_input_hash,
    read_joblib,
    read_json,
    read_parquet,
    write_joblib,
    write_json,
    write_parquet,
)
from src.prep import (
    CATEGORICAL_MODE_CARDINALITY,
    CLEANED_PARQUET,
    LOW_CARDINALITY_OHE,
    SPLITS_JSON,
    TABULAR_UNSCALED_PARQUET,
    TRAIN_BALANCED_PARQUET,
    _is_identifier_column,
    _sanitize_columns,
    _skewness,
    _treat_as_missing,
    map_labels,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


# --- paths ---
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESS_DIR = ARTIFACTS_DIR / "preprocess"
PREPROCESS_BUNDLE_PATH = PREPROCESS_DIR / "preprocess_bundle.joblib"
TRAIN_REPORTS_DIR = REPORTS_DIR / "train"
EVAL_REPORTS_DIR = REPORTS_DIR / "evaluate"
BEST_MODEL_PATH = ARTIFACTS_DIR / "models" / "best_model.joblib"
BEST_MODEL_META_PATH = ARTIFACTS_DIR / "best_model.json"
MODEL_CARD_PATH = REPORTS_DIR / "model_card.md"

# Above this row count, avoid parallel CV workers and subsample for tuning.
LARGE_TRAIN_THRESHOLD = 200_000
GRID_SEARCH_SUBSAMPLE_MAX = 100_000
MODEL_SUBSAMPLE_MAX: dict[str, int] = {
    "decision_tree": 100_000,
    "random_forest": 50_000,
    "xgboost": 100_000,
}

IDENTIFYING_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip",
    "source port",
    "destination port",
    "src port",
    "dst port",
    "timestamp",
    "stime",
    "ltime",
    "source_file",
}


class LeakageError(RuntimeError):
    """Raised when train/val/test isolation or preprocess-fit scope is violated."""


class _EncodedLabelModelWrapper(ClassifierMixin, BaseEstimator):
    """Wraps an estimator trained on integer labels; exposes string labels at predict time."""

    def __init__(self, estimator: Any, label_encoder: LabelEncoder):
        self.estimator = estimator
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X: Any) -> np.ndarray:
        raw = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(np.asarray(raw, dtype=int))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def fit(self, X: Any, y: Any) -> "_EncodedLabelModelWrapper":
        y_enc = self.label_encoder.transform(y)
        self.estimator.fit(X, y_enc)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self.estimator, name)


def tree_model_for_shap(model: Any) -> Any:
    """Return the underlying tree booster for SHAP (unwraps label-encoded XGBoost)."""
    if isinstance(model, _EncodedLabelModelWrapper):
        return model.estimator
    return model


def _fit_label_encoder(y: np.ndarray) -> LabelEncoder:
    """Fit a deterministic label encoder for estimators requiring integer classes."""
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


# --- inference cache (Streamlit / repeated predict calls) ---
class _InferenceCache:
    model: Any = None
    model_path: Path | None = None
    bundle: dict[str, Any] | None = None
    imputation_maps: dict[str, Any] | None = None
    feature_names: list[str] | None = None
    classes_: list[str] | None = None


_CACHE = _InferenceCache()


def _indices_hash(indices: list[int] | np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(sorted(int(i) for i in indices)).encode("utf-8"))
    return hasher.hexdigest()


def _split_indices_hash(splits: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    for key in ("train", "val", "test"):
        hasher.update(key.encode("utf-8"))
        hasher.update(_indices_hash(splits[key]).encode("utf-8"))
    return hasher.hexdigest()


def _train_fit_hash(splits: dict[str, Any]) -> str:
    return _indices_hash(splits["train"])


def _model_checkpoint_path(name: str) -> Path:
    return CHECKPOINTS_DIR / f"train_{name}.done"


def _write_markdown(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mark_model_done(name: str, input_hash: str, output_paths: list[Path]) -> None:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "step_name": f"train_{name}",
        "input_hash": input_hash,
        "output_paths": [str(p) for p in output_paths],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _model_checkpoint_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _model_is_done(name: str, input_hash: str) -> bool:
    done_path = _model_checkpoint_path(name)
    model_path = MODELS_DIR / f"{name}.joblib"
    if not done_path.exists() or not model_path.exists():
        return False
    try:
        payload = read_json(done_path)
    except json.JSONDecodeError:
        return False
    return payload.get("input_hash") == input_hash


def _load_splits() -> dict[str, Any]:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(f"Split indices not found: {SPLITS_JSON}")
    return read_json(SPLITS_JSON)


def _target_column(splits: dict[str, Any]) -> str:
    return str(splits.get("target_column", "threat_class"))


def _feature_matrix(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    labels: np.ndarray | None = None
    if target_col in df.columns:
        labels = df[target_col].astype(str).to_numpy()
    elif "threat_label" in df.columns:
        labels = df["threat_label"].astype(str).to_numpy()
    drop_cols = {target_col, "threat_label"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], labels


def _filter_known_labels(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    context: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep only rows whose labels are in the project taxonomy."""
    known = set(THREAT_CLASSES)
    mask = np.isin(y, list(known))
    if not mask.all():
        dropped = int((~mask).sum())
        unknown = sorted({str(v) for v in y[~mask]})
        LOGGER.warning(
            "%s: dropping %d rows with labels outside THREAT_CLASSES: %s",
            context,
            dropped,
            unknown[:10],
        )
        X = X.loc[mask].reset_index(drop=True)
        y = y[mask]
    return X, y


# --- LEAKAGE CHECK ---
def leakage_check() -> None:
    """Verify split disjointness and that preprocess artefacts were fit on train only."""
    splits = _load_splits()
    train_set = set(int(i) for i in splits["train"])
    val_set = set(int(i) for i in splits["val"])
    test_set = set(int(i) for i in splits["test"])

    overlaps = {
        "train∩val": train_set & val_set,
        "train∩test": train_set & test_set,
        "val∩test": val_set & test_set,
    }
    for name, overlap in overlaps.items():
        if overlap:
            raise LeakageError(
                f"Split leakage detected: {name} has {len(overlap)} overlapping indices "
                f"(examples: {sorted(overlap)[:5]})."
            )

    fit_input_hash = _train_fit_hash(splits)
    split_hash = _split_indices_hash(splits)

    preprocess_done = checkpoint_path("preprocess")
    if not preprocess_done.exists():
        raise LeakageError("Preprocess checkpoint missing. Run preprocess() before model steps.")

    preprocess_payload = read_json(preprocess_done)
    preprocess_params = {
        "cleaned_path": CLEANED_PARQUET,
        "splits_path": SPLITS_JSON,
    }
    expected_preprocess_hash = compute_input_hash(
        [CLEANED_PARQUET, SPLITS_JSON], preprocess_params
    )
    if preprocess_payload.get("input_hash") != expected_preprocess_hash:
        raise LeakageError(
            "Preprocessing artefacts are stale or were fit on different data/splits than "
            "the current cleaned dataset and splits.json."
        )

    if not PREPROCESS_BUNDLE_PATH.exists():
        raise LeakageError(f"Preprocess bundle missing: {PREPROCESS_BUNDLE_PATH}")

    bundle = read_joblib(PREPROCESS_BUNDLE_PATH)
    recorded_fit_hash = bundle.get("train_fit_indices_hash")
    if recorded_fit_hash is not None and recorded_fit_hash != fit_input_hash:
        raise LeakageError(
            "Preprocess fit-input hash does not match current train split indices hash. "
            "Preprocessors may have been fit on the wrong rows."
        )

    # fit-input hash must be the train component of the split-indices hash (not val/test).
    if fit_input_hash == _indices_hash(splits["val"]) or fit_input_hash == _indices_hash(splits["test"]):
        raise LeakageError("Fit-input hash matches val or test indices — train-only fit violated.")

    if val_set & train_set or test_set & train_set:
        raise LeakageError("Val/test indices overlap train indices used for preprocessing fit.")

    LOGGER.info(
        "Leakage check passed (train=%d, val=%d, test=%d, split_hash=%s…).",
        len(train_set),
        len(val_set),
        len(test_set),
        split_hash[:12],
    )


def _verify_test_holdout(splits: dict[str, Any]) -> str:
    test_hash = _indices_hash(splits["test"])
    if TABULAR_UNSCALED_PARQUET.exists():
        df = read_parquet(TABULAR_UNSCALED_PARQUET)
        used = set(int(i) for i in splits["test"])
        if len(used) != len(splits["test"]):
            raise LeakageError("Duplicate indices in test split definition.")
        if max(used, default=-1) >= len(df):
            raise LeakageError("Test indices exceed tabular_unscaled row count.")
    return test_hash


def _decision_tree_grid(n_train_rows: int) -> dict[str, list[Any]]:
    depths: list[Any] = [4, 6, 8, 10, None]
    if n_train_rows < 10_000:
        depths = [d for d in depths if d is None or d <= 6]
    return {
        "max_depth": depths,
        "min_samples_leaf": [1, 5, 20, 50],
    }


def _random_forest_grid(n_train_rows: int) -> dict[str, list[Any]]:
    if n_train_rows >= LARGE_TRAIN_THRESHOLD:
        return {
            "n_estimators": [100, 200],
            "max_depth": [12, 20],
            "min_samples_leaf": [5],
        }
    return {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 5],
    }


def _xgboost_grid(n_train_rows: int) -> dict[str, list[Any]]:
    if n_train_rows >= LARGE_TRAIN_THRESHOLD:
        return {
            "n_estimators": [200],
            "max_depth": [4, 6],
            "learning_rate": [0.1],
            "subsample": [0.8],
        }
    return {
        "n_estimators": [200, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }


def _estimator_factories(random_seed: int, *, parallel_jobs: int = -1) -> dict[str, Any]:
    factories: dict[str, Any] = {
        "decision_tree": lambda: DecisionTreeClassifier(random_state=random_seed),
        "random_forest": lambda: RandomForestClassifier(
            random_state=random_seed, n_jobs=parallel_jobs
        ),
    }
    if XGBClassifier is not None:
        xgb_kwargs: dict[str, Any] = {
            "random_state": random_seed,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "objective": "multi:softprob",
        }
        factories["xgboost"] = lambda: XGBClassifier(**xgb_kwargs)
    return factories


def _macro_roc_auc(y_true: np.ndarray, y_proba: np.ndarray, classes: list[str]) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        return float("nan")


def _score_on_split(model: Any, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    X_scoring = _compact_training_frame(X)
    preds = model.predict(X_scoring)
    proba = model.predict_proba(X_scoring) if hasattr(model, "predict_proba") else None
    classes = [str(c) for c in getattr(model, "classes_", sorted(set(y)))]
    scores = {
        "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
    }
    if proba is not None:
        scores["roc_auc_ovr_macro"] = _macro_roc_auc(y, proba, classes)
    else:
        scores["roc_auc_ovr_macro"] = float("nan")
    return scores


def _compact_training_frame(X_train: pd.DataFrame) -> pd.DataFrame:
    """Use float32 columns to halve memory while preserving feature names."""
    return X_train.astype(np.float32)


def _grid_search_settings(n_train_rows: int) -> dict[str, Any]:
    if n_train_rows >= LARGE_TRAIN_THRESHOLD:
        return {"n_jobs": 1, "cv": 3, "subsample_max": GRID_SEARCH_SUBSAMPLE_MAX}
    return {"n_jobs": -1, "cv": 5, "subsample_max": None}


def _effective_cv_folds(y: np.ndarray, requested: int) -> int:
    """Pick a CV fold count that every class can support."""
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else requested
    if min_count < 2:
        return 2
    return max(2, min(requested, min_count))


def _stratified_subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    random_seed: int,
    *,
    cv_folds: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stratified subsample that keeps enough rows per class for CV."""
    if len(y) <= max_rows:
        return X, y

    min_per_class = max(cv_folds, 2)
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)

    selected: list[int] = []
    pool_idx: list[int] = []

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) <= min_per_class * 2:
            selected.extend(cls_idx.tolist())
            continue
        guaranteed = rng.choice(cls_idx, size=min_per_class, replace=False)
        selected.extend(guaranteed.tolist())
        pool_idx.extend(np.setdiff1d(cls_idx, guaranteed).tolist())

    budget = max_rows - len(selected)
    if budget > 0 and pool_idx:
        pool = np.array(pool_idx)
        take = min(budget, len(pool))
        if take == len(pool):
            selected.extend(pool.tolist())
        else:
            extra, _ = train_test_split(
                pool,
                train_size=take,
                stratify=y[pool],
                random_state=random_seed,
            )
            selected.extend(extra.tolist())

    if len(selected) > max_rows:
        selected_arr = np.array(selected)
        trimmed, _ = train_test_split(
            selected_arr,
            train_size=max_rows,
            stratify=y[selected_arr],
            random_state=random_seed,
        )
        selected = trimmed.tolist()

    selected_arr = np.array(sorted(set(selected)))
    return X.iloc[selected_arr], y[selected_arr]


def _subsample_max_for_model(name: str, settings: dict[str, Any]) -> int | None:
    if settings["subsample_max"] is None:
        return None
    return MODEL_SUBSAMPLE_MAX.get(name, settings["subsample_max"])


def _grid_search_model(
    name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_train_rows: int,
    random_seed: int = RANDOM_SEED,
) -> GridSearchCV:
    settings = _grid_search_settings(n_train_rows)
    X_full = _compact_training_frame(X_train)
    subsample_cap = _subsample_max_for_model(name, settings) or n_train_rows
    cv_folds = _effective_cv_folds(y_train, settings["cv"])
    X_tune, y_tune = _stratified_subsample(
        X_full,
        y_train,
        subsample_cap,
        random_seed,
        cv_folds=cv_folds,
    )
    tune_cv = _effective_cv_folds(y_tune, settings["cv"])

    label_encoder: LabelEncoder | None = None
    y_tune_fit = y_tune
    y_train_fit = y_train
    if name == "xgboost":
        label_encoder = _fit_label_encoder(y_train)
        y_tune_fit = label_encoder.transform(y_tune)
        y_train_fit = label_encoder.transform(y_train)

    if settings["subsample_max"] and len(y_train) > subsample_cap:
        LOGGER.info(
            "Model '%s': tuning on stratified subsample (%d / %d rows, cv=%d) to limit memory.",
            name,
            len(y_tune),
            n_train_rows,
            tune_cv,
        )

    cv_splitter = StratifiedKFold(n_splits=tune_cv, shuffle=True, random_state=random_seed)
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=cv_splitter,
        scoring="f1_macro",
        n_jobs=settings["n_jobs"],
        refit=True,
        error_score="raise",
        pre_dispatch=2,
    )
    search.fit(X_tune, y_tune_fit)

    if settings["subsample_max"] and len(y_train) > subsample_cap:
        if name == "random_forest":
            LOGGER.info(
                "Model '%s': skipping full-data refit (too heavy at %d rows); "
                "using best estimator from tuning subsample.",
                name,
                n_train_rows,
            )
        else:
            LOGGER.info(
                "Model '%s': refitting best params on full training set (%d rows).",
                name,
                n_train_rows,
            )
            best = clone(search.best_estimator_)
            best.fit(X_full, y_train_fit)
            search.best_estimator_ = best

    if label_encoder is not None:
        search.best_estimator_ = _EncodedLabelModelWrapper(
            search.best_estimator_, label_encoder
        )

    return search


# --- TRAIN ---
@checkpointed("train")
def train(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    force: bool = bool(params.get("force", False))
    random_seed: int = int(params.get("random_seed", RANDOM_SEED))
    train_path: Path = Path(params.get("train_path", TRAIN_BALANCED_PARQUET))
    unscaled_path: Path = Path(params.get("unscaled_path", TABULAR_UNSCALED_PARQUET))

    leakage_check()

    if not train_path.exists():
        raise FileNotFoundError(f"Balanced training data not found: {train_path}")
    if not unscaled_path.exists():
        raise FileNotFoundError(f"Unscaled tabular data not found: {unscaled_path}")

    splits = _load_splits()
    target_col = _target_column(splits)

    train_df = read_parquet(train_path)
    unscaled_df = read_parquet(unscaled_path)
    val_df = unscaled_df.iloc[splits["val"]].copy()

    X_train, y_train = _feature_matrix(train_df, target_col)
    X_val, y_val = _feature_matrix(val_df, target_col)
    if y_train is None or y_val is None:
        raise ValueError("Training/validation labels missing from processed data.")

    X_train, y_train = _filter_known_labels(X_train, y_train, context="train")
    X_val, y_val = _filter_known_labels(X_val, y_val, context="val")

    n_train_rows = len(X_train)
    tune_settings = _grid_search_settings(n_train_rows)
    parallel_jobs = 1 if tune_settings["n_jobs"] == 1 else -1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_hash_base = compute_input_hash(
        [train_path, unscaled_path, SPLITS_JSON],
        {
            "random_seed": random_seed,
            "models": MODEL_NAMES,
            "label_filter": "threat_classes_only",
            "tune_strategy": "subsample_v3_xgb_labels",
        },
    )

    tuning_note = (
        f"- Tuning: GridSearchCV, cv={tune_settings['cv']}, scoring=f1_macro, "
        f"n_jobs={tune_settings['n_jobs']}."
    )
    if tune_settings["subsample_max"]:
        tuning_note += (
            f" Large train set → stratified subsample of {tune_settings['subsample_max']:,} rows "
            f"for CV, then refit best params on all {n_train_rows:,} rows."
        )

    selection_lines = [
        "# Model selection reasoning",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Training rows (balanced): {n_train_rows}",
        "- Features: **unscaled** (tree models are scale-invariant).",
        tuning_note,
        "",
    ]

    dt_grid = _decision_tree_grid(n_train_rows)
    if n_train_rows < 10_000:
        selection_lines.append(
            f"- Decision Tree: n_train_rows={n_train_rows} < 10,000 → max_depth capped at ≤6 "
            f"(grid={dt_grid['max_depth']})."
        )
    else:
        selection_lines.append(
            f"- Decision Tree: n_train_rows={n_train_rows} ≥ 10,000 → full max_depth grid "
            f"{dt_grid['max_depth']}."
        )
    selection_lines.extend(
        [
            f"- Random Forest grid: {_random_forest_grid(n_train_rows)} (n_jobs={parallel_jobs}).",
            f"- XGBoost grid: {_xgboost_grid(n_train_rows)} (tree_method=hist, eval_metric=mlogloss).",
            "",
        ]
    )

    factories = _estimator_factories(random_seed, parallel_jobs=parallel_jobs)
    val_leaderboard: list[dict[str, Any]] = []
    output_paths: list[str] = []

    grids = {
        "decision_tree": dt_grid,
        "random_forest": _random_forest_grid(n_train_rows),
        "xgboost": _xgboost_grid(n_train_rows),
    }

    for name in MODEL_NAMES:
        if name not in factories:
            LOGGER.warning("Skipping %s — estimator unavailable.", name)
            continue

        model_hash = compute_input_hash([], {**{"base": model_hash_base}, "model": name})
        model_path = MODELS_DIR / f"{name}.joblib"
        params_path = MODELS_DIR / f"{name}_params.json"
        cv_path = TRAIN_REPORTS_DIR / f"{name}_cv_results.csv"

        if not force and _model_is_done(name, model_hash):
            LOGGER.info("Model '%s' skipped, loaded from disk.", name)
            model = read_joblib(model_path)
        else:
            LOGGER.info("Training model '%s' via GridSearchCV…", name)
            search = _grid_search_model(
                name,
                factories[name](),
                grids[name],
                X_train,
                y_train,
                n_train_rows=n_train_rows,
                random_seed=random_seed,
            )
            model = search.best_estimator_
            write_joblib(model, model_path)
            write_json(
                {
                    "model": name,
                    "best_params": search.best_params_,
                    "best_cv_f1_macro": float(search.best_score_),
                },
                params_path,
            )
            pd.DataFrame(search.cv_results_).to_csv(cv_path, index=False)
            _mark_model_done(name, model_hash, [model_path, params_path, cv_path])
            selection_lines.append(
                f"- {name}: best CV f1_macro={search.best_score_:.4f}, params={search.best_params_}"
            )
            output_paths.extend([str(model_path), str(params_path), str(cv_path)])

        scores = _score_on_split(model, X_val, y_val)
        val_leaderboard.append({"model": name, **scores})

    pending: list[str] = []
    for name in MODEL_NAMES:
        if name not in factories:
            continue
        mh = compute_input_hash([], {**{"base": model_hash_base}, "model": name})
        if not _model_is_done(name, mh):
            pending.append(name)
    if pending and not force:
        raise RuntimeError(f"Training incomplete — missing checkpoints for: {pending}")

    val_leaderboard.sort(
        key=lambda row: (row["f1_macro"], row.get("roc_auc_ovr_macro", float("-inf"))),
        reverse=True,
    )
    leaderboard_lines = [
        "# Validation leaderboard",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Ranked by **val macro-F1** (primary) and **val ROC-AUC OvR macro** (secondary).",
        "",
        "| Rank | Model | Val F1 (macro) | Val ROC-AUC (OvR macro) | Val accuracy |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(val_leaderboard, start=1):
        leaderboard_lines.append(
            f"| {rank} | {row['model']} | {row['f1_macro']:.4f} | "
            f"{row.get('roc_auc_ovr_macro', float('nan')):.4f} | {row['accuracy']:.4f} |"
        )

    _write_markdown(TRAIN_REPORTS_DIR / "model_selection.md", selection_lines)
    _write_markdown(TRAIN_REPORTS_DIR / "leaderboard.md", leaderboard_lines)
    output_paths.extend(
        [str(TRAIN_REPORTS_DIR / "model_selection.md"), str(TRAIN_REPORTS_DIR / "leaderboard.md")]
    )

    LOGGER.info("Training complete for models: %s", [r["model"] for r in val_leaderboard])
    return {
        "output_paths": output_paths,
        "val_leaderboard": val_leaderboard,
        "best_val_model": val_leaderboard[0]["model"] if val_leaderboard else None,
    }


# --- evaluation helpers ---
def _per_class_dict(
    metric_name: str,
    values: np.ndarray,
    classes: list[str],
) -> dict[str, float]:
    return {str(cls): float(val) for cls, val in zip(classes, values)}


def _full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    classes: list[str],
) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": {
            "macro": float(p_macro),
            "weighted": float(p_weighted),
            "per_class": _per_class_dict("precision", precision, classes),
        },
        "recall": {
            "macro": float(r_macro),
            "weighted": float(r_weighted),
            "per_class": _per_class_dict("recall", recall, classes),
        },
        "f1": {
            "macro": float(f_macro),
            "weighted": float(f_weighted),
            "per_class": _per_class_dict("f1", f1, classes),
        },
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1:
            metrics["pr_auc"] = float(average_precision_score(y_bin, y_proba[:, 1]))
        else:
            ap_scores = [
                average_precision_score(y_bin[:, i], y_proba[:, i])
                for i in range(len(classes))
                if y_bin[:, i].sum() > 0
            ]
            metrics["pr_auc"] = float(np.mean(ap_scores)) if ap_scores else None
    else:
        metrics["roc_auc_ovr"] = None
        metrics["pr_auc"] = None

    return metrics


def _save_confusion_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    csv_path = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(csv_path)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(cm, row_sums, where=row_sums != 0)
    norm_df = pd.DataFrame(row_norm, index=classes, columns=classes)
    norm_path = out_dir / "confusion_matrix_row_normalized.csv"
    norm_df.to_csv(norm_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (counts)")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    png_path = out_dir / "confusion_matrix.png"
    plt.savefig(png_path, dpi=120)
    plt.close()
    return [csv_path, norm_path, png_path]


def _save_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> Path:
    y_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(classes):
        if y_bin.shape[1] <= i:
            continue
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=str(cls))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves (one-vs-rest)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = out_dir / "roc_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _save_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> Path:
    y_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(classes):
        if y_bin.shape[1] <= i:
            continue
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, label=str(cls))
    ax.set_title("Precision–recall curves (one-vs-rest)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    path = out_dir / "pr_curves.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def _feature_importance_frame(model: Any, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance_gain": values}).sort_values(
            "importance_gain", ascending=False
        )
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        rows = [{"feature": k, "importance_gain": v} for k, v in score.items()]
        return pd.DataFrame(rows).sort_values("importance_gain", ascending=False)
    return pd.DataFrame(columns=["feature", "importance_gain"])


def _evaluate_one_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    out_dir = EVAL_REPORTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    classes = [str(c) for c in getattr(model, "classes_", sorted(set(y_test)))]

    metrics = _full_metrics(y_test, y_pred, y_proba, classes)
    metrics_path = out_dir / "metrics.json"
    write_json(metrics, metrics_path)

    report_txt = classification_report(y_test, y_pred, labels=classes, zero_division=0)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report_txt, encoding="utf-8")

    cm_paths = _save_confusion_artifacts(y_test, y_pred, classes, out_dir)

    extra_paths: list[Path] = [metrics_path, report_path, *cm_paths]
    if y_proba is not None:
        extra_paths.append(_save_roc_curves(y_test, y_proba, classes, out_dir))
        extra_paths.append(_save_pr_curves(y_test, y_proba, classes, out_dir))

    fi = _feature_importance_frame(model, feature_names)
    fi_path = out_dir / "feature_importances.csv"
    fi.to_csv(fi_path, index=False)
    extra_paths.append(fi_path)

    perm_path = out_dir / "permutation_importance_val.csv"
    try:
        perm = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            scoring="f1_macro",
        )
        perm_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(perm_path, index=False)
        extra_paths.append(perm_path)
    except Exception as exc:
        LOGGER.warning("Permutation importance failed for %s: %s", name, exc)

    return {
        "model": name,
        "metrics": metrics,
        "output_paths": [str(p) for p in extra_paths],
        "f1_macro": metrics["f1"]["macro"],
    }


def _write_model_card(winner: str, all_results: list[dict[str, Any]]) -> Path:
    winner_metrics = next(r for r in all_results if r["model"] == winner)
    per_class = winner_metrics["metrics"]["f1"]["per_class"]
    lines = [
        "# Model card",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Intended use",
        "",
        "- **Defensive network-flow analysis** for SOC / analyst decision support.",
        "- Classifies tabular flow records (shared multi-corpus schema) into the project threat taxonomy.",
        "- Outputs are advisory signals only — not autonomous blocking or enforcement.",
        "",
        "## Training data",
        "",
        "- Standardized multi-corpus flows (CICIDS-2017, and UNSW-NB15 / KDD Cup 99 when present)",
        "  after shared-schema integrate → clean → split → preprocess → balance.",
        "- Models trained on **balanced train split only**; validation used for model selection;",
        "  **test split held out** until final evaluation.",
        "- Tree models use **unscaled** features.",
        "",
        "## Selected model",
        "",
        f"- **{winner}** (highest test macro-F1 among the fixed trio: Decision Tree, Random Forest, XGBoost).",
        "",
        "## Per-class test F1",
        "",
        "| Class | F1 |",
        "| --- | ---: |",
    ]
    for cls in THREAT_CLASSES:
        if cls in per_class:
            lines.append(f"| {cls} | {per_class[cls]:.4f} |")
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            "- Minority attack classes may be under-detected; SMOTE/balancing can inflate precision on synthetic points.",
            "- Post-hoc explanations (SHAP/LIME) are not causal.",
            "- Performance degrades on out-of-distribution traffic or adversarially crafted flows.",
            "- Identifier columns are removed by design; predictions rely on statistical flow features only.",
            "",
            "## Dual-use and responsible disclosure",
            "",
            "- This system is developed **only for defensive cybersecurity research** (intrusion detection",
            "  awareness, analyst training, and academic evaluation).",
            "- **Do not use** to develop, tune, or validate offensive capabilities (evasion, targeting,",
            "  or automated attack generation).",
            "- **Model weights are not published** per the dissertation proposal; share metrics and",
            "  explanations only in defensive framing.",
            "- **Tuned hyperparameters are not published** in forms that would enable adversarial evasion;",
            "  selection rationale is documented in `reports/train/model_selection.md` only.",
            "- Human analyst oversight is required for any operational use.",
            "",
        ]
    )
    return _write_markdown(MODEL_CARD_PATH, lines)


# --- EVALUATE ---
@checkpointed("evaluate")
def evaluate(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    unscaled_path: Path = Path(params.get("unscaled_path", TABULAR_UNSCALED_PARQUET))

    leakage_check()
    splits = _load_splits()
    test_hash = _verify_test_holdout(splits)
    LOGGER.info("Evaluating on held-out TEST split (test_indices_hash=%s…).", test_hash[:12])

    if not unscaled_path.exists():
        raise FileNotFoundError(f"Unscaled tabular data not found: {unscaled_path}")

    df = read_parquet(unscaled_path)
    test_df = df.iloc[splits["test"]].copy()
    val_df = df.iloc[splits["val"]].copy()
    target_col = _target_column(splits)

    X_test, y_test = _feature_matrix(test_df, target_col)
    X_val, y_val = _feature_matrix(val_df, target_col)
    if y_test is None or y_val is None:
        raise ValueError("Test/val labels missing from processed data.")

    feature_names = list(X_test.columns)
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    all_output_paths: list[str] = []

    for name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}.joblib"
        if not model_path.exists():
            LOGGER.warning("Model artifact missing, skipping evaluate for %s", name)
            continue
        model = read_joblib(model_path)
        result = _evaluate_one_model(
            name, model, X_test, y_test, X_val, y_val, feature_names
        )
        results.append(result)
        all_output_paths.extend(result["output_paths"])

    if not results:
        raise RuntimeError("No trained models found for evaluation.")

    results.sort(key=lambda r: r["f1_macro"], reverse=True)
    winner = results[0]["model"]

    lb_lines = [
        "# Final test leaderboard",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Test indices hash: `{test_hash}`",
        "",
        "| Rank | Model | Test F1 (macro) | Test accuracy | ROC-AUC OvR | MCC |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(results, start=1):
        m = row["metrics"]
        lb_lines.append(
            f"| {rank} | {row['model']} | {m['f1']['macro']:.4f} | {m['accuracy']:.4f} | "
            f"{m.get('roc_auc_ovr') or float('nan'):.4f} | {m['mcc']:.4f} |"
        )
    final_lb_path = EVAL_REPORTS_DIR / "final_leaderboard.md"
    _write_markdown(final_lb_path, lb_lines)
    all_output_paths.append(str(final_lb_path))

    winner_path = MODELS_DIR / f"{winner}.joblib"
    write_joblib(read_joblib(winner_path), BEST_MODEL_PATH)
    write_json({"best_model": winner, "metric": "f1_macro", "test_f1_macro": results[0]["f1_macro"]}, BEST_MODEL_META_PATH)
    all_output_paths.extend([str(BEST_MODEL_PATH), str(BEST_MODEL_META_PATH)])

    model_card_path = _write_model_card(winner, results)
    all_output_paths.append(str(model_card_path))

    LOGGER.info("Evaluation complete. Winner: %s (test macro-F1=%.4f)", winner, results[0]["f1_macro"])
    return {
        "output_paths": all_output_paths,
        "winner": winner,
        "test_hash": test_hash,
        "results": results,
    }


# --- inference / PREDICT ---
def _normalize_col(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def _identifying_columns_present(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if _normalize_col(col) in IDENTIFYING_COLUMNS]


def _build_imputation_maps(train_df: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    maps: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for col in numeric_cols:
        if col not in train_df.columns:
            continue
        fit_series = _treat_as_missing(train_df[col])
        skew = _skewness(fit_series)
        strategy = "median" if skew > 1 else "mean"
        fill_value = fit_series.median() if strategy == "median" else fit_series.mean()
        maps["numeric"][col] = {"strategy": strategy, "value": float(fill_value)}
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
        fit_series = train_df[col].astype(str)
        nunique = fit_series.nunique(dropna=True)
        if nunique < CATEGORICAL_MODE_CARDINALITY:
            mode = fit_series.mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "MISSING"
        else:
            fill_value = "MISSING"
        maps["categorical"][col] = fill_value
    return maps


def _load_inference_context(model_path: Path | None = None) -> None:
    resolved_model_path = Path(model_path or BEST_MODEL_PATH)
    if (
        _CACHE.model is not None
        and _CACHE.model_path == resolved_model_path
        and _CACHE.bundle is not None
    ):
        return

    if not PREPROCESS_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"Preprocess bundle not found: {PREPROCESS_BUNDLE_PATH}")
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model not found: {resolved_model_path}")

    bundle = read_joblib(PREPROCESS_BUNDLE_PATH)
    splits = _load_splits()
    target_col = _target_column(splits)

    if CLEANED_PARQUET.exists():
        cleaned = read_parquet(CLEANED_PARQUET)
        train_df = cleaned.iloc[splits["train"]]
    else:
        unscaled = read_parquet(TABULAR_UNSCALED_PARQUET)
        train_df = unscaled.iloc[splits["train"]]

    imputation_maps = _build_imputation_maps(train_df, bundle)
    model = read_joblib(resolved_model_path)

    _CACHE.model = model
    _CACHE.model_path = resolved_model_path
    _CACHE.bundle = bundle
    _CACHE.imputation_maps = imputation_maps
    _CACHE.feature_names = list(bundle["feature_names"])
    _CACHE.classes_ = [str(c) for c in getattr(model, "classes_", [])]


def _prepare_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    from src.data import (
        SHARED_FEATURE_COLUMNS,
        SUPPORTED_DATASET_TYPES,
        _detect_dataset_type,
        _project_to_shared_schema,
    )

    out = _sanitize_columns(df.copy())
    cols = [str(c) for c in out.columns]
    has_shared = sum(1 for c in SHARED_FEATURE_COLUMNS if c in out.columns) >= 3
    if not has_shared:
        detected = _detect_dataset_type(Path("upload.csv"), cols)
        if detected in SUPPORTED_DATASET_TYPES:
            out = _project_to_shared_schema(out, detected, "upload")

    try:
        label_col_present = any(
            _normalize_col(c) in {"label", "class", "threat class", "threat label"} for c in out.columns
        )
        if label_col_present:
            out = map_labels(out)
    except Exception:
        pass

    id_cols = [c for c in out.columns if _is_identifier_column(c)]
    if id_cols:
        out = out.drop(columns=id_cols)

    meta_drop = [c for c in ("source_file", "dataset_source") if c in out.columns]
    if meta_drop:
        out = out.drop(columns=meta_drop)

    return out


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    if _CACHE.bundle is None or _CACHE.imputation_maps is None or _CACHE.feature_names is None:
        raise RuntimeError("Inference cache not initialized.")

    bundle = _CACHE.bundle
    maps = _CACHE.imputation_maps
    numeric_cols = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    low_card = bundle["low_card_ohe"]
    high_card = bundle["high_card_freq"]
    one_hot = bundle.get("one_hot")
    freq_encoder = bundle.get("freq_encoder")

    feature_cols = numeric_cols + categorical_cols
    imputed = df.reindex(columns=feature_cols).copy()

    for col, spec in maps["numeric"].items():
        if col in imputed.columns:
            imputed[col] = _treat_as_missing(imputed[col]).fillna(spec["value"])
    for col, fill_value in maps["categorical"].items():
        if col in imputed.columns:
            imputed[col] = imputed[col].astype(str).fillna(fill_value)

    parts: list[pd.DataFrame] = [imputed[numeric_cols]]
    if low_card and one_hot is not None:
        ohe = one_hot.transform(imputed[low_card].astype(str))
        ohe_cols = one_hot.get_feature_names_out(low_card).tolist()
        parts.append(pd.DataFrame(ohe, columns=ohe_cols, index=imputed.index))
    if high_card and freq_encoder is not None:
        parts.append(freq_encoder.transform(imputed[high_card]))

    features = pd.concat(parts, axis=1)
    features = features.reindex(columns=_CACHE.feature_names, fill_value=0.0)
    return features


def predict_from_dataframe(
    df: pd.DataFrame,
    model_path: Path | str | None = None,
    *,
    audit: bool = True,
) -> pd.DataFrame:
    """Apply saved preprocessors and predict; returns labels, probabilities, and confidence."""
    _load_inference_context(Path(model_path) if model_path else None)

    raw = _prepare_raw_dataframe(df)
    id_cols = _identifying_columns_present(df)
    id_frame = df[[c for c in id_cols if c in df.columns]].copy() if id_cols else pd.DataFrame(index=df.index)

    X = _encode_features(raw)
    model = _CACHE.model
    assert model is not None

    preds = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    classes = _CACHE.classes_ or [str(c) for c in getattr(model, "classes_", [])]

    out = df.copy()
    out["predicted_label"] = preds
    if proba is not None:
        for i, cls in enumerate(classes):
            out[f"predicted_proba_{cls}"] = proba[:, i]
        out["confidence"] = proba.max(axis=1)
    else:
        out["confidence"] = 1.0

    for col in id_frame.columns:
        if col not in out.columns:
            out[col] = id_frame[col]

    if audit:
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        audit_path = PREDICTIONS_DIR / f"{ts}.parquet"
        write_parquet(out, audit_path)
        LOGGER.info("Prediction audit saved: %s", audit_path)

    return out


def predict_from_file(
    path: Path | str,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load a raw CICIDS-style CSV and return predictions."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return predict_from_dataframe(df, model_path=model_path)


def predict(
    records: pd.DataFrame,
    model_path: Path = BEST_MODEL_PATH,
    preprocessor_path: Path | None = None,
) -> np.ndarray:
    """Backwards-compatible helper returning label array only (Streamlit UI)."""
    del preprocessor_path  # legacy arg; preprocess bundle is loaded from artifacts/preprocess/
    result = predict_from_dataframe(records, model_path=model_path, audit=True)
    return result["predicted_label"].to_numpy()


@checkpointed("predict")
def predict_step(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pipeline checkpoint step — run batch predict on held-out test split."""
    params = params or {}
    unscaled_path: Path = Path(params.get("unscaled_path", TABULAR_UNSCALED_PARQUET))
    splits = _load_splits()
    df = read_parquet(unscaled_path).iloc[splits["test"]]
    result = predict_from_dataframe(df, audit=True)
    out_path = PREDICTIONS_DIR / "test_predictions.parquet"
    write_parquet(result, out_path)
    return {"output_paths": [str(out_path)], "n_predictions": int(len(result))}


def run_default() -> dict[str, Any]:
    return train({})


if __name__ == "__main__":
    result = run_default()
    loaded = result.get("loaded_from_checkpoint", False)
    state = "checkpoint" if loaded else f"executed ({result.get('best_val_model', 'unknown')})"
    print(f"src.model: {state}.")
