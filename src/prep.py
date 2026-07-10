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
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.core import (
    ARTIFACTS_DIR,
    INTERIM_DIR,
    LOGGER,
    PROCESSED_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpoint_path,
    checkpointed,
    read_json,
    read_parquet,
    write_json,
    write_joblib,
    write_parquet,
)

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None
    SMOTETomek = None


# --- paths ---
INTEGRATED_PARQUET = INTERIM_DIR / "integrated.parquet"
CLEANED_PARQUET = INTERIM_DIR / "cleaned.parquet"
EDA_DIR = REPORTS_DIR / "eda"
SPLITS_JSON = PROCESSED_DIR / "splits.json"
SPLIT_REPORT_MD = REPORTS_DIR / "split_report.md"
CLEAN_DECISIONS_MD = REPORTS_DIR / "clean_decisions.md"
PREPROCESS_DECISIONS_MD = REPORTS_DIR / "preprocess_decisions.md"
PREPROCESS_ARTIFACTS_DIR = ARTIFACTS_DIR / "preprocess"
TABULAR_SCALED_PARQUET = PROCESSED_DIR / "tabular_scaled.parquet"
TABULAR_UNSCALED_PARQUET = PROCESSED_DIR / "tabular_unscaled.parquet"
TEST_SCALED_PARQUET = PROCESSED_DIR / "test_scaled.parquet"
VAL_SCALED_PARQUET = PROCESSED_DIR / "val_scaled.parquet"
TRAIN_BALANCED_PARQUET = PROCESSED_DIR / "train_balanced.parquet"
CLASS_WEIGHT_JSON = ARTIFACTS_DIR / "class_weight.json"

NEAR_ZERO_VAR_THRESHOLD = 1e-6
HIGH_CORR_THRESHOLD = 0.95
MISSING_COL_THRESHOLD = 0.60
LOW_CARDINALITY_OHE = 15
HIGH_CARDINALITY_FREQ = 20
CATEGORICAL_MODE_CARDINALITY = 20

IDENTIFIER_PATTERNS = [
    r"^flow\s*id$",
    r"^source\s*ip$",
    r"^destination\s*ip$",
    r"^src\s*ip$",
    r"^dst\s*ip$",
    r"^source\s*port$",
    r"^destination\s*port$",
    r"^src\s*port$",
    r"^dst\s*port$",
    r"^timestamp$",
    r"^stime$",
    r"^ltime$",
]

FLOW_DURATION_PATTERNS = [r"flow\s*duration", r"^duration$", r"^dur$"]
PACKET_COUNT_PATTERNS = [
    r"total\s*fwd\s*packets",
    r"total\s*backward\s*packets",
    r"total\s*length\s*of\s*fwd\s*packets",
    r"total\s*length\s*of\s*bwd\s*packets",
    r"^spkts$",
    r"^dpkts$",
    r"packet",
]

METADATA_COLUMNS = {"source_file", "label", "threat_class", "threat_label"}


# --- label mapping (kept for tests / backwards compatibility) ---
LABEL_MAP: dict[str, str] = {
    "BENIGN": "BENIGN",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack - Brute Force": "WebAttack",
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Bot": "Botnet",
}


def _normalize_col(name: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def _match_column(columns: list[str], patterns: list[str]) -> str | None:
    for col in columns:
        normalized = _normalize_col(col)
        for pattern in patterns:
            if re.search(pattern, normalized):
                return col
    return None


def _is_junk_column(name: str) -> bool:
    normalized = _normalize_col(name)
    if not normalized or normalized.startswith("unnamed"):
        return True
    if normalized.startswith("#"):
        return True
    return bool(re.fullmatch(r"=+", normalized.replace(" ", "")))


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace from column names, drop CSV junk columns, dedupe names."""
    out = df.copy()
    new_cols: list[str] = []
    seen: dict[str, int] = {}
    for raw in out.columns:
        base = str(raw).strip()
        if not base:
            base = "unnamed"
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        new_cols.append(base if idx == 0 else f"{base}_{idx}")
    out.columns = new_cols
    keep_cols = [col for col in out.columns if not _is_junk_column(col)]
    out = out[keep_cols]
    return out


def _label_series(df: pd.DataFrame, label_col: str) -> pd.Series:
    return (
        df[label_col]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(str)
        .str.strip()
    )


def _sorted_classes(series: pd.Series) -> list[str]:
    values = {str(v).strip() for v in series.dropna().unique() if str(v).strip() and str(v).strip().lower() != "nan"}
    return sorted(values, key=str.lower)


def _resolve_label_column(df: pd.DataFrame) -> str:
    normalized_to_actual = {_normalize_col(col): col for col in df.columns}
    for candidate in ["label", "class", "threat class", "threat label"]:
        actual = normalized_to_actual.get(candidate)
        if actual is not None:
            return actual
    raise ValueError("No label column found in dataset.")


def _resolve_target_column(df: pd.DataFrame) -> str:
    normalized_to_actual = {_normalize_col(col): col for col in df.columns}
    if "threat class" in normalized_to_actual:
        return normalized_to_actual["threat class"]
    if "threat label" in normalized_to_actual:
        return normalized_to_actual["threat label"]
    label_col = _resolve_label_column(df)
    mapped = df[label_col].astype(str).map(LABEL_MAP).fillna(df[label_col].astype(str))
    df["threat_class"] = mapped
    df["threat_label"] = mapped
    return "threat_class"


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw labels to the project threat taxonomy (used by tests)."""
    out = _sanitize_columns(df)
    label_col = _resolve_label_column(out)
    out["threat_label"] = out[label_col].astype(str).map(LABEL_MAP).fillna("BENIGN")
    out = out[out["threat_label"].isin(THREAT_CLASSES)].copy()
    return out


def _is_identifier_column(col: str) -> bool:
    normalized = _normalize_col(col)
    return any(re.search(pattern, normalized) for pattern in IDENTIFIER_PATTERNS)


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS and col != target_col and not _is_identifier_column(col)
    ]


def _treat_as_missing(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.replace([np.inf, -np.inf], np.nan)
    return series


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        series = _treat_as_missing(df[col])
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_pct": float(missing_count / len(df) * 100) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


def _numeric_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    cols = _feature_columns(df, target_col)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    cols = _feature_columns(df, target_col)
    return [c for c in cols if c not in _numeric_feature_cols(df, target_col)]


def _safe_variance(series: pd.Series) -> float:
    clean = _treat_as_missing(series).dropna()
    if len(clean) < 2 or clean.nunique(dropna=True) <= 1:
        return 0.0
    value = clean.var()
    return float(value) if pd.notna(value) else 0.0


def _outlier_counts(series: pd.Series) -> dict[str, int]:
    clean = _treat_as_missing(series).dropna()
    if clean.empty:
        return {"iqr_outliers": 0, "zscore_outliers": 0}
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_outliers = int(((clean < lower) | (clean > upper)).sum())
    if len(clean) < 2 or float(clean.std()) <= 1e-12:
        zscore_outliers = 0
    else:
        z_scores = np.abs(stats.zscore(clean, nan_policy="omit"))
        zscore_outliers = int((z_scores > 3).sum())
    return {"iqr_outliers": iqr_outliers, "zscore_outliers": zscore_outliers}


def _high_correlation_pairs(corr: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(float(value)) > threshold:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": float(value)})
    return pairs


def _choose_corr_drop(
    pairs: list[dict[str, Any]], variances: dict[str, float]
) -> list[dict[str, str]]:
    to_drop: set[str] = set()
    decisions: list[dict[str, str]] = []
    for pair in pairs:
        a, b = pair["feature_a"], pair["feature_b"]
        if a in to_drop or b in to_drop:
            continue
        var_a = variances.get(a, 0.0)
        var_b = variances.get(b, 0.0)
        drop_col = a if var_a <= var_b else b
        keep_col = b if drop_col == a else a
        to_drop.add(drop_col)
        decisions.append(
            {
                "dropped": drop_col,
                "kept": keep_col,
                "reason": f"|corr|={pair['correlation']:.4f}; kept higher variance",
            }
        )
    return decisions


def _flow_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    cols = _feature_columns(df, target_col)
    flow_cols = [c for c in cols if "flow" in _normalize_col(c)]
    return flow_cols if flow_cols else cols


def _write_markdown(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _verify_split_checkpoint() -> dict[str, Any]:
    done_path = checkpoint_path("split")
    if not done_path.exists():
        raise FileNotFoundError("Split checkpoint missing. Run split() before preprocess().")
    payload = read_json(done_path)
    for raw in payload.get("output_paths", []):
        if not Path(raw).exists():
            raise FileNotFoundError(f"Split output missing: {raw}")
    return payload


class FrequencyEncoder:
    """Fit frequency encoding on train; unseen categories map to 0."""

    def __init__(self) -> None:
        self.maps_: dict[str, dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> "FrequencyEncoder":
        for col in columns:
            counts = df[col].astype(str).value_counts(normalize=True)
            self.maps_[col] = {str(k): float(v) for k, v in counts.items()}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, mapping in self.maps_.items():
            out[col] = out[col].astype(str).map(mapping).fillna(0.0).astype(float)
        return out


# --- EDA ---
@checkpointed("eda")
def eda(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    integrated_path: Path = params.get("integrated_path", INTEGRATED_PARQUET)
    eda_dir: Path = params.get("eda_dir", EDA_DIR)

    if not integrated_path.exists():
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_path}")

    df = _sanitize_columns(read_parquet(integrated_path))
    target_col = _resolve_target_column(df)
    label_col = _resolve_label_column(df)
    eda_dir.mkdir(parents=True, exist_ok=True)

    shape_dtypes_path = eda_dir / "shape_dtypes_memory.json"
    missingness_path = eda_dir / "missingness.json"
    numeric_stats_path = eda_dir / "numeric_stats.json"
    corr_heatmap_path = eda_dir / "correlation_heatmap.png"
    high_corr_path = eda_dir / "high_correlation_pairs.json"
    class_dist_path = eda_dir / "class_distribution.json"
    variance_path = eda_dir / "variance.json"
    dataset_size_path = eda_dir / "dataset_size.json"

    mem_usage = {col: int(df[col].memory_usage(deep=True)) for col in df.columns}
    shape_payload = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "memory_usage_bytes": mem_usage,
        "total_memory_bytes": int(sum(mem_usage.values())),
    }
    write_json(shape_payload, shape_dtypes_path)

    miss_df = _missing_table(df)
    write_json(miss_df.to_dict(orient="records"), missingness_path)

    num_cols = _numeric_feature_cols(df, target_col)
    numeric_rows: list[dict[str, Any]] = []
    for col in num_cols:
        series = _treat_as_missing(df[col])
        outliers = _outlier_counts(series)
        has_spread = series.nunique(dropna=True) > 1
        numeric_rows.append(
            {
                "column": col,
                "count": int(series.count()),
                "mean": float(series.mean()) if series.count() else None,
                "std": float(series.std()) if series.count() and has_spread else 0.0 if series.count() else None,
                "min": float(series.min()) if series.count() else None,
                "25%": float(series.quantile(0.25)) if series.count() else None,
                "50%": float(series.quantile(0.50)) if series.count() else None,
                "75%": float(series.quantile(0.75)) if series.count() else None,
                "max": float(series.max()) if series.count() else None,
                "skew": float(series.skew()) if series.count() and has_spread else 0.0 if series.count() else None,
                "kurtosis": float(series.kurtosis()) if series.count() and has_spread else 0.0 if series.count() else None,
                **outliers,
            }
        )
    write_json(numeric_rows, numeric_stats_path)

    variance_ranking = pd.Series({col: _safe_variance(df[col]) for col in num_cols})
    top_var_cols = variance_ranking.sort_values(ascending=False).head(40).index.tolist() if num_cols else []
    high_corr_pairs: list[dict[str, Any]] = []
    if len(top_var_cols) >= 2:
        corr = df[top_var_cols].corr(numeric_only=True)
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
        plt.title("Correlation heatmap (top-40 variance features)")
        plt.tight_layout()
        plt.savefig(corr_heatmap_path, dpi=120)
        plt.close()
        high_corr_pairs = _high_correlation_pairs(corr)
    else:
        corr_heatmap_path.write_bytes(b"")
    write_json(high_corr_pairs, high_corr_path)

    label_counts = _label_series(df, label_col).value_counts().sort_values(ascending=False)
    total = int(label_counts.sum())
    class_distribution = {
        "counts": {str(k): int(v) for k, v in label_counts.items()},
        "percentages": {str(k): float(v / total * 100) for k, v in label_counts.items()},
    }
    write_json(class_distribution, class_dist_path)

    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.index.astype(str).tolist(), label_counts.to_numpy())
    plt.title("Class distribution (label)")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(eda_dir / "class_distribution.png", dpi=120)
    plt.close()

    variances = {col: _safe_variance(df[col]) for col in num_cols if df[col].notna().any()}
    near_zero = [col for col, var in variances.items() if var <= NEAR_ZERO_VAR_THRESHOLD]
    variance_payload = {"variances": variances, "near_zero_variance": near_zero}
    write_json(variance_payload, variance_path)

    flow_col = _match_column(list(df.columns), FLOW_DURATION_PATTERNS)
    packet_col = _match_column(list(df.columns), PACKET_COUNT_PATTERNS)
    if flow_col and target_col in df.columns:
        for cls in _sorted_classes(df[target_col]):
            subset = _treat_as_missing(df.loc[df[target_col].astype(str) == cls, flow_col]).dropna()
            if subset.empty:
                continue
            plt.figure(figsize=(8, 4))
            plt.hist(subset, bins=50, alpha=0.85)
            plt.title(f"Flow duration histogram — {cls}")
            plt.xlabel(flow_col)
            plt.ylabel("count")
            plt.tight_layout()
            safe_cls = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cls))
            plt.savefig(eda_dir / f"flow_duration_hist_{safe_cls}.png", dpi=120)
            plt.close()
    if packet_col and target_col in df.columns:
        for cls in _sorted_classes(df[target_col]):
            subset = _treat_as_missing(df.loc[df[target_col].astype(str) == cls, packet_col]).dropna()
            if subset.empty:
                continue
            plt.figure(figsize=(8, 4))
            plt.hist(subset, bins=50, alpha=0.85)
            plt.title(f"Packet count histogram — {cls}")
            plt.xlabel(packet_col)
            plt.ylabel("count")
            plt.tight_layout()
            safe_cls = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cls))
            plt.savefig(eda_dir / f"packet_count_hist_{safe_cls}.png", dpi=120)
            plt.close()

    target_counts = df[target_col].astype(str).value_counts()
    min_count = int(target_counts.min()) if not target_counts.empty else 0
    max_count = int(target_counts.max()) if not target_counts.empty else 0
    imbalance_ratio = float(max_count / min_count) if min_count > 0 else float("inf")
    dataset_size = {
        "n_rows": int(len(df)),
        "n_features_numeric": int(len(num_cols)),
        "n_features_categorical": int(len(_categorical_feature_cols(df, target_col))),
        "n_classes": int(target_counts.shape[0]),
        "min_class_count": min_count,
        "max_class_count": max_count,
        "imbalance_ratio": imbalance_ratio,
    }
    write_json(dataset_size, dataset_size_path)

    output_paths = [
        shape_dtypes_path,
        missingness_path,
        numeric_stats_path,
        corr_heatmap_path,
        high_corr_path,
        class_dist_path,
        variance_path,
        dataset_size_path,
        eda_dir / "class_distribution.png",
    ]
    LOGGER.info("EDA complete: %d rows, %d numeric features", len(df), len(num_cols))
    return {
        "output_paths": [str(p) for p in output_paths if Path(p).exists()],
        "rows": int(len(df)),
        "near_zero_variance": near_zero,
        "high_corr_pairs": len(high_corr_pairs),
    }


# --- CLEAN ---
@checkpointed("clean")
def clean(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    integrated_path: Path = params.get("integrated_path", INTEGRATED_PARQUET)
    cleaned_path: Path = params.get("cleaned_path", CLEANED_PARQUET)
    eda_dir: Path = params.get("eda_dir", EDA_DIR)
    decisions_path: Path = params.get("decisions_path", CLEAN_DECISIONS_MD)

    if not integrated_path.exists():
        raise FileNotFoundError(f"Integrated dataset not found: {integrated_path}")

    variance_path = eda_dir / "variance.json"
    high_corr_path = eda_dir / "high_correlation_pairs.json"
    if not variance_path.exists() or not high_corr_path.exists():
        LOGGER.warning("EDA artifacts missing; running eda() before clean().")
        eda({"integrated_path": integrated_path, "eda_dir": eda_dir})

    df = _sanitize_columns(read_parquet(integrated_path))
    target_col = _resolve_target_column(df)
    lines = [
        "# Clean decisions",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    df = df.replace([np.inf, -np.inf], np.nan)
    lines.append("- Replaced `np.inf` / `-np.inf` with `np.nan`.")

    miss_df = _missing_table(df)
    high_missing = miss_df.loc[miss_df["missing_pct"] > MISSING_COL_THRESHOLD * 100, "column"].tolist()
    if high_missing:
        df = df.drop(columns=high_missing)
        lines.append(f"- Dropped columns with >60% missing: {high_missing}")
    else:
        lines.append("- No columns exceeded 60% missing threshold.")

    flow_cols = _flow_feature_columns(df, target_col)
    if flow_cols:
        before = len(df)
        df = df.dropna(subset=flow_cols, how="all")
        dropped_rows = before - len(df)
        lines.append(f"- Dropped {dropped_rows} rows where all flow features were missing.")
    else:
        lines.append("- No flow feature columns detected for all-missing row filter.")

    variance_payload = read_json(variance_path)
    near_zero = variance_payload.get("near_zero_variance", [])
    existing_near_zero = [c for c in near_zero if c in df.columns]
    if existing_near_zero:
        df = df.drop(columns=existing_near_zero)
        lines.append(f"- Dropped near-zero-variance columns from EDA: {existing_near_zero}")

    high_corr_pairs = read_json(high_corr_path)
    num_cols = _numeric_feature_cols(df, target_col)
    variances = {col: _safe_variance(df[col]) for col in num_cols if df[col].notna().any()}
    corr_decisions = _choose_corr_drop(high_corr_pairs, variances)
    corr_drops = [d["dropped"] for d in corr_decisions if d["dropped"] in df.columns]
    if corr_drops:
        df = df.drop(columns=corr_drops)
        lines.append("- Highly correlated feature drops (|corr|>0.95, kept higher variance):")
        for decision in corr_decisions:
            lines.append(f"  - dropped `{decision['dropped']}`, kept `{decision['kept']}` ({decision['reason']})")

    identifier_cols = [c for c in df.columns if _is_identifier_column(c)]
    if identifier_cols:
        df = df.drop(columns=identifier_cols)
        lines.append(
            "- Dropped identifier / potentially leaking columns (default DROP per ethics section): "
            f"{identifier_cols}"
        )
    else:
        lines.append("- No identifier-like columns detected for removal.")

    write_parquet(df, cleaned_path)
    _write_markdown(decisions_path, lines)
    LOGGER.info("Cleaned data written: %s (%d rows, %d cols)", cleaned_path, len(df), df.shape[1])
    return {
        "output_paths": [str(cleaned_path), str(decisions_path)],
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }


# --- SPLIT ---
@checkpointed("split")
def split(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    cleaned_path: Path = params.get("cleaned_path", CLEANED_PARQUET)
    splits_path: Path = params.get("splits_path", SPLITS_JSON)
    split_report_path: Path = params.get("split_report_path", SPLIT_REPORT_MD)
    random_seed: int = int(params.get("random_seed", RANDOM_SEED))

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    df = _sanitize_columns(read_parquet(cleaned_path))
    target_col = _resolve_target_column(df)
    y = df[target_col].astype(str)
    indices = np.arange(len(df))

    stratify = y if y.nunique() > 1 else None
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_seed,
        stratify=stratify,
    )
    y_temp = y.iloc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=random_seed,
        stratify=y_temp if y_temp.nunique() > 1 else None,
    )

    splits_payload = {
        "random_seed": random_seed,
        "target_column": target_col,
        "train": [int(i) for i in train_idx],
        "val": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
        "n_rows": int(len(df)),
    }
    write_json(splits_payload, splits_path)

    lines = [
        "# Split report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Stratified 70/15/15 split on `{target_col}` (random_seed={random_seed})",
        f"- Train indices: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}",
        "",
        "## Per-class counts",
        "",
        "| Class | Train | Val | Test | Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    warnings: list[str] = []
    observed_classes = set(y.unique())
    for cls in THREAT_CLASSES:
        if cls not in observed_classes:
            warnings.append(f"Class `{cls}` not present in dataset.")
            continue
        train_c = int((y.iloc[train_idx] == cls).sum())
        val_c = int((y.iloc[val_idx] == cls).sum())
        test_c = int((y.iloc[test_idx] == cls).sum())
        total_c = int((y == cls).sum())
        lines.append(f"| {cls} | {train_c} | {val_c} | {test_c} | {total_c} |")
        if train_c == 0 or val_c == 0 or test_c == 0:
            warnings.append(
                f"Class `{cls}` missing from at least one split (train={train_c}, val={val_c}, test={test_c}). "
                "Stratification may be unreliable for rare classes — note in dissertation limitations."
            )
        if total_c < 3:
            warnings.append(f"Class `{cls}` has only {total_c} instances; stratified splitting is fragile.")

    if warnings:
        lines.extend(["", "## Warnings", ""])
        for note in warnings:
            lines.append(f"- {note}")
            LOGGER.warning(note)
    _write_markdown(split_report_path, lines)

    LOGGER.info("Split indices saved to %s", splits_path)
    return {
        "output_paths": [str(splits_path), str(split_report_path)],
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "warnings": warnings,
    }


# --- PREPROCESS ---
def _skewness(series: pd.Series) -> float:
    clean = _treat_as_missing(series).dropna()
    if clean.empty:
        return 0.0
    return float(clean.skew())


def _build_preprocessed_frames(
    df: pd.DataFrame,
    train_idx: list[int],
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    feature_cols = _feature_columns(df, target_col)
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    train_df = df.iloc[train_idx].copy()
    decisions: list[str] = [
        "# Preprocess decisions",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Missing data",
        "",
    ]

    imputed = df[feature_cols].copy()
    for col in numeric_cols:
        fit_series = _treat_as_missing(train_df[col])
        skew = _skewness(fit_series)
        strategy = "median" if skew > 1 else "mean"
        fill_value = fit_series.median() if strategy == "median" else fit_series.mean()
        imputed[col] = _treat_as_missing(imputed[col]).fillna(fill_value)
        decisions.append(f"- `{col}` (numeric): {strategy} imputation (train skew={skew:.3f})")
    for col in categorical_cols:
        fit_series = train_df[col].astype(str)
        nunique = fit_series.nunique(dropna=True)
        if nunique < CATEGORICAL_MODE_CARDINALITY:
            mode = fit_series.mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "MISSING"
            strategy = "mode"
        else:
            fill_value = "MISSING"
            strategy = "literal_MISSING"
        imputed[col] = imputed[col].astype(str).fillna(fill_value)
        decisions.append(f"- `{col}` (categorical): {strategy} (cardinality={nunique})")

    decisions.extend(["", "## Encoding", ""])
    low_card = [c for c in categorical_cols if imputed[c].nunique(dropna=True) <= LOW_CARDINALITY_OHE]
    high_card = [c for c in categorical_cols if c not in low_card]

    encoded_parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    one_hot: OneHotEncoder | None = None
    if low_card:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe_train = one_hot.fit_transform(imputed.loc[train_idx, low_card].astype(str))
        ohe_all = one_hot.transform(imputed[low_card].astype(str))
        ohe_cols = one_hot.get_feature_names_out(low_card).tolist()
        encoded_parts.append(pd.DataFrame(ohe_all, columns=ohe_cols, index=imputed.index))
        feature_names.extend(ohe_cols)
        decisions.append(f"- OneHotEncoder on low-cardinality columns (≤{LOW_CARDINALITY_OHE}): {low_card}")

    freq_encoder = FrequencyEncoder()
    if high_card:
        freq_encoder.fit(imputed.loc[train_idx], high_card)
        freq_part = freq_encoder.transform(imputed[high_card])
        encoded_parts.append(freq_part)
        feature_names.extend(high_card)
        decisions.append(
            f"- Frequency encoding on high-cardinality columns (>{LOW_CARDINALITY_OHE}): {high_card}. "
            "TargetEncoder avoided due to SMOTE downstream risk."
        )

    numeric_part = imputed[numeric_cols].copy()
    continuous_cols = numeric_cols.copy()
    feature_names = numeric_cols + feature_names

    if encoded_parts:
        encoded_df = pd.concat(encoded_parts, axis=1)
        unscaled_features = pd.concat([numeric_part, encoded_df], axis=1)
    else:
        unscaled_features = numeric_part.copy()

    unscaled_features = unscaled_features.reindex(columns=feature_names)
    scaler = MinMaxScaler()
    scaler.fit(unscaled_features.iloc[train_idx])
    scaled_features = pd.DataFrame(
        scaler.transform(unscaled_features),
        columns=feature_names,
        index=unscaled_features.index,
    )

    decisions.extend(
        [
            "",
            "## Scaling",
            "",
            "- MinMaxScaler fit on TRAIN only for all model-ready features.",
            "- Unscaled copy retained for tree models (DT, RF, XGBoost).",
        ]
    )

    target = df[target_col].astype(str)
    unscaled_out = unscaled_features.copy()
    unscaled_out[target_col] = target.values
    unscaled_out["threat_label"] = target.values

    scaled_out = scaled_features.copy()
    scaled_out[target_col] = target.values
    scaled_out["threat_label"] = target.values

    artifacts = {
        "feature_names": feature_names,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "low_card_ohe": low_card,
        "high_card_freq": high_card,
        "continuous_cols": continuous_cols,
        "one_hot": one_hot,
        "freq_encoder": freq_encoder,
        "scaler": scaler,
        "train_fit_indices_hash": hashlib.sha256(
            json.dumps(sorted(int(i) for i in train_idx)).encode("utf-8")
        ).hexdigest(),
    }
    return scaled_out, unscaled_out, artifacts, decisions


@checkpointed("preprocess")
def preprocess(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    cleaned_path: Path = params.get("cleaned_path", CLEANED_PARQUET)
    splits_path: Path = params.get("splits_path", SPLITS_JSON)
    eda_dir: Path = params.get("eda_dir", EDA_DIR)
    scaled_path: Path = params.get("scaled_path", TABULAR_SCALED_PARQUET)
    unscaled_path: Path = params.get("unscaled_path", TABULAR_UNSCALED_PARQUET)
    artifacts_dir: Path = params.get("artifacts_dir", PREPROCESS_ARTIFACTS_DIR)
    decisions_path: Path = params.get("decisions_path", PREPROCESS_DECISIONS_MD)

    split_payload = _verify_split_checkpoint()
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Split indices not found: {splits_path}")

    df = _sanitize_columns(read_parquet(cleaned_path))
    splits = read_json(splits_path)
    train_idx = splits["train"]
    target_col = splits.get("target_column", _resolve_target_column(df))

    scaled_df, unscaled_df, artifacts, decisions = _build_preprocessed_frames(df, train_idx, target_col)
    val_idx = splits["val"]
    test_idx = splits["test"]
    write_parquet(scaled_df.iloc[val_idx], VAL_SCALED_PARQUET)
    write_parquet(scaled_df.iloc[test_idx], TEST_SCALED_PARQUET)
    decisions.append("")
    decisions.append(f"- Split checkpoint verified: `{split_payload.get('step_name', 'split')}`")
    decisions.append(f"- EDA directory consulted: `{eda_dir}`")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_joblib(artifacts, artifacts_dir / "preprocess_bundle.joblib")
    write_joblib(artifacts["scaler"], artifacts_dir / "minmax_scaler.joblib")
    write_parquet(scaled_df, scaled_path)
    write_parquet(unscaled_df, unscaled_path)
    _write_markdown(decisions_path, decisions)

    LOGGER.info(
        "Preprocess complete: scaled=%s unscaled=%s features=%d",
        scaled_path,
        unscaled_path,
        len(artifacts["feature_names"]),
    )
    return {
        "output_paths": [
            str(scaled_path),
            str(unscaled_path),
            str(VAL_SCALED_PARQUET),
            str(TEST_SCALED_PARQUET),
            str(artifacts_dir / "preprocess_bundle.joblib"),
            str(artifacts_dir / "minmax_scaler.joblib"),
            str(decisions_path),
        ],
        "n_features": int(len(artifacts["feature_names"])),
    }


# --- BALANCE ---
@checkpointed("balance")
def balance(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    unscaled_path: Path = params.get("unscaled_path", TABULAR_UNSCALED_PARQUET)
    splits_path: Path = params.get("splits_path", SPLITS_JSON)
    dataset_size_path: Path = params.get("dataset_size_path", EDA_DIR / "dataset_size.json")
    train_balanced_path: Path = params.get("train_balanced_path", TRAIN_BALANCED_PARQUET)
    class_weight_path: Path = params.get("class_weight_path", CLASS_WEIGHT_JSON)
    random_seed: int = int(params.get("random_seed", RANDOM_SEED))

    if not unscaled_path.exists():
        raise FileNotFoundError(f"Unscaled tabular data not found: {unscaled_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Split indices not found: {splits_path}")
    if not dataset_size_path.exists():
        LOGGER.warning("dataset_size.json missing; running eda() for balance decisions.")
        eda({})
    size_info = read_json(dataset_size_path)

    df = _sanitize_columns(read_parquet(unscaled_path))
    splits = read_json(splits_path)
    target_col = splits.get("target_column", "threat_class")
    train_idx = splits["train"]

    train_df = df.iloc[train_idx].copy()
    feature_cols = [c for c in train_df.columns if c not in {target_col, "threat_label"}]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(str).to_numpy()

    imbalance_ratio = float(size_info.get("imbalance_ratio", float("inf")))
    min_class_count = int(size_info.get("min_class_count", 0))
    log_lines = [
        "# Balance decisions",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- imbalance_ratio={imbalance_ratio:.3f}, min_class_count={min_class_count}",
        "",
    ]

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {str(cls): float(w) for cls, w in zip(classes, weights)}
    write_json(class_weight, class_weight_path)

    method = "none"
    if imbalance_ratio < 1.5:
        balanced_df = train_df
        log_lines.append("- Branch: imbalance_ratio < 1.5 → SMOTE skipped; saved training split as-is.")
    elif imbalance_ratio < 10 and min_class_count >= 100:
        if SMOTE is None:
            balanced_df = train_df
            log_lines.append("- Branch: SMOTE requested but imbalanced-learn unavailable; saved train as-is.")
        else:
            sampler: Any = SMOTE(random_state=random_seed)
            method = "SMOTE"
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                balanced_df = pd.DataFrame(X_res, columns=feature_cols)
                balanced_df[target_col] = y_res
                balanced_df["threat_label"] = y_res
                log_lines.append(
                    "- Branch: 1.5 ≤ ratio < 10 and min_class_count ≥ 100 → applied SMOTE on TRAIN only."
                )
            except Exception as exc:
                balanced_df = train_df
                log_lines.append(f"- SMOTE failed ({exc}); saved train as-is.")
    else:
        if SMOTETomek is None:
            balanced_df = train_df
            log_lines.append("- Branch: SMOTETomek requested but imbalanced-learn unavailable; saved train as-is.")
        else:
            method = "SMOTETomek"
            sampler = SMOTETomek(random_state=random_seed)
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                balanced_df = pd.DataFrame(X_res, columns=feature_cols)
                balanced_df[target_col] = y_res
                balanced_df["threat_label"] = y_res
                log_lines.append(
                    "- Branch: ratio ≥ 10 OR min_class_count < 100 → applied SMOTETomek on TRAIN only."
                )
            except Exception as exc:
                balanced_df = train_df
                log_lines.append(f"- SMOTETomek failed ({exc}); saved train as-is.")

    log_lines.extend(
        [
            "",
            "## Trade-off note",
            "",
            "- `class_weight.json` saved as an alternative when models support `class_weight` / `scale_pos_weight`.",
            "- SMOTE can improve recall on minority classes but may reduce minority precision; compare both in evaluation.",
            f"- Applied method: `{method}`",
            f"- Train rows before: {len(train_df)} | after: {len(balanced_df)}",
        ]
    )

    write_parquet(balanced_df, train_balanced_path)
    _write_markdown(REPORTS_DIR / "balance_decisions.md", log_lines)
    LOGGER.info("Balance complete: method=%s train_rows=%d", method, len(balanced_df))
    return {
        "output_paths": [str(train_balanced_path), str(class_weight_path), str(REPORTS_DIR / "balance_decisions.md")],
        "method": method,
        "train_rows": int(len(balanced_df)),
    }


def run_default() -> dict[str, Any]:
    eda_res = eda({})
    clean_res = clean({})
    split_res = split({})
    preprocess_res = preprocess({})
    balance_res = balance({})
    return {
        "eda": eda_res,
        "clean": clean_res,
        "split": split_res,
        "preprocess": preprocess_res,
        "balance": balance_res,
    }


if __name__ == "__main__":
    result = run_default()
    loaded = result.get("loaded_from_checkpoint", False)
    state = "checkpoint" if loaded else "executed"
    print(f"src.prep: {state}.")
