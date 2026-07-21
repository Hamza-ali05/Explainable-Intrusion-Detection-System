from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core import (
    THREAT_CLASSES,
    checkpoint_path,
    checkpointed,
    compute_input_hash,
    is_done,
    mark_done,
)
from src.model import (
    LeakageError,
    leakage_check,
    train,
)
from src.pipeline import cmd_reset, checkpoint_status_rows, STEP_ORDER
from src.prep import (
    _choose_corr_drop,
    _is_identifier_column,
    balance,
    clean,
    eda,
    map_labels,
    preprocess,
    split,
)
from src.xai import (
    _fidelity_for_model,
    _jaccard_sets,
    _lime_explainer_for_data,
    _predict_proba_for_lime,
    _shap_explainer_for_model,
    _stability_for_model,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_project_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "interim": root / "data" / "interim",
        "processed": root / "data" / "processed",
        "artifacts": root / "artifacts",
        "checkpoints": root / "checkpoints",
        "reports": root / "reports",
        "eda": root / "reports" / "eda",
        "models": root / "artifacts" / "models",
        "preprocess": root / "artifacts" / "preprocess",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _make_integrated_df(
    n_per_class: int = 40,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic CICIDS-style table for fast pipeline tests."""
    classes = classes or ["BENIGN", "DDoS", "PortScan"]
    rows: list[dict[str, Any]] = []
    for cls in classes:
        for _ in range(n_per_class):
            base = float(RNG.normal(loc=classes.index(cls) * 3.0, scale=1.0))
            rows.append(
                {
                    "Label": cls,
                    "feat_high_var": base + RNG.normal(0, 0.5),
                    "feat_low_var_corr": base * 0.99 + RNG.normal(0, 0.01),
                    "feat_constant": 7.0,
                    "feat_noisy": float(RNG.normal(0, 2.0)),
                    "feat_with_inf": float("inf") if RNG.random() < 0.05 else float(RNG.normal(0, 1)),
                    "Source IP": f"10.0.0.{RNG.integers(1, 200)}",
                    "Destination Port": int(RNG.integers(1, 65535)),
                    "Timestamp": "2024-01-01 00:00:00",
                }
            )
    return pd.DataFrame(rows)


def _write_integrated(paths: dict[str, Path], df: pd.DataFrame | None = None) -> Path:
    df = df if df is not None else _make_integrated_df()
    out = paths["interim"] / "integrated.parquet"
    df.to_parquet(out, index=False)
    return out


def _prep_params(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "integrated_path": paths["interim"] / "integrated.parquet",
        "cleaned_path": paths["interim"] / "cleaned.parquet",
        "splits_path": paths["processed"] / "splits.json",
        "split_report_path": paths["reports"] / "split_report.md",
        "eda_dir": paths["eda"],
        "decisions_path": paths["reports"] / "clean_decisions.md",
        "scaled_path": paths["processed"] / "tabular_scaled.parquet",
        "unscaled_path": paths["processed"] / "tabular_unscaled.parquet",
        "artifacts_dir": paths["preprocess"],
        "decisions_path_preprocess": paths["reports"] / "preprocess_decisions.md",
        "train_balanced_path": paths["processed"] / "train_balanced.parquet",
        "class_weight_path": paths["artifacts"] / "class_weight.json",
        "dataset_size_path": paths["eda"] / "dataset_size.json",
        "random_seed": 42,
    }


def _run_through_preprocess(paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _patch_globals(paths, monkeypatch)
    params = _prep_params(paths)
    _write_integrated(paths)
    eda({"integrated_path": params["integrated_path"]})
    clean({"integrated_path": params["integrated_path"]})
    split({"cleaned_path": params["cleaned_path"], "random_seed": params["random_seed"]})
    preprocess({"cleaned_path": params["cleaned_path"], "splits_path": params["splits_path"]})
    return params


def _patch_globals(paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.core.CHECKPOINTS_DIR", paths["checkpoints"])
    monkeypatch.setattr("src.core.ARTIFACTS_DIR", paths["artifacts"])
    monkeypatch.setattr("src.prep.CHECKPOINTS_DIR", paths["checkpoints"], raising=False)
    monkeypatch.setattr("src.prep.ARTIFACTS_DIR", paths["artifacts"])
    monkeypatch.setattr("src.prep.PREPROCESS_ARTIFACTS_DIR", paths["preprocess"])
    monkeypatch.setattr("src.model.CHECKPOINTS_DIR", paths["checkpoints"])
    monkeypatch.setattr("src.model.ARTIFACTS_DIR", paths["artifacts"])
    monkeypatch.setattr("src.prep.INTEGRATED_PARQUET", paths["interim"] / "integrated.parquet")
    monkeypatch.setattr("src.prep.CLEANED_PARQUET", paths["interim"] / "cleaned.parquet")
    monkeypatch.setattr("src.prep.SPLITS_JSON", paths["processed"] / "splits.json")
    monkeypatch.setattr("src.prep.EDA_DIR", paths["eda"])
    monkeypatch.setattr("src.prep.TABULAR_SCALED_PARQUET", paths["processed"] / "tabular_scaled.parquet")
    monkeypatch.setattr("src.prep.TABULAR_UNSCALED_PARQUET", paths["processed"] / "tabular_unscaled.parquet")
    monkeypatch.setattr("src.prep.VAL_SCALED_PARQUET", paths["processed"] / "val_scaled.parquet")
    monkeypatch.setattr("src.prep.TRAIN_BALANCED_PARQUET", paths["processed"] / "train_balanced.parquet")
    monkeypatch.setattr("src.model.SPLITS_JSON", paths["processed"] / "splits.json")
    monkeypatch.setattr("src.model.CLEANED_PARQUET", paths["interim"] / "cleaned.parquet")
    monkeypatch.setattr("src.model.TABULAR_UNSCALED_PARQUET", paths["processed"] / "tabular_unscaled.parquet")
    monkeypatch.setattr("src.model.TRAIN_BALANCED_PARQUET", paths["processed"] / "train_balanced.parquet")
    monkeypatch.setattr("src.model.MODELS_DIR", paths["models"])
    monkeypatch.setattr("src.model.PREPROCESS_BUNDLE_PATH", paths["preprocess"] / "preprocess_bundle.joblib")
    monkeypatch.setattr("src.model.PREPROCESS_DIR", paths["preprocess"])
    monkeypatch.setattr("src.xai.TABULAR_UNSCALED_PARQUET", paths["processed"] / "tabular_unscaled.parquet")
    monkeypatch.setattr("src.xai.TRAIN_BALANCED_PARQUET", paths["processed"] / "train_balanced.parquet")
    monkeypatch.setattr("src.xai.MODELS_DIR", paths["models"])
    monkeypatch.setattr("src.xai.PREPROCESS_BUNDLE_PATH", paths["preprocess"] / "preprocess_bundle.joblib")


# --- core (existing) ---
def test_input_hash_changes_with_params(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("abc", encoding="utf-8")
    h1 = compute_input_hash([file_path], {"a": 1})
    h2 = compute_input_hash([file_path], {"a": 2})
    assert h1 != h2


def test_label_mapping_to_taxonomy() -> None:
    df = pd.DataFrame({"Label": ["BENIGN", "DoS Hulk", "Bot", "UnknownAttack"]})
    mapped = map_labels(df)
    assert set(mapped["threat_label"].unique()).issubset(set(THREAT_CLASSES))
    assert mapped["threat_label"].iloc[1] == "DoS"
    assert mapped["threat_label"].iloc[2] == "Botnet"


# --- checkpoint ---
def test_checkpoint_create_skip_force_and_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_project_dirs(tmp_path / "ckpt")
    monkeypatch.setattr("src.core.CHECKPOINTS_DIR", paths["checkpoints"])
    monkeypatch.setattr("src.pipeline.CHECKPOINTS_DIR", paths["checkpoints"])

    input_file = paths["root"] / "input.txt"
    input_file.write_text("data", encoding="utf-8")
    out_file = paths["root"] / "output.txt"
    params = {"input_path": input_file}
    input_hash = compute_input_hash([input_file], params)
    calls = {"n": 0}

    @checkpointed("demo_step")
    def demo_step(p: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        out_file.write_text("done", encoding="utf-8")
        return {"output_paths": [str(out_file)], "value": calls["n"]}

    first = demo_step(params)
    assert calls["n"] == 1
    assert first["value"] == 1
    assert checkpoint_path("demo_step").exists()

    second = demo_step(params)
    assert calls["n"] == 1
    assert second.get("loaded_from_checkpoint") is True

    third = demo_step(params, force=True)
    assert calls["n"] == 2
    assert third["value"] == 2

    # downstream reset: mark demo + downstream markers, reset from demo
    for step in ("demo_step", "split", "preprocess", "train"):
        mark_done(step, input_hash, [out_file], params)
    assert all(checkpoint_path(s).exists() for s in ("demo_step", "split", "preprocess", "train"))

    reset_info = cmd_reset("split", confirm=False)
    assert "split" in reset_info["deleted_markers"]
    assert "preprocess" in reset_info["deleted_markers"]
    assert "train" in reset_info["deleted_markers"]
    assert not checkpoint_path("split").exists()
    assert checkpoint_path("demo_step").exists()

    done, payload = is_done("demo_step", input_hash)
    assert done and payload is not None


# --- clean ---
def test_clean_inf_nan_identifiers_variance_and_correlation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_project_dirs(tmp_path / "clean")
    _patch_globals(paths, monkeypatch)
    params = _prep_params(paths)

    df = _make_integrated_df(n_per_class=30)
    integrated = _write_integrated(paths, df)

    eda({"integrated_path": integrated, "eda_dir": params["eda_dir"]})
    result = clean(
        {
            "integrated_path": integrated,
            "cleaned_path": params["cleaned_path"],
            "eda_dir": params["eda_dir"],
            "decisions_path": params["decisions_path"],
        }
    )

    cleaned = pd.read_parquet(params["cleaned_path"])
    assert not np.isinf(cleaned.select_dtypes(include=[np.number])).any().any()

    for col in ("Source IP", "Destination Port", "Timestamp"):
        assert col not in cleaned.columns

    assert "feat_constant" not in cleaned.columns

    high_corr = json.loads((params["eda_dir"] / "high_correlation_pairs.json").read_text(encoding="utf-8"))
    num_cols = [c for c in cleaned.columns if c not in {"Label", "threat_class", "threat_label", "source_file"}]
    variances = {c: float(cleaned[c].var()) for c in num_cols if pd.api.types.is_numeric_dtype(cleaned[c])}
    decisions = _choose_corr_drop(high_corr, variances)
    for decision in decisions:
        assert decision["dropped"] not in cleaned.columns

    assert result["rows"] > 0


def test_identifier_column_detection() -> None:
    assert _is_identifier_column("Source IP")
    assert _is_identifier_column("dst port")
    assert not _is_identifier_column("Total Fwd Packets")


# --- split ---
def test_split_stratification_disjoint_and_class_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    paths = _make_project_dirs(tmp_path / "split")
    _patch_globals(paths, monkeypatch)
    params = _prep_params(paths)

    _write_integrated(paths)
    eda({"integrated_path": params["integrated_path"], "eda_dir": params["eda_dir"]})
    clean(
        {
            "integrated_path": params["integrated_path"],
            "cleaned_path": params["cleaned_path"],
            "eda_dir": params["eda_dir"],
        }
    )

    with caplog.at_level(logging.WARNING):
        result = split(
            {
                "cleaned_path": params["cleaned_path"],
                "splits_path": params["splits_path"],
                "random_seed": 42,
            }
        )

    assert any("not present in dataset" in rec.message for rec in caplog.records)

    splits = json.loads(params["splits_path"].read_text(encoding="utf-8"))
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train) + len(val) + len(test) == splits["n_rows"]

    cleaned = pd.read_parquet(params["cleaned_path"])
    target = splits["target_column"]
    full_counts = cleaned[target].astype(str).value_counts(normalize=True)
    for idx_set in (train, val, test):
        subset = cleaned.iloc[list(idx_set)][target].astype(str).value_counts(normalize=True)
        for cls in full_counts.index:
            if cls in subset.index:
                assert abs(subset[cls] - full_counts[cls]) < 0.08

    observed = set(cleaned[target].astype(str).unique())
    for cls in observed:
        assert (cleaned.iloc[list(train)][target].astype(str) == cls).any()
        assert (cleaned.iloc[list(val)][target].astype(str) == cls).any()
        assert (cleaned.iloc[list(test)][target].astype(str) == cls).any()

    assert result["train_size"] > 0


# --- preprocess ---
def test_preprocess_train_fit_and_scaled_unscaled_artefacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_project_dirs(tmp_path / "preprocess")
    params = _run_through_preprocess(paths, monkeypatch)

    assert params["scaled_path"].exists()
    assert params["unscaled_path"].exists()
    assert (paths["processed"] / "val_scaled.parquet").exists()

    splits = json.loads(params["splits_path"].read_text(encoding="utf-8"))
    scaled = pd.read_parquet(params["scaled_path"])
    unscaled = pd.read_parquet(params["unscaled_path"])
    target = splits["target_column"]
    feature_cols = [c for c in unscaled.columns if c not in {target, "threat_label"}]

    from src.core import read_joblib

    bundle = read_joblib(paths["preprocess"] / "preprocess_bundle.joblib")
    scaler = bundle["scaler"]
    train_idx = splits["train"]
    val_idx = splits["val"]

    train_fit_hash = bundle["train_fit_indices_hash"]
    expected_hash = __import__("hashlib").sha256(
        json.dumps(sorted(int(i) for i in train_idx)).encode("utf-8")
    ).hexdigest()
    assert train_fit_hash == expected_hash

    val_manual = scaler.transform(unscaled.iloc[val_idx][feature_cols])
    val_saved = scaled.iloc[val_idx][feature_cols].to_numpy()
    np.testing.assert_allclose(val_saved, val_manual, rtol=1e-5, atol=1e-5)


# --- balance ---
def test_balance_skips_smote_when_ratio_low_and_preserves_val_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_project_dirs(tmp_path / "balance")
    params = _run_through_preprocess(paths, monkeypatch)

    dataset_size = {"imbalance_ratio": 1.2, "min_class_count": 50}
    (paths["eda"] / "dataset_size.json").write_text(json.dumps(dataset_size), encoding="utf-8")

    splits_before = json.loads(params["splits_path"].read_text(encoding="utf-8"))
    unscaled_before = pd.read_parquet(params["unscaled_path"])

    result = balance(
        {
            "unscaled_path": params["unscaled_path"],
            "splits_path": params["splits_path"],
            "dataset_size_path": params["dataset_size_path"],
            "random_seed": 42,
        }
    )
    assert result["method"] == "none"

    splits_after = json.loads(params["splits_path"].read_text(encoding="utf-8"))
    assert splits_after == splits_before
    assert pd.read_parquet(params["unscaled_path"]).equals(unscaled_before)

    train_idx = splits_before["train"]
    target = splits_before["target_column"]
    orig = unscaled_before.iloc[train_idx]
    balanced = pd.read_parquet(params["train_balanced_path"])
    orig_ratio = orig[target].astype(str).value_counts(normalize=True).sort_index()
    bal_ratio = balanced[target].astype(str).value_counts(normalize=True).sort_index()
    pd.testing.assert_series_equal(orig_ratio, bal_ratio)


# --- leakage ---
def test_leakage_detects_train_test_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_project_dirs(tmp_path / "leakage")
    params = _run_through_preprocess(paths, monkeypatch)

    splits = json.loads(params["splits_path"].read_text(encoding="utf-8"))
    splits["test"] = [splits["train"][0], *splits["test"][:5]]
    params["splits_path"].write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(LeakageError, match="Split leakage detected"):
        leakage_check()


# --- train ---
def test_train_all_models_and_skip_on_second_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("xgboost")
    paths = _make_project_dirs(tmp_path / "train")
    params = _run_through_preprocess(paths, monkeypatch)

    dataset_size = {"imbalance_ratio": 1.2, "min_class_count": 50}
    (paths["eda"] / "dataset_size.json").write_text(json.dumps(dataset_size), encoding="utf-8")
    balance(
        {
            "unscaled_path": params["unscaled_path"],
            "splits_path": params["splits_path"],
            "dataset_size_path": params["dataset_size_path"],
        }
    )

    train_params = {
        "train_path": params["train_balanced_path"],
        "unscaled_path": params["unscaled_path"],
        "random_seed": 42,
    }
    first = train(train_params)
    assert first.get("loaded_from_checkpoint") is not True
    for name in ("decision_tree", "random_forest", "xgboost"):
        assert (paths["models"] / f"{name}.joblib").exists()

    second = train(train_params)
    assert second.get("loaded_from_checkpoint") is True


# --- xai ---
def test_xai_shap_lime_fidelity_and_stability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("shap")
    pytest.importorskip("lime")
    pytest.importorskip("xgboost")

    paths = _make_project_dirs(tmp_path / "xai")
    params = _run_through_preprocess(paths, monkeypatch)

    dataset_size = {"imbalance_ratio": 1.2, "min_class_count": 50}
    (paths["eda"] / "dataset_size.json").write_text(json.dumps(dataset_size), encoding="utf-8")
    balance(
        {
            "unscaled_path": params["unscaled_path"],
            "splits_path": params["splits_path"],
            "dataset_size_path": params["dataset_size_path"],
        }
    )
    train(
        {
            "train_path": params["train_balanced_path"],
            "unscaled_path": params["unscaled_path"],
            "random_seed": 42,
        }
    )

    monkeypatch.setattr("src.xai.STABILITY_INSTANCES", 8)
    monkeypatch.setattr("src.xai.STABILITY_LIME_RUNS", 3)
    monkeypatch.setattr("src.xai.ABLATION_K_VALUES", [1, 3, 5])

    from src.core import read_joblib
    from src.xai import _load_data_context

    ctx = _load_data_context(params["unscaled_path"])
    sample_idx = np.arange(min(12, len(ctx["y_test"])))
    X_sample = ctx["X_test"].iloc[sample_idx]
    n_samples = len(X_sample)
    n_features = len(ctx["feature_names"])

    for model_name in ("decision_tree", "random_forest", "xgboost"):
        model = read_joblib(paths["models"] / f"{model_name}.joblib")
        explainer = _shap_explainer_for_model(model, model_name)
        shap_values = explainer.shap_values(np.asarray(X_sample, dtype=np.float64))
        n_classes = len(model.classes_)

        if isinstance(shap_values, list):
            assert len(shap_values) == n_classes
            for sv in shap_values:
                assert sv.shape == (n_samples, n_features)
        else:
            arr = np.asarray(shap_values)
            assert arr.shape == (n_samples, n_features, n_classes)

        lime_explainer = _lime_explainer_for_data(ctx["X_train"], [str(c) for c in model.classes_])
        row = np.asarray(X_sample.iloc[[0]], dtype=np.float64).ravel()
        exp = lime_explainer.explain_instance(
            data_row=row,
            predict_fn=_predict_proba_for_lime(model, ctx["feature_names"]),
            num_features=10,
            top_labels=1,
        )
        pairs = exp.as_list(label=exp.top_labels[0])[:10]
        assert len(pairs) > 0

        if model_name in ("decision_tree", "random_forest"):
            global_dir = paths["reports"] / "xai" / model_name / "global"
            global_dir.mkdir(parents=True, exist_ok=True)
            np.save(global_dir / "shap_values.npy", shap_values, allow_pickle=True)

            fid_df = _fidelity_for_model(model_name, ctx, global_dir)
            drops = fid_df.sort_values("k_removed")["f1_drop"].to_numpy()
            assert drops[-1] >= drops[0] - 0.05
            assert np.nanmax(drops) >= 0.0

        stab_df = _stability_for_model(model_name, ctx)
        for col in ("jaccard_mean", "perturbed_jaccard_mean"):
            values = stab_df[col].dropna().to_numpy()
            assert len(values) > 0
            assert np.all((values >= 0.0) & (values <= 1.0))

    assert callable(_jaccard_sets)
    assert _jaccard_sets({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_checkpoint_status_rows_match_step_order() -> None:
    rows = checkpoint_status_rows()
    assert [row["Step"] for row in rows] == STEP_ORDER
    assert all(row["Status"] in {"Done", "Pending"} for row in rows)


def test_detect_dataset_type_requires_kdd_schema() -> None:
    from src.data import _detect_dataset_type

    assert _detect_dataset_type(Path("kdd_fake.csv"), ["foo", "bar", "label"]) == "unknown"
    assert (
        _detect_dataset_type(
            Path("kdd_train.csv"),
            ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "label"],
        )
        == "kddcup_csv"
    )


def test_map_threat_class_multi_corpus() -> None:
    from src.prep import map_threat_class

    assert map_threat_class("Heartbleed") == "DoS"
    assert map_threat_class("neptune.") == "DoS"
    assert map_threat_class("Normal") == "BENIGN"
    assert map_threat_class("Reconnaissance") == "PortScan"
    assert map_threat_class("guess_passwd") == "BruteForce"


def test_project_and_integrate_shared_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.data import (
        SHARED_FEATURE_COLUMNS,
        _project_to_shared_schema,
        integrate,
    )
    from src.core import write_json, write_parquet

    paths = _make_project_dirs(tmp_path)
    loaded_dir = paths["interim"] / "loaded"
    loaded_dir.mkdir(parents=True, exist_ok=True)

    cicids = pd.DataFrame(
        {
            "Flow Duration": [100, 200],
            "Total Length of Fwd Packets": [10, 20],
            "Total Length of Bwd Packets": [5, 15],
            "Total Fwd Packets": [2, 4],
            "Total Backward Packets": [1, 3],
            "Label": ["BENIGN", "DDoS"],
        }
    )
    unsw = pd.DataFrame(
        {
            "dur": [1.0],
            "sbytes": [100],
            "dbytes": [50],
            "Spkts": [3],
            "Dpkts": [2],
            "proto": ["tcp"],
            "service": ["http"],
            "attack_cat": ["DoS"],
            "label": [1],
        }
    )
    kdd = pd.DataFrame(
        {
            "duration": [0],
            "protocol_type": ["tcp"],
            "service": ["http"],
            "flag": ["SF"],
            "src_bytes": [100],
            "dst_bytes": [200],
            "label": ["neptune."],
        }
    )
    fraud = pd.DataFrame({"# comment": [1], "Unnamed: 1": [2]})

    cicids_pq = loaded_dir / "cicids.parquet"
    unsw_pq = loaded_dir / "unsw.parquet"
    kdd_pq = loaded_dir / "kdd.parquet"
    fraud_pq = loaded_dir / "fraud.parquet"
    write_parquet(cicids, cicids_pq)
    write_parquet(unsw, unsw_pq)
    write_parquet(kdd, kdd_pq)
    write_parquet(fraud, fraud_pq)

    projected = _project_to_shared_schema(cicids, "cicids_csv", "cicids.csv")
    assert list(projected.columns) == [
        "duration",
        "src_bytes",
        "dst_bytes",
        "src_packets",
        "dst_packets",
        "protocol",
        "service",
        "label",
        "threat_class",
        "source_file",
        "dataset_source",
    ]
    assert projected["threat_class"].tolist() == ["BENIGN", "DDoS"]

    load_index = {
        "files": [
            {
                "relative_path": "CICIDS2017/a.csv",
                "detected_type": "cicids_csv",
                "parquet_path": str(cicids_pq),
            },
            {
                "relative_path": "UNSW/b.csv",
                "detected_type": "unsw_nb15_csv",
                "parquet_path": str(unsw_pq),
            },
            {
                "relative_path": "KDD/c.csv",
                "detected_type": "kddcup_csv",
                "parquet_path": str(kdd_pq),
            },
            {
                "relative_path": "KDD Cup 99/fraud.csv",
                "detected_type": "unknown",
                "parquet_path": str(fraud_pq),
            },
        ]
    }
    index_path = paths["interim"] / "load_index.json"
    write_json(load_index, index_path)
    integrated_path = paths["interim"] / "integrated.parquet"
    report_path = paths["interim"] / "integration_report.md"
    mapping_path = paths["reports"] / "label_mapping.json"

    monkeypatch.setattr("src.core.CHECKPOINTS_DIR", paths["checkpoints"])
    monkeypatch.setattr("src.data.PROVENANCE_MD", paths["reports"] / "provenance.md")
    (paths["reports"] / "provenance.md").write_text("# Data provenance\n", encoding="utf-8")

    result = integrate(
        {
            "load_index_path": index_path,
            "integrated_path": integrated_path,
            "integration_report_path": report_path,
            "label_mapping_path": mapping_path,
        }
    )
    assert set(result["merged_types"]) == {"cicids_csv", "unsw_nb15_csv", "kddcup_csv"}
    out = pd.read_parquet(integrated_path)
    assert set(SHARED_FEATURE_COLUMNS).issubset(out.columns)
    assert "fraud.csv" not in out["source_file"].astype(str).str.cat(sep=" ")
    assert set(out["dataset_source"]) == {"cicids", "unsw", "kdd"}
    assert set(out["threat_class"]).issubset(set(THREAT_CLASSES))
    assert len(out) == 4  # 2 cicids + 1 unsw + 1 kdd
