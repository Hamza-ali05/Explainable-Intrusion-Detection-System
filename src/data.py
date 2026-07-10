from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core import (
    INTERIM_DIR,
    LOGGER,
    RAW_DIR,
    REPORTS_DIR,
    THREAT_CLASSES,
    checkpointed,
    read_json,
    read_parquet,
    write_json,
    write_parquet,
)
from src.prep import LABEL_MAP


DISCOVER_MANIFEST_JSON = INTERIM_DIR / "manifest.json"
DISCOVER_MANIFEST_MD = INTERIM_DIR / "manifest.md"
PROVENANCE_MD = REPORTS_DIR / "provenance.md"
LOADED_DIR = INTERIM_DIR / "loaded"
LOAD_INDEX_JSON = INTERIM_DIR / "loaded" / "load_index.json"
LOAD_FAILURES_JSON = INTERIM_DIR / "load_failures.json"
INTEGRATED_PARQUET = INTERIM_DIR / "integrated.parquet"
INTEGRATION_REPORT_MD = INTERIM_DIR / "integration_report.md"
LABEL_MAPPING_JSON = REPORTS_DIR / "label_mapping.json"

UNSW_NB15_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes", "sttl",
    "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]
KDDCUP_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]


# --- DISCOVER ---
def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _strip_columns(columns: list[str]) -> list[str]:
    return [str(col).strip() for col in columns]


def _detect_dataset_type(path: Path, columns: list[str]) -> str:
    lowered_name = path.name.lower()
    colset = {c.lower() for c in _strip_columns(columns)}
    if (
        "label" in colset
        and "flow duration" in colset
    ) or any(day in lowered_name for day in ["monday-workinghours", "tuesday-workinghours", "wednesday-workinghours", "thursday-workinghours", "friday-workinghours"]):
        return "cicids_csv"
    if {"attack_cat", "label"}.issubset(colset):
        return "unsw_nb15_csv"
    if {"protocol_type", "service", "flag"}.issubset(colset) or "kdd" in lowered_name:
        return "kddcup_csv"
    return "unknown"


def _candidate_label_columns(columns: list[str]) -> list[str]:
    ranked: list[str] = []
    for col in columns:
        normalized = re.sub(r"[\s_\-]+", "", col.lower())
        if normalized in {"label", "class", "attackcat", "attack", "target"}:
            ranked.append(col)
    return ranked


def _count_rows(path: Path) -> int:
    rows = 0
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        rows += len(chunk)
    return int(rows)


def _preview_rows(path: Path, n: int = 5) -> list[dict[str, Any]]:
    preview_df = pd.read_csv(path, low_memory=False, nrows=n)
    preview_df.columns = _strip_columns([str(c) for c in preview_df.columns])
    return preview_df.fillna("").to_dict(orient="records")


def _unique_label_values(path: Path, columns: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, low_memory=False)
    actual_cols = [str(c) for c in header.columns]
    stripped_cols = _strip_columns(actual_cols)
    candidates = _candidate_label_columns(stripped_cols)
    if not candidates:
        return []
    label_col = actual_cols[stripped_cols.index(candidates[0])]
    uniques: set[str] = set()
    for chunk in pd.read_csv(path, usecols=[label_col], chunksize=100_000, low_memory=False):
        series = chunk[label_col].dropna().astype(str).str.strip()
        uniques.update(v for v in series.unique() if v)
    return sorted(uniques)


def _render_manifest_md(records: list[dict[str, Any]], target: Path) -> None:
    lines = [
        "# Data Manifest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Relative path | Type | Rows | Cols | Size (bytes) | SHA256 | Labels |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for rec in records:
        labels = ", ".join(rec.get("label_values", []))
        lines.append(
            f"| `{rec['relative_path']}` | `{rec['detected_type']}` | {rec['row_count']} | {rec['column_count']} | {rec['size_bytes']} | `{rec['sha256']}` | {labels} |"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_provenance(records: list[dict[str, Any]], target: Path) -> None:
    dataset_names = sorted({rec["detected_type"] for rec in records if rec["detected_type"] != "unknown"})
    dataset_name = ", ".join(dataset_names) if dataset_names else "unknown"
    is_cicids = "cicids_csv" in dataset_names
    source_url = (
        "https://www.unb.ca/cic/datasets/ids-2017.html" if is_cicids else
        "https://research.unsw.edu.au/projects/unsw-nb15-dataset" if "unsw_nb15_csv" in dataset_names else
        "https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html" if "kddcup_csv" in dataset_names else
        "unknown"
    )
    licence_note = (
        "CICIDS-2017 is provided by the Canadian Institute for Cybersecurity (UNB) for academic "
        "and research use. Verify current licence terms at the source URL before redistribution "
        "or publication of derived data."
        if is_cicids
        else "Verify and retain original dataset licensing terms before publication/distribution."
    )
    if not target.exists():
        header = [
            "# Data provenance",
            "",
            "## Source dataset",
            "",
            "- **Dataset:** CICIDS-2017 (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017)"
            if is_cicids
            else f"- **Dataset:** {dataset_name}",
            f"- **URL:** {source_url}",
            f"- **Licence:** {licence_note}",
            "",
            "## Raw file checksums (SHA-256)",
            "",
            "Checksums are recorded at discovery time for every CSV under `data/raw/`.",
            "",
            "## Label mapping (project taxonomy)",
            "",
            "Raw CICIDS-2017 labels are normalised to the dissertation threat taxonomy. "
            "Observed mappings from integration are appended after the `integrate` step; "
            "see also `reports/label_mapping.json`.",
            "",
            "| Raw label | Threat class |",
            "| --- | --- |",
        ]
        for raw_label in sorted(LABEL_MAP):
            header.append(f"| `{raw_label}` | `{LABEL_MAP[raw_label]}` |")
        header.append("")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(header), encoding="utf-8")

    lines = [
        "",
        f"## Discovery entry ({datetime.now(timezone.utc).isoformat()})",
        "",
        f"- Detected dataset type(s): {dataset_name}",
        f"- Download URL: {source_url}",
        f"- Licence note: {licence_note}",
        "- File checksums:",
    ]
    for rec in records:
        lines.append(f"  - `{rec['relative_path']}`: `{rec['sha256']}`")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _append_label_mapping_to_provenance(mapping: dict[str, str], target: Path = PROVENANCE_MD) -> None:
    if not mapping:
        return
    lines = [
        "",
        f"## Label mapping ({datetime.now(timezone.utc).isoformat()})",
        "",
        "Raw CICIDS-2017 labels are mapped to the project threat taxonomy "
        "(see also `reports/label_mapping.json`).",
        "",
        "| Raw label | Threat class |",
        "| --- | --- |",
    ]
    for raw_label in sorted(mapping):
        lines.append(f"| `{raw_label}` | `{mapping[raw_label]}` |")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@checkpointed("discover")
def discover(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    raw_dir: Path = params.get("raw_dir", RAW_DIR)
    manifest_json_path: Path = params.get("manifest_json_path", DISCOVER_MANIFEST_JSON)
    manifest_md_path: Path = params.get("manifest_md_path", DISCOVER_MANIFEST_MD)
    provenance_path: Path = params.get("provenance_path", PROVENANCE_MD)

    csv_paths = sorted(raw_dir.rglob("*.csv"))
    records: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        try:
            header = pd.read_csv(csv_path, low_memory=False, nrows=0)
            columns = _strip_columns([str(c) for c in header.columns])
            rec = {
                "relative_path": str(csv_path.relative_to(raw_dir)).replace("\\", "/"),
                "size_bytes": int(csv_path.stat().st_size),
                "sha256": _sha256_file(csv_path),
                "row_count": _count_rows(csv_path),
                "column_count": int(len(columns)),
                "detected_type": _detect_dataset_type(csv_path, columns),
                "preview_rows": _preview_rows(csv_path, n=5),
                "label_values": _unique_label_values(csv_path, columns),
            }
            records.append(rec)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Discovery failed for %s: %s", csv_path, exc)
            records.append(
                {
                    "relative_path": str(csv_path.relative_to(raw_dir)).replace("\\", "/"),
                    "size_bytes": int(csv_path.stat().st_size),
                    "sha256": _sha256_file(csv_path),
                    "row_count": 0,
                    "column_count": 0,
                    "detected_type": "unknown",
                    "preview_rows": [],
                    "label_values": [],
                    "error": str(exc),
                }
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "files": records,
    }
    write_json(payload, manifest_json_path)
    _render_manifest_md(records, manifest_md_path)
    _append_provenance(records, provenance_path)
    LOGGER.info("Discovered %d CSV files under %s", len(records), raw_dir)
    return {
        "output_paths": [str(manifest_json_path), str(manifest_md_path), str(provenance_path)],
        "n_files": int(len(records)),
    }


# --- LOAD ---
def _safe_source_name(relative_path: str) -> str:
    normalized = relative_path.replace("\\", "/")
    if normalized.lower().endswith(".csv"):
        normalized = normalized[:-4]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", normalized).strip("_") or "source"


def _find_label_column(columns: list[str]) -> str | None:
    candidates = _candidate_label_columns(columns)
    return candidates[0] if candidates else None


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        coverage = float(converted.notna().sum()) / float(non_null.shape[0])
        if coverage >= 0.95:
            df[col] = converted
    return df


def _load_cicids(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        na_values=["NaN", "nan"],
        keep_default_na=True,
    )
    df.columns = _strip_columns([str(c) for c in df.columns])
    df = df.replace({"Infinity": np.inf, "-Infinity": -np.inf, " inf": np.inf, " -inf": -np.inf})
    return _coerce_numeric_columns(df)


def _read_header_columns(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0, low_memory=False)
    return [str(c) for c in header.columns]


def _load_with_expected_schema(path: Path, expected_columns: list[str], dataset_name: str) -> tuple[pd.DataFrame, str | None]:
    raw_columns = _read_header_columns(path)
    stripped = _strip_columns(raw_columns)
    error_note: str | None = None
    if stripped == expected_columns:
        df = pd.read_csv(path, low_memory=False)
        df.columns = stripped
        return df, None
    # Header does not match expected schema; still read with existing header and log warning.
    df = pd.read_csv(path, low_memory=False)
    df.columns = _strip_columns([str(c) for c in df.columns])
    if len(df.columns) == len(expected_columns):
        error_note = (
            f"{dataset_name} schema mismatch: header names differ from published schema "
            f"(file={path.name})."
        )
    else:
        error_note = (
            f"{dataset_name} schema mismatch: expected {len(expected_columns)} cols, "
            f"found {len(df.columns)} (file={path.name})."
        )
    return df, error_note


@checkpointed("load")
def load(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    raw_dir: Path = params.get("raw_dir", RAW_DIR)
    manifest_json_path: Path = params.get("manifest_json_path", DISCOVER_MANIFEST_JSON)
    loaded_dir: Path = params.get("loaded_dir", LOADED_DIR)
    load_index_path: Path = params.get("load_index_path", LOAD_INDEX_JSON)
    failures_path: Path = params.get("failures_path", LOAD_FAILURES_JSON)

    if manifest_json_path.exists():
        manifest = read_json(manifest_json_path)
    else:
        discover_result = discover({"raw_dir": raw_dir, "manifest_json_path": manifest_json_path})
        manifest = read_json(manifest_json_path)
        LOGGER.info("Manifest generated during load step: %s", discover_result)

    files = manifest.get("files", [])
    loaded_dir.mkdir(parents=True, exist_ok=True)
    old_index = read_json(load_index_path) if load_index_path.exists() else {"files": []}
    old_by_rel = {entry["relative_path"]: entry for entry in old_index.get("files", [])}

    new_index_entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    loaded_count = 0
    skipped_count = 0

    for rec in files:
        relative_path = rec["relative_path"]
        sha256 = rec["sha256"]
        detected_type = rec.get("detected_type", "unknown")
        source_name = _safe_source_name(relative_path)
        parquet_path = loaded_dir / f"{source_name}.parquet"
        source_path = raw_dir / Path(relative_path.replace("/", "\\"))

        old_entry = old_by_rel.get(relative_path)
        unchanged = (
            old_entry is not None
            and old_entry.get("sha256") == sha256
            and Path(old_entry.get("parquet_path", "")).exists()
        )
        if unchanged:
            skipped_count += 1
            new_index_entries.append(old_entry)
            continue

        try:
            schema_note: str | None = None
            if detected_type == "cicids_csv":
                df = _load_cicids(source_path)
            elif detected_type == "unsw_nb15_csv":
                df, schema_note = _load_with_expected_schema(source_path, UNSW_NB15_COLUMNS, "UNSW-NB15")
            elif detected_type == "kddcup_csv":
                df, schema_note = _load_with_expected_schema(source_path, KDDCUP_COLUMNS, "KDDCup")
            else:
                df = pd.read_csv(source_path, low_memory=False)
                df.columns = _strip_columns([str(c) for c in df.columns])

            df["source_file"] = relative_path
            write_parquet(df, parquet_path)
            entry = {
                "relative_path": relative_path,
                "source_name": source_name,
                "detected_type": detected_type,
                "sha256": sha256,
                "parquet_path": str(parquet_path),
                "rows": int(len(df)),
                "columns": [str(c) for c in df.columns],
                "schema_note": schema_note,
            }
            if schema_note:
                LOGGER.warning(schema_note)
            new_index_entries.append(entry)
            loaded_count += 1
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to load %s: %s", source_path, exc)
            failures.append(
                {
                    "relative_path": relative_path,
                    "detected_type": detected_type,
                    "sha256": sha256,
                    "error": str(exc),
                }
            )

    index_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": new_index_entries,
    }
    write_json(index_payload, load_index_path)
    write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "failures": failures,
        },
        failures_path,
    )
    LOGGER.info(
        "Load step finished. loaded=%d skipped=%d failures=%d",
        loaded_count,
        skipped_count,
        len(failures),
    )
    return {
        "output_paths": [str(load_index_path), str(failures_path), str(loaded_dir)],
        "loaded_count": int(loaded_count),
        "skipped_count": int(skipped_count),
        "failure_count": int(len(failures)),
    }


# --- INTEGRATE ---
def _standardize_label_text(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).strip()


def _map_to_threat_class(raw_label: str) -> str:
    label = raw_label.strip()
    lowered = label.lower()
    if lowered == "benign":
        return "BENIGN"
    if lowered == "ddos":
        return "DDoS"
    if lowered in {"portscan", "port scan"}:
        return "PortScan"
    if lowered in {"ftp-patator", "ssh-patator"}:
        return "BruteForce"
    if lowered.startswith("dos "):
        return "DoS"
    if lowered.startswith("web attack"):
        return "WebAttack"
    if lowered == "bot":
        return "Botnet"
    if lowered == "infiltration":
        return "Infiltration"
    return label


def _pick_primary_dataset(entries: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for entry in entries:
        dataset_type = entry.get("detected_type", "unknown")
        counts[dataset_type] = counts.get(dataset_type, 0) + 1
    ranked_types = sorted(
        ((k, v) for k, v in counts.items() if k != "unknown"),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked_types[0][0] if ranked_types else "unknown"


def _render_integration_report(
    report_path: Path,
    primary_dataset: str,
    before_rows: int,
    after_rows: int,
    contributions: dict[str, int],
    raw_distribution: dict[str, int],
    mapped_distribution: dict[str, int],
    schema_notes: list[str],
) -> None:
    lines = [
        "# Integration Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Primary dataset: `{primary_dataset}`",
        f"- Rows before deduplication: {before_rows}",
        f"- Rows after deduplication: {after_rows}",
        "",
        "## Per-file contribution",
        "",
    ]
    for source_file, count in sorted(contributions.items(), key=lambda kv: kv[0]):
        lines.append(f"- `{source_file}`: {count} rows")

    lines.extend(["", "## Class distribution (raw labels)", ""])
    for cls, count in sorted(raw_distribution.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{cls}`: {count}")

    lines.extend(["", "## Class distribution (mapped taxonomy)", ""])
    for cls, count in sorted(mapped_distribution.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- `{cls}`: {count}")

    lines.extend(["", "## Schema alignment notes", ""])
    if schema_notes:
        for note in schema_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No schema mismatches detected.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@checkpointed("integrate")
def integrate(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    load_index_path: Path = params.get("load_index_path", LOAD_INDEX_JSON)
    integrated_path: Path = params.get("integrated_path", INTEGRATED_PARQUET)
    integration_report_path: Path = params.get("integration_report_path", INTEGRATION_REPORT_MD)
    label_mapping_path: Path = params.get("label_mapping_path", LABEL_MAPPING_JSON)

    if not load_index_path.exists():
        load_result = load({})
        LOGGER.info("Load was executed during integrate: %s", load_result)

    load_index = read_json(load_index_path)
    entries = load_index.get("files", [])
    if not entries:
        empty_df = pd.DataFrame(columns=["label", "threat_class"])
        write_parquet(empty_df, integrated_path)
        _render_integration_report(
            integration_report_path,
            "unknown",
            before_rows=0,
            after_rows=0,
            contributions={},
            raw_distribution={},
            mapped_distribution={},
            schema_notes=["No loaded files found."],
        )
        write_json({}, label_mapping_path)
        return {
            "output_paths": [str(integrated_path), str(integration_report_path), str(label_mapping_path)],
            "rows": 0,
        }

    primary_dataset = _pick_primary_dataset(entries)
    primary_entries = [e for e in entries if e.get("detected_type") == primary_dataset]
    if not primary_entries:
        primary_entries = entries

    frames: list[pd.DataFrame] = []
    contributions: dict[str, int] = {}
    schema_notes: list[str] = []
    common_columns: set[str] | None = None

    for entry in primary_entries:
        parquet_path = Path(entry["parquet_path"])
        if not parquet_path.exists():
            schema_notes.append(f"Missing parquet for `{entry['relative_path']}`.")
            continue
        df = read_parquet(parquet_path)
        label_col = _find_label_column([str(c) for c in df.columns])
        if label_col and label_col != "label":
            df = df.rename(columns={label_col: "label"})
        if "label" not in df.columns:
            df["label"] = "UNKNOWN"
            schema_notes.append(f"`{entry['relative_path']}` has no label column; defaulted to UNKNOWN.")
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            prev = set(common_columns)
            common_columns = common_columns.intersection(set(df.columns))
            if set(df.columns) != prev:
                schema_notes.append(
                    f"Schema mismatch for `{entry['relative_path']}`; aligned by intersection."
                )
        contributions[entry["relative_path"]] = int(len(df))
        frames.append(df)

    if not frames:
        raise RuntimeError("Integration failed: no readable parquet files were found for primary dataset.")

    assert common_columns is not None
    keep_cols = sorted(common_columns)
    aligned_frames = [df[keep_cols].copy() for df in frames]
    integrated = pd.concat(aligned_frames, ignore_index=True)

    integrated["label"] = integrated["label"].map(_standardize_label_text)
    raw_distribution_series = integrated["label"].value_counts(dropna=False)
    integrated["threat_class"] = integrated["label"].map(_map_to_threat_class)
    integrated["threat_class"] = integrated["threat_class"].where(
        integrated["threat_class"].isin(THREAT_CLASSES),
        integrated["threat_class"],
    )
    mapped_distribution_series = integrated["threat_class"].value_counts(dropna=False)

    before_rows = int(len(integrated))
    integrated = integrated.drop_duplicates(ignore_index=True)
    after_rows = int(len(integrated))

    observed_mapping: dict[str, str] = {}
    for raw_label, mapped in zip(integrated["label"], integrated["threat_class"]):
        if raw_label not in observed_mapping:
            observed_mapping[raw_label] = str(mapped)

    write_parquet(integrated, integrated_path)
    write_json(observed_mapping, label_mapping_path)
    _append_label_mapping_to_provenance(observed_mapping)
    _render_integration_report(
        report_path=integration_report_path,
        primary_dataset=primary_dataset,
        before_rows=before_rows,
        after_rows=after_rows,
        contributions=contributions,
        raw_distribution={str(k): int(v) for k, v in raw_distribution_series.to_dict().items()},
        mapped_distribution={str(k): int(v) for k, v in mapped_distribution_series.to_dict().items()},
        schema_notes=schema_notes,
    )
    LOGGER.info(
        "Integrated dataset saved to %s (rows before dedup=%d, after=%d).",
        integrated_path,
        before_rows,
        after_rows,
    )
    return {
        "output_paths": [str(integrated_path), str(integration_report_path), str(label_mapping_path)],
        "rows_before_dedup": before_rows,
        "rows_after_dedup": after_rows,
        "primary_dataset": primary_dataset,
    }


def run_default() -> dict[str, Any]:
    discover({})
    load({})
    return integrate({})


if __name__ == "__main__":
    result = run_default()
    loaded = result.get("loaded_from_checkpoint", False)
    state = "checkpoint" if loaded else "executed"
    print(f"src.data: {state}.")

