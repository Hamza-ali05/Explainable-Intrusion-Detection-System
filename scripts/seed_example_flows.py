"""Seed data/manual/example_flows.csv from raw CICIDS-2017 CSVs (one row per threat class)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core import MANUAL_DIR, RAW_DIR, THREAT_CLASSES
from src.data import _map_to_threat_class, _standardize_label_text

TARGET_PER_CLASS = 3
OUTPUT_PATH = MANUAL_DIR / "example_flows.csv"


def _find_label_column(columns: list[str]) -> str | None:
    for col in columns:
        if col.strip().lower() == "label":
            return col
    return None


def _sample_from_file(csv_path: Path, threat_class: str, n: int) -> list[pd.Series]:
    header = pd.read_csv(csv_path, nrows=0)
    label_col = _find_label_column([str(c) for c in header.columns])
    if label_col is None:
        return []

    chunksize = 50_000
    collected: list[pd.Series] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        label_col = _find_label_column(list(chunk.columns)) or label_col
        mapped = chunk[label_col].map(_standardize_label_text).map(_map_to_threat_class)
        subset = chunk.loc[mapped == threat_class]
        for _, row in subset.iterrows():
            collected.append(row)
            if len(collected) >= n:
                return collected
    return collected


def main() -> None:
    cicids_dir = RAW_DIR / "CICIDS2017"
    if not cicids_dir.exists():
        raise SystemExit(f"CICIDS raw directory not found: {cicids_dir}")

    csv_files = sorted(cicids_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files under {cicids_dir}")

    rows: list[pd.Series] = []
    for threat_class in THREAT_CLASSES:
        need = TARGET_PER_CLASS
        for csv_path in csv_files:
            if need <= 0:
                break
            sampled = _sample_from_file(csv_path, threat_class, need)
            rows.extend(sampled)
            need -= len(sampled)

        if need > 0:
            print(f"Warning: only found {TARGET_PER_CLASS - need} rows for {threat_class}")

    if not rows:
        raise SystemExit("No example rows collected.")

    df = pd.DataFrame(rows).reset_index(drop=True)
    if " Label" in df.columns and "Label" not in df.columns:
        df = df.rename(columns={" Label": "Label"})

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    if "Label" in df.columns:
        mapped = df["Label"].map(_standardize_label_text).map(_map_to_threat_class)
        print(mapped.value_counts().to_string())


if __name__ == "__main__":
    main()
