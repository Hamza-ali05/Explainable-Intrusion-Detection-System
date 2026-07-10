# Explainable AI for cybersecurity threat classification

Minimal MSc dissertation project for tabular cyber threat classification and post-hoc explainability on CICIDS-2017 network flows.

## Layout

- `src/core.py`: configuration, paths, logging, checkpointing.
- `src/data.py`: CSV discovery/loading and integration.
- `src/prep.py`: cleaning, taxonomy mapping, split, preprocessing, balancing.
- `src/model.py`: train and evaluate Decision Tree, Random Forest, XGBoost.
- `src/xai.py`: SHAP, LIME, fidelity, stability, comprehensibility metrics.
- `src/pipeline.py`: CLI orchestration, status table, timing, reproducibility snapshots.
- `ui/app.py`: Streamlit interface with tabs in one file.
- `tests/test_all.py`: consolidated test file.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Obtaining CICIDS-2017

1. Visit the [CICIDS-2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html).
2. Download the CSV edition (MachineLearningCSV.zip or per-day PCAP-derived CSVs).
3. Extract the CSV files into `data/raw/CICIDS2017/` (eight daily files, e.g. `Monday-WorkingHours.pcap_ISCX.csv`).
4. Run `python -m src.pipeline discover` to record SHA-256 checksums in `reports/provenance.md`.

`data/raw/` is gitignored; only checksums and provenance metadata are committed.

## Usage

### CLI subcommands

**Pipeline steps** (run individually or via `all`):

| Step | Description |
| --- | --- |
| `discover` | Scan raw CSVs, write manifest and provenance |
| `load` | Load raw CSVs to interim parquet |
| `integrate` | Merge primary dataset, map labels |
| `eda` | Exploratory analysis reports |
| `clean` | Remove identifiers, inf/nan, high-correlation features |
| `split` | Stratified train/val/test indices |
| `preprocess` | Imputation, encoding, unscaled tabular export |
| `balance` | SMOTE on training split only |
| `train` | Tune and fit three tree models |
| `evaluate` | Test metrics, select best model, write model card |
| `explain-global` | Global SHAP (per model) |
| `explain-local` | Local SHAP + LIME (per model) |
| `explanation-quality` | Fidelity, stability, comprehensibility |
| `predict` | Batch predict on held-out test split |

**Orchestration:**

```bash
python -m src.pipeline all              # run all steps in order
python -m src.pipeline status             # checkpoint table (same as Status tab)
python -m src.pipeline reset --step train # delete train + downstream markers (prompts to confirm)
python -m src.pipeline evaluate           # run a single step
```

After a successful `all` run, `reports/timing.json`, `reports/env.json`, and `requirements.lock.txt` are written automatically.

### Streamlit UI

```bash
streamlit run ui/app.py
```

Tabs: **Status** (checkpoint table), **Run** (background pipeline), **Evaluate**, **Explain**, **Predict** (upload CSV or demo `data/manual/example_flows.csv`), **Reports**.

### Seed manual demo flows

```bash
python scripts/seed_example_flows.py
```

Creates `data/manual/example_flows.csv` (~24 rows, 3 per threat class) sampled from raw CICIDS-2017.

## Taxonomy

Threat classes used across the project:

`["BENIGN", "DDoS", "DoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet"]`

## Resume behaviour

Rerunning `python -m src.pipeline all` is safe and idempotent — completed steps are skipped via checkpoint markers in `checkpoints/`.

- To force a single step to rerun, delete `checkpoints/<step>.done`.
- To rerun a step and everything that depends on it: `python -m src.pipeline reset --step <name>` (deletes that marker **and** every downstream marker; prompts for confirmation in an interactive terminal).

## Verify tests

```bash
pytest -q
```

The suite in `tests/test_all.py` covers checkpoint skip/force/reset, cleaning, stratified splits, preprocessing, balancing, leakage detection, model training skip logic, XAI quality checks (SHAP, LIME, fidelity, stability), and CLI status rows.

## Ethics

See `ETHICS.md` for REC approval, UK Data Protection Act 2018 / BCS references, and dual-use mitigation.
