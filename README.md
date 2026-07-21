# Explainable AI for cybersecurity threat classification

Minimal MSc dissertation project for tabular cyber threat classification and post-hoc explainability on standardized multi-corpus network flows (CICIDS-2017, UNSW-NB15, KDD Cup 99).

## Quick Start

1. Place CICIDS-2017 day CSVs in `data/raw/CICIDS2017/`.
2. `pip install -r requirements.txt`
3. `python -m src.pipeline all`
4. `uvicorn api.main:app --reload --port 8000`
5. `cd frontend && npm install && npm run dev`
6. Open `http://localhost:3000`
7. Results: `reports/model_card.md` and `reports/xai/quality/`

## Layout

- `src/core.py`: configuration, paths, logging, checkpointing.
- `src/data.py`: CSV discovery/loading and shared-schema integration.
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

## Obtaining datasets

Place IDS CSVs under `data/raw/` (subfolders allowed). Supported types:

1. **CICIDS-2017** — [dataset page](https://www.unb.ca/cic/datasets/ids-2017.html); extract day CSVs into `data/raw/CICIDS2017/`.
2. **UNSW-NB15** — CSV edition with `attack_cat` + `label` columns under e.g. `data/raw/UNSW-NB15/`.
3. **KDD Cup 99** — 42-column schema including `protocol_type`, `service`, `flag`, and `label`.

`integrate` projects every supported corpus onto one shared feature schema (`duration`, `src_bytes`, `dst_bytes`, `src_packets`, `dst_packets`, `protocol`, `service`) and maps labels into the project threat taxonomy. Unknown / non-IDS CSVs are skipped.

Run `python -m src.pipeline discover` to record SHA-256 checksums in `reports/provenance.md`.

`data/raw/` is gitignored; only checksums and provenance metadata are committed.

## Usage

### CLI subcommands

**Pipeline steps** (run individually or via `all`):

| Step | Description |
| --- | --- |
| `discover` | Scan raw CSVs, write manifest and provenance |
| `load` | Load raw CSVs to interim parquet |
| `integrate` | Project supported corpora onto shared schema, map labels |
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
