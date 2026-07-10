# Ethics and responsible use

This project is for academic research on explainable cybersecurity classification (MSc dissertation). It has been reviewed under the institution's research ethics process.

## Research ethics approval

- **Research Ethics Committee (REC):** This study uses publicly available, de-identified network-flow metadata only. No human subjects, personal data, or live production traffic are collected. REC approval reference: *[insert your committee reference number before submission]*.

## Legal and professional frameworks

- **UK Data Protection Act 2018:** Raw CICIDS-2017 records contain no directly identifying personal data in the form used here; `data/raw/` is gitignored and must not be republished. Manual demo rows in `data/manual/` must remain de-identified.
- **BCS Code of Conduct:** Analysis and reporting follow the BCS duty to act in the public interest, maintain competence, and avoid harm — including responsible handling of security-sensitive artefacts.

## Scope

- The system supports analysis of network-flow records only.
- Outputs are decision-support signals, not autonomous enforcement actions.

## Data governance

- Raw datasets are stored in `data/raw/` and are never modified.
- Provenance checksums and label mappings are recorded in `reports/provenance.md` and `reports/label_mapping.json`.
- Any manual records in `data/manual/` should be de-identified before use.

## Model risks

- Misclassification can occur for minority attack classes.
- Explanations are post-hoc and may not represent causality.
- Deployment should include human analyst oversight.

## Dual-use mitigation strategy

Cybersecurity models can be misused to craft evasion techniques. This project mitigates dual-use risk as follows:

1. **Defensive framing only** — documentation, model cards, and dissertation narrative describe SOC analyst training and intrusion *detection* awareness, not offensive tooling.
2. **No weight publication** — trained `.joblib` artefacts are gitignored (`artifacts/models/*.joblib`) and kept locally for reproducibility only; they are not pushed to public repositories.
3. **No hyperparameter disclosure for evasion** — tuned hyperparameters that could enable adversarial evasion are not published; aggregate metrics and explanation quality are shared instead.
4. **Human-in-the-loop** — predictions and explanations require analyst interpretation before any operational response.
5. **Dataset licensing** — CICIDS-2017 terms are respected; derived data is not redistributed without verifying licence conditions.

See also `reports/model_card.md` for the operational model card dual-use statement.

## Reproducibility and transparency

- All pipeline steps use checkpoint metadata in `checkpoints/`.
- Environment snapshots (`reports/env.json`, `requirements.lock.txt`) and timing (`reports/timing.json`) support audit and replication.
- Metrics and explanation artefacts are written to `reports/`.
