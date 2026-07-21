# Model Card

## 1 — Project

This project trains Decision Tree, Random Forest, and XGBoost classifiers on the CICIDS-2017 network traffic dataset (`data/raw/CICIDS2017/`) for eight-class cybersecurity threat detection. Models are evaluated on a held-out test split and explained with SHAP and LIME post-hoc methods.

## 2 — Classification Performance Table

| Model | Accuracy | Macro F1 | Weighted F1 | MCC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| decision_tree | 0.9987 | 0.8421 | 0.9987 | 0.9957 | 0.9032 |
| random_forest | 0.9963 | 0.6985 | 0.9961 | 0.9873 | 0.9261 |
| xgboost | 0.9989 | 0.8518 | 0.9989 | 0.9964 | 0.9650 |

## 3 — Per-Class F1 (XGBoost)

| Threat Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| BENIGN | 0.9996 | 0.9991 | 0.9994 | 322258 |
| Botnet | 0.8225 | 0.7747 | 0.7979 | 293 |
| BruteForce | 1.0000 | 0.9949 | 0.9974 | 1373 |
| DDoS | 1.0000 | 0.9998 | 0.9999 | 19203 |
| DoS | 0.9967 | 0.9988 | 0.9978 | 29062 |
| Infiltration | 1.0000 | 0.8333 | 0.9091 | 6 |
| PortScan | 0.9900 | 0.9995 | 0.9947 | 13623 |
| WebAttack | 0.9809 | 0.9595 | 0.9701 | 321 |

## 4 — XAI Fidelity Results

| Model | k=1 F1 Drop | k=3 F1 Drop | k=5 F1 Drop | k=10 F1 Drop |
| --- | ---: | ---: | ---: | ---: |
| decision_tree | 0.2889 | 0.3563 | 0.3665 | 0.5624 |
| random_forest | 0.0005 | 0.0771 | 0.1791 | 0.5942 |
| xgboost | 0.0461 | 0.2981 | 0.5564 | 0.6406 |

## 5 — XAI Stability Results

Mean LIME Jaccard across 16 local instances: **0.4241**. Mean SHAP Jaccard: **1.0000** (tree SHAP is deterministic).

| Mean LIME Jaccard | Mean LIME Spearman | Mean SHAP Jaccard | Mean SHAP Spearman |
| ---: | ---: | ---: | ---: |
| 0.4241 | 0.7085 | 1.0000 | 1.0000 |

## 6 — XAI Comprehensibility (IoC Alignment)

Mean IoC alignment score averaged across seven attack classes (excluding BENIGN).

| Model | Mean IoC Alignment Score |
| --- | ---: |
| decision_tree | 0.0143 |
| random_forest | 0.0000 |
| xgboost | 0.0571 |

## 7 — Key Findings

- **Best model by Macro F1:** XGBoost (0.8518).
- **More faithful explanations at k=5:** Random Forest (F1 drop 0.1791) vs XGBoost (0.5564); lower drop indicates higher fidelity.
- **More stable explanations:** SHAP (mean Jaccard 1.0000) vs LIME (mean Jaccard 0.4241).
- **Highest IoC alignment:** 0.1000 for DDoS (Decision Tree), and for DoS, BruteForce, Botnet, and Infiltration (XGBoost).
- **Lowest IoC alignment:** 0.0000 for PortScan and WebAttack across all three models.

## 8 — Limitations

- Trained only on CICIDS-2017; generalisation to other network environments not tested.
- Rare classes (Infiltration: 6 test instances, Botnet: 293 test instances) may produce unreliable per-class metrics.
- LIME instability is an inherent limitation; stability scores above reflect this.
- Identifier columns (IPs, ports, timestamps) were dropped per ethics policy; some IoC alignment features may be absent.
- Comprehensibility is assessed against static IoC reference lists; real IoCs evolve.

## 9 — Ethical and Defensive Framing

This system is intended for defensive network monitoring and research only; no operational attack evasion guidance is published. CICIDS-2017 is a publicly available academic benchmark used in accordance with its licence.
