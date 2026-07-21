# XAI Quality Summary

Generated: 2026-07-21

## Fidelity

Feature-ablation fidelity measures how much test-set macro F1 drops when the top-k SHAP-ranked features are replaced with training-set means. Random Forest shows the smallest F1 drop at k=5 (0.1791), indicating its top SHAP features are most aligned with decision-critical inputs. XGBoost exhibits a larger k=5 drop (0.5564) despite higher classification accuracy, suggesting its predictions depend on a broader feature set. Decision Tree shows the largest sensitivity overall (k=5 drop 0.3665; k=10 drop 0.5624).

## Stability

LIME explanations were run five times per instance on 16 selected test flows; SHAP was re-run five times as a deterministic reference. Mean LIME Jaccard similarity is 0.4241, while SHAP Jaccard is 1.0000 across all instances. LIME rank correlation (mean Spearman 0.7085) is higher than set overlap but still well below SHAP consistency, confirming that tree-based SHAP is the stability baseline and LIME varies between runs.

## Comprehensibility

IoC alignment scores the fraction of class-specific top-10 SHAP features that match static cybersecurity indicator reference lists. Mean alignment is low overall (XGBoost highest at 0.0571 across seven attack classes). Packet-statistics features dominate SHAP rankings more often than canonical IoC names such as Flow Duration or Destination Port, especially for PortScan and WebAttack (0.0000 alignment across all models).

---

## Fidelity Table

| model | k_ablated | baseline_accuracy | ablated_accuracy | accuracy_drop | baseline_f1 | ablated_f1 | f1_drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decision_tree | 1 | 0.9987 | 0.8081 | 0.1906 | 0.8421 | 0.5532 | 0.2889 |
| decision_tree | 3 | 0.9987 | 0.8812 | 0.1176 | 0.8421 | 0.4858 | 0.3563 |
| decision_tree | 5 | 0.9987 | 0.9007 | 0.0980 | 0.8421 | 0.4756 | 0.3665 |
| decision_tree | 10 | 0.9987 | 0.8349 | 0.1639 | 0.8421 | 0.2797 | 0.5624 |
| random_forest | 1 | 0.9963 | 0.9959 | 0.0004 | 0.6985 | 0.6980 | 0.0005 |
| random_forest | 3 | 0.9963 | 0.9872 | 0.0091 | 0.6985 | 0.6214 | 0.0771 |
| random_forest | 5 | 0.9963 | 0.9570 | 0.0393 | 0.6985 | 0.5194 | 0.1791 |
| random_forest | 10 | 0.9963 | 0.8356 | 0.1606 | 0.6985 | 0.1043 | 0.5942 |
| xgboost | 1 | 0.9989 | 0.9874 | 0.0115 | 0.8518 | 0.8058 | 0.0461 |
| xgboost | 3 | 0.9989 | 0.8980 | 0.1010 | 0.8518 | 0.5537 | 0.2981 |
| xgboost | 5 | 0.9989 | 0.8596 | 0.1393 | 0.8518 | 0.2954 | 0.5564 |
| xgboost | 10 | 0.9989 | 0.8371 | 0.1619 | 0.8518 | 0.2112 | 0.6406 |

## Stability Table

| instance_idx | true_class | lime_jaccard_mean | lime_spearman_mean | shap_jaccard_mean | shap_spearman_mean |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1161188 | BENIGN | 0.4234 | 0.5964 | 1.0000 | 1.0000 |
| 800461 | BENIGN | 0.4253 | 0.7921 | 1.0000 | 1.0000 |
| 574999 | Botnet | 0.6324 | 0.8393 | 1.0000 | 1.0000 |
| 619963 | Botnet | 0.3632 | 0.6386 | 1.0000 | 1.0000 |
| 1555343 | BruteForce | 0.4645 | 0.7636 | 1.0000 | 1.0000 |
| 1571040 | BruteForce | 0.3619 | 0.7333 | 1.0000 | 1.0000 |
| 141306 | DDoS | 0.6195 | 0.7529 | 1.0000 | 1.0000 |
| 166274 | DDoS | 0.5701 | 0.8138 | 1.0000 | 1.0000 |
| 2030800 | DoS | 0.5659 | 0.8593 | 1.0000 | 1.0000 |
| 2184568 | DoS | 0.4220 | 0.9329 | 1.0000 | 1.0000 |
| 1199613 | Infiltration | 0.4053 | 0.8993 | 1.0000 | 1.0000 |
| 1209787 | Infiltration | 0.4074 | 0.5638 | 1.0000 | 1.0000 |
| 311163 | PortScan | 0.4205 | 0.9036 | 1.0000 | 1.0000 |
| 407164 | PortScan | 0.4645 | 0.5536 | 1.0000 | 1.0000 |
| 1457232 | WebAttack | 0.2436 | 0.2457 | 1.0000 | 1.0000 |
| 1467591 | WebAttack | 0.4520 | 0.8493 | 1.0000 | 1.0000 |

## Comprehensibility Table

| model | threat_class | ioc_alignment_score | top10_features |
| --- | --- | ---: | --- |
| decision_tree | DDoS | 0.1000 | Bwd Packet Length Std, Fwd Packet Length Max, Total Length of Fwd Packets, Average Packet Size, Packet Length Std, Init_Win_bytes_forward, Bwd Packet Length Mean, Fwd Packet Length Mean, Subflow Bwd Packets, Flow Duration |
| decision_tree | DoS | 0.0000 | Bwd Packet Length Std, Fwd Packet Length Max, Bwd Packet Length Mean, Total Length of Bwd Packets, Bwd Header Length, Packet Length Mean, Init_Win_bytes_forward, Init_Win_bytes_backward, Average Packet Size, Idle Min |
| decision_tree | PortScan | 0.0000 | Average Packet Size, Packet Length Std, Flow IAT Std, Fwd Packet Length Mean, PSH Flag Count, Bwd IAT Std, Bwd Packet Length Std, Fwd IAT Mean, Fwd Packet Length Std, min_seg_size_forward |
| decision_tree | BruteForce | 0.0000 | Bwd Header Length, Bwd Packet Length Max, Max Packet Length, Average Packet Size, SYN Flag Count, Packet Length Std, Bwd Packet Length Std, Fwd IAT Std, Bwd IAT Std, Packet Length Variance |
| decision_tree | WebAttack | 0.0000 | Average Packet Size, Fwd Header Length, Fwd IAT Min, Init_Win_bytes_forward, Init_Win_bytes_backward, Bwd Packets/s, Packet Length Std, Flow IAT Mean, Total Length of Fwd Packets, Bwd Packet Length Std |
| decision_tree | Botnet | 0.0000 | Init_Win_bytes_backward, Total Length of Fwd Packets, Average Packet Size, Bwd Packet Length Std, Bwd Header Length, SYN Flag Count, Bwd Packet Length Mean, Total Length of Bwd Packets, Bwd IAT Min, Fwd Header Length |
| decision_tree | Infiltration | 0.0000 | Subflow Bwd Packets, Total Length of Bwd Packets, SYN Flag Count, act_data_pkt_fwd, Average Packet Size, Bwd Packet Length Std, Bwd Header Length, Fwd Packet Length Min, Avg Bwd Segment Size, Packet Length Std |
| random_forest | DDoS | 0.0000 | Fwd Packet Length Max, Total Length of Fwd Packets, Packet Length Std, Avg Bwd Segment Size, Bwd Packet Length Std, Packet Length Variance, Average Packet Size, Bwd Packet Length Max, Fwd Packet Length Mean, Avg Fwd Segment Size |
| random_forest | DoS | 0.0000 | Bwd Packet Length Std, Packet Length Std, Packet Length Variance, Avg Bwd Segment Size, Bwd Packet Length Max, Bwd Packet Length Mean, Fwd Packet Length Max, Init_Win_bytes_backward, Fwd IAT Std, Max Packet Length |
| random_forest | PortScan | 0.0000 | Packet Length Std, Average Packet Size, Packet Length Variance, PSH Flag Count, Total Length of Fwd Packets, Packet Length Mean, Total Length of Bwd Packets, min_seg_size_forward, Fwd Packet Length Max, Bwd Packet Length Min |
| random_forest | BruteForce | 0.0000 | Bwd Header Length, Total Backward Packets, Subflow Bwd Packets, Average Packet Size, Packet Length Variance, Packet Length Std, Avg Fwd Segment Size, Total Length of Bwd Packets, Packet Length Mean, Total Length of Fwd Packets |
| random_forest | WebAttack | 0.0000 | Fwd IAT Min, Flow IAT Min, Fwd Header Length, Average Packet Size, Init_Win_bytes_backward, Init_Win_bytes_forward, Total Length of Fwd Packets, Fwd IAT Mean, Fwd IAT Std, Flow IAT Std |
| random_forest | Botnet | 0.0000 | Avg Bwd Segment Size, Bwd Packet Length Mean, Init_Win_bytes_backward, Bwd IAT Min, Average Packet Size, Fwd Packet Length Std, Max Packet Length, Packet Length Std, Total Length of Bwd Packets, Fwd Packet Length Max |
| random_forest | Infiltration | 0.0000 | Avg Bwd Segment Size, Bwd Packet Length Mean, Bwd Packet Length Max, Init_Win_bytes_backward, Idle Std, Total Backward Packets, Total Length of Fwd Packets, act_data_pkt_fwd, Bwd Packet Length Std, Subflow Bwd Packets |
| xgboost | DDoS | 0.0000 | Fwd Packet Length Max, act_data_pkt_fwd, Init_Win_bytes_forward, Avg Bwd Segment Size, Init_Win_bytes_backward, Average Packet Size, min_seg_size_forward, Subflow Bwd Packets, Total Length of Fwd Packets, Total Length of Bwd Packets |
| xgboost | DoS | 0.1000 | Bwd Packet Length Std, Init_Win_bytes_backward, Idle Min, Bwd Packets/s, Fwd Packet Length Std, Init_Win_bytes_forward, Fwd Header Length, FIN Flag Count, Fwd IAT Max, Avg Bwd Segment Size |
| xgboost | PortScan | 0.0000 | Total Length of Fwd Packets, Packet Length Std, Flow Bytes/s, Packet Length Mean, Init_Win_bytes_forward, min_seg_size_forward, Fwd Header Length, Flow IAT Min, Average Packet Size, Bwd Packets/s |
| xgboost | BruteForce | 0.1000 | Bwd Header Length, Avg Bwd Segment Size, Fwd Packet Length Max, Max Packet Length, Bwd Packets/s, Init_Win_bytes_backward, min_seg_size_forward, Avg Fwd Segment Size, Fwd PSH Flags, Fwd IAT Max |
| xgboost | WebAttack | 0.0000 | Init_Win_bytes_backward, Max Packet Length, Fwd Header Length, Flow IAT Min, Init_Win_bytes_forward, Fwd IAT Std, Bwd Packets/s, min_seg_size_forward, Idle Min, URG Flag Count |
| xgboost | Botnet | 0.1000 | Bwd Packet Length Max, Bwd IAT Min, Flow IAT Mean, Init_Win_bytes_backward, Init_Win_bytes_forward, Average Packet Size, Fwd IAT Max, Bwd Packets/s, Bwd IAT Std, Max Packet Length |
| xgboost | Infiltration | 0.1000 | Avg Bwd Segment Size, Subflow Bwd Packets, Bwd IAT Min, act_data_pkt_fwd, Active Min, Init_Win_bytes_forward, Flow IAT Min, Fwd IAT Mean, Avg Fwd Segment Size, Total Length of Fwd Packets |

---

## Quality Images

- `reports/xai/quality/fidelity_plot.png`
- `reports/xai/quality/stability_plot.png`
- `reports/xai/quality/comprehensibility_heatmap.png`
- `reports/xai/quality/fidelity_decision_tree.png`
- `reports/xai/quality/fidelity_random_forest.png`
