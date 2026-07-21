# Preprocess decisions

Generated: 2026-07-13T07:08:34.249735+00:00

## Missing data

- `feat_high_var` (numeric): mean imputation (train skew=0.065)
- `feat_noisy` (numeric): mean imputation (train skew=-0.117)
- `feat_with_inf` (numeric): mean imputation (train skew=0.143)
- `Label` (categorical): mode (cardinality=3)

## Encoding

- OneHotEncoder on low-cardinality columns (≤15): ['Label']

## Scaling

- MinMaxScaler fit on TRAIN only for all model-ready features.
- Unscaled copy retained for tree models (DT, RF, XGBoost).

- Split checkpoint verified: `split`
- EDA directory consulted: `C:\Users\hp\AppData\Local\Temp\pytest-of-hp\pytest-49\test_xai_shap_lime_fidelity_an0\xai\reports\eda`
