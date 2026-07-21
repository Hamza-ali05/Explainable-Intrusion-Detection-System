# Clean decisions

Generated: 2026-07-13T07:08:33.993215+00:00

- Replaced `np.inf` / `-np.inf` with `np.nan`.
- No columns exceeded 60% missing threshold.
- Dropped 0 rows where all flow features were missing.
- Dropped near-zero-variance columns from EDA: ['feat_constant']
- Highly correlated feature drops (|corr|>0.95, kept higher variance):
  - dropped `feat_low_var_corr`, kept `feat_high_var` (|corr|=0.9825; kept higher variance)
- Dropped identifier / potentially leaking columns (default DROP per ethics section): ['Source IP', 'Destination Port', 'Timestamp']
