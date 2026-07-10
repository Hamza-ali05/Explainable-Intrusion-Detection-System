# Clean decisions

Generated: 2026-07-10T07:25:30.587286+00:00

- Replaced `np.inf` / `-np.inf` with `np.nan`.
- No columns exceeded 60% missing threshold.
- Dropped 0 rows where all flow features were missing.
- Dropped near-zero-variance columns from EDA: ['Bwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bulk Rate', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk']
- Highly correlated feature drops (|corr|>0.95, kept higher variance):
  - dropped `Fwd IAT Total`, kept `Flow Duration` (|corr|=0.9985; kept higher variance)
  - dropped `Flow IAT Max`, kept `Fwd IAT Max` (|corr|=0.9981; kept higher variance)
  - dropped `Idle Max`, kept `Fwd IAT Max` (|corr|=0.9887; kept higher variance)
  - dropped `Idle Mean`, kept `Fwd IAT Max` (|corr|=0.9783; kept higher variance)
  - dropped `Fwd Header Length.1`, kept `Fwd Header Length` (|corr|=1.0000; kept higher variance)
  - dropped `Subflow Bwd Bytes`, kept `Total Length of Bwd Packets` (|corr|=1.0000; kept higher variance)
  - dropped `Fwd Packets/s`, kept `Flow Packets/s` (|corr|=0.9801; kept higher variance)
  - dropped `Subflow Fwd Bytes`, kept `Total Length of Fwd Packets` (|corr|=1.0000; kept higher variance)
- Dropped identifier / potentially leaking columns (default DROP per ethics section): ['Destination Port']
