# Preprocess decisions

Generated: 2026-07-10T07:26:41.037846+00:00

## Missing data

- `ACK Flag Count` (numeric): mean imputation (train skew=0.821)
- `Active Max` (numeric): median imputation (train skew=22.535)
- `Active Mean` (numeric): median imputation (train skew=32.583)
- `Active Min` (numeric): median imputation (train skew=40.316)
- `Active Std` (numeric): median imputation (train skew=38.666)
- `Average Packet Size` (numeric): median imputation (train skew=2.340)
- `Avg Bwd Segment Size` (numeric): median imputation (train skew=2.360)
- `Avg Fwd Segment Size` (numeric): median imputation (train skew=8.782)
- `Bwd Header Length` (numeric): mean imputation (train skew=-576.948)
- `Bwd IAT Max` (numeric): median imputation (train skew=4.209)
- `Bwd IAT Mean` (numeric): median imputation (train skew=7.865)
- `Bwd IAT Min` (numeric): median imputation (train skew=9.454)
- `Bwd IAT Std` (numeric): median imputation (train skew=5.520)
- `Bwd IAT Total` (numeric): median imputation (train skew=2.709)
- `Bwd Packet Length Max` (numeric): median imputation (train skew=2.764)
- `Bwd Packet Length Mean` (numeric): median imputation (train skew=2.360)
- `Bwd Packet Length Min` (numeric): median imputation (train skew=4.843)
- `Bwd Packet Length Std` (numeric): median imputation (train skew=3.225)
- `Bwd Packets/s` (numeric): median imputation (train skew=21.883)
- `CWE Flag Count` (numeric): median imputation (train skew=189.833)
- `Down/Up Ratio` (numeric): median imputation (train skew=12.711)
- `ECE Flag Count` (numeric): median imputation (train skew=61.310)
- `FIN Flag Count` (numeric): median imputation (train skew=5.371)
- `Flow Bytes/s` (numeric): median imputation (train skew=45.302)
- `Flow Duration` (numeric): median imputation (train skew=2.000)
- `Flow IAT Mean` (numeric): median imputation (train skew=8.591)
- `Flow IAT Min` (numeric): median imputation (train skew=23.695)
- `Flow IAT Std` (numeric): median imputation (train skew=3.416)
- `Flow Packets/s` (numeric): median imputation (train skew=6.963)
- `Fwd Header Length` (numeric): mean imputation (train skew=-1087.752)
- `Fwd IAT Max` (numeric): median imputation (train skew=2.736)
- `Fwd IAT Mean` (numeric): median imputation (train skew=6.771)
- `Fwd IAT Min` (numeric): median imputation (train skew=9.316)
- `Fwd IAT Std` (numeric): median imputation (train skew=3.219)
- `Fwd PSH Flags` (numeric): median imputation (train skew=4.221)
- `Fwd Packet Length Max` (numeric): median imputation (train skew=9.445)
- `Fwd Packet Length Mean` (numeric): median imputation (train skew=8.782)
- `Fwd Packet Length Min` (numeric): median imputation (train skew=19.407)
- `Fwd Packet Length Std` (numeric): median imputation (train skew=10.059)
- `Fwd URG Flags` (numeric): median imputation (train skew=189.833)
- `Idle Min` (numeric): median imputation (train skew=2.999)
- `Idle Std` (numeric): median imputation (train skew=10.005)
- `Init_Win_bytes_backward` (numeric): median imputation (train skew=4.984)
- `Init_Win_bytes_forward` (numeric): median imputation (train skew=2.554)
- `Max Packet Length` (numeric): median imputation (train skew=2.647)
- `Min Packet Length` (numeric): median imputation (train skew=10.389)
- `PSH Flag Count` (numeric): mean imputation (train skew=0.917)
- `Packet Length Mean` (numeric): median imputation (train skew=2.245)
- `Packet Length Std` (numeric): median imputation (train skew=2.714)
- `Packet Length Variance` (numeric): median imputation (train skew=5.128)
- `RST Flag Count` (numeric): median imputation (train skew=61.310)
- `SYN Flag Count` (numeric): median imputation (train skew=4.221)
- `Subflow Bwd Packets` (numeric): median imputation (train skew=222.904)
- `Subflow Fwd Packets` (numeric): median imputation (train skew=222.591)
- `Total Backward Packets` (numeric): median imputation (train skew=222.904)
- `Total Fwd Packets` (numeric): median imputation (train skew=222.591)
- `Total Length of Bwd Packets` (numeric): median imputation (train skew=222.396)
- `Total Length of Fwd Packets` (numeric): median imputation (train skew=772.456)
- `URG Flag Count` (numeric): median imputation (train skew=2.658)
- `act_data_pkt_fwd` (numeric): median imputation (train skew=261.310)
- `min_seg_size_forward` (numeric): mean imputation (train skew=-404.502)

## Encoding


## Scaling

- MinMaxScaler fit on TRAIN only for all model-ready features.
- Unscaled copy retained for tree models (DT, RF, XGBoost).

- Split checkpoint verified: `split`
- EDA directory consulted: `D:\Documents\Kamran\xai-project\reports\eda`
