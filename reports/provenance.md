# Data provenance

## Source dataset

- **Dataset:** CICIDS-2017 (Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017)
- **URL:** https://www.unb.ca/cic/datasets/ids-2017.html
- **Licence:** CICIDS-2017 is provided by the Canadian Institute for Cybersecurity (UNB) for academic and research use. Verify current licence terms at the source URL before redistribution or publication of derived data.

## Raw file checksums (SHA-256)

Checksums are recorded at discovery time for every CSV under `data/raw/`.

## Label mapping (project taxonomy)

Raw CICIDS-2017 labels are normalised to the dissertation threat taxonomy. Observed mappings from integration are appended after the `integrate` step; see also `reports/label_mapping.json`.

| Raw label | Threat class |
| --- | --- |
| `BENIGN` | `BENIGN` |
| `Bot` | `Botnet` |
| `DDoS` | `DDoS` |
| `DoS GoldenEye` | `DoS` |
| `DoS Hulk` | `DoS` |
| `DoS Slowhttptest` | `DoS` |
| `DoS slowloris` | `DoS` |
| `FTP-Patator` | `BruteForce` |
| `Infiltration` | `Infiltration` |
| `PortScan` | `PortScan` |
| `SSH-Patator` | `BruteForce` |
| `Web Attack - Brute Force` | `WebAttack` |
| `Web Attack - Sql Injection` | `WebAttack` |
| `Web Attack - XSS` | `WebAttack` |

## Discovery entry (2026-07-10T06:57:29.444403+00:00)

- Detected dataset type(s): cicids_csv
- Download URL: https://www.unb.ca/cic/datasets/ids-2017.html
- Licence note: CICIDS-2017 is provided by the Canadian Institute for Cybersecurity (UNB) for academic and research use. Verify current licence terms at the source URL before redistribution or publication of derived data.
- File checksums:
  - `CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`: `6ff1580f5f81c0ae28a26f7631721018577f5f7c5e0feac28b795fcfe7b411ee`
  - `CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`: `ca1824c51bfbb7b3c72290a11be04366ba8815878c6a1cc5c44cb1cee269e99b`
  - `CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv`: `53a41c24d570ea83b7ac55b2e94df94e7a8216aeb80a2af0246b6bc8bb543000`
  - `CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv`: `852c4beb34eda186f32561fa79df7a0747e92e1a6535b01270820dd9ffe17f34`
  - `CICIDS2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`: `6bcda3857c2504676034e3ea57762d38393cc734cb377a726bd5cb153961b1b5`
  - `CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`: `d67066211fb1689c78406f1506f4c44704ecb92088353d5c96d96d6474eb819d`
  - `CICIDS2017/Tuesday-WorkingHours.pcap_ISCX.csv`: `52b8692ae8c7d2ed04671fe2b98335693c0a92c7ab157d8c8b534d6523080851`
  - `CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv`: `893c27dc968bf7a8adef1689f90be55ca4a4dc3088fb63d6ff247ac56856df2a`
  - `KDD Cup 99/login_fraud_detection_karanam_shrivasta_2.csv`: `810381c73cbc99f9d4098f39eaaecde1a110ef348c7195469ad99bdeee1eb749`

## Label mapping (2026-07-10T07:21:32.576185+00:00)

Raw CICIDS-2017 labels are mapped to the project threat taxonomy (see also `reports/label_mapping.json`).

| Raw label | Threat class |
| --- | --- |
| `BENIGN` | `BENIGN` |
| `Bot` | `Botnet` |
| `DDoS` | `DDoS` |
| `DoS GoldenEye` | `DoS` |
| `DoS Hulk` | `DoS` |
| `DoS Slowhttptest` | `DoS` |
| `DoS slowloris` | `DoS` |
| `FTP-Patator` | `BruteForce` |
| `Heartbleed` | `Heartbleed` |
| `Infiltration` | `Infiltration` |
| `PortScan` | `PortScan` |
| `SSH-Patator` | `BruteForce` |
| `Web Attack � Brute Force` | `WebAttack` |
| `Web Attack � Sql Injection` | `WebAttack` |
| `Web Attack � XSS` | `WebAttack` |
