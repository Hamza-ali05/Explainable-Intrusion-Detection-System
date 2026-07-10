# Balance decisions

Generated: 2026-07-10T07:29:01.310235+00:00
- imbalance_ratio=195307.818, min_class_count=11

- SMOTETomek failed (Unable to allocate 6.15 GiB for an array with shape (13534830, 61) and data type float64); saved train as-is.

## Trade-off note

- `class_weight.json` saved as an alternative when models support `class_weight` / `scale_pos_weight`.
- SMOTE can improve recall on minority classes but may reduce minority precision; compare both in evaluation.
- Applied method: `SMOTETomek`
- Train rows before: 1801984 | after: 1801984
