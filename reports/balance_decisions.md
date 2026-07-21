# Balance decisions

Generated: 2026-07-13T07:08:34.617958+00:00
- imbalance_ratio=1.200, min_class_count=50

- Branch: imbalance_ratio < 1.5 → SMOTE skipped; saved training split as-is.

## Trade-off note

- `class_weight.json` saved as an alternative when models support `class_weight` / `scale_pos_weight`.
- SMOTE can improve recall on minority classes but may reduce minority precision; compare both in evaluation.
- Applied method: `none`
- Train rows before: 84 | after: 84
