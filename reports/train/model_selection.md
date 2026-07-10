# Model selection reasoning

Generated: 2026-07-10T07:35:24.161310+00:00

- Training rows (balanced): 1801976
- Features: **unscaled** (tree models are scale-invariant).
- Tuning: GridSearchCV, cv=3, scoring=f1_macro, n_jobs=1. Large train set → stratified subsample of 100,000 rows for CV, then refit best params on all 1,801,976 rows.

- Decision Tree: n_train_rows=1801976 ≥ 10,000 → full max_depth grid [4, 6, 8, 10, None].
- Random Forest grid: {'n_estimators': [100, 200], 'max_depth': [12, 20], 'min_samples_leaf': [5]} (n_jobs=1).
- XGBoost grid: {'n_estimators': [200], 'max_depth': [4, 6], 'learning_rate': [0.1], 'subsample': [0.8]} (tree_method=hist, eval_metric=mlogloss).

- decision_tree: best CV f1_macro=0.8207, params={'max_depth': None, 'min_samples_leaf': 1}
- random_forest: best CV f1_macro=0.7940, params={'max_depth': 20, 'min_samples_leaf': 5, 'n_estimators': 200}
- xgboost: best CV f1_macro=0.8339, params={'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 200, 'subsample': 0.8}
