# Model selection reasoning

Generated: 2026-07-13T07:08:34.759926+00:00

- Training rows (balanced): 84
- Features: **unscaled** (tree models are scale-invariant).
- Tuning: GridSearchCV, cv=5, scoring=f1_macro, n_jobs=-1.

- Decision Tree: n_train_rows=84 < 10,000 → max_depth capped at ≤6 (grid=[4, 6, None]).
- Random Forest grid: {'n_estimators': [100, 300, 500], 'max_depth': [None, 12, 20], 'min_samples_leaf': [1, 5]} (n_jobs=-1).
- XGBoost grid: {'n_estimators': [200, 500], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0]} (tree_method=hist, eval_metric=mlogloss).

- decision_tree: best CV f1_macro=1.0000, params={'max_depth': 4, 'min_samples_leaf': 1}
- random_forest: best CV f1_macro=1.0000, params={'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 100}
- xgboost: best CV f1_macro=1.0000, params={'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.8}
