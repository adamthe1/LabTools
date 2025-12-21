# Fixed presets from original_full_regressions.py (non-tuned defaults)
FIXED_PARAM_PRESETS = {
    # LGBM regressor - tuned for ~800K rows, ~1000 features
    "LGBM": dict(
        objective="regression",
        min_child_weight=16000,  # ~0.02 × 800K - prevents overfitting on large data
        max_depth=3,
        n_estimators=1000,
        num_leaves=300,
        feature_fraction=0.15,
        bagging_fraction=0.5,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_jobs=1,
        verbose=-1,
    ),

    # LGBM classifier - tuned for ~800K rows
    "LGBM_classifier": dict(
        objective="binary",
        min_child_weight=16000,  # ~0.02 × 800K
        n_estimators=1000,
        max_depth=3,
        subsample=0.5,
        colsample_bytree=0.1,
        n_jobs=1,
        verbose=-1,
    ),

    # XGB regressor - tuned for ~800K rows
    "XGB": dict(
        max_depth=3,
        min_child_weight=16000,  # ~0.02 × 800K
        n_estimators=1000,
        reg_alpha=0.5,
        reg_lambda=0.5,
        n_jobs=1,
    ),

    # Ridge regression - from original
    "LR_ridge": dict(
        alpha=1.0,
    ),

    # Lasso regression
    "LR_lasso": dict(
        alpha=1.0,
    ),

    # Elastic net
    "LR_elastic": dict(
        alpha=11.0,
        l1_ratio=0.2,
        max_iter=10000,
    ),

    # Logistic regression - from original
    "Logit": dict(
        C=1.0,
        penalty='l2',
        max_iter=10000,
        solver='lbfgs',
    ),

    # SVM regressor - from original
    "SVM_regression": dict(
        C=1.0,
        epsilon=0.1,
        loss='squared_epsilon_insensitive',
        dual=False,
    ),

    # SVM classifier - from original
    "SVM_classifier": dict(
        C=1.0,
        loss='squared_hinge',
        penalty='l2',
        dual=False,
    ),

    # Ordinal regression
    "Ordinal_logit": dict(
        distr='logit',
    ),
}

# Dynamic params for hyperparameter tuning (RandomizedSearchCV)
TUNABLE_PARAM_RANGES = {
    # Tuned for ~800K rows, ~1000 features
    "LGBM": dict(
        min_child_weight=[0.01*4000, 0.02*4000, 0.03*4000],  # 0.01-0.05 × 800K - prevents overfitting
        max_depth=[3, 4],
        n_estimators=[1000],
        num_leaves=[100, 300],
        feature_fraction=[0.1, 0.15, 0.2],  # ~100-200 features per tree
        bagging_fraction=[0.3, 0.5, 0.7],  # lower for large datasets
        reg_alpha=[0.2, 0.5, 1.0],
        reg_lambda=[0.2, 0.5, 1.0],
    ),

    "LGBM_classifier": dict(
        min_child_weight=[0.01*4000, 0.02*4000, 0.03*4000],  # scaled for 800K rows
        n_estimators=[1000],
        max_depth=[3, 4],
        subsample=[0.3, 0.5, 0.8],
        colsample_bytree=[0.1, 0.2],  # ~100-200 features
    ),

    "XGB": dict(
        min_child_weight=[0.01*4000, 0.02*4000, 0.03*4000],  # scaled for 800K rows
        max_depth=[3, 4],
        n_estimators=[1000],
        reg_alpha=[0.2, 0.5, 1.0],
        reg_lambda=[0.2, 0.5, 1.0],
    ),

    "LR_ridge": dict(
        alpha=[0.1, 1.0, 10.0],
    ),

    "LR_lasso": dict(
        alpha=[0.1, 1.0, 5.0, 10.0, 50.0],
    ),
    "LR_elastic": dict(
        alpha=[0.1, 1.0, 5.0, 10.0, 50.0],
        l1_ratio=[0.1, 0.5, 0.9],
        max_iter=[10000],
    ),

    "Logit": dict(
        C=[0.01, 0.1, 1.0, 10.0],
        penalty=['l2'],
        max_iter=[10000],
        solver=['lbfgs'],
    ),

    "SVM_regression": dict(
        C=[0.1, 1.0, 5.0],
        epsilon=[0.1, 0.2],
        loss=['squared_epsilon_insensitive'],
        dual=[False],
    ),

    "SVM_classifier": dict(
        C=[0.1, 1.0, 5.0],
        loss=['squared_hinge'],
        penalty=['l2'],
        dual=[False],
    ),
}