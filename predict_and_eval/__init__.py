# predict_and_eval package
# Downstream prediction and evaluation pipeline

from .loading_features.load_feature_df import (
    load_dataset_filenames_dict,
    load_body_system_df,
    load_columns_as_df,
    get_body_system_column_names,
    add_body_system_csv,
    create_body_system_from_other_systems_csv,
    remove_body_system_csv,
)
from .loading_features.preprocess_features import PreprocessFeatures
from .loading_features.temp_feature import create_merged_df_bundle
from .regression_seeding.seeding import seeding
from .regression_seeding.Regressions import Regressions
from .regression_seeding.metrics_collector import MetricsCollector
from .utils.ids_folds import (
    ids_folds,
    stratified_ids_folds,
    id_fold_with_stratified_threshold,
    create_cv_folds,
    create_cv_folds_by_seed,
)
from .utils.evaluate_predictions import (
    evaluate_regression,
    evaluate_classification,
    evaluate_regression_with_gender_split,
    evaluate_classification_with_gender_split,
)
from .utils.categorical_utils import CategoricalUtils
from .correct_and_collect_results.compare_results import compare_and_collect_results
from .correct_and_collect_results.fix_pvals import fix_pvals

__all__ = [
    # Loading features
    'load_dataset_filenames_dict',
    'load_body_system_df',
    'load_columns_as_df',
    'get_body_system_column_names',
    'add_body_system_csv',
    'create_body_system_from_other_systems_csv',
    'remove_body_system_csv',
    'PreprocessFeatures',
    'create_merged_df_bundle',
    # Regression seeding
    'seeding',
    'Regressions',
    'MetricsCollector',
    # Utils
    'ids_folds',
    'stratified_ids_folds',
    'id_fold_with_stratified_threshold',
    'create_cv_folds',
    'create_cv_folds_by_seed',
    'evaluate_regression',
    'evaluate_classification',
    'evaluate_regression_with_gender_split',
    'evaluate_classification_with_gender_split',
    'CategoricalUtils',
    # Results
    'compare_and_collect_results',
    'fix_pvals',
]





