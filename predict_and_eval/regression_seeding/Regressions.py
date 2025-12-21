# Setup
import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV
from sklearn import metrics
from scipy.stats import pearsonr

from ..utils.seed_utils import average_scores_by_subject_id
from ..utils.model_and_pipeline import ModelAndPipeline
from ..utils.categorical_utils import CategoricalUtils
from ..utils.evaluate_predictions import (
    evaluate_regression_with_gender_split, 
    evaluate_classification_with_gender_split, 
    evaluate_ordinal_with_gender_split,
    evaluate_regression, 
    evaluate_classification,
    evaluate_ordinal,
    average_scores_by_subject_id_research_stage,
    get_gender_for_index
)
from ..utils.ids_folds import create_cv_folds
from ..params.model_params import FIXED_PARAM_PRESETS
from scipy.stats import spearmanr

warnings.simplefilter(action='ignore', category=FutureWarning)

# Threading strategy for 64-core machine:
# - N_THREADS_MODEL: threads per individual model (tree models benefit from this)
# - N_THREADS_CV: parallel CV folds / hyperparameter search iterations
# Total threads used = N_THREADS_MODEL * N_THREADS_CV (avoid exceeding core count)
N_THREADS_MODEL = 1   # Tree models (LGBM/XGB) benefit from multi-threading
N_THREADS_CV = 16     # 16 parallel CV fits × 4 threads each = 64 threads max

# Use fewer folds for hyperparameter tuning (faster, still uses 80/20 splits)
N_TUNING_CV_FOLDS = 2

NO_TUNING = False

import time
from contextlib import contextmanager

@contextmanager
def stage_timer(timings: dict, key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - start)

class Regressions:
    def __init__(self):
        pass
    
    @staticmethod
    def get_reg_models():
        return ["LR_ridge", "LGBM_regression"]

    @staticmethod
    def get_clas_models():
        return ['Logit', 'LGBM_classifier'] # SVM_clas
    
    @staticmethod
    def get_ordinal_models():
        return ['Ordinal_logit']
    
    @staticmethod
    def get_preset_params(model_type: str, x) -> dict:
        """
        Get preset parameters for a model type from FIXED_PARAM_PRESETS.
        
        Args:
            model_type: Model type name (e.g., 'LGBM', 'LR_ridge', 'Ordinal_logit')
            
        Returns:
            Dict of preset parameters, or empty dict if not found.
            
        Usage:
            params = Regressions.get_preset_params('LGBM')
            regressions.cross_validate_model(x, y, folds, model_key='LGBM', params=params)
        """
        # Handle model type aliases (e.g., 'LGBM_regression' -> 'LGBM')
        lookup_key = model_type.replace('_regression', '').replace('_classifier', '')
        return ModelAndPipeline.get_model_type_static_params(model_type, x)

    def cross_validate_model(self, 
                             x: pd.DataFrame,
                             y: pd.DataFrame,
                             cv_fold: list,
                             model_key: str='all',
                             params=None,
                             testing=False
                             ):
        """
        Perform cross-validation on a specified model.
        Label type (regression/categorical/ordinal) is auto-detected from y.
        
        Args:
            x: Feature DataFrame Index is (RegistrationCode, research_stage)
            y: Target DataFrame
            cv_fold: List of folds length of n_splits [(train_reg_codes, test_reg_codes), ...]
            model_key: Model type to use ('all' to tune across all models, or specific model name)
            params: Model hyperparameters (if None, will be tuned)
        """
        y_col = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        
        # Auto-detect label type from column name and values
        label_type = CategoricalUtils.get_label_type(y_col)
        categorical_cols, numeric_cols = CategoricalUtils.detect_categorical_and_numeric_columns(x)
        
        # Check for sufficient classes in categorical/ordinal targets
        n_unique = y_col.nunique()
        if x.ndim == 2 and x.shape[1] == 0:
            return {
                "model_key": None,
                "model": None,
                "params": None,
                "predictions": None,
                "cv_fold": cv_fold,
                "label_type": label_type,
                "skipped": True,
                "skip_reason": "No input features available (shape[1] == 0)",
            }
        if label_type in ('categorical', 'ordinal') and n_unique < 2:
            return {
                "model_key": None,
                "model": None,
                "params": None,
                "predictions": None,
                "cv_fold": cv_fold,
                "label_type": label_type,
                "skipped": True,
                "skip_reason": f"Only {n_unique} class(es) in target, need at least 2",
            }
        if label_type == 'categorical':
            unique_vals, counts = np.unique(y_col.dropna(), return_counts=True)
            if len(unique_vals) >= 2:
                minority_count = counts.min()
                if minority_count < 8:
                    return {
                        "model_key": None,
                        "model": None,
                        "params": None,
                        "predictions": None,
                        "cv_fold": cv_fold,
                        "label_type": label_type,
                        "skipped": True,
                        "skip_reason": f"Minority class n={minority_count} < 8, too few samples",
                    }
        
        # Convert subject-level folds to sample indices (must do before .values)
        subject_ids = x.index.get_level_values(0).values
        sample_folds = create_cv_folds(cv_fold, subject_ids)
        
        x = x.copy().values
        y = y.copy().values.flatten()

        if testing:
            if label_type == 'categorical':
                model_key = self.get_clas_models()[0]
            elif label_type == 'ordinal':
                model_key = self.get_ordinal_models()[0]
            else:
                model_key = self.get_reg_models()[0]
            params = self.get_preset_params(model_key, x)

        # If model_key is 'all', tune across all available models
        if model_key == 'all' and params is None:
            if label_type == 'categorical':
                available_models = self.get_clas_models()
            elif label_type == 'ordinal':
                available_models = self.get_ordinal_models()
            else:
                available_models = self.get_reg_models()
            
            model_key, params = self.tune_model_and_hyperparams(
                model_types=available_models, 
                x=x, 
                y=y, 
                cv_fold=sample_folds,
                label_type=label_type,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols
            )
        else:
            # Tune hyperparams for specific model if not provided
            if params is None:
                params = self.tune_hyperparams_on_specific_target(model_type=model_key, x=x, y=y, cv_fold=sample_folds, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
        
        # Initialize model and pipeline
        pipe = ModelAndPipeline.initialize_model_and_pipeline(model_key, n_jobs=N_THREADS_MODEL, params=params, categorical_cols=categorical_cols, numeric_cols=numeric_cols)

        # Select prediction method based on label type
        if label_type == 'categorical':
            method = "predict_proba"
        else:
            # Regression and ordinal both use predict (ordinal returns expected value)
            method = "predict"
        
        predictions = cross_val_predict(pipe, x, y, cv=sample_folds, n_jobs=N_THREADS_CV, method=method)

        payload = {
            "model_key": model_key,
            "model": pipe['model'],
            "params": params,
            "predictions": predictions,
            "cv_fold": cv_fold,
            "label_type": label_type,
        }
        return payload

    def cross_validate_model_time(self, 
                         x: pd.DataFrame,
                         y: pd.DataFrame,
                         cv_fold: list,
                         model_key: str='all',
                         params=None,
                         testing=False
                         ):
        """
        Perform cross-validation on a specified model with timing.
        Returns payload + 'timings' dict (seconds).
        """
        timings = {}
        with stage_timer(timings, "total"):

            with stage_timer(timings, "pre_detect_label"):
                y_col = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
                label_type = CategoricalUtils.get_label_type(y_col)
                categorical_cols, numeric_cols = CategoricalUtils.detect_categorical_and_numeric_columns(x)

            with stage_timer(timings, "sanity_checks"):
                n_unique = y_col.nunique()
                if x.ndim == 2 and x.shape[1] == 0:
                    return {
                        "model_key": None,
                        "model": None,
                        "params": None,
                        "predictions": None,
                        "cv_fold": cv_fold,
                        "label_type": label_type,
                        "skipped": True,
                        "skip_reason": "No input features available (shape[1] == 0)",
                        "timings": timings,
                    }
                if label_type in ('categorical', 'ordinal') and n_unique < 2:
                    return {
                        "model_key": None,
                        "model": None,
                        "params": None,
                        "predictions": None,
                        "cv_fold": cv_fold,
                        "label_type": label_type,
                        "skipped": True,
                        "skip_reason": f"Only {n_unique} class(es) in target, need at least 2",
                        "timings": timings,
                    }
                if label_type == 'categorical':
                    unique_vals, counts = np.unique(y_col.dropna(), return_counts=True)
                    if len(unique_vals) >= 2:
                        minority_count = counts.min()
                        if minority_count < 8:
                            return {
                                "model_key": None,
                                "model": None,
                                "params": None,
                                "predictions": None,
                                "cv_fold": cv_fold,
                                "label_type": label_type,
                                "skipped": True,
                                "skip_reason": f"Minority class n={minority_count} < 8, too few samples",
                                "timings": timings,
                            }

            with stage_timer(timings, "cv_index_mapping"):
                subject_ids = x.index.get_level_values(0).values
                sample_folds = create_cv_folds(cv_fold, subject_ids)

            with stage_timer(timings, "to_numpy"):
                x = x.copy().values
                y = y.copy().values.flatten()

            # Testing shortcut model selection
            if testing:
                with stage_timer(timings, "testing_model_selection"):
                    if label_type == 'categorical':
                        model_key = self.get_clas_models()[0]
                    elif label_type == 'ordinal':
                        model_key = self.get_ordinal_models()[0]
                    else:
                        model_key = self.get_reg_models()[0]
                    params = self.get_preset_params(model_key, x)

            # Tune model type & hyperparams
            if model_key == 'all' and params is None:
                with stage_timer(timings, "tune_model_and_hyperparams"):
                    if label_type == 'categorical':
                        available_models = self.get_clas_models()
                    elif label_type == 'ordinal':
                        available_models = self.get_ordinal_models()
                    else:
                        available_models = self.get_reg_models()

                    model_key, params = self.tune_model_and_hyperparams(
                        model_types=available_models, 
                        x=x, 
                        y=y, 
                        cv_fold=sample_folds,
                        label_type=label_type,
                        categorical_cols=categorical_cols,
                        numeric_cols=numeric_cols
                    )
            else:
                if params is None:
                    with stage_timer(timings, "tune_hyperparams_specific"):
                        params = self.tune_hyperparams_on_specific_target(
                            model_type=model_key, x=x, y=y, cv_fold=sample_folds,
                            categorical_cols=categorical_cols, numeric_cols=numeric_cols
                        )

            with stage_timer(timings, "initialize_pipeline"):
                pipe = ModelAndPipeline.initialize_model_and_pipeline(
                    model_key, n_jobs=N_THREADS_MODEL, params=params,
                    categorical_cols=categorical_cols, numeric_cols=numeric_cols
                )

            with stage_timer(timings, "cross_val_predict"):
                if label_type == 'categorical':
                    method = "predict_proba"
                else:
                    method = "predict"

                predictions = cross_val_predict(
                    pipe, x, y, cv=sample_folds, n_jobs=N_THREADS_CV, method=method
                )

        print(timings)
        
        payload = {
            "model_key": model_key,
            "model": pipe['model'],
            "params": params,
            "predictions": predictions,
            "cv_fold": cv_fold,
            "label_type": label_type,
            "timings": timings,
        }
        return payload
    
    
    def evaluate_predictions(self,
                             x: pd.DataFrame,
                             y: pd.DataFrame,
                             predictions: np.ndarray,
                             gender_split_evaluation: bool = True, 
                             average_by_subject_id: bool = True,
                             label_type: str = None):
        """
        Evaluate predictions using appropriate metrics based on label type.
        
        Args:
            x: Feature DataFrame with index
            y: Target DataFrame
            predictions: Model predictions
            gender_split_evaluation: Whether to split evaluation by gender
            average_by_subject_id: Whether to average by subject
            label_type: Explicit label type. If None, auto-detected.
        """
        y_col = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
        
        if label_type is None:
            label_type = CategoricalUtils.get_label_type(y_col)

        id_research_pairs = x.index.copy()
        y = y.copy().values

        if average_by_subject_id:
            id_research_pairs, y, predictions = average_scores_by_subject_id_research_stage(id_research_pairs, y, predictions)

        flat_y = y.flatten()
        
        # Get gender for the (possibly averaged) index
        gender = get_gender_for_index(x, id_research_pairs) if gender_split_evaluation else None

        if label_type == 'categorical':
            # predictions from predict_proba are probabilities, pass as y_pred_proba
            if gender_split_evaluation:
                eval_metrics = evaluate_classification_with_gender_split(gender, flat_y, y_pred=None, y_pred_proba=predictions)
            else:
                eval_metrics = evaluate_classification(flat_y, y_pred=None, y_pred_proba=predictions)
        elif label_type == 'ordinal':
            if gender_split_evaluation:
                eval_metrics = evaluate_ordinal_with_gender_split(gender, flat_y, predictions)
            else:
                eval_metrics = evaluate_ordinal(flat_y, predictions)
        else:
            # regression
            if gender_split_evaluation:
                eval_metrics = evaluate_regression_with_gender_split(gender, flat_y, predictions)
            else:
                eval_metrics = evaluate_regression(flat_y, predictions)
        
        # Add n_subjects (and n_positive for classification)
        n_subjects = len(np.unique(id_research_pairs.get_level_values(0)))
        eval_metrics['n_subjects'] = n_subjects
        if label_type == 'categorical':
            eval_metrics['n_positive'] = int(np.sum(flat_y == 1))
        
        payload = {
            "metrics": {**{f'{key}': value for key, value in eval_metrics.items() if "male" not in key}},
            "metrics_male": {**{f'{key}': value for key, value in eval_metrics.items() if "male" in key and "female" not in key}},
            "metrics_female": {**{f'{key}': value for key, value in eval_metrics.items() if "female" in key}},
            "predictions": predictions,
            "id_research_pairs": id_research_pairs,
            "true_values": y  # Return (possibly averaged) true values
        }
        return payload
        

    def tune_hyperparams_on_specific_target(self,
                                            model_type: str,
                                            x: np.array,
                                            y: np.array,
                                            cv_fold: list = None,
                                            categorical_cols: list = None,
                                            numeric_cols: list = None):
        """Tune hyperparameters for a specific model type. Returns {} if no tunable params."""
        if NO_TUNING:
            return self.get_preset_params(model_type, x)
        try:
            params = ModelAndPipeline.get_model_type_dynamic_params(model_type, x)
        except (KeyError, FileNotFoundError):
            # Model has no tunable hyperparameters (e.g., Ordinal_logit)
            print(f"{model_type} model has no tunable hyperparameters, using preset params")
            return {}
        
        if not params:
            return {}
        
        # Use fewer folds for tuning (faster, keeps same 80/20 train/val ratio)
        tuning_cv = cv_fold[:N_TUNING_CV_FOLDS]
        pipe = ModelAndPipeline.initialize_model_and_pipeline(model_type, n_jobs=N_THREADS_MODEL, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
        distribution = {f'model__{key}': val for key, val in params.items()}
        print("tuning params: ", distribution)
        best_params = RandomizedSearchCV(
            estimator=pipe, param_distributions=distribution, cv=tuning_cv, n_jobs=N_THREADS_CV, n_iter=30).fit(
            x, y).best_params_
        best_params = {key.replace('model__', ''): val for key, val in best_params.items()}
        return best_params
    
    def tune_model_and_hyperparams(self,
                                   model_types: list,
                                   x: np.array,
                                   y: np.array,
                                   cv_fold: list = None,
                                   label_type: str = 'regression',
                                   categorical_cols: list = None,
                                   numeric_cols: list = None):
        """
        Tune both model type and hyperparameters together.
        Tries each model type with hyperparameter tuning and returns the best combination.
        
        Args:
            model_types: List of model types to try
            x: Features (numpy array)
            y: Target
            cv_fold: CV folds
            label_type: 'regression', 'categorical', or 'ordinal'
            categorical_cols: Pre-detected categorical column indices
            numeric_cols: Pre-detected numeric column indices
            no_tuning: If True, use preset params instead of tuning (still compares models)
            
        Returns:
            Tuple of (best_model_type, best_params)
        """
        best_score = -np.inf
        best_model_type = None
        best_params = None
        no_tuning = NO_TUNING
        for model_type in model_types:
            try:
                # Get params: either tune or use presets
                if no_tuning:
                    print("using preset params for ", model_type)
                    params = self.get_preset_params(model_type, x)
                else:
                    params = self.tune_hyperparams_on_specific_target(model_type, x, y, cv_fold, categorical_cols, numeric_cols)
                

                # Evaluate with these params
                pipe = ModelAndPipeline.initialize_model_and_pipeline(model_type, n_jobs=N_THREADS_MODEL, params=params, categorical_cols=categorical_cols, numeric_cols=numeric_cols)
                
                if label_type == 'categorical':
                    predictions = cross_val_predict(pipe, x, y, cv=cv_fold, n_jobs=N_THREADS_CV, method='predict_proba')
                    score = metrics.roc_auc_score(y, predictions[:, 1])
                elif label_type == 'ordinal':
                    predictions = cross_val_predict(pipe, x, y, cv=cv_fold, n_jobs=N_THREADS_CV)
                    score = spearmanr(y.flatten(), predictions.flatten())[0]
                else:
                    predictions = cross_val_predict(pipe, x, y, cv=cv_fold, n_jobs=N_THREADS_CV)
                    score = pearsonr(y.flatten(), predictions.flatten())[0]
                
                if score > best_score:
                    best_score = score
                    best_model_type = model_type
                    best_params = params
                    
            except Exception as e:
                print(f"Failed to tune model {model_type}: {e}")
                continue
        
        if best_model_type is None:
            raise ValueError("All model types failed during tuning")
        
        return best_model_type, best_params

  