"""
Seeding function for running cross-validation across multiple seeds.

The seeding function runs cross-validation across multiple random seeds and collects
all predictions and metrics using the MetricsCollector class. This ensures consistent
evaluation across different random initializations.

Usage Example:
    from deep_learning.downstream.regression_seeding.seeding import seeding
    from deep_learning.downstream.utils.ids_folds import ids_folds
    
    # Create fold definitions
    folds = ids_folds(x.index.get_level_values(0), seeds=range(10), n_splits=5)
    
    # Run seeding (assumes 'self' is a Regressions instance)
    results = seeding(
        self=regression_instance,
        x=X_features,
        y=y_target,
        folds=folds,
        model_key='LGBM_regression',
        average_by_subject_id=True,
        gender_split_evaluation=False  # Set True if gender column exists
    )
    
    # Access results
    predictions_df = results['all_predictions']  # Predictions for each seed
    scores_df = results['scores_pvalues']        # Scores and metrics per seed
"""

import os
import warnings
import pandas as pd
import numpy as np
from ..utils.ids_folds import ids_folds, stratified_ids_folds, save_folds
from ..utils.decorators import timeout
from .metrics_collector import MetricsCollector
from .gait_metrics_collector import GaitMetricsCollector
from .Regressions import Regressions
from .explain_ml_model import explain_model
import joblib


@timeout(48 * 60 * 60)  # 48 hour timeout for the seeding function
def seeding(x: pd.DataFrame,
            y: pd.DataFrame,
            folds: list,
            model_key: str = 'all',
            average_by_subject_id: bool = True,
            gender_split_evaluation: bool = True,
            save_dir: str = None,
            params: dict = None,
            testing: bool = False,
            run_explain_model: bool = False) -> dict:
    """
    Perform cross-validation across multiple seeds and collect metrics.
    
    Label type (regression/categorical/ordinal) is auto-detected from y column name and values.
    
    Args:
        x: Feature DataFrame with MultiIndex (RegistrationCode, research_stage)
        y: Target DataFrame
        folds: List of fold definitions [seed_index][fold_index] = (train_ids, test_ids)
        model_key: Model type to use ('all' to tune across all models, or specific model name)
        average_by_subject_id: Whether to average predictions by (RegistrationCode, research_stage)
        gender_split_evaluation: Whether to evaluate separately by gender
        params: Model hyperparameters (if None, will be tuned)
        testing: If True, use preset params from model_params.py (skips tuning for faster testing)
    
    Returns:
        Dictionary with:
            - 'all_predictions': DataFrame with predictions for each seed
            - 'scores_pvalues': DataFrame with scores, pvalues, and metrics per seed
    """
    if gender_split_evaluation and 'gender' not in x.columns:
        gender_split_evaluation = False

    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    regressions = Regressions()
    metrics_collector = None
    example_model = None
    example_model_key = None
    example_seed = None
    for fold_index, fold in enumerate(folds):
        
        # Run cross-validation (label_type auto-detected from y)
        payload = regressions.cross_validate_model(x=x, y=y, cv_fold=fold, model_key=model_key, params=params, testing=testing)
        if example_model is None:
            example_model = payload['model']
            example_model_key = payload['model_key']
            example_seed = fold_index
        # Handle skipped targets (e.g., single class)
        if payload.get('skipped'):
            print(f"Skipping target: {payload.get('skip_reason', 'unknown reason')}")
            if save_dir is not None:
                pd.DataFrame().to_csv(f"{save_dir}/predictions.csv")
                pd.DataFrame({'skipped': [True], 'reason': [payload.get('skip_reason')]}).to_csv(f"{save_dir}/metrics.csv")
            return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}
        
        # Initialize collector on first successful fold (now we know model_key)
        if metrics_collector is None:
            metrics_collector = MetricsCollector(seeds=range(len(folds)), model_key=payload['model_key'])
        
        # Evaluate predictions (use label_type from payload to ensure consistency)
        evaluate_payload = regressions.evaluate_predictions(
            x=x,
            y=y,
            predictions=payload['predictions'],
            gender_split_evaluation=gender_split_evaluation,
            average_by_subject_id=average_by_subject_id,
            label_type=payload['label_type'])

        # Collect results
        metrics_collector.add_seed_results(
            seed=fold_index,
            predictions=evaluate_payload['predictions'],
            id_research_pairs=evaluate_payload['id_research_pairs'],
            metrics=evaluate_payload['metrics'],
            metrics_male=evaluate_payload.get('metrics_male', {}),
            metrics_female=evaluate_payload.get('metrics_female', {}),
            true_values=evaluate_payload.get('true_values'),
        )

    # Get final results
    results = metrics_collector.get_results()

    if save_dir is not None:
        results['predictions'].to_csv(f"{save_dir}/predictions.csv")
        results['metrics'].to_csv(f"{save_dir}/metrics.csv")
        save_folds(folds, save_dir)
        # dump the model
        joblib.dump(example_model, os.path.join(save_dir, f'example_model_seed_{example_seed}.pkl'))


    if run_explain_model:
        explain_model(example_model, example_model_key, folds[example_seed], x, y, save_dir, 'age')

    
    return results


def seeding_gait(x: pd.DataFrame,
            y: pd.DataFrame,
            folds: list,
            model_key: str = 'all',
            gender_split_evaluation: bool = True,
            save_dir: str = None,
            params: dict = None,
            testing: bool = False,
            run_explain_model: bool = False) -> dict:
    """
    Gait-specific seeding: collects raw predictions, then evaluates across multiple subsets.
    
    Subsets evaluated:
    - ensemble: all data averaged by RegistrationCode
    - activity_X: per activity, averaged by RegistrationCode
    - activity_X_first33/mid33/last33: per activity + sequence position tercile
    
    Args:
        x: Feature DataFrame with MultiIndex (RegistrationCode, research_stage)
           Must contain 'activity' and 'seq_idx' columns
        y: Target DataFrame
        folds: List of fold definitions [seed_index][fold_index] = (train_ids, test_ids)
        model_key: Model type to use ('all' to tune across all models, or specific model name)
        gender_split_evaluation: Whether to evaluate separately by gender
        save_dir: Base directory - results saved to {save_dir}/../{subset_name}/
        params: Model hyperparameters (if None, will be tuned)
        testing: If True, use preset params (skips tuning for faster testing)
    
    Returns:
        Dictionary with subset results
    """
    if gender_split_evaluation and 'gender' not in x.columns:
        print("gender split evaluation is True but gender column is not in x.columns, setting gender_split_evaluation to False")
        gender_split_evaluation = False

    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    regressions = Regressions()
    label_type = None
    gait_collector = None
    example_model = None
    example_model_key = None
    example_seed = None
    if testing:
        print("testing gait seeding")
    # Phase 1: Collect raw predictions from all folds
    for fold_index, fold in enumerate(folds):
        
        # Run cross-validation (label_type auto-detected from y)
        payload = regressions.cross_validate_model_time(x=x, y=y, cv_fold=fold, model_key=model_key, params=params, testing=testing)
        if example_model is None:
            example_model = payload['model']
            example_model_key = payload['model_key']
            example_seed = fold_index
        # Handle skipped targets (e.g., single class)
        if payload.get('skipped'):
            print(f"Skipping target: {payload.get('skip_reason', 'unknown reason')}")
            if save_dir is not None:
                base_dir = os.path.dirname(save_dir)
                ensemble_dir = os.path.join(base_dir, 'ensemble')
                os.makedirs(ensemble_dir, exist_ok=True)
                pd.DataFrame().to_csv(os.path.join(ensemble_dir, 'predictions.csv'))
                pd.DataFrame({'skipped': [True], 'reason': [payload.get('skip_reason')]}).to_csv(os.path.join(ensemble_dir, 'metrics.csv'))
            return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame(), 'skipped': True}
        
        # Initialize collector on first successful fold
        if gait_collector is None:
            label_type = payload['label_type']
            model_key = payload['model_key']
            gait_collector = GaitMetricsCollector(label_type=label_type, seeds=range(len(folds)), model_key=model_key)
        
        # Add raw predictions to collector
        gait_collector.add_seed_predictions(
            seed=fold_index,
            x=x,
            y=y,
            predictions=payload['predictions']
        )

    # Phase 2: Evaluate all subsets and save
    if save_dir is not None and gait_collector is not None:
        gait_collector.save_all(save_dir, original_save_dir=save_dir, folds=folds)
        # save the example model, can get the fold from the folds
        joblib.dump(example_model, os.path.join(save_dir, f'example_model_seed_{example_seed}.pkl'))



    if run_explain_model:
        explain_model(example_model, example_model_key, folds[example_seed], x, y, save_dir, 'age')
    
    # Return results for compatibility
    if gait_collector is not None:
        results = gait_collector.evaluate_all_subsets(gender_split=gender_split_evaluation)
        return {'subsets': results, 'label_type': label_type}
    
    return {'predictions': pd.DataFrame(), 'metrics': pd.DataFrame()}