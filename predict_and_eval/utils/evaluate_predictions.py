import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, balanced_accuracy_score
)
import pandas as pd
from typing import Dict, Any, Tuple


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive regression evaluation metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all regression metrics:
        - pearson_r: Pearson correlation coefficient
        - pearson_pvalue: P-value for Pearson correlation
        - spearman_r: Spearman correlation coefficient
        - spearman_pvalue: P-value for Spearman correlation
        - r2: R-squared score
        - mae: Mean Absolute Error
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
    """
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    metrics = {
        'pearson_r': float(pearson_r),
        'pearson_pvalue': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }
    
    return metrics


def evaluate_classification(y_true: np.ndarray, 
                           y_pred: np.ndarray = None,
                           y_pred_proba: np.ndarray = None,
                           average: str = 'binary') -> Dict[str, Any]:
    """
    Comprehensive classification evaluation metrics.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_pred_proba: Predicted probabilities (optional, needed for AUC)
        average: Averaging method for multiclass ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
        Dictionary with all classification metrics:
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (handles class imbalance)
        - f1: F1 score
        - precision: Precision score
        - recall: Recall score
        - auc: Area Under ROC Curve (if y_pred_proba provided)
    """

    if y_pred is None and y_pred_proba is None:
        raise ValueError("Either y_pred or y_pred_proba must be provided")
    if y_pred is None:
        # Handle both 1D (binary proba of positive class) and 2D (class probabilities) arrays
        if y_pred_proba.ndim == 1:
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Compute AUC first (primary metric for classification)
    auc_val = None
    auc_error = None
    if y_pred_proba is not None:
        try:
            proba_for_auc = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
            if average == 'binary':
                auc_val = float(roc_auc_score(y_true, proba_for_auc))
            else:
                auc_val = float(roc_auc_score(y_true, y_pred_proba, average=average, multi_class='ovr'))
        except (ValueError, IndexError) as e:
            auc_error = str(e)
    
    # AUC first, then other metrics
    metrics = {'auc': auc_val}
    if auc_error:
        metrics['auc_error'] = auc_error
    
    metrics.update({
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0))
    })
    
    return metrics


def evaluate_regression_with_gender_split(gender: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate regression performance with gender split.
    
    Args:
        gender: Array of gender values (1=male, 0/2=female), same length as y_true/y_pred
        y_true: True target values
        y_pred: Predicted values
    """
    gender = np.asarray(gender)
    male_mask = gender == 1
    female_mask = ~male_mask
    
    y_true_male = y_true[male_mask]
    y_true_female = y_true[female_mask]
    y_pred_male = y_pred[male_mask]
    y_pred_female = y_pred[female_mask]
    
    metrics_male = evaluate_regression(y_true_male, y_pred_male)
    metrics_female = evaluate_regression(y_true_female, y_pred_female)
    metrics_combined = evaluate_regression(y_true, y_pred)
    
    metrics = {
        **{f'male_{key}': value for key, value in metrics_male.items()},
        **{f'female_{key}': value for key, value in metrics_female.items()},
        **{f'{key}': value for key, value in metrics_combined.items()}
    }
    return metrics

def evaluate_classification_with_gender_split(gender: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray = None, y_pred_proba: np.ndarray = None, average: str = 'binary') -> Dict[str, Any]:
    """
    Evaluate classification performance with gender split.
    
    Args:
        gender: Array of gender values (1=male, 0/2=female), same length as y_true/y_pred
        y_true: True class labels
        y_pred: Predicted class labels (optional if y_pred_proba provided)
        y_pred_proba: Predicted probabilities (optional if y_pred provided)
        average: Averaging method for multiclass
    """
    gender = np.asarray(gender)
    male_mask = gender == 1
    female_mask = ~male_mask
    
    y_true_male = y_true[male_mask]
    y_true_female = y_true[female_mask]
    y_pred_male = y_pred[male_mask] if y_pred is not None else None
    y_pred_female = y_pred[female_mask] if y_pred is not None else None
    y_pred_proba_male = y_pred_proba[male_mask] if y_pred_proba is not None else None
    y_pred_proba_female = y_pred_proba[female_mask] if y_pred_proba is not None else None
    
    metrics_male = evaluate_classification(y_true_male, y_pred_male, y_pred_proba_male, average)
    metrics_female = evaluate_classification(y_true_female, y_pred_female, y_pred_proba_female, average)
    metrics_combined = evaluate_classification(y_true, y_pred, y_pred_proba, average)
    
    metrics = {
        **{f'male_{key}': value for key, value in metrics_male.items()},
        **{f'female_{key}': value for key, value in metrics_female.items()},
        **{f'{key}': value for key, value in metrics_combined.items()}
    }
    return metrics


def evaluate_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate ordinal prediction using Spearman correlation.
    
    Ordinal targets are ordered categories (e.g., questionnaire responses 1-5).
    Spearman correlation is appropriate as it measures monotonic relationships
    without assuming equal intervals between categories.
    
    Args:
        y_true: True ordinal values (integer class labels)
        y_pred: Predicted values (can be continuous expected values or class predictions)
        
    Returns:
        Dictionary with ordinal metrics:
        - spearman_r: Spearman correlation coefficient
        - spearman_pvalue: P-value for Spearman correlation
    """
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    return {
        'spearman_r': float(spearman_r),
        'spearman_pvalue': float(spearman_p),
    }


def evaluate_ordinal_with_gender_split(gender: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate ordinal prediction performance with gender split.
    
    Args:
        gender: Array of gender values (1=male, 0/2=female), same length as y_true/y_pred
        y_true: True ordinal values
        y_pred: Predicted values
    """
    gender = np.asarray(gender)
    male_mask = gender == 1
    female_mask = ~male_mask
    
    y_true_male = y_true[male_mask]
    y_true_female = y_true[female_mask]
    y_pred_male = y_pred[male_mask]
    y_pred_female = y_pred[female_mask]
    
    metrics_male = evaluate_ordinal(y_true_male, y_pred_male)
    metrics_female = evaluate_ordinal(y_true_female, y_pred_female)
    metrics_combined = evaluate_ordinal(y_true, y_pred)
    
    return {
        **{f'male_{key}': value for key, value in metrics_male.items()},
        **{f'female_{key}': value for key, value in metrics_female.items()},
        **{f'{key}': value for key, value in metrics_combined.items()}
    }


def average_scores_by_subject_id_research_stage(all_subject_ids: pd.MultiIndex, y: np.ndarray, predictions: np.ndarray) -> Tuple[pd.MultiIndex, np.ndarray, np.ndarray]:
    """
    Average scores by (RegistrationCode, research_stage) pairs.
    
    Args:
        all_subject_ids: MultiIndex with (RegistrationCode, research_stage)
        y: Target values
        predictions: Predicted values (1D for regression/ordinal, 2D for classification proba)
        
    Returns:
        Tuple of (unique_pairs, averaged_y, averaged_predictions)
    """
    unique_pairs = all_subject_ids.unique()
    
    y_avg = []
    predictions_avg = []
    is_2d = predictions.ndim == 2
    
    for pair in unique_pairs:
        mask = all_subject_ids == pair
        y_avg.append(np.mean(y[mask]))
        # Preserve 2D shape for classification probabilities
        if is_2d:
            predictions_avg.append(np.mean(predictions[mask], axis=0))
        else:
            predictions_avg.append(np.mean(predictions[mask]))
    
    return unique_pairs, np.array(y_avg), np.array(predictions_avg)


def get_gender_for_index(x: pd.DataFrame, index: pd.MultiIndex) -> np.ndarray:
    """
    Extract gender values from x for the given unique index pairs.
    
    Args:
        x: DataFrame with 'gender' column and MultiIndex (RegistrationCode, research_stage)
        index: MultiIndex of unique pairs to extract gender for
        
    Returns:
        Array of gender values, one per unique pair in index
    """
    # Group by index and take first gender value (gender is constant per subject)
    gender_per_pair = x.groupby(level=[0, 1])['gender'].first()
    return gender_per_pair.loc[index].values