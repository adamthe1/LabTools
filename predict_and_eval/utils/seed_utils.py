import os
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

N_THREADS_MODEL = 1


def benjamini_hochberg_correction(pvalues):
    """
    Apply Benjamini-Hochberg correction to p-values.

    Args:
        pvalues: List or array of p-values

    Returns:
        adjusted_pvalues: Array of adjusted p-values
    """
    pvalues = np.array(pvalues)
    n = len(pvalues)

    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = pvalues[sorted_indices]

    # Calculate adjusted p-values
    adjusted_pvalues = np.empty_like(pvalues)
    for i, p in enumerate(sorted_pvalues):
        adjusted_pvalues[sorted_indices[i]] = min(1, p * n / (i + 1))

    # Ensure monotonicity
    for i in range(n - 2, -1, -1):
        adjusted_pvalues[sorted_indices[i]] = min(adjusted_pvalues[sorted_indices[i]],
                                                  adjusted_pvalues[sorted_indices[i + 1]])

    return adjusted_pvalues


def list_files_by_subdirectory(base_directory):
    """List files organized by subdirectory."""
    files_dict = {}
    for root, dirs, files in os.walk(base_directory):
        if root != base_directory:
            subdirectory = os.path.basename(root)
            file_paths = [os.path.join(root, file) for file in files]
            files_dict[subdirectory] = file_paths
    return files_dict


def silence_lightgbm_warnings():
    """Configure logging to suppress specific LightGBM warnings."""
    if lgb is None:
        return
    
    import logging

    class LGBMFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            if "Unknown parameter" in message:
                return False
            if "[LightGBM] [Warning]" in message and ("remove these" in message or "auto_col_wise" in message):
                return False
            if "[LightGBM] [Warning] No further splits with positive gain" in message:
                return False
            if "[LightGBM] [Warning]" in message and "best gain: -inf" in message:
                return False
            return True

    lgbm_logger = logging.getLogger('lightgbm')
    lgbm_logger.addFilter(LGBMFilter())

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


def average_scores_by_subject_id(all_subject_ids, all_true, all_pred, all_pred_probas=None):
    """
    Average predictions and true values by subject ID, with consistent sorting.

    Args:
        all_subject_ids: List of subject IDs (may contain duplicates)
        all_true: List of true values corresponding to all_subject_ids
        all_pred: List of predictions corresponding to all_subject_ids
        all_pred_probas: Optional list of probability predictions

    Returns:
        unique_ids: Array of unique subject IDs (sorted)
        avg_true: Array of averaged true values per unique ID
        avg_pred: Array of averaged predictions per unique ID
    """
    all_subject_ids = np.array(all_subject_ids)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    unique_ids = np.sort(np.unique(all_subject_ids))
    avg_true = np.zeros(len(unique_ids))
    avg_pred = np.zeros(len(unique_ids))

    for i, uid in enumerate(unique_ids):
        mask = (all_subject_ids == uid)
        avg_true[i] = np.mean(all_true[mask])
        avg_pred[i] = np.mean(all_pred[mask])

    return unique_ids, avg_true, avg_pred


def remove_processed(save_dir, labels, task_types):
    """Filter out already processed targets from labels list."""
    processed_targets = []

    for root, dirs, files in os.walk(save_dir):
        for directory in dirs:
            if directory in labels:
                processed_targets.append(directory)

    print(f"Already processed {len(processed_targets)} targets:")
    print(processed_targets)
    keep = ['age', 'gender', 'bmi']
    remaining_labels = [label for label in labels if label not in processed_targets or label in keep]
    remaining_task_types = [task_types[labels.index(label)] for label in remaining_labels]
    return remaining_labels, remaining_task_types
