import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from typing import List, Tuple


def ids_folds(reg_code_labels_df: pd.DataFrame, seeds: range = range(10), n_splits: int = 5) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create deterministic subject-level cross-validation fold definitions.
    
    PURPOSE: Creates the "blueprint" for how subjects should be split into train/test sets.
    Call this ONCE at the beginning, then reuse these fold definitions across all experiments.
    
    WHY: 
    - REPRODUCIBILITY: Same seeds always produce same folds
    - CONSISTENCY: Use same subject splits across different experiments/features/targets
    - NO DATA LEAKAGE: Subjects are split at ID level, not sample level
    - MULTIPLE RUNS: Generate multiple random splits (one per seed) for robust evaluation
    
    EXAMPLE:
    subject_ids = ['A', 'A', 'A', 'B', 'B', 'C']  # 6 samples, 3 subjects
    folds = ids_folds(subject_ids, seeds=range(2), n_splits=3)
    
    folds[0][0] = (['A', 'B'], ['C'])  # Seed 0, Fold 0: train on A,B, test on C
    folds[1][0] = (['B', 'C'], ['A'])  # Seed 1, Fold 0: different shuffle
    
    Args:
        reg_code_labels_df: DataFrame with index as (RegistrationCode, research_stage(optional)) and label columns
        seeds: Range of random seeds (e.g., range(10) for 10 configurations)
        n_splits: Number of CV folds (e.g., 5 for 5-fold CV)
        
    Returns:
        [seed_index][fold_index] = (train_reg_codes, test_reg_codes)
    """
    # Extract RegistrationCode (level 0 of MultiIndex or just the index)
    if isinstance(reg_code_labels_df.index, pd.MultiIndex):
        subject_ids = reg_code_labels_df.index.get_level_values(0).values
    else:
        subject_ids = reg_code_labels_df.index.values
    
    unique_subject_ids = np.sort(np.unique(subject_ids))
    all_folds = []
    
    for seed in seeds:
        rng = np.random.RandomState(seed)
        shuffled_ids = rng.permutation(unique_subject_ids)
        kf = KFold(n_splits=n_splits, shuffle=False)
        
        seed_folds = []
        for train_idx, test_idx in kf.split(shuffled_ids):
            train_subjects = shuffled_ids[train_idx]
            test_subjects = shuffled_ids[test_idx]
            seed_folds.append((train_subjects, test_subjects))
        
        all_folds.append(seed_folds)
    
    return all_folds

def stratified_ids_folds(reg_code_labels_df: pd.DataFrame, seeds: range = range(10), n_splits: int = 5) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create stratified subject-level cross-validation fold definitions. (classification only)
    Important: If a subject has multiple labels, the label with the most occurrences will be used.
    
    PURPOSE: Like ids_folds() but maintains balanced class distribution across folds.
    Creates the "blueprint" for stratified subject splits that can be reused across experiments.
    
    WHY:
    - REPRODUCIBILITY: Same seeds always produce same stratified folds
    - CONSISTENCY: Use same subject splits across different experiments
    - NO DATA LEAKAGE: Subjects are split at ID level, not sample level
    - BALANCED FOLDS: Each fold has similar class distribution
    - MULTIPLE RUNS: Generate multiple random stratified splits for robust evaluation
    
    EXAMPLE:
    df = pd.DataFrame({
        'label': [0, 0, 1, 1, 1, 0, 0, 1]
    }, index=pd.MultiIndex.from_tuples([
        ('A', 'baseline'), ('A', 'followup'),
        ('B', 'baseline'), ('B', 'followup'), ('B', 'followup2'),
        ('C', 'baseline'), ('C', 'followup'),
        ('D', 'baseline')
    ], names=['RegistrationCode', 'research_stage']))
    
    folds = stratified_ids_folds(df, seeds=range(2), n_splits=2)
    # Subject A: 2 samples, both label 0 → majority label = 0
    # Subject B: 3 samples, 2 are label 1 → majority label = 1
    # Subject C: 2 samples, both label 0 → majority label = 0
    # Subject D: 1 sample, label 1 → majority label = 1
    # Result: 2 subjects with label 0, 2 subjects with label 1
    # Each fold will have 1 subject from each class
    
    Args:
        reg_code_labels_df: DataFrame with:
            - Index: (RegistrationCode, research_stage) or just RegistrationCode
            - Columns: label column(s) - will use first column
        seeds: Range of random seeds (e.g., range(10) for 10 configurations)
        n_splits: Number of CV folds (e.g., 5 for 5-fold CV)
        
    Returns:
        [seed_index][fold_index] = (train_subject_ids, test_subject_ids)
    """
    import pandas as pd
    
    # Extract RegistrationCode (level 0 of MultiIndex or just the index)
    if isinstance(reg_code_labels_df.index, pd.MultiIndex):
        subject_ids = reg_code_labels_df.index.get_level_values(0).values
    else:
        subject_ids = reg_code_labels_df.index.values
    
    # Get labels (use first column)
    labels = reg_code_labels_df.iloc[:, 0].values
    
    # Get unique subjects and their majority class label
    unique_subjects = np.sort(np.unique(subject_ids))
    subject_labels = []
    
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        subject_sample_labels = labels[subject_mask]
        
        # Use majority vote for subject-level label
        unique_vals, counts = np.unique(subject_sample_labels, return_counts=True)
        majority_label = unique_vals[np.argmax(counts)]
        subject_labels.append(majority_label)
    
    subject_labels = np.array(subject_labels)
    
    # Create stratified folds for each seed
    all_folds = []
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        seed_folds = []
        for train_subject_idx, test_subject_idx in skf.split(unique_subjects, subject_labels):
            train_subjects = unique_subjects[train_subject_idx]
            test_subjects = unique_subjects[test_subject_idx]
            seed_folds.append((train_subjects, test_subjects))
        
        all_folds.append(seed_folds)
    
    return all_folds

def id_fold_with_stratified_threshold(reg_code_labels_df: pd.DataFrame, seeds: range = range(10), n_splits: int = 5, stratified_threshold: float = 0.2) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Create subject-level cross-validation fold definitions with stratified threshold.
    
    Stratification is only applied for binary classification when minority class
    proportion is below the threshold. Ordinal and regression targets use regular folds.
    """
    if stratified_threshold == -1:
        return ids_folds(reg_code_labels_df, seeds=seeds, n_splits=n_splits)
    
    labels = reg_code_labels_df.iloc[:, 0].values.ravel()  # Flatten to 1D
    
    # Filter NaN regardless of dtype (pd.notna works for any dtype)
    valid_mask = pd.notna(labels)
    valid_labels = labels[valid_mask]
    unique_values = np.unique(valid_labels)
    
    # Only stratify for binary classification (exactly 2 unique values)
    if len(unique_values) != 2:
        return ids_folds(reg_code_labels_df, seeds=seeds, n_splits=n_splits)
    
    # Check minority class proportion for binary targets (using valid labels only)
    n_positive = np.sum(valid_labels == unique_values[1])
    n_negative = np.sum(valid_labels == unique_values[0])
    minority_ratio = min(n_positive, n_negative) / max(n_positive, n_negative)
    
    if minority_ratio < stratified_threshold:
        return stratified_ids_folds(reg_code_labels_df, seeds=seeds, n_splits=n_splits)
    else:
        return ids_folds(reg_code_labels_df, seeds=seeds, n_splits=n_splits)
    

def create_cv_folds(id_folds: List[Tuple[np.ndarray, np.ndarray]], 
                    subject_ids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert subject-level fold definitions to sample-level indices.
    
    Maps abstract fold definitions (which subjects go where) to concrete sample indices
    that you can use to slice your X and y arrays. Handles duplicate subject IDs correctly.
    
    EXAMPLE:
    subject_ids = ['A', 'A', 'B', 'B', 'C', 'C']  # 6 samples from 3 subjects
    id_folds = [(['A', 'B'], ['C'])]  # Train on A,B subjects, test on C
    cv_folds = create_cv_folds(id_folds, subject_ids)
    
    cv_folds[0] = ([0, 1, 2, 3], [4, 5])  # Train indices for A,B samples, test indices for C samples
    # Now use: X_train = X[cv_folds[0][0]], X_test = X[cv_folds[0][1]]
    
    Args:
        id_folds: List of (train_subjects, test_subjects) tuples from ids_folds()
        subject_ids: Array of subject IDs for your current dataset (can have duplicates)
        
    Returns:
        List of (train_indices, test_indices) tuples ready for array slicing
    """
    # Pre-compute subject -> indices mapping once (O(n) instead of O(n*k) for k folds)
    subject_to_indices = {}
    for idx, subj in enumerate(subject_ids):
        if subj not in subject_to_indices:
            subject_to_indices[subj] = []
        subject_to_indices[subj].append(idx)
    
    cv_folds = []
    for train_subjects, test_subjects in id_folds:
        train_idx = np.concatenate([subject_to_indices[s] for s in train_subjects if s in subject_to_indices])
        test_idx = np.concatenate([subject_to_indices[s] for s in test_subjects if s in subject_to_indices])
        cv_folds.append((train_idx, test_idx))
    return cv_folds


def create_cv_folds_by_seed(id_folds: List[List[Tuple[np.ndarray, np.ndarray]]], 
                            subject_ids: np.ndarray, 
                            seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Get CV folds for a specific random seed.
    
    Convenience function that selects fold configuration for specified seed
    and converts to sample-level indices in one call.
    
    Args:
        id_folds: All fold configurations from ids_folds()
        subject_ids: Array of subject IDs for your current dataset
        seed: Index of which seed configuration to use (0 to len(seeds)-1)
        
    Returns:
        List of (train_indices, test_indices) tuples for the specified seed
    """
    if not 0 <= seed < len(id_folds):
        raise ValueError(f"seed must be between 0 and {len(id_folds)-1}, got {seed}")
    return create_cv_folds(id_folds[seed], subject_ids)


def create_stratified_cv_folds(subject_ids: np.ndarray, 
                               labels: np.ndarray, 
                               n_splits: int = 5,
                               random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create stratified subject-level cross-validation folds on-the-fly.
    
    Unlike ids_folds(), this creates folds immediately based on subject-level class labels,
    ensuring balanced label distribution across folds. For classification tasks only.
    
    EXAMPLE:
    subject_ids = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    labels = [1, 1, 1, 0, 0, 1, 1, 0, 0]  # Binary classification labels
    cv_folds = create_stratified_cv_folds(subject_ids, labels, n_splits=2)
    # Returns folds where each fold has balanced class distribution
    
    Args:
        subject_ids: Array of subject IDs (can have duplicates - one per sample)
        labels: Array of class labels (one per sample, same length as subject_ids)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) tuples ready for array slicing
    """
    
    # Get unique subjects and their majority class label
    unique_subjects = np.unique(subject_ids)
    subject_labels = []
    
    for subject in unique_subjects:
        subject_mask = subject_ids == subject
        subject_sample_labels = labels[subject_mask]
        
        # Use majority vote for subject-level label
        unique_vals, counts = np.unique(subject_sample_labels, return_counts=True)
        majority_label = unique_vals[np.argmax(counts)]
        subject_labels.append(majority_label)
    
    subject_labels = np.array(subject_labels)
    
    # Create stratified folds at subject level
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_folds = []
    for train_subject_idx, test_subject_idx in skf.split(unique_subjects, subject_labels):
        train_subjects = unique_subjects[train_subject_idx]
        test_subjects = unique_subjects[test_subject_idx]
        
        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        
        cv_folds.append((train_idx, test_idx))
    
    return cv_folds


def create_simple_cv_folds(subject_ids: np.ndarray, 
                           n_splits: int = 5,
                           random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create simple subject-level cross-validation folds on-the-fly (non-stratified).
    
    Quick and simple subject-level CV without pre-creating folds or stratification.
    Use this when you don't need to reuse folds across experiments and don't need
    balanced label distribution.
    
    EXAMPLE:
    subject_ids = ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    cv_folds = create_simple_cv_folds(subject_ids, n_splits=3, random_state=42)
    
    for train_idx, test_idx in cv_folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    Args:
        subject_ids: Array of subject IDs (can have duplicates - one per sample)
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) tuples ready for array slicing
    """
    # Get unique subjects and shuffle them
    unique_subjects = np.sort(np.unique(subject_ids))
    rng = np.random.RandomState(random_state)
    shuffled_subjects = rng.permutation(unique_subjects)
    
    # Create folds at subject level
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    cv_folds = []
    for train_subject_idx, test_subject_idx in kf.split(shuffled_subjects):
        train_subjects = shuffled_subjects[train_subject_idx]
        test_subjects = shuffled_subjects[test_subject_idx]
        
        # Map to sample indices
        train_idx = np.where(np.isin(subject_ids, train_subjects))[0]
        test_idx = np.where(np.isin(subject_ids, test_subjects))[0]
        
        cv_folds.append((train_idx, test_idx))
    
    return cv_folds


def save_folds(folds: list, save_dir: str):
    """Save fold definitions to JSON for reproducibility."""
    folds_serializable = [
        [(train.tolist(), test.tolist()) for train, test in seed_folds]
        for seed_folds in folds
    ]
    with open(os.path.join(save_dir, 'folds.json'), 'w') as f:
        json.dump(folds_serializable, f)