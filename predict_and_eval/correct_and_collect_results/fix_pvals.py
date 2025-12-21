import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold


def fix_pvals(dirs):
    """
    Collect all p-value columns from metrics.csv files across directories,
    apply FDR correction separately for each (feature_system, pvalue_column) combination.
    
    FDR correction is applied per:
    - feature_system: each feature/run system is a separate hypothesis family
    - pvalue_column: implicitly separates by gender (all, male, female via column name)
    
    Args:
        dirs: List of directory paths containing metrics.csv files
              Expected path format: .../target_system/label_name/feature_name/metrics.csv
    """
    # Collect all p-value data grouped by (feature_system, pvalue_column)
    pvalue_data = []
    
    for dir_path in dirs:
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        if not os.path.exists(metrics_file):
            continue
        
        # Extract feature_system from path (last directory component)
        feature_system = os.path.basename(dir_path)
            
        df = pd.read_csv(metrics_file)
        
        # Find all columns containing 'pvalue' in the name (exclude already corrected)
        pvalue_cols = [col for col in df.columns if 'pvalue' in col.lower() and 'fdr' not in col.lower()]
        
        for col in pvalue_cols:
            for idx, row in df.iterrows():
                pvalue_data.append({
                    'dir': dir_path,
                    'row_idx': idx,
                    'feature_system': feature_system,
                    'pvalue_col': col,
                    'pvalue': row[col]
                })
    
    if not pvalue_data:
        return
    
    pvalue_df = pd.DataFrame(pvalue_data)
    
    # Apply FDR correction separately for each (feature_system, pvalue_col) group
    dir_updates = {}
    
    for (feature_system, pvalue_col), group in pvalue_df.groupby(['feature_system', 'pvalue_col']):
        pvals = group['pvalue'].values
        valid_mask = ~np.isnan(pvals)
        
        if not valid_mask.any():
            continue
        
        # Apply FDR correction (Benjamini-Hochberg)
        corrected = np.full(len(pvals), np.nan)
        _, corrected[valid_mask], _, _ = multipletests(pvals[valid_mask], method='fdr_bh')
        
        fdr_col_name = pvalue_col.replace('pvalue', 'pvalue_fdr')
        
        # Map corrected p-values back to their locations
        for i, (_, item) in enumerate(group.iterrows()):
            dir_path = item['dir']
            row_idx = item['row_idx']
            
            if dir_path not in dir_updates:
                dir_updates[dir_path] = {}
            if fdr_col_name not in dir_updates[dir_path]:
                dir_updates[dir_path][fdr_col_name] = {}
            
            dir_updates[dir_path][fdr_col_name][row_idx] = corrected[i]
    
    # Write corrected p-values back to files
    for dir_path, fdr_columns in dir_updates.items():
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        df = pd.read_csv(metrics_file)
        
        # Add FDR-corrected columns
        for fdr_col_name, row_values in fdr_columns.items():
            df[fdr_col_name] = [row_values.get(i, np.nan) for i in range(len(df))]
        
        df.to_csv(metrics_file, index=False)


def fix_pvals_grouped(dirs):
    """
    Apply FDR correction with equal budget per target_system group (weighted BH).
    
    Uses weighted BH where w_i = m / (G * m_g):
    - m = total tests across all groups
    - G = number of groups (target_systems)
    - m_g = tests in group g
    
    This gives each target_system equal total weight (m/G), preventing larger
    groups from dominating smaller ones in the FDR correction.
    
    Args:
        dirs: List of directory paths containing metrics.csv files
              Expected path format: .../target_system/label_name/feature_name/metrics.csv
    """
    # Collect p-values with target_system info
    pvalue_data = []
    
    for dir_path in dirs:
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        if not os.path.exists(metrics_file):
            continue
        
        # Extract from path: .../target_system/label/feature/metrics.csv
        path_parts = dir_path.split(os.sep)
        feature_system = path_parts[-1]  # last component
        target_system = path_parts[-3]   # 3rd from end
            
        df = pd.read_csv(metrics_file)
        
        # Find pvalue columns (exclude already corrected ones)
        pvalue_cols = [col for col in df.columns if 'pvalue' in col.lower() and 'fdr' not in col.lower()]
        
        for col in pvalue_cols:
            for idx, row in df.iterrows():
                pvalue_data.append({
                    'dir': dir_path,
                    'row_idx': idx,
                    'feature_system': feature_system,
                    'target_system': target_system,
                    'pvalue_col': col,
                    'pvalue': row[col]
                })
    
    if not pvalue_data:
        return
    
    pvalue_df = pd.DataFrame(pvalue_data)
    dir_updates = {}
    
    # Apply weighted FDR separately for each (feature_system, pvalue_col) combination
    for (feature_system, pvalue_col), group in pvalue_df.groupby(['feature_system', 'pvalue_col']):
        group = group.reset_index(drop=True)  # Reset for clean indexing
        valid_mask = ~np.isnan(group['pvalue'].values)
        if not valid_mask.any():
            continue
        
        # Compute weights: w_i = m / (G * m_g)
        m = valid_mask.sum()  # total valid tests
        valid_targets = group.loc[valid_mask, 'target_system']
        group_counts = valid_targets.value_counts()
        G = len(group_counts)  # number of groups
        
        if G == 0:
            continue
        
        # Map each test to its weight
        weights = np.full(len(group), np.nan)
        for i, row in group.iterrows():
            if valid_mask[i]:
                m_g = group_counts[row['target_system']]
                weights[i] = m / (G * m_g)
        
        # Transform p-values: p* = min(1, p / w)
        pvals = group['pvalue'].values
        weighted_pvals = np.full(len(pvals), np.nan)
        weighted_pvals[valid_mask] = np.minimum(1.0, pvals[valid_mask] / weights[valid_mask])
        
        # Apply standard BH on weighted p-values
        corrected = np.full(len(pvals), np.nan)
        _, corrected[valid_mask], _, _ = multipletests(weighted_pvals[valid_mask], method='fdr_bh')
        
        fdr_col_name = pvalue_col.replace('pvalue', 'pvalue_fdr_grouped')
        
        # Map corrected values back to locations
        for i, item in group.iterrows():
            dir_path = item['dir']
            row_idx = item['row_idx']
            
            if dir_path not in dir_updates:
                dir_updates[dir_path] = {}
            if fdr_col_name not in dir_updates[dir_path]:
                dir_updates[dir_path][fdr_col_name] = {}
            
            dir_updates[dir_path][fdr_col_name][row_idx] = corrected[i]
    
    # Write back to files
    for dir_path, fdr_columns in dir_updates.items():
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        df = pd.read_csv(metrics_file)
        
        for fdr_col_name, row_values in fdr_columns.items():
            df[fdr_col_name] = [row_values.get(i, np.nan) for i in range(len(df))]
        
        df.to_csv(metrics_file, index=False)

def fix_pvals_ihw(dirs, n_folds=5, alpha=0.05):
    """
    Apply IHW (Independent Hypothesis Weighting) FDR correction.
    
    Data-driven weights learned via cross-validation:
    - Groups with better signal (lower pi0) get higher weights
    - Groups with mostly noise get lower weights
    - Prevents overfitting by learning weights on held-out folds
    
    Args:
        dirs: List of directory paths containing metrics.csv files
              Expected path format: .../target_system/label_name/feature_name/metrics.csv
        n_folds: Number of CV folds for weight learning
        alpha: FDR level for BH procedure
    """
    # Collect p-values with target_system (covariate)
    pvalue_data = []
    
    for dir_path in dirs:
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        if not os.path.exists(metrics_file):
            continue
        
        path_parts = dir_path.split(os.sep)
        feature_system = path_parts[-1]
        target_system = path_parts[-3]
            
        df = pd.read_csv(metrics_file)
        pvalue_cols = [col for col in df.columns if 'pvalue' in col.lower() and 'fdr' not in col.lower()]
        
        for col in pvalue_cols:
            for idx, row in df.iterrows():
                pvalue_data.append({
                    'dir': dir_path,
                    'row_idx': idx,
                    'feature_system': feature_system,
                    'target_system': target_system,
                    'pvalue_col': col,
                    'pvalue': row[col]
                })
    
    if not pvalue_data:
        return
    
    pvalue_df = pd.DataFrame(pvalue_data)
    dir_updates = {}
    
    for (feature_system, pvalue_col), group in pvalue_df.groupby(['feature_system', 'pvalue_col']):
        group = group.reset_index(drop=True)
        valid_mask = ~np.isnan(group['pvalue'].values)
        
        if valid_mask.sum() < 10:  # Need minimum tests for IHW
            continue
        
        pvals = group['pvalue'].values
        groups_arr = group['target_system'].values
        
        # Get valid data only
        valid_pvals = pvals[valid_mask]
        valid_groups = groups_arr[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        # Check we have enough groups for stratified CV
        unique_groups, group_counts = np.unique(valid_groups, return_counts=True)
        group_count_map = dict(zip(unique_groups, group_counts))
        
        if len(unique_groups) < 2:
            # Single group - fall back to standard BH
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid_mask], _, _ = multipletests(valid_pvals, method='fdr_bh')
        else:
            # Separate "orphan" groups (size < 2) from CV-eligible groups
            orphan_groups = {g for g, c in group_count_map.items() if c < 2}
            cv_groups = {g for g, c in group_count_map.items() if c >= 2}
            
            # If not enough CV-eligible groups, fall back to standard BH
            if len(cv_groups) < 2:
                corrected = np.full(len(pvals), np.nan)
                _, corrected[valid_mask], _, _ = multipletests(valid_pvals, method='fdr_bh')
            else:
                # IHW with cross-validation on eligible groups only
                corrected = np.full(len(pvals), np.nan)
                weights = np.ones(len(valid_pvals))  # Default neutral weight
                
                # Mask for CV-eligible samples
                cv_mask = np.array([g in cv_groups for g in valid_groups])
                cv_pvals = valid_pvals[cv_mask]
                cv_groups_arr = valid_groups[cv_mask]
                cv_indices = np.where(cv_mask)[0]
                
                # Compute n_folds based on CV-eligible groups only
                cv_group_counts = np.array([group_count_map[g] for g in cv_groups])
                min_cv_group_size = cv_group_counts.min()
                actual_folds = min(n_folds, min_cv_group_size)
                actual_folds = max(2, actual_folds)
                
                # Convert groups to numeric for stratified split
                cv_unique = sorted(cv_groups)
                group_to_idx = {g: i for i, g in enumerate(cv_unique)}
                numeric_groups = np.array([group_to_idx[g] for g in cv_groups_arr])
                
                # K-fold CV: learn weights on K-1 folds, apply to held-out fold
                skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
                cv_weights = np.zeros(len(cv_pvals))
                
                for train_idx, test_idx in skf.split(cv_pvals, numeric_groups):
                    train_pvals = cv_pvals[train_idx]
                    train_groups = cv_groups_arr[train_idx]
                    test_groups = cv_groups_arr[test_idx]
                    
                    # Learn weights from training fold
                    train_weights = _compute_ihw_weights(train_pvals, train_groups)
                    
                    # Compute weight per group from training
                    group_weight_map = {}
                    for g in cv_unique:
                        g_mask = train_groups == g
                        if g_mask.any():
                            group_weight_map[g] = train_weights[g_mask].mean()
                        else:
                            group_weight_map[g] = 1.0
                    
                    # Apply learned group weights to test fold
                    for i, idx in enumerate(test_idx):
                        g = test_groups[i]
                        cv_weights[idx] = group_weight_map.get(g, 1.0)
                
                # Map CV weights back to full valid array
                for i, idx in enumerate(cv_indices):
                    weights[idx] = cv_weights[i]
                
                # Orphan groups keep weight=1.0 (neutral)
                
                # Normalize weights so sum = m
                m = len(valid_pvals)
                weights = weights * (m / weights.sum())
                
                # Apply weighted BH
                adj_pvals = _apply_weighted_bh(valid_pvals, weights, alpha)
                
                # Map back to full array
                for i, idx in enumerate(valid_indices):
                    corrected[idx] = adj_pvals[i]
        
        fdr_col_name = pvalue_col.replace('pvalue', 'pvalue_fdr_ihw')
        
        for i, item in group.iterrows():
            dir_path = item['dir']
            row_idx = item['row_idx']
            
            if dir_path not in dir_updates:
                dir_updates[dir_path] = {}
            if fdr_col_name not in dir_updates[dir_path]:
                dir_updates[dir_path][fdr_col_name] = {}
            
            dir_updates[dir_path][fdr_col_name][row_idx] = corrected[i]
    
    # Write IHW results back to files
    for dir_path, fdr_columns in dir_updates.items():
        metrics_file = os.path.join(dir_path, 'metrics.csv')
        df = pd.read_csv(metrics_file)
        
        for fdr_col_name, row_values in fdr_columns.items():
            df[fdr_col_name] = [row_values.get(i, np.nan) for i in range(len(df))]
        
        df.to_csv(metrics_file, index=False)


def apply_ihw_to_wilcox(df, n_folds=5, alpha=0.05):
    """
    Apply IHW to Wilcoxon p-values in comparison DataFrame.
    
    Uses target_system as covariate for data-driven weight learning.
    Groups with better signal get higher weights.
    
    Args:
        df: DataFrame with 'wilcox_pvalue', 'target_system', 'gender', 'features_name' columns
        n_folds: Number of CV folds
        alpha: FDR level
    
    Returns:
        DataFrame with 'wilcox_pvalue_ihw' column added
    """
    df = df.copy()
    df['wilcox_pvalue_ihw'] = np.nan
    
    # Apply IHW separately for each (gender, features_name) combination
    for (gender, features_name), group in df.groupby(['gender', 'features_name']):
        mask = (df['gender'] == gender) & (df['features_name'] == features_name)
        pvals = df.loc[mask, 'wilcox_pvalue'].values
        groups = df.loc[mask, 'target_system'].values
        valid = ~np.isnan(pvals)
        
        if valid.sum() < 10:  # Need minimum for IHW
            continue
        
        valid_pvals = pvals[valid]
        valid_groups = groups[valid]
        
        unique_groups, group_counts = np.unique(valid_groups, return_counts=True)
        group_count_map = dict(zip(unique_groups, group_counts))
        
        if len(unique_groups) < 2:
            # Single group - standard BH
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid], _, _ = multipletests(valid_pvals, method='fdr_bh')
            df.loc[mask, 'wilcox_pvalue_ihw'] = corrected
            continue
        
        # Separate orphan groups (size < 2) from CV-eligible groups
        orphan_groups = {g for g, c in group_count_map.items() if c < 2}
        cv_groups = {g for g, c in group_count_map.items() if c >= 2}
        
        if len(cv_groups) < 2:
            # Not enough CV-eligible groups - standard BH
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid], _, _ = multipletests(valid_pvals, method='fdr_bh')
            df.loc[mask, 'wilcox_pvalue_ihw'] = corrected
            continue
        
        # IHW with cross-validation on eligible groups only
        weights = np.ones(len(valid_pvals))  # Default neutral weight
        
        cv_mask = np.array([g in cv_groups for g in valid_groups])
        cv_pvals = valid_pvals[cv_mask]
        cv_groups_arr = valid_groups[cv_mask]
        cv_indices = np.where(cv_mask)[0]
        
        cv_group_counts = np.array([group_count_map[g] for g in cv_groups])
        min_cv_group_size = cv_group_counts.min()
        actual_folds = min(n_folds, min_cv_group_size)
        actual_folds = max(2, actual_folds)
        
        cv_unique = sorted(cv_groups)
        group_to_idx = {g: i for i, g in enumerate(cv_unique)}
        numeric_groups = np.array([group_to_idx[g] for g in cv_groups_arr])
        
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        cv_weights = np.zeros(len(cv_pvals))
        
        for train_idx, test_idx in skf.split(cv_pvals, numeric_groups):
            train_pvals = cv_pvals[train_idx]
            train_groups = cv_groups_arr[train_idx]
            test_groups = cv_groups_arr[test_idx]
            
            train_weights = _compute_ihw_weights(train_pvals, train_groups)
            
            group_weight_map = {}
            for g in cv_unique:
                g_mask = train_groups == g
                if g_mask.any():
                    group_weight_map[g] = train_weights[g_mask].mean()
                else:
                    group_weight_map[g] = 1.0
            
            for i, idx in enumerate(test_idx):
                g = test_groups[i]
                cv_weights[idx] = group_weight_map.get(g, 1.0)
        
        # Map CV weights back
        for i, idx in enumerate(cv_indices):
            weights[idx] = cv_weights[i]
        
        # Normalize weights
        m = len(valid_pvals)
        weights = weights * (m / weights.sum())
        
        # Apply weighted BH
        adj_pvals = _apply_weighted_bh(valid_pvals, weights, alpha)
        
        # Map back
        corrected = np.full(len(pvals), np.nan)
        corrected[valid] = adj_pvals
        df.loc[mask, 'wilcox_pvalue_ihw'] = corrected
    
    return df



# ============================================================================
# IHW Helper Functions
# ============================================================================

def _estimate_pi0(pvals, lambda_val=0.5):
    """
    Estimate proportion of true null hypotheses using Storey's method.
    
    pi0 = #{p > lambda} / (m * (1 - lambda))
    
    Args:
        pvals: Array of p-values (no NaNs)
        lambda_val: Threshold for estimation (default 0.5)
    
    Returns:
        Estimated pi0 (clipped to [0.1, 1.0] for stability)
    """
    m = len(pvals)
    if m == 0:
        return 1.0
    
    n_above = np.sum(pvals > lambda_val)
    pi0 = n_above / (m * (1 - lambda_val))
    
    # Clip for numerical stability
    return np.clip(pi0, 0.1, 1.0)


def _compute_ihw_weights(pvals, groups, min_weight=0.1):
    """
    Compute IHW weights based on estimated signal density per group.
    
    Groups with more signal (lower pi0) get higher weights.
    Weights are normalized so sum(weights) = m.
    
    Args:
        pvals: Array of p-values
        groups: Array of group labels (same length as pvals)
        min_weight: Minimum weight to prevent zeros
    
    Returns:
        Array of weights (same length as pvals)
    """
    unique_groups = np.unique(groups)
    m = len(pvals)
    
    # Estimate pi0 (null proportion) per group
    group_pi0 = {}
    for g in unique_groups:
        mask = groups == g
        group_pvals = pvals[mask]
        group_pi0[g] = _estimate_pi0(group_pvals)
    
    # Signal proportion = 1 - pi0 (higher is better)
    # Use inverse of pi0 as raw weight (more signal -> higher weight)
    group_raw_weights = {}
    for g in unique_groups:
        # Weight proportional to signal density
        group_raw_weights[g] = max(min_weight, 1.0 / group_pi0[g])
    
    # Assign raw weights to each test
    raw_weights = np.array([group_raw_weights[g] for g in groups])
    
    # Normalize so sum(weights) = m
    weights = raw_weights * (m / raw_weights.sum())
    
    return weights


def _apply_weighted_bh(pvals, weights, alpha=0.05):
    """
    Apply weighted Benjamini-Hochberg procedure.
    
    Transform p-values: p* = min(1, p / w)
    Then apply standard BH on p*.
    
    Args:
        pvals: Array of p-values
        weights: Array of weights (sum should equal len(pvals))
        alpha: FDR level
    
    Returns:
        Array of adjusted p-values
    """
    # Transform p-values
    weighted_pvals = np.minimum(1.0, pvals / weights)
    
    # Apply standard BH
    _, adj_pvals, _, _ = multipletests(weighted_pvals, alpha=alpha, method='fdr_bh')
    
    return adj_pvals


