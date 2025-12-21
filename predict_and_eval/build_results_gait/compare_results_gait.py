"""
Compare results for gait experiments with mixed flat/nested structure support.

Handles both:
- Gait features (nested): main_dir/target_system/label/feature/sub_model/metrics.csv
- Non-gait features (flat): main_dir/target_system/label/feature/metrics.csv
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from ..loading_features.load_feature_df import get_column_info
from ..correct_and_collect_results.fix_pvals import fix_pvals, fix_pvals_grouped, fix_pvals_ihw, apply_ihw_to_wilcox

MIN_SEEDS_FOR_WILCOX = 6


def compare_and_collect_results_gait(main_dir, baseline_features_system_name,
                                     main_reg_score='r2', main_reg_pvalue='pearson_pvalue',
                                     main_clas_score='auc',
                                     main_ordinal_score='spearman_r', main_ordinal_pvalue='spearman_pvalue',
                                     significance_threshold=0.05,
                                     pvalue_correction='fdr'):
    """
    Compare feature systems to baseline, handling mixed flat/nested structures.
    
    Args:
        main_dir: Root directory containing target_system folders
        baseline_features_system_name: Name of baseline feature system
        main_reg_score: Main regression score column
        main_reg_pvalue: Main regression p-value column
        main_clas_score: Main classification score column
        main_ordinal_score: Main ordinal score column
        main_ordinal_pvalue: Main ordinal p-value column
        significance_threshold: Threshold for significance
        pvalue_correction: Method for score p-value correction:
            'fdr' - standard FDR (Benjamini-Hochberg)
            'fdr_grouped' - weighted BH giving equal budget per target_system
            'ihw' - IHW (data-driven weights via cross-validation)
            'all' - apply all methods (fdr, fdr_grouped, ihw)
    
    Returns:
        DataFrame with all comparisons
    """
    # Collect all metrics directories and apply FDR correction to score p-values
    all_dirs = _collect_metrics_dirs_gait(main_dir)
    
    if pvalue_correction == 'all':
        fix_pvals(all_dirs)
        fix_pvals_grouped(all_dirs)
        fix_pvals_ihw(all_dirs)
    elif pvalue_correction == 'fdr':
        fix_pvals(all_dirs)
    elif pvalue_correction == 'fdr_grouped':
        fix_pvals(all_dirs)  # Always include standard FDR
        fix_pvals_grouped(all_dirs)
    elif pvalue_correction == 'ihw':
        fix_pvals(all_dirs)  # Always include standard FDR
        fix_pvals_ihw(all_dirs)
    
    all_results = []
    label_info = []
    
    for target_system in os.listdir(main_dir):
        target_system_path = os.path.join(main_dir, target_system)
        if not os.path.isdir(target_system_path):
            continue
        
        for label_name in os.listdir(target_system_path):
            label_path = os.path.join(target_system_path, label_name)
            if not os.path.isdir(label_path):
                continue
            
            label_results = _process_label_directory_gait(
                label_path, target_system, label_name, baseline_features_system_name,
                main_reg_score, main_reg_pvalue, main_clas_score,
                main_ordinal_score, main_ordinal_pvalue
            )
            
            if label_results:
                all_results.extend(label_results)
                label_info.extend([label_path] * len(label_results))
    
    if not all_results:
        return None
    
    df = pd.DataFrame(all_results)
    df['_label_path'] = label_info
    
    # Apply FDR correction to Wilcoxon p-values (standard BH)
    df = _apply_fdr_correction_to_wilcox_gait(df)
    
    # Apply IHW to Wilcoxon p-values if requested
    if pvalue_correction in ['ihw', 'all']:
        df = apply_ihw_to_wilcox(df)
    
    # is_significant: NaN if wilcox wasn't run (use IHW if available, else FDR)
    wilcox_col = 'wilcox_pvalue_ihw' if 'wilcox_pvalue_ihw' in df.columns else 'wilcox_pvalue_fdr'
    df['is_significant'] = np.where(df['wilcox_pvalue'].isna(), np.nan,
                                     df[wilcox_col] < significance_threshold)
    
    df_sorted = df.sort_values(['target_system', 'sub_model', 'delta'], ascending=[True, True, False])
    
    # Save main summary
    main_summary = df_sorted.drop(columns=['_label_path'])
    main_summary.to_csv(os.path.join(main_dir, 'all_comparisons.csv'), index=False)
    
    # Save per-target-system summaries
    for target_system in df['target_system'].unique():
        system_df = df[df['target_system'] == target_system].drop(columns=['_label_path'])
        system_df.sort_values(['sub_model', 'delta'], ascending=[True, False]).to_csv(
            os.path.join(main_dir, target_system, 'system_summary.csv'), index=False)
    
    # Save per-label summaries
    for label_path in df['_label_path'].unique():
        label_df = df[df['_label_path'] == label_path].drop(columns=['_label_path'])
        label_df.sort_values(['sub_model', 'delta'], ascending=[True, False]).to_csv(
            os.path.join(label_path, 'comparison_summary.csv'), index=False)
    
    return main_summary


def _get_feature_structure(feature_path):
    """
    Detect if feature is flat or nested (gait).
    
    Returns:
        List of (sub_model_name, metrics_df) tuples
    """
    results = []
    
    # Check for flat structure (metrics.csv directly in feature_path)
    flat_metrics = os.path.join(feature_path, 'metrics.csv')
    if os.path.exists(flat_metrics):
        df = pd.read_csv(flat_metrics)
        results.append(('flat', df))
    
    # Check for nested structure (subdirectories with metrics.csv)
    for sub_model in os.listdir(feature_path):
        sub_model_path = os.path.join(feature_path, sub_model)
        if not os.path.isdir(sub_model_path):
            continue
        
        nested_metrics = os.path.join(sub_model_path, 'metrics.csv')
        if os.path.exists(nested_metrics):
            df = pd.read_csv(nested_metrics)
            results.append((sub_model, df))
    
    return results


def _process_label_directory_gait(label_path, target_system, label_name, baseline_name,
                                   main_reg_score, main_reg_pvalue, main_clas_score,
                                   main_ordinal_score, main_ordinal_pvalue):
    """Process a single label directory with mixed flat/nested structures."""
    # Collect all feature systems with their sub_models
    # Structure: {feature_name: [(sub_model, metrics_df), ...]}
    feature_systems = {}
    baseline_data = {}  # {sub_model: metrics_df}
    
    for feature_name in os.listdir(label_path):
        feature_path = os.path.join(label_path, feature_name)
        if not os.path.isdir(feature_path):
            continue
        
        sub_models = _get_feature_structure(feature_path)
        if not sub_models:
            continue
        
        if feature_name == baseline_name:
            baseline_data = {sm: df for sm, df in sub_models}
        else:
            feature_systems[feature_name] = sub_models
    
    if not baseline_data:
        return []
    
    # Determine score type from baseline
    sample_baseline = list(baseline_data.values())[0]
    is_clas = main_clas_score in sample_baseline.columns
    # Check for regression first (has r2 or pearson_r but not auc)
    is_regression = (main_reg_score in sample_baseline.columns or 'pearson_r' in sample_baseline.columns) and not is_clas
    # Ordinal only has spearman_r without r2/pearson_r
    is_ordinal = main_ordinal_score in sample_baseline.columns and not is_clas and not is_regression
    
    if is_clas:
        main_score, main_pvalue, score_type = main_clas_score, None, 'auc'
    elif is_regression:
        main_score, main_pvalue, score_type = main_reg_score, main_reg_pvalue, main_reg_score
    elif is_ordinal:
        main_score, main_pvalue, score_type = main_ordinal_score, main_ordinal_pvalue, 'spearman'
    else:
        # Fallback to regression
        main_score, main_pvalue, score_type = main_reg_score, main_reg_pvalue, main_reg_score
    
    col_info = get_column_info(label_name)
    
    results = []
    for feature_name, sub_models in feature_systems.items():
        for sub_model, feature_df in sub_models:
            # Find matching baseline sub_model, fall back to 'flat' if not found
            if sub_model in baseline_data:
                baseline_df = baseline_data[sub_model]
            elif 'flat' in baseline_data:
                baseline_df = baseline_data['flat']
            else:
                continue
            
            genders = ['all']
            if f'male_{main_score}' in feature_df.columns:
                genders.extend(['male', 'female'])
            
            for gender in genders:
                row = _compare_to_baseline_gait(
                    feature_name, feature_df, baseline_name, baseline_df,
                    target_system, label_name, sub_model, gender,
                    main_score, main_pvalue, score_type, col_info
                )
                if row:
                    results.append(row)
    
    return results


def _compare_to_baseline_gait(feature_name, feature_df, baseline_name, baseline_df,
                               target_system, label_name, sub_model, gender,
                               main_score, main_pvalue, score_type, col_info):
    """Compare feature system to baseline for a specific sub_model."""
    score_col = main_score if gender == 'all' else f'{gender}_{main_score}'
    
    if score_col not in feature_df.columns or score_col not in baseline_df.columns:
        return None
    
    feat_scores = feature_df[score_col].values
    base_scores = baseline_df[score_col].values
    n_seeds = len(feat_scores)
    
    feat_mean = np.mean(feat_scores)
    base_mean = np.mean(base_scores)
    delta = feat_mean - base_mean
    
    # Wilcoxon only with enough seeds
    wilcox_pval = np.nan
    if n_seeds >= MIN_SEEDS_FOR_WILCOX:
        try:
            _, wilcox_pval = wilcoxon(feat_scores, base_scores, alternative='greater')
        except Exception:
            pass
    
    # Score p-value (None for classification)
    score_pvalue = np.nan
    if main_pvalue:
        pval_col = main_pvalue if gender == 'all' else f'{gender}_{main_pvalue}'
        if pval_col in feature_df.columns:
            score_pvalue = np.mean(feature_df[pval_col].values)
    
    # Get n_subjects from metrics
    n_subjects = int(feature_df['n_subjects'].mean()) if 'n_subjects' in feature_df.columns else np.nan
    
    # Get n_positive for classification
    n_positive = np.nan
    if 'n_positive' in feature_df.columns:
        n_positive = int(feature_df['n_positive'].mean())
    
    return {
        'target_system': target_system,
        'label': label_name,
        'description': col_info['description'],
        'type': col_info['type'],
        'features_name': feature_name,
        'baseline_name': baseline_name,
        'sub_model': sub_model,
        'gender': gender,
        'score_type': score_type,
        'score': feat_mean,
        'baseline_score': base_mean,
        'score_pvalue': score_pvalue,
        'delta': delta,
        'was_better': delta > 0,
        'wilcox_pvalue': wilcox_pval,
        'n_seeds': n_seeds,
        'n_subjects': n_subjects,
        'n_positive': n_positive
    }


def _apply_fdr_correction_to_wilcox_gait(df):
    """Apply FDR correction to Wilcoxon p-values per (gender, features_name, sub_model) group."""
    df['wilcox_pvalue_fdr'] = np.nan
    
    for (gender, features_name, sub_model), group in df.groupby(['gender', 'features_name', 'sub_model']):
        mask = (df['gender'] == gender) & (df['features_name'] == features_name) & (df['sub_model'] == sub_model)
        pvals = df.loc[mask, 'wilcox_pvalue'].values
        valid = ~np.isnan(pvals)
        
        if valid.any():
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid], _, _ = multipletests(pvals[valid], method='fdr_bh')
            df.loc[mask, 'wilcox_pvalue_fdr'] = corrected
    
    return df


def _collect_metrics_dirs_gait(main_dir):
    """Collect all directories containing metrics.csv files (handles both flat and nested)."""
    all_dirs = []
    for target_system in os.listdir(main_dir):
        target_path = os.path.join(main_dir, target_system)
        if not os.path.isdir(target_path):
            continue
        for label_name in os.listdir(target_path):
            label_path = os.path.join(target_path, label_name)
            if not os.path.isdir(label_path):
                continue
            for feature_name in os.listdir(label_path):
                feature_path = os.path.join(label_path, feature_name)
                if not os.path.isdir(feature_path):
                    continue
                
                # Check flat structure
                if os.path.exists(os.path.join(feature_path, 'metrics.csv')):
                    all_dirs.append(feature_path)
                
                # Check nested structure
                for sub_model in os.listdir(feature_path):
                    sub_model_path = os.path.join(feature_path, sub_model)
                    if os.path.isdir(sub_model_path) and os.path.exists(os.path.join(sub_model_path, 'metrics.csv')):
                        all_dirs.append(sub_model_path)
    
    return all_dirs

