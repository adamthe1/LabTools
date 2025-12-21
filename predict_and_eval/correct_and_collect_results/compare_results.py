import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from ..loading_features.load_feature_df import get_column_info
from .fix_pvals import fix_pvals, fix_pvals_grouped, fix_pvals_ihw, apply_ihw_to_wilcox

MIN_SEEDS_FOR_WILCOX = 6


def compare_and_collect_results(main_dir, baseline_features_system_name, 
                                main_reg_score='r2', main_reg_pvalue='pearson_pvalue', 
                                main_clas_score='auc', 
                                main_ordinal_score='spearman_r', main_ordinal_pvalue='spearman_pvalue',
                                significance_threshold=0.05):
    """
    Compare feature systems to baseline and collect results.
    For <6 seeds: only compares means, wilcox/pvalue/is_significant are NaN.
    Also applies FDR correction to p-values in metrics.csv files.
    """
    # Collect all metrics directories and apply FDR corrections
    all_dirs = _collect_metrics_dirs(main_dir)
    fix_pvals(all_dirs)                # Standard BH
    fix_pvals_grouped(all_dirs)        # Weighted BH with equal budget per target_system
    fix_pvals_ihw(all_dirs)            # IHW: data-driven weights based on signal density
    
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
            
            label_results = _process_label_directory(
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
    df = _apply_fdr_correction_to_wilcox(df)
    df = apply_ihw_to_wilcox(df)  # IHW correction for Wilcoxon p-values
    
    # is_significant: NaN if wilcox wasn't run (using standard FDR)
    df['is_significant'] = np.where(df['wilcox_pvalue'].isna(), np.nan, 
                                     df['wilcox_pvalue_fdr'] < significance_threshold)
    # is_significant_ihw: using IHW correction
    df['is_significant_ihw'] = np.where(df['wilcox_pvalue'].isna(), np.nan, 
                                         df['wilcox_pvalue_ihw'] < significance_threshold)
    
    df_sorted = df.sort_values(['target_system', 'delta'], ascending=[True, False])
    
    # Save main summary
    main_summary = df_sorted.drop(columns=['_label_path'])
    main_summary.to_csv(os.path.join(main_dir, 'all_comparisons.csv'), index=False)
    
    # Save per-target-system summaries
    for target_system in df['target_system'].unique():
        system_df = df[df['target_system'] == target_system].drop(columns=['_label_path'])
        system_df.sort_values('delta', ascending=False).to_csv(
            os.path.join(main_dir, target_system, 'system_summary.csv'), index=False)
    
    # Save per-label summaries
    for label_path in df['_label_path'].unique():
        label_df = df[df['_label_path'] == label_path].drop(columns=['_label_path'])
        label_df.sort_values('delta', ascending=False).to_csv(
            os.path.join(label_path, 'comparison_summary.csv'), index=False)
    
    return main_summary


def _process_label_directory(label_path, target_system, label_name, baseline_name,
                             main_reg_score, main_reg_pvalue, main_clas_score,
                             main_ordinal_score, main_ordinal_pvalue):
    """Process a single label directory."""
    feature_systems = {}
    baseline_data = None
    
    for feature_name in os.listdir(label_path):
        feature_path = os.path.join(label_path, feature_name)
        metrics_file = os.path.join(feature_path, 'metrics.csv')
        if not os.path.isdir(feature_path) or not os.path.exists(metrics_file):
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        if feature_name == baseline_name:
            baseline_data = metrics_df
        else:
            feature_systems[feature_name] = metrics_df
    
    if baseline_data is None:
        return []
    
    # Determine score type
    is_clas = main_clas_score in baseline_data.columns
    # Check for regression first (has r2 or pearson_r but not auc)
    is_regression = (main_reg_score in baseline_data.columns or 'pearson_r' in baseline_data.columns) and not is_clas
    # Ordinal only has spearman_r without r2/pearson_r
    is_ordinal = main_ordinal_score in baseline_data.columns and not is_clas and not is_regression
    
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
    for feature_name, feature_df in feature_systems.items():
        genders = ['all']
        if f'male_{main_score}' in feature_df.columns:
            genders.extend(['male', 'female'])
        
        for gender in genders:
            row = _compare_to_baseline(
                feature_name, feature_df, baseline_name, baseline_data,
                target_system, label_name, gender, main_score, main_pvalue,
                score_type, col_info
            )
            if row:
                results.append(row)
    
    return results


def _compare_to_baseline(feature_name, feature_df, baseline_name, baseline_df,
                         target_system, label_name, gender, main_score, main_pvalue,
                         score_type, col_info):
    """Compare feature system to baseline."""
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
    
    # Get n_subjects from metrics (mean across seeds, should be same)
    n_subjects = int(feature_df['n_subjects'].mean()) if 'n_subjects' in feature_df.columns else np.nan
    
    # Get n_positive for classification (mean across seeds)
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


def _apply_fdr_correction_to_wilcox(df):
    """Apply FDR correction to Wilcoxon p-values per (gender, features_name) group.
    
    FDR is applied separately for each combination because:
    - Each gender (all, male, female) is a separate analysis
    - Each model/features is a separate hypothesis family
    """
    df['wilcox_pvalue_fdr'] = np.nan
    
    for (gender, features_name), group in df.groupby(['gender', 'features_name']):
        mask = (df['gender'] == gender) & (df['features_name'] == features_name)
        pvals = df.loc[mask, 'wilcox_pvalue'].values
        valid = ~np.isnan(pvals)
        
        if valid.any():
            corrected = np.full(len(pvals), np.nan)
            _, corrected[valid], _, _ = multipletests(pvals[valid], method='fdr_bh')
            df.loc[mask, 'wilcox_pvalue_fdr'] = corrected
    
    return df


def _collect_metrics_dirs(main_dir):
    """Collect all directories containing metrics.csv files."""
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
                if os.path.isdir(feature_path) and os.path.exists(os.path.join(feature_path, 'metrics.csv')):
                    all_dirs.append(feature_path)
    return all_dirs
