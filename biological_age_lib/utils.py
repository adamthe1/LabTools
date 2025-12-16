"""
Biological Age Analysis - Utility Functions.

Helper functions for data filtering, loading, standardization, and internal analysis.
"""

import os
import pandas as pd
from typing import Dict, List, Union
from sklearn.preprocessing import StandardScaler

from body_system_loader.load_feature_df import (
    _merge_closest_research_stage,
    load_body_system_df,
    load_columns_as_df,
    load_system_description_json
)


# =============================================================================
# Standardization
# =============================================================================

def standardize_features(
    df: pd.DataFrame, 
    columns: list = None,
    reference_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Standardize numeric columns to z-scores (mean=0, std=1).
    
    Args:
        df: DataFrame to standardize
        columns: Specific columns to standardize. If None, uses all numeric columns.
        reference_df: DataFrame to compute mean/std from. If None, uses df itself.
    
    Returns:
        DataFrame with standardized values
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if not columns:
        return df_scaled
    
    ref = reference_df if reference_df is not None else df
    
    scaler = StandardScaler()
    scaler.fit(ref[columns])
    df_scaled[columns] = scaler.transform(df[columns])
    
    return df_scaled


# =============================================================================
# Filtering
# =============================================================================

def filter_predictions(
    df: pd.DataFrame,
    min_age: int = None,
    keep_first_visit_only: bool = True,
    visit_priority: tuple = ('baseline', '02_00_visit', '04_00_visit', '06_00_visit'),
    age_column: str = 'real_age'
) -> pd.DataFrame:
    """
    Filter predictions DataFrame by age and visit.
    
    Args:
        df: DataFrame with MultiIndex (RegistrationCode, research_stage)
            and columns including age_column
        min_age: Remove subjects below this age (None to skip)
        keep_first_visit_only: Keep only first available visit per subject
        visit_priority: Order of visit preference (first in list = highest priority)
        age_column: Name of the age column
    
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    original_count = len(df_filtered)
    
    # Filter by minimum age
    if min_age is not None and age_column in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[age_column] >= min_age]
        print(f"  Removed {original_count - len(df_filtered)} subjects under age {min_age}")
    
    # Keep only first visit per subject
    if keep_first_visit_only and isinstance(df_filtered.index, pd.MultiIndex):
        before_count = len(df_filtered)
        df_filtered = _keep_first_visit(df_filtered, visit_priority)
        print(f"  Kept first visit only: {before_count} -> {len(df_filtered)} rows")
    
    return df_filtered


def _keep_first_visit(
    df: pd.DataFrame,
    visit_priority: tuple = ('baseline', '02_00_visit', '04_00_visit', '06_00_visit')
) -> pd.DataFrame:
    """
    Keep only the first available visit per subject based on priority order.
    
    Priority: baseline > 02_00_visit > 04_00_visit > 06_00_visit
    """
    # Create priority mapping (lower = higher priority)
    priority_map = {visit: i for i, visit in enumerate(visit_priority)}
    default_priority = len(visit_priority)  # Unknown visits get lowest priority
    
    df_reset = df.reset_index()
    
    # Get research_stage column name
    stage_col = 'research_stage' if 'research_stage' in df_reset.columns else df_reset.columns[1]
    reg_col = 'RegistrationCode' if 'RegistrationCode' in df_reset.columns else df_reset.columns[0]
    
    # Add priority column
    df_reset['_visit_priority'] = df_reset[stage_col].map(
        lambda x: priority_map.get(x, default_priority)
    )
    
    # Sort by subject and priority, keep first (highest priority)
    df_reset = df_reset.sort_values([reg_col, '_visit_priority'])
    df_reset = df_reset.drop_duplicates(subset=[reg_col], keep='first')
    df_reset = df_reset.drop(columns=['_visit_priority'])
    
    # Restore index
    df_reset = df_reset.set_index([reg_col, stage_col])
    
    return df_reset


# =============================================================================
# Label Loading Utilities (list-of-dicts format)
# =============================================================================

def _is_csv_file(path: str) -> bool:
    """Check if path points to a valid CSV file."""
    return isinstance(path, str) and os.path.isfile(path) and path.endswith('.csv')


def _get_label_name(item: Union[str, Dict]) -> str:
    """Extract name from label item (string or dict key)."""
    if isinstance(item, dict):
        return list(item.keys())[0]
    return item


def _load_single_label(item: Union[str, Dict], index: pd.Index) -> pd.DataFrame:
    """
    Load DataFrame for a single label item, aligned to index.
    
    Args:
        item: String (body system name) or dict with columns/csv path
        index: Index to align labels to
    
    Returns:
        DataFrame with labels aligned to index
    """
    if isinstance(item, str):
        # Body system name
        systems = load_system_description_json()
        if item in systems:
            labels_df = load_body_system_df(item)
        else:
            # Single column name
            labels_df = load_columns_as_df([item])
    else:
        # Dict: {name: value}
        name = list(item.keys())[0]
        value = list(item.values())[0]
        
        if _is_csv_file(value):
            # CSV path
            labels_df = pd.read_csv(value, index_col=[0, 1])
        elif isinstance(value, list):
            # Column list
            labels_df = load_columns_as_df(value)
        else:
            raise ValueError(f"Invalid label format for '{name}': {type(value)}")
    
    # Align to predictions index using closest research_stage merge
    target_df = pd.DataFrame(index=index)
    merged = _merge_closest_research_stage(target_df, labels_df)
    return merged


def load_labels_from_list(
    labels_list: List[Union[str, Dict]],
    index: pd.Index
) -> Dict[str, pd.DataFrame]:
    """
    Load labels from list-of-dicts format.
    
    Args:
        labels_list: List of label items, each can be:
            - String: body system name (e.g., 'frailty')
            - Dict with columns: {'name': ['col1', 'col2']}
            - Dict with CSV path: {'name': '/path/to/data.csv'}
        index: Index to align labels to (from predictions_df)
    
    Returns:
        Dict mapping system_name -> DataFrame
    """
    result = {}
    for item in labels_list:
        name = _get_label_name(item)
        try:
            df = _load_single_label(item, index)
            if not df.empty:
                result[name] = df
                print(f"  Loaded '{name}': {len(df.columns)} columns, {len(df)} subjects")
            else:
                print(f"  WARNING: No data found for '{name}'")
        except Exception as e:
            print(f"  WARNING: Failed to load '{name}': {e}")
    return result


# =============================================================================
# Validation & Preparation
# =============================================================================

def validate_predictions_df(df: pd.DataFrame) -> None:
    """Validate predictions DataFrame has required structure."""
    required_cols = ['real_age', 'predicted_age']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"predictions_df missing required columns: {missing}. "
            f"Expected columns: {required_cols}"
        )
    
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            "predictions_df must have MultiIndex (RegistrationCode, research_stage). "
            f"Got index type: {type(df.index)}"
        )


def add_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add gender column by loading from Age_Gender_BMI body system.
    
    Args:
        df: DataFrame with MultiIndex (RegistrationCode, research_stage)
    
    Returns:
        DataFrame with 'gender' column added
    """
    try:
        gender_df = load_columns_as_df(['gender'])
        
        # Align to predictions index
        common_idx = df.index.intersection(gender_df.index)
        
        if len(common_idx) == 0:
            print("  WARNING: Could not match any subjects for gender. Disabling gender split.")
            return df
        
        df_with_gender = df.copy()
        df_with_gender['gender'] = gender_df.loc[common_idx, 'gender']
        
        n_matched = df_with_gender['gender'].notna().sum()
        print(f"  Loaded gender for {n_matched}/{len(df)} subjects")
        
        return df_with_gender
        
    except Exception as e:
        print(f"  WARNING: Could not load gender: {e}. Disabling gender split.")
        return df


def prepare_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for internal analysis functions."""
    analysis_df = df.copy()

    # Rename columns for internal compatibility
    analysis_df = analysis_df.rename(columns={
        'real_age': 'true_values',
        'predicted_age': 'predictions'
    })
    
    # Validate MultiIndex
    if not isinstance(analysis_df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have MultiIndex (RegistrationCode, research_stage)")
    
    return analysis_df


# =============================================================================
# Visualization Helpers
# =============================================================================

def create_prediction_plots(
    analysis_df: pd.DataFrame,
    metrics: Dict,
    save_dir: str,
    config,  # BiologicalAgeConfig
    original_df: pd.DataFrame
) -> None:
    """Create prediction scatter plots, optionally split by gender."""
    from .visualization import AgeVisualization
    
    viz = AgeVisualization()
    figs_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)
    
    # All subjects plot
    viz.create_gradient_scatter_plot(
        analysis_df, metrics,
        save_path=os.path.join(figs_dir, 'predictions_all'),
        figsize=config.scatter_figsize,
        scatter_title=config.scatter_title,
        residuals_title=config.residuals_title,
        error_dist_title=config.error_dist_title
    )
    
    # Gender-split plots
    if config.gender_split and 'gender' in original_df.columns:
        for gender_val, gender_name in [(1, 'male'), (0, 'female')]:
            gender_mask = original_df['gender'] == gender_val
            gender_idx = original_df[gender_mask].index
            gender_df = analysis_df[analysis_df.index.isin(gender_idx)]
            
            if len(gender_df) > 10:
                _, gender_metrics = viz.analyze_predictions(gender_df)
                viz.create_gradient_scatter_plot(
                    gender_df, gender_metrics,
                    save_path=os.path.join(figs_dir, f'predictions_{gender_name}'),
                    figsize=config.scatter_figsize,
                    scatter_title=config.scatter_title,
                    residuals_title=config.residuals_title,
                    error_dist_title=config.error_dist_title
                )


# =============================================================================
# Index Conversion & Column Validation
# =============================================================================

def _convert_indices(indices: List, target_index: pd.Index) -> List:
    """Convert indices to match target index format."""
    if not indices:
        return []
    
    # Check if target is MultiIndex
    if isinstance(target_index, pd.MultiIndex):
        # Indices might be tuples or composite strings
        if isinstance(indices[0], tuple):
            return indices
        elif '_' in str(indices[0]):
            # Composite string like "1234567890_baseline"
            converted = []
            for idx in indices:
                parts = str(idx).split('_', 1)
                if len(parts) == 2:
                    reg_code = '10K_' + parts[0] if not parts[0].startswith('10K_') else parts[0]
                    converted.append((reg_code, parts[1]))
            return converted
    
    return indices


def _get_valid_columns(df: pd.DataFrame, max_nan_frac: float = 0.5) -> List[str]:
    """Get columns with sufficient data and variance."""
    valid_cols = []
    for col in df.columns:
        col_data = df[col].dropna()
        nan_frac = 1 - len(col_data) / len(df)
        if nan_frac < max_nan_frac and len(col_data) > 1 and col_data.std() > 0:
            valid_cols.append(col)
    return valid_cols


# =============================================================================
# Volcano Analysis Helpers
# =============================================================================

def _split_results_by_system(
    results: pd.DataFrame,
    col_to_system: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """Split volcano results DataFrame by system based on feature-to-system mapping."""
    system_results = {}
    
    for system_name in set(col_to_system.values()):
        system_cols = [col for col, sys in col_to_system.items() if sys == system_name]
        mask = results['feature'].isin(system_cols)
        if mask.any():
            system_results[system_name] = results[mask].copy().reset_index(drop=True)
    
    return system_results


def _save_and_plot_volcano(
    results: pd.DataFrame,
    volcano,  # VolcanoAnalyzer
    labels_tuple: tuple,
    gender_name: str,
    save_dir: str,
    config  # BiologicalAgeConfig
) -> None:
    """Save CSV and create plot for volcano results."""
    n_sig = results['significant'].sum()
    print(f"    {gender_name}: {n_sig} significant biomarkers")
    
    if n_sig > 0:
        sig = results[results['significant']].sort_values('adj_p_value')
        print("      Top significant:")
        for _, row in sig.head(5).iterrows():
            direction = "↑" if row['delta_z'] > 0 else "↓"
            print(f"        {row['feature']}: {direction} Δz={row['delta_z']:.2f}, q={row['adj_p_value']:.3f}")
    
    if save_dir:
        # Save CSV
        results.to_csv(os.path.join(save_dir, f'volcano_results_{gender_name}.csv'), index=False)
        
        # Create plot
        if config.save_figures:
            save_path = os.path.join(save_dir, f'volcano_{gender_name}')
            volcano.plot(
                results, labels=labels_tuple, save_path=save_path, gender=gender_name,
                figsize=config.volcano_figsize,
                upregulated_color=config.upregulated_color,
                downregulated_color=config.downregulated_color,
                labels_fontsize=config.volcano_label_fontsize,
                title=config.volcano_title
            )


# =============================================================================
# Results Saving
# =============================================================================

def save_results(results: Dict, save_dir: str) -> None:
    """Save analysis results to disk."""
    # Save filtered predictions
    if 'filtered_df' in results:
        results['filtered_df'].to_csv(os.path.join(save_dir, 'filtered_predictions.csv'))
    
    # Save metrics
    if 'metrics' in results:
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(os.path.join(save_dir, 'prediction_metrics.csv'), index=False)
    
    # Save bin results summary
    if 'bin_results' in results:
        bin_summary = {
            'n_top': len(results['bin_results'].get('aggregated_top', [])),
            'n_bottom': len(results['bin_results'].get('aggregated_bottom', []))
        }
        pd.DataFrame([bin_summary]).to_csv(os.path.join(save_dir, 'bin_summary.csv'), index=False)
