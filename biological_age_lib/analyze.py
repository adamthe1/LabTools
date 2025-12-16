"""
Biological Age Analysis - Main Entry Point.

Provides a clean, intuitive interface for analyzing biological age predictions
against biomarkers from body systems.
"""

import os
import warnings
import pandas as pd
from typing import Dict, List, Union

from .config import BiologicalAgeConfig
from .age_binning import AgeBinAnalyzer
from .volcano_analysis import VolcanoAnalyzer
from .visualization import AgeVisualization
from .utils import (
    filter_predictions,
    load_labels_from_list,
    validate_predictions_df,
    add_gender_column,
    prepare_analysis_df,
    create_prediction_plots,
    save_results,
    standardize_features,
    _get_valid_columns,
    _convert_indices,
    _split_results_by_system,
    _save_and_plot_volcano
)


def analyze_biological_age(
    predictions_df: pd.DataFrame,
    labels: List[Union[str, Dict]] = None,
    config: BiologicalAgeConfig = None,
    save_dir: str = None
) -> Dict:
    """
    Main entry point for biological age analysis.
    
    Analyzes biological age predictions by:
    1. Filtering data (age cutoff, repeated visits)
    2. Binning subjects by chronological age
    3. Identifying top/bottom percentiles within each bin
    4. Comparing biomarkers between groups via volcano plots (per system + combined)
    
    Args:
        predictions_df: DataFrame with MultiIndex (RegistrationCode, research_stage)
                       and columns 'real_age', 'predicted_age'
        labels: List of label systems (list-of-dicts format). Each item can be:
            - String: body system name (e.g., 'frailty')
            - Dict with columns: {'name': ['col1', 'col2']}
            - Dict with CSV path: {'name': '/path/to/data.csv'}
            If None, only age bin analysis is performed.
        config: Analysis configuration (uses defaults if None)
        save_dir: Directory to save results and figures (None = don't save)
    
    Returns:
        Dictionary with:
            - 'filtered_df': Filtered predictions DataFrame
            - 'bin_results': Age bin analysis results
            - 'volcano_results': Dict with results per system + 'all_systems' combined
            - 'metrics': Prediction performance metrics
    
    Example:
        >>> from biological_age_lib import analyze_biological_age, BiologicalAgeConfig
        >>> 
        >>> # Load predictions (index: RegistrationCode, research_stage)
        >>> predictions_df = pd.read_csv('predictions.csv', index_col=[0, 1])
        >>> 
        >>> # Configure analysis
        >>> config = BiologicalAgeConfig(
        ...     min_age_cutoff=40,
        ...     keep_first_visit_only=True,
        ...     gender_split=True
        ... )
        >>> 
        >>> # Run analysis with multiple label systems
        >>> results = analyze_biological_age(
        ...     predictions_df,
        ...     labels=[
        ...         'frailty',  # body system name
        ...         {'frailty_select': ['hand_grip_right', 'hand_grip_left']},
        ...         {'proteomics_select': ['TNF', 'IL6', 'IGF1R']},
        ...     ],
        ...     config=config,
        ...     save_dir='/path/to/output'
        ... )
    """
    # Use default config if not provided
    if config is None:
        config = BiologicalAgeConfig()
    
    # Validate input
    validate_predictions_df(predictions_df)
    
    print("=" * 60)
    print("Biological Age Analysis")
    print("=" * 60)
    
    # Create output directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Load gender from body systems if needed and not in predictions
    if config.gender_split and 'gender' not in predictions_df.columns:
        print("\n[Step 0] Loading gender from body systems...")
        predictions_df = add_gender_column(predictions_df)
    
    # Step 1: Filter predictions
    print("\n[Step 1] Filtering predictions...")
    filtered_df = filter_predictions(
        predictions_df,
        min_age=config.min_age_cutoff,
        keep_first_visit_only=config.keep_first_visit_only,
        visit_priority=config.visit_priority
    )
    
    # Prepare DataFrame for analysis (rename columns for internal use)
    analysis_df = prepare_analysis_df(filtered_df)
    
    # Step 2: Calculate metrics
    print("\n[Step 2] Calculating prediction metrics...")
    viz = AgeVisualization()
    _, metrics = viz.analyze_predictions(analysis_df)
    viz.print_metrics_summary(metrics)
    
    # Step 3: Age bin analysis
    print("\n[Step 3] Analyzing age bins...")
    
    # Auto-detect min/max age from data if not specified
    min_age = config.min_age
    max_age = config.max_age
    if min_age is None:
        min_age = int(analysis_df['true_values'].min())
        print(f"  Auto-detected min_age: {min_age}")
    if max_age is None:
        max_age = int(analysis_df['true_values'].max()) + 1  # +1 to include max value in last bin
        print(f"  Auto-detected max_age: {max_age}")
    
    age_analyzer = AgeBinAnalyzer(
        min_age=min_age,
        max_age=max_age,
        bin_width=config.bin_width,
        percentile=config.percentile
    )
    bin_results = age_analyzer.analyze(analysis_df)
    age_analyzer.print_summary(bin_results)
    
    results = {
        'filtered_df': filtered_df,
        'analysis_df': analysis_df,
        'bin_results': bin_results,
        'metrics': metrics,
        'config': config
    }
    
    # Step 4: Create prediction visualizations
    if save_dir and config.save_figures:
        print("\n[Step 4] Creating visualizations...")
        create_prediction_plots(analysis_df, metrics, save_dir, config, filtered_df)
    
    # Step 5: Volcano analysis (if labels provided)
    if labels is not None:
        print("\n[Step 5] Running volcano analysis...")
        volcano_results = run_volcano_analysis(
            bin_results=bin_results,
            labels=labels,
            filtered_df=filtered_df,
            config=config,
            save_dir=save_dir
        )
        results['volcano_results'] = volcano_results
    
    # Save results
    if save_dir:
        save_results(results, save_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    if save_dir:
        print(f"Results saved to: {save_dir}")
    print("=" * 60)
    
    return results


def run_volcano_analysis(
    bin_results: Dict,
    labels: List[Union[str, Dict]],
    filtered_df: pd.DataFrame,
    config: BiologicalAgeConfig,
    save_dir: str = None
) -> Dict:
    """
    Run volcano plot analysis comparing top/bottom groups.
    
    Runs statistical tests once on all features combined, then splits results
    per system. FDR is applied based on config.run_fdr_on ('all' or 'per_system').
    
    Args:
        bin_results: Results from AgeBinAnalyzer with 'aggregated_top' and 'aggregated_bottom'
        labels: List of label systems (body system names, column lists, or CSV paths)
        filtered_df: Filtered predictions DataFrame with subject indices
        config: BiologicalAgeConfig with analysis parameters
        save_dir: Directory to save results (None = don't save)
    
    Returns:
        Dict with results per system + 'all_systems' combined
    """
    # Load all label systems
    print("  Loading label systems...")
    labels_dict = load_labels_from_list(labels, filtered_df.index)
    
    if not labels_dict:
        print("  WARNING: No labels found for analysis")
        return {'error': 'No labels found'}
    
    # Get top/bottom indices
    top_indices = bin_results.get('aggregated_top', [])
    bottom_indices = bin_results.get('aggregated_bottom', [])
    
    if not top_indices or not bottom_indices:
        print("  WARNING: No top/bottom groups found")
        return {'error': 'No groups found'}
    
    # Setup
    volcano_dir = os.path.join(save_dir, 'volcano') if save_dir else None
    percent = int(config.percentile * 100)
    labels_tuple = (f'Young ({percent}%)', f'Old ({percent}%)')
    
    # Gender splits configuration
    gender_splits = [('all', None)]
    if config.gender_split and 'gender' in filtered_df.columns:
        gender_splits.extend([('male', 1), ('female', 0)])
    
    print(f"  FDR correction mode: {config.run_fdr_on}")
    
    # Step 1: Load and standardize all systems, build column-to-system mapping
    print("\n  Standardizing all systems...")
    scaled_dfs = {}
    col_to_system = {}  # Maps column name -> system name
    
    for system_name, labels_df in labels_dict.items():
        # Standardize on whole population
        whole_pop_df = labels_df.loc[labels_df.index.isin(filtered_df.index)]
        valid_cols = _get_valid_columns(whole_pop_df, max_nan_frac=config.max_nan_frac)
        
        if not valid_cols:
            print(f"    {system_name}: No valid columns, skipping")
            continue
        
        whole_pop_df = whole_pop_df[valid_cols]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_df = standardize_features(whole_pop_df)
        
        scaled_dfs[system_name] = scaled_df
        for col in scaled_df.columns:
            col_to_system[col] = system_name
        
        print(f"    {system_name}: {len(valid_cols)} features")
    
    if not scaled_dfs:
        print("  WARNING: No valid systems after standardization")
        return {'error': 'No valid systems'}
    
    # Step 2: Combine all scaled data
    combined_df = pd.concat(scaled_dfs.values(), axis=1)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    print(f"\n  Combined: {len(combined_df.columns)} total features")
    
    # Convert indices
    converted_top = _convert_indices(top_indices, combined_df.index)
    converted_bottom = _convert_indices(bottom_indices, combined_df.index)
    
    # Step 3: Run volcano analysis per gender
    all_results = {}
    
    for gender_name, gender_val in gender_splits:
        print(f"\n  {'='*50}")
        print(f"  Gender: {gender_name.upper()}")
        print(f"  {'='*50}")
        
        # Filter indices by gender
        top_available = [i for i in converted_top if i in combined_df.index]
        bottom_available = [i for i in converted_bottom if i in combined_df.index]
        
        if gender_val is not None:
            gender_mask = filtered_df['gender'] == gender_val
            gender_idx = set(filtered_df[gender_mask].index)
            top_gender = [i for i in top_available if i in gender_idx]
            bottom_gender = [i for i in bottom_available if i in gender_idx]
        else:
            top_gender = top_available
            bottom_gender = bottom_available
        
        min_subj = config.min_subjects_per_group
        if len(top_gender) < min_subj or len(bottom_gender) < min_subj:
            print(f"    Not enough subjects (need {min_subj})")
            continue
        
        print(f"    Top: {len(top_gender)}, Bottom: {len(bottom_gender)}")
        
        # Get data for analysis
        top_data = combined_df.loc[top_gender]
        bottom_data = combined_df.loc[bottom_gender]
        
        # Run volcano once on ALL features (no FDR yet)
        volcano = VolcanoAnalyzer(
            alpha=config.alpha,
            fc_threshold=config.fc_threshold,
            pre_scaled=True
        )
        raw_results = volcano.compare(bottom_data, top_data, labels=labels_tuple, apply_fdr=False)
        
        # Apply FDR based on mode
        if config.run_fdr_on == 'all':
            # Apply FDR on all features together, then split
            all_with_fdr = volcano.apply_fdr(raw_results)
            system_results = _split_results_by_system(all_with_fdr, col_to_system)
        else:
            # Split first, then apply FDR per system
            system_raw = _split_results_by_system(raw_results, col_to_system)
            system_results = {
                sys: volcano.apply_fdr(df) for sys, df in system_raw.items()
            }
        
        # Save and plot per-system results
        for system_name, sys_results in system_results.items():
            sys_dir = os.path.join(volcano_dir, system_name) if volcano_dir else None
            if sys_dir:
                os.makedirs(sys_dir, exist_ok=True)
            
            _save_and_plot_volcano(
                results=sys_results,
                volcano=volcano,
                labels_tuple=labels_tuple,
                gender_name=gender_name,
                save_dir=sys_dir,
                config=config
            )
            
            # Store in all_results
            if system_name not in all_results:
                all_results[system_name] = {}
            all_results[system_name][gender_name] = {
                'results': sys_results,
                'n_significant': sys_results['significant'].sum()
            }
        
        # Combined "all_systems" plot (always uses FDR on all)
        all_sys_dir = os.path.join(volcano_dir, 'all_systems') if volcano_dir else None
        if all_sys_dir:
            os.makedirs(all_sys_dir, exist_ok=True)
        
        # For combined plot, always apply FDR on all features
        all_with_fdr = volcano.apply_fdr(raw_results) if config.run_fdr_on == 'per_system' else all_with_fdr
        
        _save_and_plot_volcano(
            results=all_with_fdr,
            volcano=volcano,
            labels_tuple=labels_tuple,
            gender_name=gender_name,
            save_dir=all_sys_dir,
            config=config
        )
        
        if 'all_systems' not in all_results:
            all_results['all_systems'] = {}
        all_results['all_systems'][gender_name] = {
            'results': all_with_fdr,
            'n_significant': all_with_fdr['significant'].sum()
        }
    
    return all_results
