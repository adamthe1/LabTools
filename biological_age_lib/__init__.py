"""
Biological Age Analysis Library

A clean, intuitive tool for analyzing biological age predictions against biomarkers.

Quick Start:
    from biological_age_lib import analyze_biological_age, BiologicalAgeConfig
    
    # Load predictions (index: RegistrationCode, research_stage)
    # Must have columns: real_age, predicted_age
    predictions_df = pd.read_csv('predictions.csv', index_col=[0, 1])
    
    # Configure analysis
    config = BiologicalAgeConfig(
        min_age_cutoff=40,           # Remove subjects under 40
        keep_first_visit_only=True,  # Keep baseline > 02 > 04 > 06
        gender_split=True            # Analyze by gender
    )
    
    # Run analysis with labels from a body system
    results = analyze_biological_age(
        predictions_df,
        labels='frailty',  # or ['hand_grip_right', 'TNF', 'IL6']
        config=config,
        save_dir='/path/to/output'
    )

Main Entry Points:
- analyze_biological_age: Main function for running the full analysis pipeline
- BiologicalAgeConfig: Configuration dataclass for all analysis parameters
- filter_predictions: Utility to filter by age and keep first visit only

Core Components:
- AgeBinAnalyzer: Age stratification and percentile grouping
- VolcanoAnalyzer: Statistical comparisons with FDR correction
- AgeVisualization: Scatter plots, residuals, and metrics display

Utilities:
- load_labels_from_list: Load labels from list-of-dicts format
"""

# Main entry points (most users only need these)
from .analyze import (
    analyze_biological_age,
    filter_predictions,
    load_labels_from_list
)
from .config import BiologicalAgeConfig

# Core analysis components
from .age_binning import AgeBinAnalyzer
from .volcano_analysis import VolcanoAnalyzer
from .visualization import (
    AgeVisualization,
    create_scatter_plot_by_gender,
    create_gender_comparison_summary
)

__all__ = [
    # Main entry points
    'analyze_biological_age',
    'BiologicalAgeConfig',
    'filter_predictions',
    'load_labels_from_list',
    
    # Core components
    'AgeBinAnalyzer',
    'VolcanoAnalyzer',
    'AgeVisualization',
    'create_scatter_plot_by_gender',
    'create_gender_comparison_summary',
]
