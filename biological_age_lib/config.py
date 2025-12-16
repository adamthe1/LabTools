"""
Configuration for Biological Age Analysis.

Provides a dataclass with all configurable parameters for the analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BiologicalAgeConfig:
    """
    Configuration for biological age analysis.
    
    Attributes:
        # Age binning
        min_age: Minimum age for age binning (None = auto-detect from data)
        max_age: Maximum age for age binning (None = auto-detect from data)
        bin_width: Width of each age bin in years
        percentile: Fraction for top/bottom percentile groups (e.g., 0.25 = 25%)
        
        # Data filtering
        min_age_cutoff: Remove subjects below this age (None = no filtering)
        keep_first_visit_only: Keep only first available visit per subject
        visit_priority: Order of visit preference (first available is kept)
        
        # Analysis options
        gender_split: Run analysis separately for each gender
        save_figures: Save generated figures to disk
        
        # Statistical parameters
        alpha: Significance level for statistical tests (FDR correction)
        fc_threshold: Fold change threshold for volcano plots
        min_subjects_per_group: Minimum subjects in top/bottom groups per gender
        run_fdr_on: FDR correction scope - 'all' (across all features) or 'per_system'
        
        # Visualization parameters
        volcano_figsize: Figure size for volcano plots (width, height)
        scatter_figsize: Figure size for scatter plots (width, height)
        volcano_label_fontsize: Font size for volcano plot labels
        upregulated_color: Color for upregulated points in volcano
        downregulated_color: Color for downregulated points in volcano
        
        # Plot titles (use {gender}, {n}, {n1}, {n2} as placeholders)
        scatter_title: Title for scatter plots
        volcano_title: Title for volcano plots
        residuals_title: Title for residuals plot
        error_dist_title: Title for error distribution plot
    """
    # Age binning parameters (None = auto-detect from data)
    min_age: int = None
    max_age: int = None
    bin_width: int = 4
    percentile: float = 0.25
    
    # Data filtering options (None = no filtering)
    min_age_cutoff: int = None
    keep_first_visit_only: bool = True
    visit_priority: Tuple[str, ...] = ('baseline', '02_00_visit', '04_00_visit', '06_00_visit')
    
    # Analysis options
    gender_split: bool = True
    save_figures: bool = True
    
    # Statistical parameters
    alpha: float = 0.05
    fc_threshold: float = 0.0
    min_subjects_per_group: int = 5
    run_fdr_on: str = 'per_system'  # 'all' | 'per_system'
    max_nan_frac: float = 0.7  # Max fraction of NaN values allowed per column
    
    # Visualization parameters
    volcano_figsize: Tuple[int, int] = (14, 10)
    scatter_figsize: Tuple[int, int] = (12, 10)
    volcano_label_fontsize: int = 8
    upregulated_color: str = '#FF5252'
    downregulated_color: str = '#4CAF50'
    
    # Plot titles (placeholders: {gender}, {n}, {n1}, {n2}, {label1}, {label2})
    scatter_title: str = 'Predicted vs True Age'
    volcano_title: str = 'Volcano: {label2} vs {label1}'
    residuals_title: str = 'Residuals Plot'
    error_dist_title: str = 'Error Distribution'
    
    def __post_init__(self):
        """Validate configuration values."""
        # Skip min/max age validation if they will be auto-detected
        if self.min_age is not None and self.max_age is not None:
            if self.min_age >= self.max_age:
                raise ValueError(f"min_age ({self.min_age}) must be less than max_age ({self.max_age})")
        if not 0 < self.percentile < 0.5:
            raise ValueError(f"percentile ({self.percentile}) must be between 0 and 0.5")
        if self.bin_width <= 0:
            raise ValueError(f"bin_width ({self.bin_width}) must be positive")
        if self.min_subjects_per_group < 2:
            raise ValueError(f"min_subjects_per_group ({self.min_subjects_per_group}) must be at least 2")
        if self.run_fdr_on not in ('all', 'per_system'):
            raise ValueError(f"run_fdr_on must be 'all' or 'per_system', got '{self.run_fdr_on}'")

