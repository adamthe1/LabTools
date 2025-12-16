"""
Test run of biological age analysis with real data.

Configure all parameters below and run to analyze biological age predictions.
"""

import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
os.chdir('/home/adamgab/PycharmProjects/LabTools')

from biological_age_lib.analyze import analyze_biological_age
from biological_age_lib.config import BiologicalAgeConfig
from body_system_loader.load_feature_df import load_system_description_json

# =============================================================================
# CONFIGURATION - Edit these parameters
# =============================================================================

# Input/Output paths
PREDICTIONS_PATH = '/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_multiple/all_together_no_covariates/ensemble_predictions_format.csv'
SAVE_DIR = '/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/results/biological_age_on_all_systems'
os.makedirs(SAVE_DIR, exist_ok=True)

# Get all body system names from dataset description
ALL_SYSTEMS = list(load_system_description_json(no_temp=True).keys())
# filter out rna, metabolites, microbiome
ALL_SYSTEMS = [sys for sys in ALL_SYSTEMS if sys not in ['rna', 'metabolites', 'microbiome', 'gait']]
print(ALL_SYSTEMS)

# Label systems to analyze - all body systems
LABELS = ALL_SYSTEMS

# Age binning parameters
MIN_AGE = 40          # Minimum age for binning
MAX_AGE = 72          # Maximum age for binning
BIN_WIDTH = 4         # Years per bin
PERCENTILE = 0.25     # Top/bottom percentile (0.25 = 25%)

# Data filtering
MIN_AGE_CUTOFF = 40           # Remove subjects below this age
KEEP_FIRST_VISIT_ONLY = True  # Keep only first visit per subject
VISIT_PRIORITY = ('baseline', '02_00_visit', '04_00_visit', '06_00_visit')

# Analysis options
GENDER_SPLIT = True           # Run separate analysis per gender
SAVE_FIGURES = True           # Save plots to disk

# Statistical parameters
ALPHA = 0.05                  # Significance level for FDR correction
FC_THRESHOLD = 0.0            # Fold change threshold for volcano plots
MIN_SUBJECTS_PER_GROUP = 5    # Minimum subjects in top/bottom groups

# Visualization parameters
VOLCANO_FIGSIZE = (14, 10)    # Volcano plot size (width, height)
SCATTER_FIGSIZE = (12, 10)    # Scatter plot size (width, height)
VOLCANO_LABEL_FONTSIZE = 8    # Font size for labels on volcano plots
UPREGULATED_COLOR = '#FF5252'    # Red for upregulated
DOWNREGULATED_COLOR = '#4CAF50'  # Green for downregulated

# =============================================================================
# MAIN - No need to edit below
# =============================================================================

def main():
    print(f"Loading predictions from: {PREDICTIONS_PATH}")
    df = pd.read_csv(PREDICTIONS_PATH, index_col=['RegistrationCode', 'research_stage'])
    
    print(f"Loaded {len(df)} predictions")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Age range: {df['real_age'].min():.1f} - {df['real_age'].max():.1f}")
    
    # Build config from parameters
    config = BiologicalAgeConfig(
        # Age binning
        min_age=MIN_AGE,
        max_age=MAX_AGE,
        bin_width=BIN_WIDTH,
        percentile=PERCENTILE,
        # Data filtering
        min_age_cutoff=MIN_AGE_CUTOFF,
        keep_first_visit_only=KEEP_FIRST_VISIT_ONLY,
        visit_priority=VISIT_PRIORITY,
        # Analysis options
        gender_split=GENDER_SPLIT,
        save_figures=SAVE_FIGURES,
        # Statistical parameters
        alpha=ALPHA,
        fc_threshold=FC_THRESHOLD,
        min_subjects_per_group=MIN_SUBJECTS_PER_GROUP,
        # Visualization
        volcano_figsize=VOLCANO_FIGSIZE,
        scatter_figsize=SCATTER_FIGSIZE,
        volcano_label_fontsize=VOLCANO_LABEL_FONTSIZE,
        upregulated_color=UPREGULATED_COLOR,
        downregulated_color=DOWNREGULATED_COLOR,
    )

    print(f"\nRunning analysis, saving to: {SAVE_DIR}")
    results = analyze_biological_age(
        df,
        labels=LABELS,
        config=config,
        save_dir=SAVE_DIR
    )
    
    print("\nDone!")
    return results


if __name__ == '__main__':
    results = main()

