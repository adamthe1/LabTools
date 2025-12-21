"""
Run cross-validation prediction pipeline on body systems.

Configure all parameters below and run to evaluate feature systems against
target systems using cross-validation with multiple seeds.

Usage:
    python predict_and_eval/run_prediction.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, '/home/adamgab/PycharmProjects/LabTools')
os.chdir('/home/adamgab/PycharmProjects/LabTools')

from predict_and_eval.build_results.run_on_systems import BuildResults

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

SAVE_DIR = '/tmp/predict_test_frailty_lipids/'  # Required: set your output directory

# =============================================================================
# FEATURE SYSTEMS TO EVALUATE
# =============================================================================
# Options:
#   - String: existing body system name (e.g., 'sleep', 'blood_lipids')
#   - Dict with CSV path: {'my_features': '/path/to/features.csv'}
#   - Dict with column list: {'custom': ['col1', 'col2', 'col3']}
#   - must be a unique name for each feature system
RUN_LIST = [
    'frailty',  # existing body system
    # {'my_embeddings': '/path/to/embeddings.csv'},  # custom CSV
    # {'custom_features': ['feature1', 'feature2']},  # specific columns
]

# Column type overrides for feature columns (optional)
# Options: "regression", "classification", "ordinal"
RUN_COLUMN_DESCRIPTIONS = {
    # "activity": "classification",
    # "score": "ordinal",
}

# =============================================================================
# TARGET SYSTEMS TO PREDICT
# =============================================================================
# Options:
#   - String: existing body system name - predicts ALL columns in that system
#   - Dict with column list: {'my_targets': ['glucose', 'hba1c', 'insulin']}
#   - Dict with CSV path: {'custom_targets': '/path/to/targets.csv'}

TARGET_SYSTEMS = [
    'blood_tests_lipids',                          # body system (all columns)
    # {'glycemic_markers': ['glucose', 'hba1c']},  # specific columns only
    # {'my_outcomes': '/path/to/outcomes.csv'},    # custom CSV
]

# Column type overrides for target columns (optional)
TARGET_COLUMN_DESCRIPTIONS = {
    # "diabetes_status": "classification",
}

# =============================================================================
# BASELINE CONFIGURATION
# =============================================================================

BASELINE = 'Age_Gender_BMI'  # Baseline system to compare against
CONFOUNDERS = ['age', 'gender', 'bmi']  # Confounders added to feature set

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Available models:
#   - 'LGBM'           : LightGBM (recommended for large datasets)
#   - 'XGB'            : XGBoost
#   - 'LR_ridge'       : Ridge regression
#   - 'LR_lasso'       : Lasso regression
#   - 'LR_elastic'     : Elastic net
#   - 'Logit'          : Logistic regression (for classification)
#   - 'SVM_regression' : Support Vector Regression
#   - 'SVM_classifier' : Support Vector Classifier
#
# Use 'all' to tune across all models, a single model name, or a list of models

MODEL_KEY = 'LGBM'                     # Tune across all models (slowest, best)
# MODEL_KEY = 'LGBM'                   # Single model (faster)
# MODEL_KEY = ['LGBM', 'XGB']          # List of models to try

TUNE_MODELS = False  # True: hyperparameter tuning via RandomizedSearchCV (slower, better)
                     # False: use preset params from model_params.py (faster, for testing)

# =============================================================================
# CROSS-VALIDATION PARAMETERS
# =============================================================================

NUM_SEEDS = 3               # Number of random seeds (more = more stable results, slower)
NUM_SPLITS = 3              # K-fold splits per seed (5 or 10 typical)
STRATIFIED_THRESHOLD = 0.2  # Use stratified folds if minority class < 20% of total
                            # Set to -1 to disable stratified splitting

# =============================================================================
# COMPUTE RESOURCES
# =============================================================================

NUM_THREADS = 16            # Threads per job (for parallel model fitting)
USE_QUEUE = False           # True: use LabQueue for distributed runs
                            # False: run locally (single machine)
USE_CACHE = False           # True: cache merged DataFrames to disk
                            # False: load data directly into memory

# =============================================================================
# DATA HANDLING
# =============================================================================

MERGE_CLOSEST_RESEARCH_STAGE = True  # If True, match on closest research_stage 
                                      # when exact match is unavailable

# =============================================================================
# MAIN - No need to edit below
# =============================================================================

def main():
    """Run the prediction pipeline with configured parameters."""
    
    # Validate save_dir
    if SAVE_DIR == '/path/to/your/results/' or not SAVE_DIR:
        raise ValueError("Please set SAVE_DIR to your output directory")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize builder
    builder = BuildResults()
    
    # Apply configuration
    builder.save_dir = SAVE_DIR
    builder.run_list = RUN_LIST
    builder.run_column_descriptions = RUN_COLUMN_DESCRIPTIONS
    builder.target_systems = TARGET_SYSTEMS
    builder.target_column_descriptions = TARGET_COLUMN_DESCRIPTIONS
    builder.baseline = BASELINE
    builder.confounders = CONFOUNDERS
    builder.model_key = MODEL_KEY
    builder.num_seeds = NUM_SEEDS
    builder.num_splits = NUM_SPLITS
    builder.stratified_minority_threshold = STRATIFIED_THRESHOLD
    builder.num_threads = NUM_THREADS
    builder.merge_closest_research_stage = MERGE_CLOSEST_RESEARCH_STAGE
    builder.use_cache = USE_CACHE
    builder.testing = not TUNE_MODELS  # testing mode uses preset params
    
    # Print configuration summary
    print("=" * 60)
    print("PREDICTION PIPELINE CONFIGURATION")
    print("=" * 60)
    print(f"Save directory: {SAVE_DIR}")
    print(f"Feature systems: {[builder._get_name(r) for r in RUN_LIST]}")
    print(f"Target systems: {len(TARGET_SYSTEMS)} systems")
    print(f"Baseline: {BASELINE}")
    print(f"Models: {MODEL_KEY}")
    print(f"Tune models: {TUNE_MODELS}")
    print(f"Seeds: {NUM_SEEDS}, Splits: {NUM_SPLITS}")
    print(f"Use queue: {USE_QUEUE}")
    print("=" * 60)
    
    # Run pipeline
    builder.run(with_queue=USE_QUEUE)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {SAVE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
