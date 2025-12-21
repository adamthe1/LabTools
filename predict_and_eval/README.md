# Predict and Evaluate

Cross-validation prediction pipeline for evaluating feature systems against body system targets.

## Overview

This tool runs cross-validated predictions using your features to predict targets from various body systems. It compares your feature system against a baseline (e.g., Age/Gender/BMI) and generates summary statistics with statistical significance testing.

## Environment Setup

### 1. Activate Conda Environment

```bash
# Using conda
source /specific/netapp5_3/d/segal_lab/miniforge3/etc/profile.d/conda.sh
conda activate gait

# Or using mamba (faster)
source /specific/netapp5_3/d/segal_lab/miniforge3/etc/profile.d/mamba.sh
mamba activate gait
```

### 2. Set Environment Variables

Create a `.env` file in the project root with:

```bash
BODY_SYSTEMS="/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/body_systems"
TEMP_SYSTEMS="/net/mraid20/ifs/wisdom/segal_lab/genie/LabData/Analyses/10K_Trajectories/body_systems/temp_systems"
```

## Quick Start

1. Edit `run_prediction.py` with your configuration
2. Run the pipeline:

```bash
python predict_and_eval/run_prediction.py
```

## Configuration Reference

### Output Configuration

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `SAVE_DIR` | str | Yes | Output directory for results |

### Feature Systems (RUN_LIST)

Define which features to evaluate. Options:

```python
RUN_LIST = [
    'sleep',                                        # Existing body system
    {'my_embeddings': '/path/to/embeddings.csv'},  # Custom CSV file
    {'custom_features': ['col1', 'col2', 'col3']}, # Specific columns from any system
]
```

### Target Systems (TARGET_SYSTEMS)

Define what to predict. Options:

```python
TARGET_SYSTEMS = [
    'blood_lipids',                              # Body system (all columns)
    {'glycemic': ['glucose', 'hba1c']},         # Specific columns only
    {'outcomes': '/path/to/targets.csv'},       # Custom CSV file
]
```

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MODEL_KEY` | str/list | `'all'` | Model(s) to use |
| `TUNE_MODELS` | bool | `True` | Whether to tune hyperparameters |

**Available Models:**

| Model Key | Description | Best For |
|-----------|-------------|----------|
| `LGBM` | LightGBM | Large datasets, fast training |
| `XGB` | XGBoost | General purpose |
| `LR_ridge` | Ridge regression | Linear relationships |
| `LR_lasso` | Lasso regression | Feature selection |
| `LR_elastic` | Elastic net | Mixed L1/L2 regularization |
| `Logit` | Logistic regression | Classification |
| `SVM_regression` | Support Vector Regression | Small datasets |

**Examples:**

```python
MODEL_KEY = 'all'              # Tune across all models (slowest, best results)
MODEL_KEY = 'LGBM'             # Single model (faster)
MODEL_KEY = ['LGBM', 'XGB']    # List of models to compare
```

### Cross-Validation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUM_SEEDS` | int | `20` | Random seeds for stability |
| `NUM_SPLITS` | int | `5` | K-fold CV splits per seed |
| `STRATIFIED_THRESHOLD` | float | `0.2` | Use stratified if minority < 20% |

### Compute Resources

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `NUM_THREADS` | int | `16` | Threads per job |
| `USE_QUEUE` | bool | `True` | Use LabQueue for distributed runs |
| `USE_CACHE` | bool | `False` | Cache merged DataFrames to disk |

### Data Handling

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `MERGE_CLOSEST_RESEARCH_STAGE` | bool | `True` | Match on closest visit if exact unavailable |
| `CONFOUNDERS` | list | `['age', 'gender', 'bmi']` | Variables added to feature set |

## Output Structure

```
SAVE_DIR/
├── queue_logs/                 # LabQueue logs (if using queue)
├── target_system_name/
│   ├── label_name/
│   │   ├── feature_name/
│   │   │   ├── predictions.csv  # Cross-validated predictions
│   │   │   ├── metrics.csv      # Performance metrics per seed
│   │   │   └── example_model_seed_0.pkl  # Trained model
│   │   └── Age_Gender_BMI/      # Baseline results
│   │       └── ...
│   └── system_summary.csv       # Summary for this target system
└── all_comparisons.csv          # Full comparison summary
```

### Output Files

**predictions.csv**: Cross-validated predictions with columns:
- `RegistrationCode`, `research_stage`: Subject identifiers
- `prediction`: Model prediction
- `true_value`: Actual value
- `seed`: Random seed used

**metrics.csv**: Performance metrics per seed:
- `r2`, `pearson_r`, `pearson_pvalue`: Regression metrics
- `auc`, `accuracy`: Classification metrics
- `spearman_r`, `spearman_pvalue`: Ordinal metrics
- `n_subjects`: Number of subjects

**all_comparisons.csv**: Summary with statistical tests:
- `delta`: Improvement over baseline
- `wilcox_pvalue`: Wilcoxon signed-rank test p-value
- `wilcox_pvalue_fdr`: FDR-corrected p-value
- `is_significant`: Whether improvement is significant

## Example Configurations

### Basic: Evaluate embeddings on blood biomarkers

```python
SAVE_DIR = '/path/to/results/embeddings_vs_blood'

RUN_LIST = [
    {'my_embeddings': '/path/to/embeddings.csv'},
]

TARGET_SYSTEMS = [
    'blood_lipids',
    'glycemic_status',
    'liver',
]

MODEL_KEY = 'LGBM'
TUNE_MODELS = True
NUM_SEEDS = 20
USE_QUEUE = True
```

### Advanced: Multiple feature systems, specific targets

```python
SAVE_DIR = '/path/to/results/multi_comparison'

RUN_LIST = [
    'sleep',
    {'embeddings_v1': '/path/to/v1.csv'},
    {'embeddings_v2': '/path/to/v2.csv'},
]

TARGET_SYSTEMS = [
    {'metabolic': ['glucose', 'hba1c', 'insulin', 'triglycerides']},
    {'cardiovascular': ['systolic_bp', 'diastolic_bp', 'heart_rate']},
]

MODEL_KEY = ['LGBM', 'XGB', 'LR_elastic']
TUNE_MODELS = True
NUM_SEEDS = 10  # Fewer seeds for faster iteration
USE_QUEUE = True
```

### Quick Testing: Fast local run

```python
SAVE_DIR = '/tmp/test_run'

RUN_LIST = ['sleep']
TARGET_SYSTEMS = ['Age_Gender_BMI']

MODEL_KEY = 'LGBM'
TUNE_MODELS = False  # Use presets
NUM_SEEDS = 3
NUM_SPLITS = 3
USE_QUEUE = False  # Run locally
```

## Troubleshooting

### LabQueue not available

If you see `ImportError: LabQueue is required to run with queue`, either:
- Set `USE_QUEUE = False` to run locally
- Ensure LabQueue is installed and accessible

### Column not found

If columns are missing, they will be automatically filtered out with a warning. Check that:
- Column names match exactly (case-sensitive)
- The body system containing the column exists

### Insufficient subjects

If you see "insufficient_subjects" in the output, the target has too few valid subjects for cross-validation. Try:
- Increasing the target population
- Reducing `NUM_SPLITS`
