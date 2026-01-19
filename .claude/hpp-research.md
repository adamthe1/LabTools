# HPP Dataset Research Skill

You are an expert data scientist working with the Human Phenotype Project (HPP) / 10K dataset. This skill provides comprehensive guidelines for conducting rigorous, reproducible, publication-quality research.

## Environment Setup

Always activate the conda environment before running Python:

```bash
source /home/adamgab/miniconda3/etc/profile.d/conda.sh && conda activate NewtonModels && export PYTHONPATH=/home/adamgab/PycharmProjects/LabTools
```

For running Python files:
- Regular scripts: `python <file.py>`
- Modules: `python -m <module.path>`

## Core Principles

### 1. Data Leakage Prevention (CRITICAL)

**NEVER split data at the row level. ALWAYS split at the subject level.**

The same subject (RegistrationCode) can have multiple samples (different research stages/visits). If samples from the same subject appear in both train and test sets, information leaks between splits.

```python
# WRONG - Row-level split causes data leakage
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)  # NEVER DO THIS

# CORRECT - Subject-level split using ids_folds
from predict_and_eval.utils.ids_folds import ids_folds, create_cv_folds

# Create subject-level fold definitions
id_folds = ids_folds(df_with_labels, seeds=range(10), n_splits=5)

# Convert to sample indices for your data
subject_ids = df.index.get_level_values(0).values
cv_folds = create_cv_folds(id_folds[seed_index], subject_ids)

# Use the folds
for train_idx, test_idx in cv_folds:
    X_train, X_test = X[train_idx], X[test_idx]
```

### 2. Subject ID Understanding

These identifiers all refer to the same thing - a unique participant:
- `RegistrationCode` - Primary identifier in body systems data
- `subject_id` - Alternative name
- `10K_id` - Alternative name

Format: `"10K_" + {10 digits}` (e.g., "10K_1234567890")

**Note:** Some datasets may have transition IDs from original tests - these are different and should be mapped or excluded.

### 3. Data Index Structure

All body system DataFrames use a MultiIndex: `(RegistrationCode, research_stage)`

```python
# Example index structure
# RegistrationCode    research_stage
# 10K_0000000001      baseline
# 10K_0000000001      followup_1
# 10K_0000000002      baseline
```

## Data Access

### Loading Data from Body Systems

Use the `body_system_loader` module:

```python
from body_system_loader.load_feature_df import (
    load_body_system_df,           # Load entire body system
    load_columns_as_df,            # Load specific columns from any system
    load_feature_target_systems_as_df,  # Load feature + target systems merged
    get_body_system_column_names,  # List columns in a system
    filter_existing_columns,       # Check which columns exist
)

# Load a body system
df = load_body_system_df('cardiovascular_system')

# Load specific columns from any system
df = load_columns_as_df(['Age', 'BMI', 'HbA1c'])

# Load features and target together
df = load_feature_target_systems_as_df(
    feature_system='blood_lipids',
    target_system='cardiovascular_system',
    confounders=['Age', 'gender']
)
```

### Available Body Systems

```
Age_Gender_BMI, blood_lipids, blood_tests_lipids, body_composition,
bone_density, cardiovascular_system, diet, diet_questions,
exercise_logging, family_history, family_medical_conditions, frailty,
gait, glycemic_status, hematopoietic, immune_system, lifestyle, liver,
medical_conditions, medications, mental, metabolites, microbiome,
nightingale, proteomics, renal_function, rna, sleep, study_type
```

### Exploring Biomarkers

Use the BiomarkerBrowser for understanding available targets without deep data exploration:

```python
from body_system_loader.biomarker_browser import BiomarkerBrowser

browser = BiomarkerBrowser()

# List all systems
systems = browser.list_systems()

# Get system summary (column counts by type)
summary = browser.get_system_summary()

# Get detailed info about specific systems
info = browser.get_system_info(['frailty', 'cardiovascular_system'])

# Filter by column type
regression_cols = browser.get_columns_by_type('proteomics', 'regression')
binary_cols = browser.get_columns_by_type('medical_conditions', 'binary_classification')

# Search for biomarkers
diabetes_related = browser.search_columns('diabetes')

# Get classification targets with sufficient samples
targets = browser.get_classification_targets(min_positives=100)

# Get regression targets with sufficient variability
targets = browser.get_regression_targets(min_unique=50)
```

## Data Exploration Guidelines

**IMPORTANT: Do NOT look deep into individual data points. Use summary statistics.**

```python
# CORRECT - Summary statistics
df.describe()
df.isna().sum()
df.nunique()
df.dtypes

# CORRECT - Use biomarker browser
browser.get_system_summary()
browser.get_column_details('frailty', 'grip_strength')

# AVOID - Looking at individual rows (unless debugging specific issues)
df.head(100)  # Only when necessary
df.iloc[0]    # Only when necessary
```

## Cross-Validation Workflow

### Standard K-Fold CV with Hyperparameter Optimization

```python
from predict_and_eval.utils.ids_folds import ids_folds, id_fold_with_stratified_threshold
from predict_and_eval.regression_seeding.Regressions import Regressions

# 1. Load data
from body_system_loader.load_feature_df import load_feature_target_systems_as_df
df = load_feature_target_systems_as_df(feature_system, target_system, confounders)

# 2. Prepare X and y
X = df[feature_columns]
y = df[[target_column]]

# 3. Create subject-level folds (10 seeds x 5 folds = 50 train/test splits)
# Use stratified folds for imbalanced classification (auto-detects)
id_folds = id_fold_with_stratified_threshold(y, seeds=range(10), n_splits=5)

# 4. Run cross-validation with automatic model selection
regressions = Regressions()

# For a single seed
cv_fold = id_folds[0]  # Use seed 0
result = regressions.cross_validate_model(X, y, cv_fold, model_key='all')

# model_key options:
# - 'all': Tune across all models (LR_ridge + LGBM for regression, Logit + LGBM for classification)
# - 'LR_ridge': Linear model only
# - 'LGBM_regression': Tree model only

# 5. Evaluate predictions
eval_result = regressions.evaluate_predictions(
    X, y,
    result['predictions'],
    gender_split_evaluation=True,
    average_by_subject_id=True
)

print(f"Pearson R: {eval_result['metrics']['pearson_r']:.3f}")
print(f"RMSE: {eval_result['metrics']['rmse']:.3f}")
```

### Model Types

**Regression:**
- `LR_ridge` - Ridge regression (linear)
- `LGBM_regression` - LightGBM (tree-based)

**Classification:**
- `Logit` - Logistic regression (linear)
- `LGBM_classifier` - LightGBM classifier (tree-based)

**Ordinal:**
- `Ordinal_logit` - Ordinal logistic regression

### Label Type Detection

Label types are auto-detected from the target column:
- `regression` - Continuous numeric values
- `categorical` - Binary (2 unique values) or multi-class
- `ordinal` - Ordered categories (detected by column name or values)

## Scientific Rigor Requirements

### 1. Reproducibility
- Always use fixed random seeds
- Save fold definitions with `save_folds()`
- Document all preprocessing steps

### 2. Statistical Validity
- Report confidence intervals (use multiple seeds)
- Use appropriate metrics for task type
- Account for multiple comparisons when testing many targets

### 3. Proper Train/Test Separation
- Fit preprocessing (scaling, imputation) on train only
- The pipeline in `Regressions` handles this automatically

### 4. Reporting
- Always report n_subjects, not n_samples
- For classification, report n_positives
- Split results by gender when relevant

## Creating New Research Tools

When developing new analysis tools (e.g., batch effect analysis, new visualization types):

1. Create a new folder in LabTools:
```bash
mkdir -p /home/adamgab/PycharmProjects/LabTools/<tool_name>
```

2. Create `__init__.py` and main module files

3. Create a corresponding skill file:
```bash
/home/adamgab/PycharmProjects/LabTools/.claude/<tool_name>.md
```

4. Document the tool's purpose, usage, and integration with existing codebase

## Quick Reference

### Common Data Loading Patterns

```python
# Load demographics + one body system
from body_system_loader.load_feature_df import load_columns_as_df, load_body_system_df

demographics = load_columns_as_df(['Age', 'gender', 'BMI'])
features = load_body_system_df('cardiovascular_system')
df = demographics.join(features, how='inner')
```

### Checking Data Availability

```python
from body_system_loader.load_feature_df import filter_existing_columns

requested = ['Age', 'BMI', 'NonExistentColumn']
available = filter_existing_columns(requested)
# Warns about missing columns, returns only valid ones
```

### Environment Variables Required

Set in `.env` file:
```
BODY_SYSTEMS=/path/to/body_systems
JAFAR_BASE=/path/to/jafar  # Optional
```

## Anti-Patterns to Avoid

1. **Row-level splitting** - Always use subject-level folds
2. **Looking at test data** - Never peek at test set distributions
3. **Fitting on full data** - Preprocessing must be train-only
4. **Ignoring subject duplicates** - Same person can have multiple visits
5. **Hardcoding paths** - Use environment variables and config
6. **Skipping validation** - Always verify your splits don't leak

## Metrics Reference

### Regression
- `pearson_r` - Pearson correlation
- `rmse` - Root mean squared error
- `mae` - Mean absolute error
- `r2` - R-squared

### Classification
- `auroc` - Area under ROC curve
- `auprc` - Area under precision-recall curve
- `accuracy` - Classification accuracy
- `f1` - F1 score

### Ordinal
- `spearman_r` - Spearman correlation
- `mae` - Mean absolute error
