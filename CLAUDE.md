# LabTools - HPP Dataset Research

## Quick Start

Activate environment before any Python:
```bash
source /home/adamgab/miniconda3/etc/profile.d/conda.sh && conda activate NewtonModels && export PYTHONPATH=/home/adamgab/PycharmProjects/LabTools
```

## Critical Rules

### 1. Data Leakage Prevention
**ALWAYS split on RegistrationCode (subject) level, NEVER on row level.**

```python
# CORRECT
from predict_and_eval.utils.ids_folds import ids_folds, create_cv_folds
id_folds = ids_folds(df_with_labels, seeds=range(10), n_splits=5)

# WRONG - causes data leakage
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, y)  # NEVER DO THIS
```

### 2. Subject IDs
- `RegistrationCode`, `subject_id`, `10K_id` are the same thing
- Format: `"10K_" + {10 digits}`
- Data index: `(RegistrationCode, research_stage)` MultiIndex

### 3. Data Exploration
- Use summary statistics and `BiomarkerBrowser`, not deep data inspection
- Everything must be verifiable and scientific article level

### 4. Age Distribution Warning
**The HPP 10K dataset has HIGHLY UNEVEN age distribution:**
- 94% of subjects are aged 40-70 years
- Ages <40 and >75 have very low sample sizes (unreliable estimates)
- **ALWAYS limit age-related analyses to 40-70** unless specifically studying young/old populations
- When analyzing age effects, always check sample size per age bin first

```python
# Check age distribution before any analysis
age_bins = [40, 45, 50, 55, 60, 65, 70]
df['age_bin'] = pd.cut(df['age'], bins=age_bins, right=False)
print(df.groupby('age_bin', observed=True).size())  # Should have n>500 per bin
```

## Key Modules

### Data Loading (`body_system_loader/`)
```python
from body_system_loader.load_feature_df import (
    load_body_system_df,           # Load entire body system
    load_columns_as_df,            # Load specific columns
    load_feature_target_systems_as_df,  # Load feature + target merged
)
from body_system_loader.biomarker_browser import BiomarkerBrowser
```

### Cross-Validation (`predict_and_eval/`)
```python
from predict_and_eval.utils.ids_folds import ids_folds, id_fold_with_stratified_threshold
from predict_and_eval.regression_seeding.Regressions import Regressions

# Auto model selection between linear (LR_ridge/Logit) and tree (LGBM)
regressions = Regressions()
result = regressions.cross_validate_model(X, y, cv_fold, model_key='all')
eval_result = regressions.evaluate_predictions(X, y, result['predictions'])
```

## Creating New Research Tools

When creating new analysis capabilities:
1. Create folder: `LabTools/<tool_name>/`
2. Create skill: `.claude/<tool_name>.md`
3. Document integration with existing patterns

## Available Skills

| Skill | File | Description |
|-------|------|-------------|
| Data Quality Check | `.claude/data-quality-check.md` | Check distributions of age, gender, BMI; choose correct ranges; when to stratify |
| Disease Groups | `.claude/disease-groups.md` | WHO-based disease groupings, helper functions, color schemes |
| Study Cohorts | `.claude/study-cohorts.md` | HPP cohort types (10K, BRCA, T1D, etc.); when to filter/stratify by cohort |
| Family History | `.claude/family-history.md` | Family medical conditions (48 features) and background (13 features); hereditary risk |
| Create Report | `.claude/create-report.md` | Guidelines for creating clean PDF reports with figures |

## Available Body Systems
Age_Gender_BMI, blood_lipids, body_composition, bone_density, cardiovascular_system,
diet, family_history, family_medical_conditions, frailty, gait, glycemic_status,
hematopoietic, immune_system, lifestyle, liver, medical_conditions, medications,
mental, metabolites, microbiome, nightingale, proteomics, renal_function, rna,
sleep, study_type

See `.claude/hpp-research.md` for complete documentation.
