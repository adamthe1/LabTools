# Skill: Family Medical History

Family history data capturing hereditary risk factors and personal background for predicting disease susceptibility.

## Overview

Two body systems provide family history information:

| Body System | Features | Description |
|-------------|----------|-------------|
| `family_medical_conditions` | 48 | Binary indicators for 1st-degree relative conditions |
| `family_history` | 13 | Personal/family background and early-life factors |

---

## family_medical_conditions

Binary indicators for whether **any 1st-degree relative** (parents, siblings, or children) has each medical condition. Captures hereditary risk factors clinically relevant for disease prediction.

### Loading Data

```python
from body_system_loader.load_feature_df import load_body_system_df, load_columns_as_df

# Load all family medical conditions
family_conditions = load_body_system_df('family_medical_conditions')

# Load specific conditions
cardiovascular_family = load_columns_as_df([
    'Hypertension_family',
    'Ischemic Heart Disease_family',
    'Stroke_family'
])
```

### Condition Categories by Prevalence

| Category | Prevalence | Key Conditions |
|----------|------------|----------------|
| Cardiovascular | ~38% | Hypertension, Ischemic Heart Disease, Stroke |
| Metabolic | ~28% | Type 2 Diabetes, Hyperlipidemia |
| Autoimmune | ~13% | Autoimmune disorders, Asthma, Coeliac, Crohn |
| Cancer | ~8% | Breast, Lung, Colon, Prostate, Melanoma, Ovarian |
| Neurological | ~6.5% | Alzheimer, Parkinson |

### Available Conditions

```python
FAMILY_CONDITIONS = {
    'Cardiovascular': [
        'Hypertension_family',
        'Ischemic Heart Disease_family',
        'Stroke_family',
        'Heart Failure_family',
        'Arrhythmia_family'
    ],

    'Metabolic': [
        'Type 2 Diabetes_family',
        'Type 1 Diabetes_family',
        'Hyperlipidemia_family',
        'Obesity_family'
    ],

    'Cancer': [
        'Breast Cancer_family',
        'Lung Cancer_family',
        'Colon Cancer_family',
        'Prostate Cancer_family',
        'Melanoma_family',
        'Ovarian Cancer_family',
        'Pancreatic Cancer_family',
        'Stomach Cancer_family',
        'Liver Cancer_family',
        'Kidney Cancer_family',
        'Bladder Cancer_family',
        'Lymphoma_family',
        'Leukemia_family'
    ],

    'Neurological': [
        'Alzheimer_family',
        'Parkinson_family',
        'Dementia_family'
    ],

    'Autoimmune': [
        'Autoimmune disorders_family',
        'Asthma_family',
        'Coeliac_family',
        'Crohn_family',
        'Rheumatoid Arthritis_family'
    ],

    'Mental Health': [
        'Depression_family',
        'Anxiety_family',
        'Schizophrenia_family',
        'Bipolar_family'
    ],

    'Other': [
        'Osteoporosis_family',
        'Glaucoma_family',
        'Kidney Disease_family'
    ]
}
```

### Grouped Conditions

Pre-computed grouped indicators for increased statistical power:

```python
GROUPED_CONDITIONS = [
    'Cardiovascular_family',    # Any cardiovascular condition
    'Cancer_any_family',        # Any cancer
    'Metabolic_family',         # Any metabolic condition
    'Neurological_family',      # Any neurological condition
    'Autoimmune_family',        # Any autoimmune condition
    'Mental_family',            # Any mental health condition
    'Any_family_condition'      # Any condition at all
]
```

### Data Characteristics

- **Propagation:** Family history is propagated forward across research stages (family history doesn't disappear)
- **1st-degree relatives:** Parents, siblings, children only
- **Binary format:** 1 = at least one relative has condition, 0 = no relatives have condition
- **Missing data:** Subjects who didn't complete family history questionnaire will have NaN

---

## family_history

Personal and family background features including demographics, early-life factors, and physical traits.

### Loading Data

```python
from body_system_loader.load_feature_df import load_body_system_df

family_background = load_body_system_df('family_history')
```

### Available Features

| Feature | Type | Description |
|---------|------|-------------|
| `mother_alive` | Binary | Mother living status |
| `father_alive` | Binary | Father living status |
| `mother_age_at_death` | Numeric | Mother's age at death (if applicable) |
| `father_age_at_death` | Numeric | Father's age at death (if applicable) |
| `n_siblings` | Numeric | Number of siblings |
| `n_children` | Numeric | Number of children |
| `birth_order` | Numeric | Birth order among siblings |
| `weight_at_10` | Categorical | Self-reported weight at age 10 (underweight/normal/overweight) |
| `height_at_10` | Categorical | Self-reported height at age 10 (short/average/tall) |
| `hair_color` | Categorical | Natural hair color |
| `skin_color` | Categorical | Skin tone |
| `handedness` | Categorical | Left/right/ambidextrous |
| `breastfed` | Binary | Whether breastfed as infant |

### Data Characteristics

- **Missing values:** ~46% missing - not all subjects completed this questionnaire
- **Self-reported:** Early-life factors are retrospective self-reports
- **Use cases:** Environmental/genetic context, longevity studies, early-life programming research

---

## Research Applications

### Hereditary Risk Prediction

```python
# Predict diabetes risk using family history
from body_system_loader.load_feature_df import load_feature_target_systems_as_df

df = load_feature_target_systems_as_df(
    feature_system='family_medical_conditions',
    target_system='medical_conditions',
    confounders=['Age', 'gender', 'BMI']
)

# Family history of diabetes as predictor
X = df[['Type 2 Diabetes_family', 'Metabolic_family', 'Age', 'gender', 'BMI']]
y = df['Diabetes']
```

### Parental Longevity Studies

```python
# Study association between parental age at death and offspring health
family_bg = load_body_system_df('family_history')
health = load_body_system_df('cardiovascular_system')

df = family_bg.join(health, how='inner')

# Filter to subjects with deceased parents
df_analysis = df[df['father_alive'] == 0]
# Analyze father_age_at_death vs offspring cardiovascular measures
```

### Cancer Risk Stratification

```python
# Identify subjects with strong family cancer history
family_cond = load_body_system_df('family_medical_conditions')

# Multiple family members with cancer (using grouped indicator)
high_risk = family_cond[family_cond['Cancer_any_family'] == 1]

# Specific cancer types
breast_cancer_family = family_cond[family_cond['Breast Cancer_family'] == 1]
```

### Early-Life Factors Analysis

```python
# Study early-life factors and adult health
family_bg = load_body_system_df('family_history')
body_comp = load_body_system_df('body_composition')

df = family_bg.join(body_comp, how='inner')

# Weight at age 10 vs adult BMI
df.groupby('weight_at_10')['BMI'].describe()
```

---

## Helper Functions

```python
def get_family_condition_category(condition):
    """Get category for a family condition."""
    for category, conditions in FAMILY_CONDITIONS.items():
        if condition in conditions:
            return category
    return 'Other'


def filter_subjects_with_family_history(df, condition):
    """Filter to subjects with family history of condition."""
    col = f"{condition}_family" if not condition.endswith('_family') else condition
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")
    return df[df[col] == 1]


def get_family_risk_score(df, conditions):
    """Calculate simple family risk score (count of family conditions)."""
    cols = [c if c.endswith('_family') else f"{c}_family" for c in conditions]
    available = [c for c in cols if c in df.columns]
    return df[available].sum(axis=1)


def check_family_history_coverage(df):
    """Check what proportion of subjects have family history data."""
    family_cols = [c for c in df.columns if c.endswith('_family')]
    if not family_cols:
        print("No family history columns found")
        return

    # Check non-null rate
    coverage = df[family_cols].notna().any(axis=1).mean()
    print(f"Family history coverage: {coverage*100:.1f}%")

    # Per-condition coverage
    for col in sorted(family_cols):
        n_valid = df[col].notna().sum()
        n_positive = (df[col] == 1).sum()
        if n_valid > 0:
            prevalence = n_positive / n_valid * 100
            print(f"  {col}: n={n_valid:,}, positive={n_positive:,} ({prevalence:.1f}%)")
```

---

## Important Considerations

### Missing Data

```python
# Check family history completeness before analysis
def check_family_data_completeness(df):
    """Assess missing data in family history."""
    family_cols = [c for c in df.columns if '_family' in c]

    n_total = len(df)
    n_with_data = df[family_cols].notna().any(axis=1).sum()

    print(f"Total subjects: {n_total:,}")
    print(f"With family history: {n_with_data:,} ({n_with_data/n_total*100:.1f}%)")
    print(f"Missing family history: {n_total - n_with_data:,} ({(n_total-n_with_data)/n_total*100:.1f}%)")

    return n_with_data / n_total

# ~46% missing is expected for family_history system
```

### Recall Bias

- Early-life factors (weight_at_10, height_at_10) are self-reported retrospectively
- Family conditions depend on subject's knowledge of relatives' health
- May underreport conditions in relatives not in regular contact

### Sex-Specific Conditions

```python
# Some conditions are sex-specific in relatives too
FEMALE_RELATIVE_CONDITIONS = [
    'Breast Cancer_family',
    'Ovarian Cancer_family'
]

MALE_RELATIVE_CONDITIONS = [
    'Prostate Cancer_family'
]

# These may have different interpretations based on family composition
```

### Combining with Personal Medical History

```python
# Compare personal vs family history
personal = load_body_system_df('medical_conditions')
family = load_body_system_df('family_medical_conditions')

df = personal.join(family, how='inner', lsuffix='', rsuffix='_fam')

# Subjects with diabetes AND family history
diabetes_with_family = df[(df['Diabetes'] == 1) & (df['Type 2 Diabetes_family'] == 1)]

# Subjects with diabetes but NO family history (potential sporadic cases)
diabetes_no_family = df[(df['Diabetes'] == 1) & (df['Type 2 Diabetes_family'] == 0)]
```

---

## Quick Reference

### Load Commands

```python
# All family medical conditions
family_cond = load_body_system_df('family_medical_conditions')

# All family background
family_bg = load_body_system_df('family_history')

# Specific family conditions
cardio_family = load_columns_as_df(['Cardiovascular_family', 'Hypertension_family'])
```

### Common Filters

```python
# Subjects with any family cancer history
cancer_family = df[df['Cancer_any_family'] == 1]

# Subjects with cardiovascular family history
cvd_family = df[df['Cardiovascular_family'] == 1]

# Subjects with complete family history data
has_family_data = df[df['Type 2 Diabetes_family'].notna()]
```

### Prevalence Summary

| Grouped Condition | Approximate Prevalence |
|-------------------|------------------------|
| Cardiovascular_family | ~38% |
| Metabolic_family | ~28% |
| Autoimmune_family | ~13% |
| Cancer_any_family | ~8% |
| Neurological_family | ~6.5% |
